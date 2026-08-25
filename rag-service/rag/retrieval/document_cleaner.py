"""数据清洗层 — 五步清洗 + 无损归一化（module-064 / ADR-0014 WP2/WP3）

在整个 ingestion 链路中的位置：
  解析层 Markdown → [清洗层 clean()] → [归一化 normalize()] → 分块 → 嵌入

五步清洗（clean）：
  ① 格式清理 —— 去页眉页脚/页码/水印残留/控制字符，Unicode NFKC 统一标点
  ② 冗余过滤 —— 无意义短段落/纯符号段落过滤（可配置长度下限）
  ③ 结构恢复 ⭐ —— 合并 PDF 断行切碎的段落、还原标题层级（`#` → `##` 对齐
     MarkdownHeaderTextSplitter）、标题/表格前补空行
  ④ 语义修复 —— 规则级（OCR 常见错字表 OCR_TYPO_MAP 留空待补，诚实声明）
  ⑤ 分块准备 —— 标题层级规范化对齐 chunker 的 [("##","section"),("###","subsection")]

无损归一化（normalize，WP3）：
  NFKC / 去零宽与不可见控制符 / 统一空白（块外折叠多空格+去行尾空白）/ 表格保持
  Markdown 表格 / 超长截断（chunker 已把父块限制 4000/子块 300 防 embedding 截断，
  此处 max_chars 为病态超长文档的兜底，默认 None 不截）。

白名单哲学（关键纪律，ADR-0014 决策 2）：
  清洗规则按块类型作用——代码块（```/~~~ 围栏）、行内代码（`...`）、表格（| 行）、
  行内数学（$...$）、显示数学（$$/\\begin 块）、URL 全部用区域/占位符保护，不误伤：
    代码块符号不清、表格合并单元格不拆、URL/LaTeX 原样保留。
"""
import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ── 块级信号 ────────────────────────────────────────────────────────────
_CODE_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s|\S)")           # 至少 # 开头
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?")
_LIST_MARKER_RE = re.compile(r"^\s{0,3}(?:[-*+]\s|\d+[.)]\s)")
_THEMATIC_BREAK_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_LATEX_ENV_RE = re.compile(r"^\s*\\begin\{[^}]*\}")
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://[^\s<>\"'，。；）)\]，]+")

# 页眉页脚/页码残留（PDF 提取常见噪声；水印因语境相关不做自动删——诚实声明）
# `--- Page i/N ---` 是 PyMuPDF 回退路径（document_parser._parse_pdf_pymupdf）的
# 分页标记，属噪声一并移除。
_PAGE_FURNITURE_RE = re.compile(
    r"^(?:第\s*\d+\s*页"
    r"|page\s*\d+(?:\s*(?:of|/)\s*\d+)?"
    r"|-{3,}\s*page\s*\d+(?:\s*(?:of|/)\s*\d+)?[\s-]*"
    r"|页码[:：]\s*\d+)\s*$",
    re.IGNORECASE,
)

# 纯符号噪声（冗余过滤）
_NOISE_CHARS = set("…-·—_|#*~`<>{}[]()")

# 句末标点（PDF 断行合并判定）
_SENTENCE_END = set("。！？；.!?;:")
# 中文字符区间
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# 语义修复：OCR 常见错字表（规则级，先留空待补——OCR 组件默认关暂无错字语料，
# 诚实声明：需要标注语料后填充，当前为空表零生效）
OCR_TYPO_MAP: dict[str, str] = {}


def _strip_control(text: str) -> str:
    """去控制字符（保留 \n / \t；含零宽、方向符等不可见 C 类字符）"""
    return "".join(
        ch for ch in text
        if ch in "\n\t" or (unicodedata.category(ch)[0] != "C")
    )


def _is_page_furniture(stripped: str) -> bool:
    """判断是否为页码/页脚残留行"""
    if _PAGE_FURNITURE_RE.match(stripped):
        return True
    s = stripped.strip("-·.—:：")
    return bool(s.isdigit() and len(s) <= 4)


def _is_noise(text: str) -> bool:
    """冗余过滤：无意义段落判定

    - 空/纯空白
    - 不含 CJK/字母/数字（纯标点符号）
    - 全部由噪声符号组成（如 "…", "----", "····"）
    """
    t = text.strip()
    if not t:
        return True
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t):
        return True
    if all(ch in _NOISE_CHARS for ch in t):
        return True
    return False


def _tokenize_regions(md: str) -> list[tuple[str, list[str]]]:
    """把 Markdown 切成区域列表 [(type, lines), ...]

    type:
      code  —— ``` / ~~~ 围栏代码块（内容原样保留）
      math  —— $$ 显示数学 / \\begin{env} LaTeX 块（内容原样保留）
      table —— 连续 | 行表格（结构保留，不拆合并单元格）
      body  —— 其余正文区域（应用全部清洗规则）
    """
    lines = md.split("\n")
    regions: list[tuple[str, list[str]]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        # 围栏代码块
        m = _CODE_FENCE_RE.match(line)
        if m:
            fence = m.group(1)
            block = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i])
                i += 1
            regions.append(("code", block))
            continue
        # 显示数学 $$...$$
        if stripped.startswith("$$"):
            block = [line]
            i += 1
            while i < n and "$$" not in lines[i]:
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i])
                i += 1
            regions.append(("math", block))
            continue
        # LaTeX 环境块
        if _LATEX_ENV_RE.match(line):
            block = [line]
            i += 1
            while i < n and not re.search(r"\\end\{", lines[i]):
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i])
                i += 1
            regions.append(("math", block))
            continue
        # 表格块（连续 | 行）
        if _TABLE_LINE_RE.match(line):
            block = []
            while i < n and _TABLE_LINE_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            regions.append(("table", block))
            continue
        # 正文区域（收集到下一个区域起点为止）
        block = []
        while i < n:
            ln = lines[i]
            if (_CODE_FENCE_RE.match(ln) or ln.strip().startswith("$$")
                    or _LATEX_ENV_RE.match(ln) or _TABLE_LINE_RE.match(ln)):
                break
            block.append(ln)
            i += 1
        regions.append(("body", block))
    return regions


# ── 正文内联元素保护（行内代码 / 行内数学 / URL） ───────────────────────
_INLINE_PROTECT_RE = re.compile(
    r"(`[^`\n]+`|\$[^$\n]+\$|https?://[^\s<>\"'，。；）\]，]+)"
)
# 占位符定界符：⟦⟧（U+27E6/27E7，数学开/闭括号，非控制符——能通过 _strip_control
# 的 C 类过滤，且 NFKC 稳定）。行内代码/URL/数学必须先于 NFKC 保护（否则全角
# 括号/空格会被 NFKC 改写，破坏代码/URL 原样——白名单哲学）。
_PROTECT_OPEN, _PROTECT_CLOSE = "⟦", "⟧"


def _protect_inline(line: str) -> tuple[str, list[str]]:
    """把行内代码/行内数学/URL 替换为占位符，返回 (保护后行, 原始片段列表)"""
    parts: list[str] = []
    def _repl(m: re.Match) -> str:
        parts.append(m.group(0))
        return f"{_PROTECT_OPEN}{len(parts) - 1}{_PROTECT_CLOSE}"
    return _INLINE_PROTECT_RE.sub(_repl, line), parts


def _restore_inline(line: str, parts: list[str]) -> str:
    def _repl(m: re.Match) -> str:
        idx = int(m.group(1))
        return parts[idx] if idx < len(parts) else ""
    return re.sub(
        rf"{re.escape(_PROTECT_OPEN)}(\d+){re.escape(_PROTECT_CLOSE)}",
        _repl, line,
    )


def _starts_block(stripped: str) -> bool:
    """行首是否是块级信号（不参与 PDF 断行合并）"""
    return bool(
        _HEADING_RE.match(stripped)
        or _LIST_MARKER_RE.match(stripped)
        or _BLOCKQUOTE_RE.match(stripped)
        or _THEMATIC_BREAK_RE.match(stripped)
        or _CODE_FENCE_RE.match(stripped)
        or _TABLE_LINE_RE.match(stripped)
        or stripped.startswith("$$")
        or _LATEX_ENV_RE.match(stripped)
    )


def _merge_paragraph_lines(lines: list[str]) -> list[str]:
    """⭐ 结构恢复：合并 PDF 断行切碎的段落

    以空行分隔段落边界；段落内连续行合并为流动文本（除非行首是块级信号）。
    连接规则：
      - 前一行以连字符 - 结尾 → 无空格并入（断词合并）
      - 前一行末字符与当前行首字符含 CJK → 无空格并入（中文不需要空格）
      - 否则单空格并入
    """
    paragraphs: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        if not line.strip():
            if cur:
                paragraphs.append(cur)
                cur = []
            continue
        # 当前行是块级信号（标题/列表/引用…）或前一行是块级信号（如标题行）
        # → 另起一段，避免 "标题" 与下一行正文粘连（标题/内容合并 bug）
        if cur and (_starts_block(line.strip()) or _starts_block(cur[-1].strip())):
            paragraphs.append(cur)
            cur = [line]
            continue
        cur.append(line)
    if cur:
        paragraphs.append(cur)

    out: list[str] = []
    for para in paragraphs:
        if not para:
            continue
        merged = para[0].rstrip()
        for ln in para[1:]:
            prev = merged
            nxt = ln.strip()
            if not nxt:
                continue
            if prev.endswith("-") or prev.endswith("－"):
                merged = prev + nxt
            elif _CJK_RE.search(prev[-1:] + nxt[:1]):
                merged = prev + nxt
            else:
                merged = prev + " " + nxt
        out.append(merged)
    return out


def _clean_body_line(line: str) -> str:
    """正文单行清洗：去控制符 + NFKC + 页眉页脚/页码行 + 行尾空白

    行内代码/URL/LaTeX 在调用方已占位保护，此处不动它们。
    """
    cleaned = _strip_control(line)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    stripped = cleaned.strip()
    if _is_page_furniture(stripped):
        return ""
    return stripped.rstrip()


def _semantic_repair(text: str) -> str:
    """④ 语义修复（规则级）：应用 OCR_TYPO_MAP 错字表（当前空表，零生效）"""
    if not OCR_TYPO_MAP:
        return text
    for wrong, right in OCR_TYPO_MAP.items():
        text = text.replace(wrong, right)
    return text


def _normalize_heading(line: str) -> str:
    """⑤ 分块准备：标题层级规范化

    单 `#`（H1）提升为 `##`（section 级）——对齐 chunker 的
    [("##","section"),("###","subsection")]，否则 MarkdownHeaderTextSplitter
    不会在 H1 处分块（整篇作单一父块）。`####`+ 深层级保留不动（chunker 原样
    不识别，属该 ### 父块内容，不误伤）。
    """
    m = re.match(r"^(\s*)#(?!#)(\s+\S.*)$", line)
    if m:
        return f"{m.group(1)}##{m.group(2)}"
    return line


def clean(markdown: str, source_format: str = "") -> str:
    """五步清洗入口（白名单哲学：按块类型作用）

    Args:
        markdown: 解析层输出的 Markdown 文本
        source_format: 来源格式（pdf/docx/text/...，仅用于日志）

    Returns:
        清洗后的 Markdown 文本。清洗异常由调用方捕获降级为原始 Markdown
        （fail-open，不阻断入库）。
    """
    if not markdown or not markdown.strip():
        return markdown or ""

    regions = _tokenize_regions(markdown)
    out_lines: list[str] = []
    for rtype, lines in regions:
        if rtype in ("code", "math"):
            # 白名单：代码/数学内容原样保留（只去控制符 + 行尾空白，
            # 不 NFKC、不剥行首缩进——代码缩进语义关键）
            out_lines.extend(_strip_control(ln).rstrip() for ln in lines)
            continue
        if rtype == "table":
            # 白名单：表格结构保留，不拆合并单元格；只做 NFKC + 控制符 + 行尾空白
            for ln in lines:
                cleaned = _clean_body_line(ln)
                if cleaned:
                    out_lines.append(cleaned)
            continue
        # body 区域：保护内联（先于 NFKC，白名单）→ 逐行清洗（去控制符/NFKC/页码）
        # → 合并断行 → 冗余过滤 → 语义修复（占位保护文本上做，不碰代码/URL）→ 还原内联
        parts_global: list[str] = []

        def _protect_global(ln: str) -> str:
            def _repl(m: re.Match) -> str:
                parts_global.append(m.group(0))
                return f"{_PROTECT_OPEN}{len(parts_global) - 1}{_PROTECT_CLOSE}"
            return _INLINE_PROTECT_RE.sub(_repl, ln)

        # 先保护内联（行内代码/URL/数学不被 NFKC 改写），再逐行清洗
        clean_lines = [_clean_body_line(_protect_global(ln)) for ln in lines]
        # 合并断行（结构恢复 ⭐）
        raw_merged = _merge_paragraph_lines(clean_lines)
        for merged in raw_merged:
            restored = _restore_inline(merged, parts_global)
            if _is_noise(restored):
                continue
            # 语义修复在占位保护文本上做——行内代码/URL/数学不误伤（白名单）
            repaired = _semantic_repair(merged)
            out_lines.append(_restore_inline(repaired, parts_global))

    text = "\n".join(out_lines)
    # ⑤ 分块准备：标题规范化 + 标题/表格前补空行（防标题粘进上一段）
    text = _heading_prep(text)
    # 合并连续空行（保留最多 1 空行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    logger.debug("clean: source_format=%s, %d → %d chars", source_format,
                 len(markdown), len(text))
    return text


def _heading_prep(text: str) -> str:
    """⑤ 分块准备：标题 `#`→`##` + 标题/表格前补空行

    逐行处理：标题行前若无空行则补一个；表格块起点前若无空行也补（保结构）。
    """
    lines = text.split("\n")
    out: list[str] = []
    prev_blank = True  # 首行前视为空行，标题不额外补
    for line in lines:
        stripped = line.strip()
        is_heading = bool(_HEADING_RE.match(stripped)) and not _THEMATIC_BREAK_RE.match(stripped)
        is_table_start = _TABLE_LINE_RE.match(line) and not prev_blank and (
            not out or not _TABLE_LINE_RE.match(out[-1])
        )
        if (is_heading or is_table_start) and not prev_blank:
            out.append("")
        if is_heading:
            line = _normalize_heading(line)
        out.append(line)
        prev_blank = (not stripped)
    return "\n".join(out)


# ── WP3 无损归一化 ───────────────────────────────────────────────────────
_SPACE_COLLAPSE_RE = re.compile(r"(?<![ \t])[ \t]{2,}")  # 折叠块外连续 2+ 空格（保留行首缩进）


def normalize(text: str, max_chars: Optional[int] = None) -> str:
    """无损归一化：NFKC / 去零宽控制符 / 统一空白；表格保持 Markdown；超长截断

    只改表示不改语义（改词/换词/总结不做）。按块类型保护：代码/数学/表格内容
    只做 NFKC + 控制符 + 行尾空白，正文折叠多空格。

    Args:
        text: 清洗后的 Markdown
        max_chars: 超长截断上限（None=不截；chunker 已限父块 4000/子块 300 防
            嵌入截断，此处为病态超长文档兜底）。截断在段落边界对齐。
    """
    if not text:
        return text or ""

    # 1. 全局 NFKC + 去零宽/不可见控制符（表示层归一，安全不破坏结构）
    chars = []
    for ch in text:
        if ch in "\n\t":
            chars.append(ch)
            continue
        if unicodedata.category(ch)[0] == "C":
            continue
        chars.append(ch)
    text = unicodedata.normalize("NFKC", "".join(chars))

    # 2. 统一空白（块外折叠多空格 + 去行尾空白；代码/数学区域原样）
    regions = _tokenize_regions(text)
    out_lines: list[str] = []
    for rtype, lines in regions:
        if rtype in ("code", "math"):
            out_lines.extend(ln.rstrip() for ln in lines)
            continue
        for ln in lines:
            collapsed = _SPACE_COLLAPSE_RE.sub(" ", ln)
            out_lines.append(collapsed.rstrip())

    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. 超长截断（段落边界）
    if max_chars and len(text) > max_chars:
        text = _truncate_at_boundary(text, max_chars)
    return text.strip()


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """截断到 max_chars 以内，尽量在段落/句末边界断（防切开代码块语义）"""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    # 在最近的 \n\n 段落边界断（向前找，最多回退 2000 字符）
    boundary = head.rfind("\n\n", max(0, len(head) - 2000))
    if boundary > 0:
        return head[:boundary].strip()
    # 退化：按行断
    line_break = head.rfind("\n")
    if line_break > 0:
        return head[:line_break].strip()
    return head.strip()
