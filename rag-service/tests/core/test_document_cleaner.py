"""module-064 WP2/WP3 清洗层 + 无损归一化测试（ADR-0014 决策 2/3）

覆盖：
- 白名单哲学（AC 2.2）：代码块符号不清、表格合并单元格不拆、URL/LaTeX/行内
  代码不误伤
- 五步清洗：格式清理（页码/控制符/NFKC 标点）、冗余过滤（纯符号/噪声短段）、
  结构恢复 ⭐（合并 PDF 断行 + 标题不粘连正文 + `#`→`##` + 标题前补空行）、
  语义修复（错字表空表零生效）、分块准备（标题规范化）
- 无损归一化（AC 3.1/3.2）：NFKC / 去零宽控制符 / 统一空白 / 表格保持 MD /
  超长截断；不改语义
"""
import unicodedata

import pytest

from rag.retrieval import document_cleaner
from rag.retrieval.document_cleaner import clean, normalize, _is_noise, _merge_paragraph_lines


# ── 白名单哲学：不误伤 ──────────────────────────────────────────────────
def test_code_block_preserved():
    """代码块内容原样保留（符号不清不误伤）"""
    md = "```python\nif x == 'a':\n    print(x)\n```"
    out = clean(md, "text")
    assert "if x == 'a':" in out
    assert "    print(x)" in out


def test_table_preserved():
    """表格结构保留（合并单元格不拆、分隔行不动）"""
    md = "| 名称 | 值 |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
    out = clean(md, "text")
    assert "| 名称 | 值 |" in out
    assert "| --- | --- |" in out
    assert "| A | 1 |" in out


def test_url_preserved():
    """URL 原样保留（不被空格折叠/标点替换误伤）"""
    md = "详情见 https://example.com/path?a=1&b=2 了解更多"
    out = clean(md, "text")
    assert "https://example.com/path?a=1&b=2" in out


def test_latex_inline_preserved():
    """行内 LaTeX $...$ 原样保留"""
    md = "公式 $E = mc^2$ 是质能方程"
    out = clean(md, "text")
    assert "$E = mc^2$" in out


def test_inline_code_preserved():
    """行内代码 `...` 原样保留（NFKC 不把里面标点转半角）"""
    md = "调用 `func(" + "）" + "` 结束"  # 行内代码内全角括号保持
    out = clean(md, "text")
    assert "`func(" + "）" + "`" in out


def test_blank_line_around_heading_added():
    """标题前无空行 → 补空行（防标题粘进上一段）"""
    md = "上一段内容\n## 标题\n下一段"
    out = clean(md, "text")
    assert "\n\n## 标题\n" in out


# ── 五步清洗：格式清理 ──────────────────────────────────────────────────
def test_page_furniture_removed():
    """页码/页脚残留行移除"""
    md = "正文第一行\n第 3 页\nPage 2 of 10\n继续正文"
    out = clean(md, "pdf")
    assert "第 3 页" not in out
    assert "Page 2 of 10" not in out
    assert "继续正文" in out


def test_pymupdf_page_marker_removed():
    """PyMuPDF 回退路径的 `--- Page i/N ---` 分页标记移除"""
    md = "--- Page 1/3 ---\n第一页内容\n\n--- Page 2/3 ---\n第二页内容"
    out = clean(md, "pdf")
    assert "--- Page 1/3 ---" not in out
    assert "--- Page 2/3 ---" not in out
    assert "第一页内容" in out
    assert "第二页内容" in out


def test_control_chars_removed():
    """控制字符移除（保留 \\n / \\t）"""
    md = "正常文本\x00\x01\x02垃圾\t制表符"
    out = clean(md, "text")
    assert "\x00" not in out and "\x01" not in out
    assert "正常文本" in out


def test_nfkc_punctuation_normalized():
    """NFKC 统一标点：全角逗号 → 半角"""
    md = "你好，世界。"
    out = clean(md, "text")
    assert "你好,世界。" in out


# ── 五步清洗：冗余过滤 ──────────────────────────────────────────────────
def test_noise_paragraph_dropped():
    """纯符号/噪声短段被过滤"""
    md = "正文\n…\n----\n继续正文"
    out = clean(md, "text")
    assert "…" not in out
    assert "----" not in out
    assert "继续正文" in out


def test_meaningful_short_kept():
    """有意义的短文本保留（不误删）"""
    md = "你好\n这是一个测试"
    out = clean(md, "text")
    assert "你好" in out


def test_is_noise_cases():
    assert _is_noise("")
    assert _is_noise("···")
    assert _is_noise("……")
    assert _is_noise("-")
    assert not _is_noise("你好")
    assert not _is_noise("2026年")


# ── 五步清洗：结构恢复 ⭐ ───────────────────────────────────────────────
def test_merge_pdf_broken_lines():
    """PDF 断行切碎的段落合并为流动文本"""
    md = "Java 是一种面向对象\n编程语言，它支持\n垃圾回收机制。"
    out = clean(md, "pdf")
    assert "Java 是一种面向对象编程语言" in out
    assert "垃圾回收机制。" in out


def test_heading_not_merged_with_content():
    """标题行不与正文粘连"""
    md = "# 大标题\n内容"
    out = clean(md, "text")
    assert "## 大标题" in out
    assert "\n内容" in out


def test_list_items_not_merged():
    """列表项保持独立（不合并成一段）"""
    md = "- 第一项\n- 第二项\n- 第三项"
    out = clean(md, "text")
    assert out.count("- 第") == 3


def test_h1_normalized_to_h2():
    """`#` → `##`（对齐 chunker MarkdownHeaderTextSplitter）"""
    md = "# 唯一标题\n内容"
    out = clean(md, "text")
    assert out.startswith("## 唯一标题")


def test_deep_heading_not_touched():
    """`####`+ 深层级标题保留原样（不误伤）"""
    md = "#### 深层小节\n内容"
    out = clean(md, "text")
    assert out.startswith("#### 深层小节")


# ── 五步清洗：语义修复 ──────────────────────────────────────────────────
def test_semantic_repair_empty_map_noop():
    """OCR 错字表为空 → 零生效（诚实声明）"""
    assert document_cleaner.OCR_TYPO_MAP == {}
    md = "正文内容"
    out = clean(md, "text")
    assert out == "正文内容"


# ── WP3 无损归一化 ──────────────────────────────────────────────────────
def test_normalize_nfkc_and_zero_width():
    """NFKC + 去零宽字符"""
    md = "ＡＢＣ\u200b\u200c全角\u00ad"  # fullwidth + zero-width + soft hyphen
    out = normalize(md)
    assert "ABC" in out          # 全角 → 半角（NFKC）
    assert "\u200b" not in out    # 去零宽
    assert "\u200c" not in out
    assert "\u00ad" not in out    # 软连字符（Cf）被去


def test_normalize_table_kept_markdown():
    """表格保持 Markdown 表格（不转纯文本）"""
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    out = normalize(md)
    assert "| A | B |" in out
    assert "| --- | --- |" in out


def test_normalize_code_block_whitespace_untouched():
    """代码块缩进不被空白折叠破坏"""
    md = "```python\nif x:\n    pass\n```"
    out = normalize(md)
    assert "    pass" in out


def test_normalize_does_not_change_semantics():
    """归一化不改词不改语义（改词/换词/总结不做）"""
    md = "Java 线程池的核心参数"
    out = normalize(md)
    assert "Java" in out and "线程池" in out and "核心参数" in out


def test_normalize_truncate_at_boundary():
    """超长截断：在段落边界断，保留头部"""
    md = "## 标题\n\n" + "内容。" * 100 + "\n\n尾部段"
    out = normalize(md, max_chars=120)
    assert len(out) <= 120
    assert "## 标题" in out
    # 尾部段被截掉（截断在段落边界）
    assert "尾部段" not in out


def test_normalize_no_truncation_when_under():
    md = "短文本"
    assert normalize(md, max_chars=100) == "短文本"


# ── chunker 输出结构零破坏（AC 2.3）────────────────────────────────────
def test_cleaned_output_chunks_normally():
    """清洗后文本能被 chunker 正常分块（标题结构保留）"""
    from rag.retrieval.chunker import chunker
    md = "## 第一节\n\n" + "内容。" * 200 + "\n\n## 第二节\n\n" + "内容。" * 200
    out = clean(md, "text")
    res = chunker.chunk(out, source="t")
    assert len(res["parents"]) == 2
    assert res["parents"][0]["title"] == "第一节"
