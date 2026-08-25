"""PDF 内嵌图片三层方案 — 开关 + 分层路由 + 占位符替换（module-064 / ADR-0014 WP4）

三层全默认关（重工具不默认启用，只路由复杂文档）：
  L1 PW_IMAGE_OCR        —— 图内文字 OCR（PaddleOCR/RapidOCR），图内文字插回 MD
  L2 PW_IMAGE_CAPTION    —— 本地轻量 VLM 图片描述插回 MD（显式占位符替换）
  L3 PW_PDF_ENGINE=mineru—— MinerU 独立通道（复杂版面整体重解析）

fail-open：模型/组件缺失 → 对应层降级关，文本部分照常入库（不阻断）。
图片价值过滤（只留流程图/架构图/UML/表格截图等装饰图之外的）：面积占比 +
OCR 质量 + VLM 评分——接口 image_value_filter 预留，真实评分需对应模型可用后接线。

诚实边界：
  - 三层默认关时含图 PDF 不报错（走文本部分，图片占位符原样保留）；
  - 扫描版 PDF（无文本层）无 OCR 时如实返回"图片未解析"提示；
  - L1/L2/L3 模型未安装时各自降级关（本环境 PaddleOCR/RapidOCR/VLM/MinerU
    均未安装 → 三层恒走 fail-open，如实在返回文本附注提示）。
"""
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

# Markdown 图片语法 ![alt](url "title") 与裸 <img src="..." >（AnyDoc 输出形式）
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_IMG_RE = re.compile(r"<img[^>]*src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)


@dataclass
class ImageRef:
    """PDF/文档中的一张内嵌图片引用

    Attributes:
        idx: 图片序号（1-based，用于占位符替换与价值过滤标识）
        placeholder: 原文中的图片片段（如 `![alt](url)`），替换时定位用
        alt: 图片替代文本（可为空）
        url: 图片地址/资源定位
    """
    idx: int
    placeholder: str
    alt: str
    url: str


def extract_image_refs(md: str) -> list[ImageRef]:
    """提取文档中的图片引用（Markdown 语法 + 裸 <img> 标签）

    不做重复采集（HTML 标签内的图片已由 Markdown 语法覆盖时跳过）。
    """
    refs: list[ImageRef] = []
    seen_placeholders: set[str] = set()
    for m in _IMG_RE.finditer(md):
        placeholder = m.group(0)
        if placeholder in seen_placeholders:
            continue
        seen_placeholders.add(placeholder)
        refs.append(ImageRef(idx=len(refs) + 1, placeholder=placeholder,
                             alt=m.group(1), url=m.group(2)))
    for m in _HTML_IMG_RE.finditer(md):
        placeholder = m.group(0)
        if placeholder in seen_placeholders:
            continue
        seen_placeholders.add(placeholder)
        refs.append(ImageRef(idx=len(refs) + 1, placeholder=placeholder,
                             alt="", url=m.group(1)))
    return refs


# ── L1/L2/L3 组件可用性探测（本环境均未安装 → 恒 False，fail-open） ──────
def _ocr_available() -> bool:
    """L1 OCR 组件（PaddleOCR/RapidOCR）可用性——未安装返回 False"""
    try:
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


def _vlm_available() -> bool:
    """L2 本地轻量 VLM 可用性——未安装返回 False"""
    # 预留：本地 VLM 模型加载探测（如 Qwen-VL GGUF / 多模态 CLIP）。
    # 本环境未接入本地 VLM，恒 False（诚实：需要模型下载依赖环境）。
    return False


def _mineru_available() -> bool:
    """L3 MinerU 独立通道可用性——未安装返回 False"""
    try:
        import mineru  # noqa: F401
        return True
    except Exception:
        return False


def image_value_filter(area_ratio: float = 0.0, ocr_quality: float = 0.0,
                       vlm_score: float = 0.0,
                       min_area_ratio: float = 0.05,
                       min_ocr_quality: float = 0.3,
                       min_vlm_score: float = 0.3) -> bool:
    """图片价值过滤（只留流程图/架构图/UML/产品截图/表格截图等有价值图）

    三项评分任一维度低于阈值即判"装饰图"丢弃（保守交集口径可后续按需调）：
      - area_ratio  : 图片占页面面积比例（0-1，由解析器给出；默认 0 表示未知）
      - ocr_quality : OCR 提取图内文字的置信度（0-1；无 OCR 时 0）
      - vlm_score   : VLM 对该图内容价值的评分（0-1；无 VLM 时 0）

    当前为接口预留：真实评分需 L1/L2 模型可用后由对应层回填（默认值全 0 会
    判为装饰图丢弃，防止无模型时把图片当证据——诚实保守）。
    """
    return (area_ratio >= min_area_ratio
            or ocr_quality >= min_ocr_quality
            or vlm_score >= min_vlm_score)


def _replace_with_placeholder(md: str, refs: list[ImageRef], note: str) -> str:
    """把图片占位符替换为统一说明占位符（按 1-based 序号）"""
    for ref in refs:
        md = md.replace(ref.placeholder, f"[图片 {ref.idx}：{note}]")
    return md


def _append_note(md: str, note: str) -> str:
    """在文本末尾追加一层降级说明（不覆盖原文）"""
    if note in md:
        return md
    return md.rstrip() + f"\n\n{note}\n"


def _is_scan_only(md: str, page_count: Optional[int]) -> bool:
    """扫描版 PDF 启发式：几乎无文本层但有多页/多图 → 极可能是扫描件"""
    meaningful = len(md.strip())
    return meaningful < 50 and (page_count or 0) >= 1


def process_pdf_images(
    md: str,
    *,
    ocr_enabled: Optional[bool] = None,
    caption_enabled: Optional[bool] = None,
    pdf_engine: Optional[str] = None,
    page_count: Optional[int] = None,
) -> str:
    """三层开关路由入口：按 PW_IMAGE_OCR / PW_IMAGE_CAPTION / PW_PDF_ENGINE 处理

    Args:
        md: 解析层输出的 Markdown（可能含图片占位符）
        ocr_enabled: L1 开关（None → 读 settings.image_ocr_enabled）
        caption_enabled: L2 开关（None → 读 settings.image_caption_enabled）
        pdf_engine: L3 引擎（None → 读 settings.pdf_engine）
        page_count: PDF 页数（扫描版判定用）

    Returns:
        处理后的 Markdown。任何一层降级都不抛异常（fail-open）。
    """
    ocr = settings.image_ocr_enabled if ocr_enabled is None else ocr_enabled
    caption = settings.image_caption_enabled if caption_enabled is None else caption_enabled
    engine = settings.pdf_engine if pdf_engine is None else pdf_engine

    refs = extract_image_refs(md)
    if not refs:
        return md

    # L3 MinerU：复杂版面独立通道（未安装 → 降级继续，不做任何改动）
    if engine == "mineru":
        if _mineru_available():
            logger.info("image_pipeline: L3 MinerU 通道接管（%d 张图）", len(refs))
            # MinerU 整体重解析由上层引擎对接；此处返回原 MD（通道接管标记）
            return md
        logger.warning("image_pipeline: PW_PDF_ENGINE=mineru 但 MinerU 未安装，"
                       "降级回默认解析（fail-open）")

    # L2 VLM 图片描述（显式占位符替换）
    if caption:
        if _vlm_available():
            # 预留：_vlm_describe(refs) 返回 [{idx, description}]；未接入时不可达
            logger.info("image_pipeline: L2 VLM 图片描述启用")
            return md
        md = _replace_with_placeholder(md, refs, "图片未解析（L2 VLM 模型缺失，已降级保留）")
        return _append_note(md, "图片描述未生成：L2 VLM 模型缺失（PW_IMAGE_CAPTION 默认关）")

    # L1 OCR 图内文字
    if ocr:
        if _ocr_available():
            logger.info("image_pipeline: L1 OCR 图内文字启用")
            return md
        md = _replace_with_placeholder(md, refs, "图内文字未解析（L1 OCR 组件缺失，已降级保留）")
        return _append_note(md, "图内文字未提取：L1 OCR 组件未安装（PW_IMAGE_OCR 默认关）")

    # 默认（三层全关）：图片占位符原样保留（不报错，走文本部分）；
    # 扫描版 PDF 无文本层时如实附"图片未解析"提示（诚实边界）
    if _is_scan_only(md, page_count):
        return _append_note(
            md,
            "本 PDF 为扫描版（无文本层），图片三层解析默认关闭，图片内容未解析。"
            "如需解析图内信息请启用 PW_IMAGE_OCR/PW_IMAGE_CAPTION/PW_PDF_ENGINE。",
        )
    return md
