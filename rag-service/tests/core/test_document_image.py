"""module-064 WP4 PDF 内嵌图片三层开关单测（ADR-0014 决策 4 / AC §4）

Review 修复 2：image_pipeline 原先零直接单测（仅 test_document_ingest.py 用
lambda 打桩 process_pdf_images 恒返回 md，changelog §五"开关逻辑 + 占位符替换
单测 ✓"声明与事实不符）。本文件补齐 AC §4 四项 + Reviewer 指定场景：

- 三层默认关原样返回含图 md（PDF 含图不报错，走文本部分）
- L1 OCR / L2 VLM 开启但组件缺失 → 占位符替换 + 附注（fail-open）
- L3 PW_PDF_ENGINE=mineru 未装 → 降级回默认解析（不改动）
- 扫描版（有效文本 <50 字且 page_count≥1）→ 如实追加"图片未解析"提示
- image_value_filter 三阈值判定（面积占比 / OCR 质量 / VLM 评分）
- extract_image_refs 提取 Markdown / 裸 <img> 图片（含去重）

全量 hermetic：开关显式传参 + 组件可用性 monkeypatch，不依赖真实环境
（本环境 PaddleOCR/VLM/MinerU 均未装，三层恒 fail-open）。
"""
from rag.retrieval import image_pipeline
from rag.retrieval.image_pipeline import (
    extract_image_refs,
    image_value_filter,
    process_pdf_images,
)


# ── extract_image_refs（Markdown / HTML 图片提取） ───────────────────────
def test_extract_markdown_image():
    refs = extract_image_refs("![架构图](img/arch.png)")
    assert len(refs) == 1
    assert refs[0].idx == 1
    assert refs[0].alt == "架构图"
    assert refs[0].url == "img/arch.png"
    assert refs[0].placeholder == "![架构图](img/arch.png)"


def test_extract_html_image():
    refs = extract_image_refs('<img src="img/logo.png" width="100">')
    assert len(refs) == 1
    assert refs[0].url == "img/logo.png"
    assert refs[0].alt == ""  # HTML <img> 无 alt 属性
    assert refs[0].placeholder == '<img src="img/logo.png" width="100">'


def test_extract_markdown_image_with_title():
    """Markdown 图片带 title（![](... "title")）→ alt/url 正确提取"""
    refs = extract_image_refs('![截图](img/s.png "标题")')
    assert len(refs) == 1
    assert refs[0].url == "img/s.png"
    assert refs[0].alt == "截图"


def test_extract_mixed_and_dedup():
    """Markdown + HTML 混合；重复占位符去重（不重复采集）"""
    md = "![a](x.png)\n<img src=\"y.png\">\n![a](x.png)"
    refs = extract_image_refs(md)
    assert len(refs) == 2
    assert [r.idx for r in refs] == [1, 2]
    assert refs[0].url == "x.png"
    assert refs[1].url == "y.png"


def test_extract_no_image():
    assert extract_image_refs("纯文本，无图片") == []


# ── image_value_filter（面积占比 + OCR 质量 + VLM 评分 三阈值判定） ────────
def test_value_filter_all_zero_rejects():
    """默认全 0 → 判装饰图丢弃（诚实保守：无模型时不把图片当证据）"""
    assert image_value_filter() is False


def test_value_filter_area_ratio_pass():
    assert image_value_filter(area_ratio=0.1) is True
    assert image_value_filter(area_ratio=0.05) is True  # 边界 == min_area_ratio


def test_value_filter_ocr_quality_pass():
    assert image_value_filter(ocr_quality=0.5) is True
    assert image_value_filter(ocr_quality=0.3) is True  # 边界 == min_ocr_quality


def test_value_filter_vlm_score_pass():
    assert image_value_filter(vlm_score=0.9) is True


def test_value_filter_all_below_rejects():
    assert image_value_filter(area_ratio=0.04, ocr_quality=0.2, vlm_score=0.2) is False


# ── 三层默认关：含图 PDF 原样返回（不报错，走文本部分） ───────────────────
def test_default_off_returns_md_unchanged():
    """默认关 + 多页 + 有效文本≥50字 → 图片占位符原样保留（不报错）"""
    md = ("# 标题\n\n这是一段足够长的正文内容，用于验证默认关时含图文档原样返回"
          "不报错。\n\n![流程图](img/flow.png)")
    out = process_pdf_images(md, ocr_enabled=False, caption_enabled=False,
                             pdf_engine="anydoc", page_count=3)
    assert out == md


def test_default_off_no_image_returns_md():
    assert process_pdf_images("纯文本无图", ocr_enabled=False, caption_enabled=False,
                              pdf_engine="anydoc") == "纯文本无图"


# ── L1 OCR 开启但组件缺失 → 占位符替换 + 附注（fail-open） ─────────────────
def test_l1_ocr_missing_fail_open(monkeypatch):
    """PW_IMAGE_OCR=true 但 PaddleOCR/RapidOCR 未装 → 占位符替换 + 附注"""
    monkeypatch.setattr(image_pipeline, "_ocr_available", lambda: False)
    md = "正文。\n\n![图](img/x.png)"
    out = process_pdf_images(md, ocr_enabled=True, caption_enabled=False,
                             pdf_engine="anydoc")
    assert "图内文字未解析" in out          # 占位符替换文本
    assert "图内文字未提取：L1 OCR 组件未安装" in out  # 附注
    assert "![图](img/x.png)" not in out   # 原占位符已被替换
    assert "正文" in out                    # 文本部分保留


# ── L2 VLM 开启但模型缺失 → 占位符替换 + 附注（fail-open） ────────────────
def test_l2_caption_missing_fail_open(monkeypatch):
    """PW_IMAGE_CAPTION=true 但本地 VLM 未接入 → 占位符替换 + 附注"""
    monkeypatch.setattr(image_pipeline, "_vlm_available", lambda: False)
    md = "正文。\n\n![图](img/x.png)"
    out = process_pdf_images(md, ocr_enabled=False, caption_enabled=True,
                             pdf_engine="anydoc")
    assert "图片未解析（L2 VLM 模型缺失" in out
    assert "图片描述未生成：L2 VLM 模型缺失" in out
    assert "![图](img/x.png)" not in out
    assert "正文" in out


# ── L3 MinerU 未装 → 降级回默认解析（fail-open，不改动） ──────────────────
def test_l3_mineru_missing_falls_back(monkeypatch):
    """PW_PDF_ENGINE=mineru 但 MinerU 未安装 → 降级回默认解析"""
    monkeypatch.setattr(image_pipeline, "_mineru_available", lambda: False)
    md = ("# 标题\n\n这是一段足够长的正文内容，用于验证 MinerU 缺失降级回默认解析"
          "时图片原样保留。\n\n![架构图](img/arch.png)")
    out = process_pdf_images(md, ocr_enabled=False, caption_enabled=False,
                             pdf_engine="mineru", page_count=5)
    assert out == md  # 降级回默认解析 = 不做任何改动


# ── 扫描版 PDF（有效文本 <50 字且 page_count≥1）→ 如实"图片未解析"提示 ─────
def test_scan_only_append_unparsed_note():
    md = "![图](img/x.png)"
    out = process_pdf_images(md, ocr_enabled=False, caption_enabled=False,
                             pdf_engine="anydoc", page_count=3)
    assert "扫描版" in out
    assert "图片内容未解析" in out
    assert "![图](img/x.png)" in out          # 原占位符保留（诚实：未解析而非删除）
    assert out.startswith("![图](img/x.png)")  # 提示是追加，不覆盖原文


def test_scan_only_requires_page_count():
    """无 page_count（未知页数）→ 不判扫描版，原样返回"""
    md = "![图](img/x.png)"
    out = process_pdf_images(md, ocr_enabled=False, caption_enabled=False,
                             pdf_engine="anydoc")
    assert out == md
    assert "扫描版" not in out


def test_long_text_not_scan_only():
    """有效文本≥50字（有文本层）→ 即使多页也不判扫描版，原样返回"""
    md = "这是一段足够长的正文。" * 12 + "\n\n![图](img/x.png)"
    out = process_pdf_images(md, ocr_enabled=False, caption_enabled=False,
                             pdf_engine="anydoc", page_count=5)
    assert out == md
    assert "扫描版" not in out
