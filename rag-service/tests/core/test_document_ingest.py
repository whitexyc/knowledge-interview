"""module-064 ingestion 管线测试（parse→clean→normalize→原件→去重→入库）

覆盖（mock 各层）：
- 成功路径：add_document 收到 original_path / doc_content_hash / is_canonical=True
- L1 完全重复 → 直接丢弃（不落原件不入库，dup_kind=exact）
- L2 语义重复 → 入库但标簇 + is_canonical=False（dup_kind=semantic）
- 无有效文本 → IngestError（扫描版无 OCR 诚实提示）
"""
import asyncio

from rag.retrieval import document_ingest
from rag.retrieval.document_parser import ParsedDocument
from rag.retrieval.document_ingest import IngestError, ingest_document


def _run(coro):
    return asyncio.run(coro)


def _patch_layers(monkeypatch, *, parsed, clean_out, norm_out,
                  exact_id=None, semantic=None, add_result=None, doc_emb=None):
    """打桩各层，返回可断言 mock"""
    from unittest import mock
    monkeypatch.setattr(document_ingest.document_parser, "parse_document",
                        lambda data, filename="": parsed)
    monkeypatch.setattr(document_ingest.image_pipeline, "process_pdf_images",
                        lambda md, page_count=None: md)
    monkeypatch.setattr(document_ingest.document_cleaner, "clean",
                        lambda text, source_format="": clean_out)
    monkeypatch.setattr(document_ingest.document_cleaner, "normalize",
                        lambda text, max_chars=None: norm_out)
    monkeypatch.setattr(document_ingest, "_find_exact_duplicate",
                        mock.AsyncMock(return_value=exact_id))
    monkeypatch.setattr(document_ingest.document_dedup, "find_semantic_duplicate",
                        mock.AsyncMock(return_value=semantic))
    monkeypatch.setattr(document_ingest.document_dedup, "compute_doc_embedding",
                        mock.AsyncMock(return_value=doc_emb))
    from rag.engine import rag_engine
    add_mock = mock.AsyncMock(return_value=add_result or {
        "id": 1, "title": "t", "chunks": 2, "duplicate": False})
    monkeypatch.setattr(rag_engine, "add_document", add_mock)
    return add_mock


def test_ingest_success_pipeline(monkeypatch, tmp_path):
    """成功路径：解析→清洗→归一化→原件落盘→入库，字段透传"""
    monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(document_ingest.settings, "doc_dedup_semantic_enabled", False)

    parsed = ParsedDocument(text="# 标题\n\n正文", format="md", engine="text", page_count=None)
    add_mock = _patch_layers(
        monkeypatch, parsed=parsed, clean_out="清洗后", norm_out="归一化后",
        exact_id=None, semantic=None, add_result={
            "id": 11, "title": "a", "chunks": 2, "duplicate": False})

    result = _run(ingest_document(b"raw bytes", "a.md", "文档A"))
    assert result["id"] == 11
    assert result["duplicate"] is False
    # add_document 收到清洗+归一化后文本与 WP5/WP6 字段
    call = add_mock.await_args
    assert call.args[1] == "归一化后"  # content
    assert call.kwargs["is_canonical"] is True
    assert call.kwargs["doc_content_hash"]
    # 原件已落盘
    assert result["original_path"]
    import os
    assert os.path.exists(result["original_path"])
    assert "a.md" in result["original_path"]


def test_ingest_exact_duplicate_skips(monkeypatch, tmp_path):
    """L1 完全重复：直接丢弃——不落原件、不调 add_document"""
    monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
    parsed = ParsedDocument(text="重复内容", format="md", engine="text")
    add_mock = _patch_layers(monkeypatch, parsed=parsed, clean_out="重复内容",
                             norm_out="重复内容", exact_id=99)

    result = _run(ingest_document(b"dup bytes", "a.md", "重复文档"))
    assert result["duplicate"] is True
    assert result["dup_kind"] == "exact"
    assert result["id"] == 99
    assert result["chunks"] == 0
    assert result["original_path"] == ""  # 未落原件
    add_mock.assert_not_awaited()


def test_ingest_semantic_duplicate_marks_canonical(monkeypatch, tmp_path):
    """L2 语义重复：入库但标簇 + is_canonical=False（不删）"""
    monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(document_ingest.settings, "doc_dedup_semantic_enabled", True)
    parsed = ParsedDocument(text="相似内容", format="md", engine="text")
    add_mock = _patch_layers(
        monkeypatch, parsed=parsed, clean_out="相似内容", norm_out="相似内容",
        exact_id=None,
        semantic={"id": 5, "title": "旧文档", "cluster_id": "C-1", "cosine": 0.97},
        add_result={"id": 12, "title": "新文档", "chunks": 2, "duplicate": False})

    result = _run(ingest_document(b"sim bytes", "b.md", "新文档"))
    assert result["duplicate"] is True
    assert result["dup_kind"] == "semantic"
    assert result["duplicate_cluster_id"] == "C-1"
    assert result["canonical"] is False
    call = add_mock.await_args
    assert call.kwargs["duplicate_cluster_id"] == "C-1"
    assert call.kwargs["is_canonical"] is False


def test_ingest_doc_embedding_strips_boilerplate(monkeypatch, tmp_path):
    """Review 修复 1：入库侧文档向量与查询侧同口径——先剥离 Boilerplate 再 embed

    find_semantic_duplicate 查询侧对文本先 strip_boilerplate 再 embed
    （document_dedup.py:127）；入库侧存储的候选向量（根父块 embedding 列）
    必须同口径，否则共同页脚/免责声明主导相似度——同套话不同内容的文档被
    误判语义重复（标 is_canonical=false 检索抑制误隐藏），或真实重复因套话
    差异漏判（ADR-0014 决策 6 坑①）。断言 compute_doc_embedding 收到的
    文本已剥离套话行（"第 3 页"）。
    """
    monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(document_ingest.settings, "doc_dedup_semantic_enabled", True)
    parsed = ParsedDocument(text="主题内容\n第 3 页", format="md", engine="text")
    _patch_layers(
        monkeypatch, parsed=parsed, clean_out="主题内容\n第 3 页",
        norm_out="主题内容\n第 3 页", exact_id=None, semantic=None,
        add_result={"id": 1, "title": "t", "chunks": 2, "duplicate": False},
        doc_emb=[0.5, 0.5])

    _run(ingest_document(b"bytes", "a.md", "文档A"))
    call = document_ingest.document_dedup.compute_doc_embedding.await_args
    assert call is not None
    arg_text = call.args[0]
    assert "第 3 页" not in arg_text  # 套话已剥离
    assert "主题内容" in arg_text      # 正文保留


def test_ingest_no_effective_text_raises(monkeypatch):
    """扫描版/纯图片文档（清洗后无有效文本）→ 明确 IngestError"""
    parsed = ParsedDocument(text="![图](a.png)", format="pdf", engine="anydoc")
    _patch_layers(monkeypatch, parsed=parsed, clean_out="", norm_out="",
                  exact_id=None)
    import pytest
    with pytest.raises(IngestError, match="无有效文本"):
        _run(ingest_document(b"%PDF-1.4", "scan.pdf", "扫描件"))
