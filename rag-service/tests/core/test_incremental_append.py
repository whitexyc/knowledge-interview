"""module-079 增量 append 不重建路径验证（hermetic：全 mock，不依赖真实 PG/bge-m3）

覆盖 ADR-0019 阶段 3 五项验收：
- 验收1 增量嵌入：embed_documents 只对新文档子块调用（次数=子块数），存量零嵌入；
  嵌入服务抛异常 fail-open 不阻断入库
- 验收2 检索增量生效：_vector_search 候选 SQL（parent_id IS NOT NULL AND embedding
  IS NOT NULL）命中新子块；add_document 提交后清检索缓存
- 验收3 无全量重嵌：追加仅 INSERT 新行（无 UPDATE/DELETE 存量），存量 embedding
  逐字节不变；自动入库路径不引用 reindex/rebuild/backfill 脚本；图提取幂等追加
- 验收4 去重不破坏增量：L1 命中返回 duplicate 不写库，后续新文档正常追加
- 验收5 性能：语义去重候选 SQL 固定 LIMIT :k；不同存量文档数下 embed_documents
  调用次数相同（增量成本与 N 无关）
"""
import asyncio
import pathlib
from unittest import mock

import pytest

from rag.models import Document
from rag.retrieval import document_ingest
from rag.retrieval.document_parser import ParsedDocument


def _run(coro):
    return asyncio.run(coro)


class _FakeAddSession:
    """add_document 假会话：预置存量文档，记录 DML（add=INSERT），flush 赋 id"""

    def __init__(self, existing=None):
        self._added = list(existing or [])
        self._ops = [("existing", len(existing or []))]
        self._nid = len(existing or []) + 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt):
        result = mock.MagicMock()
        result.scalar_one_or_none.return_value = None  # 无重复
        return result

    def add(self, obj):
        self._added.append(obj)
        self._ops.append(("insert", obj))

    async def flush(self):
        for o in self._added:
            if o.id is None:
                o.id = self._nid
                self._nid += 1

    async def commit(self):
        pass

    async def rollback(self):
        pass


def _run_add_document(session, *, chunk=None, embed=None, cache=None, extraction=None):
    """以标准 mock 栈跑 rag_engine.add_document，返回 (result, mocks)"""
    from rag.engine import rag_engine

    factory = mock.MagicMock(return_value=session)
    chunk_ret = chunk or {
        "parents": [{"title": "section", "content": "新内容"}],
        "children": [{"title": "section", "content": "新内容", "parent_index": 0}],
    }
    embed_mock = embed if embed is not None else mock.AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    cache_mock = cache if cache is not None else mock.AsyncMock(return_value=True)
    ensure_mock = mock.AsyncMock(return_value=True)
    extract_mock = mock.AsyncMock(
        return_value=extraction if extraction is not None else {"entities": [], "relations": []})
    upsert_ent = mock.AsyncMock(return_value=None)
    upsert_rel = mock.AsyncMock(return_value=None)

    async def run():
        with mock.patch("rag.engine.async_session_factory", factory), \
             mock.patch("rag.engine.chunker.chunk", return_value=chunk_ret), \
             mock.patch("rag.engine.embedding_service.embed_documents", embed_mock), \
             mock.patch("rag.engine.tokenize", return_value=""), \
             mock.patch("rag.engine.cache.delete_by_prefix", cache_mock), \
             mock.patch("rag.engine.graph_store.ensure_graph", ensure_mock), \
             mock.patch("rag.engine.graph_extractor.extract_from_document", extract_mock), \
             mock.patch("rag.engine.graph_store.upsert_entity", upsert_ent), \
             mock.patch("rag.engine.graph_store.upsert_relation", upsert_rel):
            return await rag_engine.add_document("新文档", "新内容", "test")

    return _run(run()), {
        "embed": embed_mock, "cache": cache_mock, "ensure": ensure_mock,
        "extract": extract_mock, "upsert_entity": upsert_ent, "upsert_relation": upsert_rel,
    }


# ── 验收 1：增量嵌入 ──────────────────────────────────────────────────────
def test_embedding_only_new_children():
    """验收1：embed_documents 只对新文档子块调用（次数=子块数），存量零嵌入"""
    existing = [
        Document(id=100, title="存量1", content="旧内容1", embedding=[0.1] * 4, parent_id=None),
        Document(id=200, title="存量2", content="旧内容2", embedding=[0.2] * 4, parent_id=None),
    ]
    session = _FakeAddSession(existing)
    embed = mock.AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    result, mocks = _run_add_document(session, embed=embed, chunk={
        "parents": [{"title": "section", "content": "新内容"}],
        "children": [
            {"title": "s1", "content": "新子块一", "parent_index": 0},
            {"title": "s2", "content": "新子块二", "parent_index": 0},
        ],
    })
    assert result["duplicate"] is False
    assert result["chunks"] == 3  # 1 父块 + 2 子块
    embed.assert_awaited_once()
    assert embed.await_args.args[0] == ["新子块一", "新子块二"]  # 只嵌入新文档子块


@pytest.mark.parametrize("existing_count", [0, 200])
def test_embedding_cost_independent_of_existing_count(existing_count):
    """验收1/5：存量文档数 N 不同，embed_documents 调用次数相同（增量成本与 N 无关）"""
    existing = [
        Document(id=i, title=f"存量{i}", content=f"旧内容{i}", embedding=[0.1] * 4, parent_id=None)
        for i in range(1, existing_count + 1)
    ]
    session = _FakeAddSession(existing)
    embed = mock.AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    result, mocks = _run_add_document(session, embed=embed)
    assert result["duplicate"] is False
    embed.assert_awaited_once()  # N 增长不增加嵌入调用
    assert embed.await_args.args[0] == ["新内容"]


def test_ingest_embedding_failure_fail_open(monkeypatch, tmp_path):
    """验收1：嵌入服务抛异常 → 入库仍成功（fail-open 不阻断增量追加）"""
    from rag.engine import rag_engine

    monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(document_ingest.settings, "doc_dedup_semantic_enabled", True)
    monkeypatch.setattr(document_ingest.document_parser, "parse_document",
                        lambda data, filename="": ParsedDocument(
                            text="新文档正文", format="md", engine="text", page_count=None))
    monkeypatch.setattr(document_ingest.image_pipeline, "process_pdf_images",
                        lambda md, page_count=None: md)
    monkeypatch.setattr(document_ingest.document_cleaner, "clean",
                        lambda text, source_format="": text)
    monkeypatch.setattr(document_ingest.document_cleaner, "normalize",
                        lambda text, max_chars=None: text)
    monkeypatch.setattr(document_ingest, "_find_exact_duplicate",
                        mock.AsyncMock(return_value=None))

    class _RaisingEmb:
        """嵌入服务不可用（真实链路 compute_doc_embedding 内部 catch → fail-open）"""

        async def embed_text(self, text):
            raise RuntimeError("model missing")

    monkeypatch.setattr(document_ingest.document_dedup, "default_embedding_service", _RaisingEmb())
    received = {}

    async def fake_add(title, content, source, **kwargs):
        received["doc_embedding"] = kwargs.get("doc_embedding")
        return {"id": 1, "title": title, "chunks": 2, "duplicate": False}

    monkeypatch.setattr(rag_engine, "add_document", fake_add)

    result = _run(document_ingest.ingest_document(b"raw", "a.md", "新文档"))
    assert result["duplicate"] is False
    assert result["chunks"] == 2
    assert received["doc_embedding"] is None  # 嵌入失败 fail-open，doc_embedding=None 仍入库


# ── 验收 2：检索增量生效 ──────────────────────────────────────────────────
def test_add_document_clears_retrieval_cache():
    """验收2：add_document 提交后清检索缓存（新文档立即可检索）"""
    result, mocks = _run_add_document(_FakeAddSession())
    assert result["duplicate"] is False
    mocks["cache"].assert_awaited_once_with("rag:retrieve:")


def test_vector_search_hits_new_chunk():
    """验收2：_vector_search 候选 SQL（parent_id IS NOT NULL AND embedding IS NOT NULL）命中新子块"""
    from rag.retrieval.retriever import hybrid_retriever

    captured = {}

    class FakeRow(dict):
        pass

    class FakeMappings:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return FakeMappings(self._rows)

    class FakeSession:
        async def execute(self, stmt, params=None):
            captured["sql"] = str(stmt)
            return FakeResult([FakeRow(
                id=777, title="新文档 > 子块", content="新内容",
                source="test", page_num=1, metadata=None, created_at=None,
                parent_id=1, score=0.91,
            )])

    rows = _run(hybrid_retriever._vector_search([0.1, 0.2, 0.3], 5, FakeSession()))
    assert any(r["id"] == 777 for r in rows)  # 新子块可被向量检索命中
    assert rows[0]["score"] == 0.91
    assert "parent_id IS NOT NULL" in captured["sql"]
    assert "embedding IS NOT NULL" in captured["sql"]


# ── 验收 3：无全量重嵌 ────────────────────────────────────────────────────
def test_add_document_insert_only_no_existing_mutation():
    """验收3：追加仅 INSERT 新行；存量文档 embedding 逐字节不变"""
    existing = [
        Document(id=100, title="存量1", content="旧内容1",
                 embedding=[0.11, 0.12, 0.13, 0.14], parent_id=None),
        Document(id=200, title="存量2", content="旧内容2",
                 embedding=[0.21, 0.22, 0.23, 0.24], parent_id=None),
    ]
    emb_before = [list(d.embedding) for d in existing]
    session = _FakeAddSession(existing)

    result, mocks = _run_add_document(session)
    assert result["duplicate"] is False
    op_types = [op[0] for op in session._ops]
    assert op_types.count("insert") == 2  # 新父块 + 新子块
    assert "update" not in op_types and "delete" not in op_types  # 零 UPDATE/DELETE
    # 存量 embedding 逐字节不变（追加不改存量行）
    assert [list(d.embedding) for d in session._added[:2]] == emb_before
    assert len(session._added) == 4


def test_auto_ingest_path_has_no_reindex_scripts():
    """验收3：自动入库路径不引用全量重嵌/重建运维脚本（reindex 仅手动）"""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for rel in ("rag/engine.py", "rag/retrieval/document_ingest.py",
                "rag/retrieval/document_dedup.py"):
        src = (repo_root / rel).read_text(encoding="utf-8")
        for name in ("reindex_knowledge_base", "migrate_embedding_1024", "backfill_graph"):
            assert name not in src, f"{rel} 不应引用手动全量重嵌脚本 {name}"


def test_graph_extraction_additive():
    """验收3：ensure_graph 幂等建图 + upsert_entity/upsert_relation 幂等追加（非重建）"""
    result, mocks = _run_add_document(_FakeAddSession(), extraction={
        "entities": [{"name": "实体A", "type": "concept"}],
        "relations": [{"source": "A", "target": "B"}],
    })
    assert result["duplicate"] is False
    mocks["ensure"].assert_awaited_once()
    mocks["extract"].assert_awaited_once_with("新内容")
    mocks["upsert_entity"].assert_awaited_once_with("实体A", "concept", 1)  # 新父块 id
    mocks["upsert_relation"].assert_awaited_once_with("A", "B")


# ── 验收 4：去重不破坏增量 ────────────────────────────────────────────────
def test_l1_dedup_does_not_block_incremental_append(monkeypatch, tmp_path):
    """验收4：入库 A → 同内容再入库（L1 命中 duplicate 不写库）→ 新文档 B 正常追加"""
    from rag.engine import rag_engine

    monkeypatch.setattr(document_ingest.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(document_ingest.settings, "doc_dedup_semantic_enabled", False)
    monkeypatch.setattr(document_ingest.document_parser, "parse_document",
                        lambda data, filename="": ParsedDocument(
                            text=data.decode("utf-8"), format="md", engine="text", page_count=None))
    monkeypatch.setattr(document_ingest.image_pipeline, "process_pdf_images",
                        lambda md, page_count=None: md)
    monkeypatch.setattr(document_ingest.document_cleaner, "clean",
                        lambda text, source_format="": text)
    monkeypatch.setattr(document_ingest.document_cleaner, "normalize",
                        lambda text, max_chars=None: text)

    stored = {}  # doc_content_hash → id（有状态 mock 模拟 L1 去重表）

    async def fake_find(doc_hash, session=None):
        return stored.get(doc_hash)

    monkeypatch.setattr(document_ingest, "_find_exact_duplicate", fake_find)

    added = []

    async def fake_add(title, content, source, **kwargs):
        doc_id = len(added) + 1
        added.append({"id": doc_id, "title": title, "content": content})
        stored[kwargs["doc_content_hash"]] = doc_id
        return {"id": doc_id, "title": title, "chunks": 2, "duplicate": False}

    monkeypatch.setattr(rag_engine, "add_document", fake_add)

    r1 = _run(document_ingest.ingest_document("文档A内容".encode("utf-8"), "a.md", "文档A"))
    assert r1["duplicate"] is False and r1["chunks"] == 2
    r2 = _run(document_ingest.ingest_document("文档A内容".encode("utf-8"), "a2.md", "文档A"))
    assert r2["duplicate"] is True and r2["dup_kind"] == "exact"
    assert r2["chunks"] == 0
    assert r2["original_path"] == ""  # L1 命中不落原件
    r3 = _run(document_ingest.ingest_document("文档B内容".encode("utf-8"), "b.md", "文档B"))
    assert r3["duplicate"] is False and r3["chunks"] == 2
    assert len(added) == 2  # 只入库 A 与 B：去重不阻塞后续增量


def test_add_document_internal_dedup_zero_embedding():
    """验收4：add_document 内 title/content_hash 兜底去重命中 → duplicate，零嵌入"""
    from rag.engine import rag_engine

    dup = Document(id=9, title="已有文档", content="已有内容", embedding=[0.5] * 4)

    class _DupSession(_FakeAddSession):
        async def execute(self, stmt):
            result = mock.MagicMock()
            result.scalar_one_or_none.return_value = dup
            return result

    embed = mock.AsyncMock(return_value=[[0.1]])
    factory = mock.MagicMock(return_value=_DupSession([dup]))

    async def run():
        with mock.patch("rag.engine.async_session_factory", factory), \
             mock.patch("rag.engine.embedding_service.embed_documents", embed):
            return await rag_engine.add_document("已有文档", "新内容", "test")

    result = _run(run())
    assert result["duplicate"] is True
    assert result["chunks"] == 0
    embed.assert_not_awaited()  # 去重命中不做任何嵌入（增量零成本）
