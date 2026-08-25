"""module-064 WP6 文档去重三级测试（ADR-0014 决策 6，mock embedding/DB）

覆盖：
- L1 内容哈希（exact_hash）
- Boilerplate 先剥离（共同页脚/免责声明主导相似度前先扒；开关可关）
- L2 文档级 embedding 余弦（绝对余弦口径）：≥0.95 → 标 cluster + canonical；
  < 阈值 → 不命中；embedding 失败 → fail-open None
- canonical 选择返回（cluster_id 缺省用根 id）
- L2 候选 pgvector top-K（LIMIT :k 固定，O(N)→O(log N+K)，module-079 加固）
- backlog① 回归：ndarray 形态余弦不抛 ValueError；候选查询失败 fail-open None
- SimHash-LSH 接口预留（文档量几千+ 才启用）
"""
import asyncio

import pytest

from rag.retrieval import document_dedup
from rag.retrieval.document_dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    _cosine,
    compute_doc_embedding,
    exact_hash,
    find_semantic_duplicate,
    simhash_lsh,
    strip_boilerplate,
)

# ── L1 内容哈希 ─────────────────────────────────────────────────────────
def test_exact_hash_sha256():
    h = exact_hash("abc")
    assert len(h) == 64  # sha256 hex
    assert exact_hash("abc") == exact_hash("abc")
    assert exact_hash("abc") != exact_hash("abd")


# ── Boilerplate 先剥离 ──────────────────────────────────────────────────
def test_strip_boilerplate_removes_footers():
    text = "主题内容\n第 3 页\nPage 2 of 10\n版权所有：某公司\n继续内容"
    out = strip_boilerplate(text, enabled=True)
    assert "主题内容" in out
    assert "继续内容" in out
    assert "第 3 页" not in out
    assert "Page 2 of 10" not in out
    assert "版权所有" not in out


def test_strip_boilerplate_disabled():
    text = "主题\n免责声明：本文档仅供参考。"
    out = strip_boilerplate(text, enabled=False)
    assert "免责声明" in out


def test_strip_boilerplate_default_reads_settings(monkeypatch):
    from src.config import settings
    text = "正文\n第 5 页"
    monkeypatch.setattr(settings, "doc_dedup_boilerplate_enabled", True)
    assert "第 5 页" not in strip_boilerplate(text)
    monkeypatch.setattr(settings, "doc_dedup_boilerplate_enabled", False)
    assert "第 5 页" in strip_boilerplate(text)


# ── 余弦 ────────────────────────────────────────────────────────────────
def test_cosine():
    assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0


# ── 文档级 embedding（mock）─────────────────────────────────────────────
class FakeEmb:
    def __init__(self, vec=None, raise_exc=None):
        self._vec = vec
        self._raise = raise_exc

    async def embed_text(self, text):
        if self._raise is not None:
            raise self._raise
        return self._vec


def test_compute_doc_embedding_ok(monkeypatch):
    fake = FakeEmb(vec=[0.5, 0.5])
    out = _run(compute_doc_embedding("doc", embedding_service=fake))
    assert out == [0.5, 0.5]


def test_compute_doc_embedding_fail_open(monkeypatch):
    fake = FakeEmb(raise_exc=RuntimeError("model missing"))
    assert _run(compute_doc_embedding("doc", embedding_service=fake)) is None


# ── L2 语义去重（mock DB）───────────────────────────────────────────────
class FakeRow(dict):
    """假 SQL 行：dict 支持 row["id"] 访问；cosine 由"SQL"算好传入"""


class FakeMappings:
    """假 result.mappings()：可迭代的假行集合"""

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
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, *args, **kwargs):
        return FakeResult(self._rows)

def _run(coro):
    return asyncio.run(coro)


def test_semantic_duplicate_hit():
    """余弦=1.0 ≥ 0.95 → 返回 cluster（id + canonical 选择）"""
    existing = FakeRow(id=5, title="旧文档", duplicate_cluster_id=None, cosine=1.0)
    sess = FakeSession([existing])
    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    dup = _run(find_semantic_duplicate("新文档", embedding_service=emb,
                                       session=sess, threshold=0.95))
    assert dup is not None
    assert dup["id"] == 5
    assert dup["cluster_id"] == "5"  # 无簇 → 用根父块 id 作簇 id
    assert dup["cosine"] == 1.0


def test_semantic_duplicate_uses_existing_cluster():
    """已有 duplicate_cluster_id → 沿用（不新建）"""
    existing = FakeRow(id=9, title="旧文档", duplicate_cluster_id="C-77", cosine=1.0)
    sess = FakeSession([existing])
    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    dup = _run(find_semantic_duplicate("x", embedding_service=emb,
                                       session=sess, threshold=0.95))
    assert dup["cluster_id"] == "C-77"


def test_semantic_duplicate_miss_below_threshold():
    """余弦=0 < 0.95 → 不命中（None）"""
    existing = FakeRow(id=5, title="无关文档", duplicate_cluster_id=None, cosine=0.0)
    sess = FakeSession([existing])
    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    assert _run(find_semantic_duplicate("新", embedding_service=emb,
                                        session=sess, threshold=0.95)) is None


def test_semantic_duplicate_embedding_failure_fail_open():
    """embedding 失败 → None（fail-open，不阻断入库）"""
    sess = FakeSession([])  # embedding 失败在查询前返回，不触达 DB
    emb = FakeEmb(raise_exc=RuntimeError("boom"))
    assert _run(find_semantic_duplicate("新", embedding_service=emb,
                                        session=sess, threshold=0.95)) is None


def test_semantic_duplicate_skips_null_embedding():
    """存量旧文档 embedding=NULL → 跳过不比较"""
    # NULL embedding 由 SQL WHERE embedding IS NOT NULL 排除（不进候选）；
    # 候选余弦不存在 → 0.0 < 阈值 → None（语义不变，判定形态随 SQL 变化）
    legacy = FakeRow(id=1, title="旧格式", duplicate_cluster_id=None, cosine=0.0)
    sess = FakeSession([legacy])
    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    assert _run(find_semantic_duplicate("新", embedding_service=emb,
                                        session=sess, threshold=0.95)) is None


def test_semantic_duplicate_query_excludes_memory():
    """同源内语义去重：SQL 排除 source='memory:%'（不跨源折叠）

    旧格式单文档记忆 parent_id=None 且带向量——若被纳入候选会把知识库文档
    折叠进记忆簇（跨源 bug 防线）。fake session 不执行 SQL 过滤，故编译捕获的
    语句断言排除条件真实存在于查询（对齐 retriever._source_condition 口径）。
    """
    captured = {}

    class CapturingSession(FakeSession):
        async def execute(self, stmt, params=None, **kwargs):
            captured["sql"] = str(stmt)  # text() 语句 str 即完整 SQL（含 :vec/:k 绑定）
            return FakeResult([])

    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    _run(find_semantic_duplicate("新", embedding_service=emb,
                                 session=CapturingSession([]), threshold=0.95))
    assert "memory:%" in captured["sql"]           # 排除记忆 source
    assert "NOT LIKE" in captured["sql"]


def test_semantic_duplicate_query_filters_non_canonical():
    """候选查询过滤 is_canonical=True（module-065 minor-1）

    非 canonical 重复副本虽存文档级 embedding，但检索侧已抑制（只出
    canonical），候选比对语义对齐——副本不参与比对防低概率误判。fake
    session 不执行 SQL 过滤，故编译捕获的语句断言条件真实存在于查询。
    """
    captured = {}

    class CapturingSession(FakeSession):
        async def execute(self, stmt, params=None, **kwargs):
            captured["sql"] = str(stmt)  # text() 语句 str 即完整 SQL（含 :vec/:k 绑定）
            return FakeResult([])

    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    _run(find_semantic_duplicate("新", embedding_service=emb,
                                 session=CapturingSession([]), threshold=0.95))
    assert "is_canonical IS true" in captured["sql"]      # 候选只出 canonical

def test_semantic_duplicate_query_topk_limit():
    """验收5：候选查询为 pgvector top-K——固定 LIMIT :k，不随存量文档数增长"""
    from src.config import settings
    captured = {}

    class CapturingSession(FakeSession):
        async def execute(self, stmt, params=None, **kwargs):
            captured["sql"] = str(stmt)
            captured["params"] = params
            return FakeResult([])

    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    _run(find_semantic_duplicate("新", embedding_service=emb,
                                 session=CapturingSession([]), threshold=0.95))
    assert "ORDER BY embedding <=> :vec ASC" in captured["sql"]
    assert "LIMIT :k" in captured["sql"]
    assert "embedding IS NOT NULL" in captured["sql"]  # NULL 不进候选（SQL 层排除）
    assert captured["params"]["k"] == settings.doc_dedup_candidate_top_k  # 固定 K（默认 50）


def test_semantic_duplicate_ndarray_cosine_no_valueerror():
    """backlog① 回归：pgvector numpy 形态不抛 ValueError

    旧实现 `if not emb` 对非空 ndarray 抛 "truth value ambiguous"；新实现余弦
    由 SQL 算好（1 - (embedding <=> :vec)），Python 只判数值——ndarray 形态的
    候选余弦（np.float64）正常判阈值。
    """
    np = pytest.importorskip("numpy")
    existing = FakeRow(id=7, title="旧文档", duplicate_cluster_id=None,
                       cosine=np.float64(0.98))
    sess = FakeSession([existing])
    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    dup = _run(find_semantic_duplicate("新", embedding_service=emb,
                                       session=sess, threshold=0.95))
    assert dup is not None
    assert dup["id"] == 7
    assert float(dup["cosine"]) == 0.98


def test_semantic_duplicate_query_failure_fail_open():
    """边界：候选查询/判定失败（如 pgvector 维度不匹配）→ fail-open None，不阻断入库"""
    class FailingSession(FakeSession):
        async def execute(self, *args, **kwargs):
            raise RuntimeError("pgvector 维度不匹配")

    emb = FakeEmb(vec=[1.0, 0.0, 0.0])
    assert _run(find_semantic_duplicate("新", embedding_service=emb,
                                        session=FailingSession([]), threshold=0.95)) is None

def test_default_threshold_is_095():
    assert DEFAULT_DEDUP_THRESHOLD == 0.95


# ── L3 SimHash-LSH 接口预留 ─────────────────────────────────────────────
def test_simhash_lsh_reserved():
    """SimHash-LSH 接口预留：文档量几千+ 才启用，当前返回 None（不实现不假装）"""
    assert simhash_lsh([1.0, 2.0, 3.0]) is None
