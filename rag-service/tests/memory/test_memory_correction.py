"""Module-061 记忆纠错单元测试（P0 升级留后悔药 + P1 冲突消解 + 评测基线一致性）

覆盖（acceptance-criteria.md §2/§3/§4/§7）：
- P0：_promote_memory 升级**保留短期副本**（后悔药）+ 长期新条目
     superseded=false/updated_at + 幂等不产生垃圾；_expand_to_parents /
     _evolve_recall 过滤 superseded=true（召回侧统一口径）
- P1：nli_judge 生产封装（延迟加载/失败 None/超时 None/三分类返回）；
     _merge_duplicate 分流（矛盾 → SUPERSEDED+新增 / 一致 → 追加 /
     NLI None → 追加 / 开关关 → 完全旧行为）；save 全流程冲突新增
- 评测基线一致性：标注集结构校验（≥20/矛盾≥15/正例中性对照）+
     contradiction_metrics 纯函数 + 达标判定 + fixture 判定
- conftest autouse 钉住 memory_conflict_enabled=False（存量记忆测试零漂移）——
    本文件开关用例显式 setattr True 验证冲突分流（mock NLI，不依赖真实 557MB 模型）

实现说明：与 test_memory.py 同款模式（mock AsyncSession / _FakeSession /
_ScriptedSession / mock NLI 不加载真实模型）；同步用例内 asyncio.run 执行。
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest import mock

from rag.memory.memory import memory_service, _is_superseded
from rag.memory.nli_judge import MemoryNLIJudge, nli_judge
from eval.datasets.memory_conflict_dataset import (
    load_memory_conflict_dataset, contradiction_metrics, gate_passed, fixture_judge,
    GATE_CONTRADICTION_PRECISION, GATE_CONTRADICTION_RECALL,
)
from src.config import settings


class _FakeSession:
    """假 AsyncSession：记录 add 的对象 + 可配置 execute 结果（对齐 test_memory.py）"""

    def __init__(self, scalar=None, scalars=None):
        self.added: list = []
        self._scalar = scalar
        self._scalars = scalars or []
        self.rolled_back = False

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for i, obj in enumerate(self.added):
            if getattr(obj, "parent_id", None) is None:
                obj.id = i + 1

    async def commit(self):
        pass

    async def rollback(self):
        self.rolled_back = True

    async def execute(self, stmt):
        result = mock.MagicMock()
        result.scalar.return_value = self._scalar
        result.scalars.return_value = mock.MagicMock(
            all=mock.MagicMock(return_value=self._scalars),
        )
        result.all.return_value = []
        return result


class _ScriptedSession:
    """按执行序号返回脚本结果的会话桩（对齐 test_memory.py module-046 升级测试）"""

    def __init__(self, script):
        self.script = list(script)
        self.executed = []
        self.added = []
        self.i = 0

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for i, obj in enumerate(self.added):
            if getattr(obj, "parent_id", None) is None:
                obj.id = i + 1

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def execute(self, stmt):
        self.executed.append(stmt)
        kind, value = self.script[self.i] if self.i < len(self.script) else ("scalars", [])
        self.i += 1
        result = mock.MagicMock()
        if kind == "scalars":
            result.scalars.return_value = mock.MagicMock(all=mock.MagicMock(return_value=value))
            result.scalar.return_value = None
        else:
            result.scalar.return_value = value
            result.scalars.return_value = mock.MagicMock(all=mock.MagicMock(return_value=[]))
        result.all.return_value = []
        return result


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器"""
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


def _chunk_single(content):
    return {
        "parents": [{"title": "记忆", "content": content}],
        "children": [{"title": "记忆", "content": content, "parent_index": 0}],
    }


# ──────────────────────────────────────────────────────────────
# P0 升级留后悔药（module-061 / AC §2）
# ──────────────────────────────────────────────────────────────

class TestPromoteKeepsShortCopy:
    """module-061 P0：升级不删除短期副本（后悔药）+ 长期新条目 superseded=false/updated_at"""

    @staticmethod
    def _parent_with_children(count=2, content="常提及主题"):
        now = datetime.now(timezone.utc)
        return (
            mock.MagicMock(id=1, parent_id=None, content=content, title="t",
                           created_at=now - timedelta(days=5), last_mentioned_at=now,
                           mention_count=count, content_hash="hash-parent"),
            [
                mock.MagicMock(id=2, parent_id=1, content="子块1", title="t",
                               created_at=now, last_mentioned_at=now, mention_count=0,
                               content_hash="hash-child-1", embedding=[0.1], search_tokens="t"),
                mock.MagicMock(id=3, parent_id=1, content="子块2", title="t",
                               created_at=now, last_mentioned_at=now, mention_count=0,
                               content_hash="hash-child-2", embedding=[0.2], search_tokens="t"),
            ],
        )

    @staticmethod
    def _run_promote(script, parent, children):
        session = _ScriptedSession(script)

        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(session)):
                await memory_service._promote_memory("42", parent)

        asyncio.run(run())
        return session

    @staticmethod
    def _delete_sqls(session):
        sqls = [str(s.compile(compile_kwargs={"literal_binds": True})) for s in session.executed]
        return [s for s in sqls if s.lstrip().upper().startswith("DELETE")]

    def test_promotes_keeps_short_copy_and_stamps_superseded(self):
        parent, children = self._parent_with_children()
        session = self._run_promote([
            ("scalars", children),  # _promote_memory 查子块
            ("scalar", None),       # 幂等检查：长期层无同 content_hash 父块
        ], parent, children)
        # 复制到长期层：新父块（无向量）+ 子块（含向量），source='memory:42:'
        assert len(session.added) == 3
        new_parent = session.added[0]
        assert new_parent.source == "memory:42:"
        assert new_parent.parent_id is None and new_parent.embedding is None
        assert new_parent.superseded is False          # 长期新条目 superseded=false
        assert new_parent.updated_at is not None       # updated_at=now
        for c in session.added[1:]:
            assert c.source == "memory:42:"
            assert c.parent_id == new_parent.id
            assert c.embedding is not None
            assert c.superseded is False
        # 后悔药：不删除短期副本（无 DELETE 语句）
        assert self._delete_sqls(session) == []

    def test_promotion_idempotent_keeps_short_copy(self):
        parent, children = self._parent_with_children()
        session = self._run_promote([
            ("scalars", children),
            ("scalar", 99),        # 幂等命中：长期层已有同 hash 父块
        ], parent, children)
        assert session.added == []                     # 不重复复制（不产生垃圾行）
        assert self._delete_sqls(session) == []        # 短期副本保留


class TestSupersededRecallFilter:
    """module-061 P0：召回/检索侧过滤 superseded=true（_expand_to_parents + _evolve_recall）"""

    def test_expand_to_parents_skips_superseded_parent(self):
        # 旧记忆已被 SUPERSEDED → 子块命中也不映射回父块（不参与召回）
        superseded_parent = mock.MagicMock(id=1, content="旧说法：用户讨厌咖啡",
                                          title="t", created_at=datetime(2026, 8, 1),
                                          superseded=True)
        child = {"id": 2, "content": "子块", "parent_id": 1, "hybrid_score": 0.9}

        async def run():
            with mock.patch("rag.memory.async_session_factory",
                            _fake_factory(_FakeSession(scalars=[superseded_parent]))):
                return await memory_service._expand_to_parents([child])

        assert asyncio.run(run()) == []

    def test_expand_to_parents_keeps_active_parent(self):
        active_parent = mock.MagicMock(id=1, content="最新说法：用户喜欢咖啡",
                                       title="t", created_at=datetime(2026, 8, 1),
                                       superseded=False)
        child = {"id": 2, "content": "子块", "parent_id": 1, "hybrid_score": 0.9}

        async def run():
            with mock.patch("rag.memory.async_session_factory",
                            _fake_factory(_FakeSession(scalars=[active_parent]))):
                return await memory_service._expand_to_parents([child])

        memories = asyncio.run(run())
        assert len(memories) == 1
        assert memories[0]["content"] == "最新说法：用户喜欢咖啡"

    def test_evolve_recall_ignores_superseded_refs(self):
        # superseded 参考文档不参与进化（by_content 排除 → 该记忆保留原样不衰减）
        superseded_parent = mock.MagicMock(id=1, content="旧说法", title="t",
                                          created_at=datetime.now(timezone.utc),
                                          last_mentioned_at=None, mention_count=0,
                                          superseded=True)
        memories = [{"content": "旧说法", "score": 0.9, "title": "t", "created_at": ""}]
        child = {"id": 2, "content": "子块", "parent_id": 1, "hybrid_score": 0.9}

        async def run():
            with mock.patch("rag.memory.hybrid_retriever") as ret:
                ret.retrieve = mock.AsyncMock(return_value=[child])
                with mock.patch("rag.memory.embedding_service") as emb:
                    emb.embed_text = mock.AsyncMock(side_effect=RuntimeError("embed down"))
                    with mock.patch.object(memory_service, "_expand_to_parents",
                                           new=mock.AsyncMock(return_value=memories)):
                        with mock.patch("rag.memory.async_session_factory",
                                        _fake_factory(_FakeSession(scalars=[superseded_parent]))):
                            return await memory_service._evolve_recall(memories, [child], "42")

        out = asyncio.run(run())
        assert out == memories          # 保留原样（fail-open，不参与进化衰减）
        assert out[0]["score"] == 0.9


class TestIsSuperseded:
    """_is_superseded 辅助：仅 superseded 字段确为 Python True 才返回 True"""

    def test_true_only_when_explicit_true(self):
        assert _is_superseded(mock.MagicMock(superseded=True)) is True
        assert _is_superseded(mock.MagicMock(superseded=False)) is False
        # MagicMock 缺字段时 .superseded 返回真值 MagicMock → `is True` 为 False（不过滤）
        assert _is_superseded(mock.MagicMock()) is False
        assert _is_superseded(None) is False
        assert _is_superseded({"superseded": True}) is False


# ──────────────────────────────────────────────────────────────
# P1 冲突消解（module-061 / AC §3）
# ──────────────────────────────────────────────────────────────

class TestMergeDuplicateConflict:
    """_merge_duplicate 分流：矛盾 → SUPERSEDED+新增 / 一致 → 追加 / 降级/开关关 → 旧行为"""

    @staticmethod
    def _merge(parent, verdict, layer="", enabled=True):
        duplicate = mock.MagicMock(id=6, parent_id=parent.id)
        session = mock.MagicMock()
        session.get = mock.AsyncMock(return_value=parent)
        session.commit = mock.AsyncMock()
        out = {}

        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(session)):
                settings.memory_conflict_enabled = enabled
                try:
                    with mock.patch.object(memory_service, "_judge_conflict",
                                           new=mock.AsyncMock(return_value=verdict)) as jc:
                        out["result"] = await memory_service._merge_duplicate(
                            duplicate, "新内容", layer=layer)
                        out["jc"] = jc
                finally:
                    settings.memory_conflict_enabled = False

        asyncio.run(run())
        return out

    def test_contradiction_marks_superseded_and_returns_none(self):
        parent = mock.MagicMock(id=5, title="t", content="旧说法：用户讨厌咖啡",
                                mention_count=0, last_mentioned_at=None)
        out = self._merge(parent, verdict="contradiction")
        assert out["result"] is None                     # 触发 save 正常新增
        assert parent.superseded is True                 # 旧父块标 SUPERSEDED
        assert parent.updated_at is not None             # 刷新 updated_at
        assert "新内容" not in parent.content             # 不拼接共存
        # NLI 判 (旧父块 content, 新内容)
        assert out["jc"].call_args.args == ("旧说法：用户讨厌咖啡", "新内容")

    def test_contradiction_short_layer_no_mention_refresh(self):
        parent = mock.MagicMock(id=5, title="t", content="旧说法",
                                mention_count=0, last_mentioned_at=None)
        out = self._merge(parent, verdict="contradiction", layer="short")
        assert out["result"] is None
        assert parent.superseded is True
        assert parent.mention_count == 0                  # 矛盾路径不刷新提及（旧记忆已 SUPERSEDED）
        assert parent.last_mentioned_at is None

    def test_entailment_appends_like_before(self):
        parent = mock.MagicMock(id=5, title="t", content="用户喜欢咖啡",
                                mention_count=0, last_mentioned_at=None)
        out = self._merge(parent, verdict="entailment")
        assert out["result"]["status"] == "updated"
        assert "用户喜欢咖啡\n新内容" in parent.content    # 追加拼接（现行为）
        assert parent.superseded is not True

    def test_neutral_appends_like_before(self):
        parent = mock.MagicMock(id=5, title="t", content="用户喜欢咖啡",
                                mention_count=0, last_mentioned_at=None)
        out = self._merge(parent, verdict="neutral")
        assert out["result"]["status"] == "updated"
        assert "新内容" in parent.content
        assert parent.superseded is not True

    def test_nli_none_degrades_to_append(self):
        parent = mock.MagicMock(id=5, title="t", content="用户喜欢咖啡",
                                mention_count=0, last_mentioned_at=None)
        out = self._merge(parent, verdict=None)          # NLI 不可用
        assert out["result"]["status"] == "updated"
        assert "新内容" in parent.content                 # 追加（零回归）
        assert parent.superseded is not True

    def test_switch_off_is_fully_old_behavior(self):
        parent = mock.MagicMock(id=5, title="t", content="用户喜欢咖啡",
                                mention_count=0, last_mentioned_at=None)
        out = self._merge(parent, verdict="contradiction", enabled=False)
        assert out["result"]["status"] == "updated"      # 开关关 → 完全旧行为（追加）
        assert "新内容" in parent.content
        assert parent.superseded is not True
        out["jc"].assert_not_called()                    # NLI 完全不调用

    def test_superseded_parent_returns_none_keeps_new(self):
        # module-061 Review 修复：superseded 父块 = 非法合并目标（已被新说法取代、
        # 从召回面过滤）——即使判一致也不追加进被过滤父块（内容不可召回），走正常新增
        parent = mock.MagicMock(id=5, title="t", content="旧说法（已 SUPERSEDED）",
                                mention_count=0, last_mentioned_at=None)
        parent.superseded = True
        out = self._merge(parent, verdict="entailment")  # 守卫先于 NLI
        assert out["result"] is None                      # 走正常新增（save 入库新内容）
        assert "新内容" not in parent.content              # 不追加进 superseded 父块
        assert parent.superseded is True                  # 旧记忆保持 SUPERSEDED
        out["jc"].assert_not_called()                     # 守卫先于 NLI，NLI 不调用


class TestSaveConflictFullFlow:
    """module-061 P1：save 全流程——去重命中判矛盾 → 旧 SUPERSEDED + 新内容按正常新增"""

    def test_conflict_saves_new_as_separate_memory(self):
        fs = _FakeSession(scalar=0)
        fs.get = mock.AsyncMock(return_value=mock.MagicMock(
            id=5, title="旧", content="旧说法：用户讨厌咖啡",
            mention_count=0, last_mentioned_at=None))
        duplicate = mock.MagicMock(id=6, parent_id=5)

        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                with mock.patch("rag.memory.chunker") as chunker_mock:
                    with mock.patch("rag.memory.embedding_service") as emb_mock:
                        with mock.patch.object(memory_service, "_find_duplicate",
                                               new=mock.AsyncMock(return_value=duplicate)) as find:
                            with mock.patch.object(memory_service, "_judge_conflict",
                                                  new=mock.AsyncMock(return_value="contradiction")) as jc:
                                settings.memory_conflict_enabled = True
                                try:
                                    chunker_mock.chunk.return_value = _chunk_single("新说法：用户喜欢咖啡")
                                    emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                                    result = await memory_service.save("新说法：用户喜欢咖啡", "42")
                                finally:
                                    settings.memory_conflict_enabled = False

            assert result["status"] == "saved"           # 新内容按正常新增（非 updated 合并）
            assert result["id"] == 1
            # 父块 + 子块写入新内容（不拼接旧说法）
            contents = [getattr(d, "content", "") for d in fs.added]
            assert any("新说法：用户喜欢咖啡" in c for c in contents)
            assert not any("旧说法" in c for c in contents)
            find.assert_awaited_once()
            jc.assert_awaited_once()

        asyncio.run(run())


# ──────────────────────────────────────────────────────────────
# NLI 裁判封装（module-061 / AC §3）
# ──────────────────────────────────────────────────────────────

class TestNLIJudge:
    """nli_judge 生产封装：延迟加载/失败 None/超时 None/三分类返回"""

    def test_predict_returns_label(self):
        judge = MemoryNLIJudge(model_dir="unused")
        with mock.patch.object(judge, "_predict_sync",
                               return_value="contradiction") as ps:
            result = asyncio.run(judge.predict("旧记忆", "新事实"))
        assert result == "contradiction"
        ps.assert_called_once_with("旧记忆", "新事实")

    def test_predict_inference_failure_returns_none(self):
        judge = MemoryNLIJudge(model_dir="unused")
        with mock.patch.object(judge, "_predict_sync",
                               side_effect=RuntimeError("model down")):
            assert asyncio.run(judge.predict("旧记忆", "新事实")) is None

    def test_predict_timeout_returns_none(self):
        judge = MemoryNLIJudge(model_dir="unused")
        with mock.patch.object(judge, "_predict_sync",
                               side_effect=asyncio.TimeoutError()):
            assert asyncio.run(judge.predict("旧记忆", "新事实")) is None

    def test_predict_empty_input_returns_none(self):
        judge = MemoryNLIJudge(model_dir="unused")
        assert asyncio.run(judge.predict("", "新事实")) is None
        assert asyncio.run(judge.predict("旧记忆", "")) is None

    def test_predict_lazy_load_failure_returns_none(self):
        judge = MemoryNLIJudge(model_dir="nonexistent-dir")
        with mock.patch.object(judge, "_lazy_load", side_effect=RuntimeError("load fail")):
            assert asyncio.run(judge.predict("旧", "新")) is None

    def test_singleton_available(self):
        assert isinstance(nli_judge, MemoryNLIJudge)


class TestJudgeConflict:
    """_judge_conflict 辅助：复用 nli_judge，异常/None → None（上层降级旧行为）"""

    def test_judge_conflict_returns_none_when_unavailable(self):
        async def run():
            with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                            new=mock.AsyncMock(return_value=None)):
                return await memory_service._judge_conflict("旧", "新")

        assert asyncio.run(run()) is None

    def test_judge_conflict_propagates_verdict(self):
        async def run():
            with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                            new=mock.AsyncMock(return_value="contradiction")):
                return await memory_service._judge_conflict("旧", "新")

        assert asyncio.run(run()) == "contradiction"

    def test_judge_conflict_exception_returns_none(self):
        async def run():
            with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                            new=mock.AsyncMock(side_effect=RuntimeError("nli down"))):
                return await memory_service._judge_conflict("旧", "新")

        assert asyncio.run(run()) is None


# ──────────────────────────────────────────────────────────────
# 配置 / 评测基线一致性（AC §1/§4）
# ──────────────────────────────────────────────────────────────

class TestConfig061:
    def test_memory_conflict_disabled_by_default(self):
        assert settings.memory_conflict_enabled is False   # 不预设成功（评测达标才启用）


class TestConflictDataset:
    """记忆矛盾标注集结构 + contradiction_metrics + 达标判定 + fixture"""

    def test_dataset_structure_valid(self):
        data = load_memory_conflict_dataset()
        assert len(data) >= 20
        contradictions = sum(1 for i in data if i["verdict"] == "contradiction")
        assert contradictions >= 15
        assert any(i["verdict"] == "entailment" for i in data)
        assert any(i["verdict"] == "neutral" for i in data)
        # 五类场景齐全（改口/迁移/过时/升级冲突/正例中性）
        scenarios = {i["scenario"] for i in data}
        assert {"改口", "迁移", "过时", "升级冲突", "正例", "中性"} <= scenarios

    def test_contradiction_metrics_pure(self):
        m = contradiction_metrics(
            ["contradiction", "contradiction", "entailment"],
            ["contradiction", "entailment", "contradiction"])
        assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
        assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5

    def test_gate_requires_both_recall_and_precision(self):
        assert GATE_CONTRADICTION_RECALL == 0.8 and GATE_CONTRADICTION_PRECISION == 0.8
        assert gate_passed({"recall": 0.8, "precision": 0.8}) is True
        assert gate_passed({"recall": 0.79, "precision": 0.8}) is False
        assert gate_passed({"recall": 0.8, "precision": 0.79}) is False

    def test_fixture_judge_deterministic(self):
        assert fixture_judge("用户喜欢咖啡", "用户换成喝茶了") == "contradiction"
        assert fixture_judge("用户喜欢咖啡", "用户养了一只猫") == "neutral"
