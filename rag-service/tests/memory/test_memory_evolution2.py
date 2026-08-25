"""Module-062 记忆进化 2 单元测试（WP1 类型判断 + WP2 类型化衰减 + WP3 冷记忆降权 + WP4 矛盾 clf）

覆盖（acceptance-criteria.md §1-§7）：
- WP1 类型判断：extract_facts 输出 type（LLM few-shot + 默认 fact 兜底）/ MemoryTypeClassifier
    推理 / resolve_memory_type 按 memory_type_mode（clf/llm/none）注入 / 评测集与指标
- WP2 类型化衰减：_evolve_recall 按 type 差异化半衰期（preference 慢 / event 快 / 其余现状）/
    存量无 type 零回归 / 开关 false 回退全局 half_life / _type_half_life 映射
- WP3 冷记忆降权：_apply_cold_decay 久未召回 ×0.3-1.0 / 最近 ×1.0 / 存量无时间字段不降权 /
    开关 false 回退 / 刷新 fire-and-forget / recall 集成调用
- WP4 矛盾 clf：_judge_conflict 裁判切换（clf/nli + clf 失败回退 nli）/ MemoryConflictClassifier
    推理 / 特征拼接形状 / 训练集与评测集结构校验 / contradiction P/R/F1 指标
- DDL 幂等 + 配置默认值

实现说明：与 test_memory_correction.py / test_memory.py 同款模式（mock AsyncSession /
_FakeSession / mock LLM / mock 分类器，不依赖真实模型/DB/LLM）；同步用例内 asyncio.run。
conftest autouse 钉住：memory_type_mode='none' / memory_cold_decay_enabled=False /
memory_conflict_judge='nli'（本文件用例体内显式 setattr 覆盖验证，finally 还原）。
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest import mock

from rag.memory.memory import memory_service, _memory_type_of, _cold_ref_time
from rag.memory.memory_type_clf import (
    MemoryTypeClassifier, resolve_memory_type, memory_type_clf,
)
from rag.memory.memory_conflict_clf import (
    MemoryConflictClassifier, memory_conflict_clf,
)
from rag.memory.memory_extractor import extract_facts, MEMORY_TYPES
from eval.datasets.memory_type_dataset import (
    load_memory_type_dataset, type_metrics, gate_passed,
    GATE_TYPE_ACCURACY, fixture_judge,
)
from eval.datasets.build_memory_type_dataset import build_dataset as build_type_train
from eval.datasets.build_memory_conflict_train import build_dataset as build_conflict_train
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


def _parent(doc_id, content, **kwargs):
    """构造父块 MagicMock（含 module-062 字段默认）"""
    now = datetime.now(timezone.utc)
    base = dict(
        id=doc_id, parent_id=None, content=content, title="t",
        created_at=now - timedelta(days=1), last_mentioned_at=None,
        mention_count=0, superseded=False, last_recalled_at=None,
    )
    base.update(kwargs)
    return mock.MagicMock(**base)


# ──────────────────────────────────────────────────────────────
# WP1 类型判断：extract_facts 输出 type（AC §1）
# ──────────────────────────────────────────────────────────────

class _FakeLLM:
    """假 LLM 客户端：generate 返回原始文本（extract_facts mock 用）"""

    def __init__(self, raw: str):
        self.generate = mock.AsyncMock(return_value=raw)


class TestExtractFactsType:
    """module-062 WP1：extract_facts 输出 type（LLM few-shot，缺失/非法默认 fact）"""

    @staticmethod
    def _run(raw: str):
        fake = _FakeLLM(raw)
        async def run():
            with mock.patch("rag.memory.memory_extractor.LLMFactory.get_client",
                            return_value=fake):
                return await extract_facts("问题", "答案", [])
        return asyncio.run(run())

    def test_llm_type_parsing(self):
        facts = self._run(
            '{"facts": [{"content": "用户喜欢喝咖啡", "importance": 0.8, "type": "preference"},'
            ' {"content": "用户是Java开发", "importance": 0.7, "type": "fact"},'
            ' {"content": "用户明天去北京", "importance": 0.6, "type": "event"}]}')
        types = {f["content"]: f["type"] for f in facts}
        assert types["用户喜欢喝咖啡"] == "preference"
        assert types["用户是Java开发"] == "fact"
        assert types["用户明天去北京"] == "event"

    def test_missing_type_defaults_to_fact(self):
        facts = self._run('{"facts": [{"content": "用户是Java开发", "importance": 0.7}]}')
        assert facts[0]["type"] == "fact"

    def test_invalid_type_defaults_to_fact(self):
        facts = self._run(
            '{"facts": [{"content": "用户是Java开发", "importance": 0.7, "type": "bogus"}]}')
        assert facts[0]["type"] == "fact"

    def test_lowercase_type_normalized(self):
        facts = self._run(
            '{"facts": [{"content": "用户喜欢咖啡", "importance": 0.7, "type": "Preference"}]}')
        assert facts[0]["type"] == "preference"

    def test_backward_compat_content_importance_preserved(self):
        facts = self._run(
            '{"facts": [{"content": "用户是Java开发", "importance": 0.8, "type": "fact"}]}')
        assert facts[0]["content"] == "用户是Java开发"
        assert facts[0]["importance"] == 0.8

    def test_extract_failure_returns_empty(self):
        fake = mock.MagicMock()
        fake.generate = mock.AsyncMock(side_effect=RuntimeError("llm down"))
        async def run():
            with mock.patch("rag.memory.memory_extractor.LLMFactory.get_client",
                            return_value=fake):
                return await extract_facts("问题", "答案", [])
        assert asyncio.run(run()) == []

    def test_empty_answer_skips_llm(self):
        fake = mock.MagicMock()
        async def run():
            with mock.patch("rag.memory.memory_extractor.LLMFactory.get_client",
                            return_value=fake):
                return await extract_facts("问题", "  ", [])
        assert asyncio.run(run()) == []
        fake.generate.assert_not_called()


# ──────────────────────────────────────────────────────────────
# WP1 类型判断：resolve_memory_type 生产注入（AC §1）
# ──────────────────────────────────────────────────────────────

class TestResolveMemoryType:
    """按 memory_type_mode 决策记忆类型（clf/llm/none），失败回退 fact"""

    def test_mode_none_defaults_fact(self):
        async def run():
            settings.memory_type_mode = "none"
            try:
                return await resolve_memory_type("用户喜欢咖啡", "preference")
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) == "fact"

    def test_mode_llm_uses_extracted_type(self):
        async def run():
            settings.memory_type_mode = "llm"
            try:
                return await resolve_memory_type("用户喜欢咖啡", "preference")
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) == "preference"

    def test_mode_llm_invalid_defaults_fact(self):
        async def run():
            settings.memory_type_mode = "llm"
            try:
                assert await resolve_memory_type("用户喜欢咖啡", "bogus") == "fact"
                assert await resolve_memory_type("用户喜欢咖啡", None) == "fact"
                return True
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) is True

    def test_mode_clf_uses_classifier(self):
        async def run():
            settings.memory_type_mode = "clf"
            try:
                with mock.patch.object(memory_type_clf, "load",
                                       new=mock.AsyncMock(return_value=True)):
                    with mock.patch.object(memory_type_clf, "classify",
                                           new=mock.AsyncMock(return_value="event")):
                        return await resolve_memory_type("用户明天去北京", "fact")
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) == "event"

    def test_mode_clf_failure_falls_back_to_llm_type(self):
        async def run():
            settings.memory_type_mode = "clf"
            try:
                with mock.patch.object(memory_type_clf, "load",
                                       new=mock.AsyncMock(return_value=True)):
                    with mock.patch.object(memory_type_clf, "classify",
                                           new=mock.AsyncMock(side_effect=RuntimeError("clf down"))):
                        return await resolve_memory_type("用户喜欢咖啡", "preference")
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) == "preference"  # 回退 llm_type

    def test_mode_clf_failure_no_llm_type_defaults_fact(self):
        async def run():
            settings.memory_type_mode = "clf"
            try:
                with mock.patch.object(memory_type_clf, "load",
                                       new=mock.AsyncMock(return_value=True)):
                    with mock.patch.object(memory_type_clf, "classify",
                                           new=mock.AsyncMock(side_effect=RuntimeError("clf down"))):
                        return await resolve_memory_type("用户喜欢咖啡", None)
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) == "fact"

    def test_mode_clf_model_missing_falls_back(self):
        async def run():
            settings.memory_type_mode = "clf"
            try:
                with mock.patch.object(memory_type_clf, "load",
                                       new=mock.AsyncMock(return_value=False)):
                    return await resolve_memory_type("用户喜欢咖啡", "event")
            finally:
                settings.memory_type_mode = "none"
        assert asyncio.run(run()) == "event"  # 模型缺失 → 回退 llm_type


# ──────────────────────────────────────────────────────────────
# WP1 类型判断：MemoryTypeClassifier 推理（AC §1）
# ──────────────────────────────────────────────────────────────

class TestMemoryTypeClassifier:
    """MemoryTypeClassifier：predict_proba / classify / load / fit"""

    @staticmethod
    def _clf(proba_rows, classes):
        clf = MemoryTypeClassifier(model_path="unused", embedding_service=mock.MagicMock())
        clf._embedding_service.embed_text = mock.AsyncMock(return_value=[0.1, 0.2, 0.3])
        clf._model = mock.MagicMock()
        clf._model.predict_proba.return_value = proba_rows
        clf._model.classes_ = list(classes)
        return clf

    def test_predict_proba_keys_complete(self):
        clf = self._clf([[0.6, 0.3, 0.1]], ["event", "fact", "preference"])
        probs = asyncio.run(clf.predict_proba("用户明天去北京"))
        assert set(probs) == set(MEMORY_TYPES)
        assert probs["event"] == 0.6

    def test_classify_returns_top_class(self):
        clf = self._clf([[0.2, 0.7, 0.1]], ["event", "preference", "fact"])
        assert asyncio.run(clf.classify("用户喜欢咖啡")) == "preference"

    def test_predict_proba_unloaded_raises(self):
        clf = MemoryTypeClassifier(model_path="unused", embedding_service=mock.MagicMock())
        try:
            asyncio.run(clf.predict_proba("用户喜欢咖啡"))
            raise AssertionError("模型未加载应抛 RuntimeError")
        except RuntimeError:
            pass

    def test_load_missing_model_returns_false(self):
        clf = MemoryTypeClassifier(model_path="nonexistent.joblib",
                                   embedding_service=mock.MagicMock())
        assert asyncio.run(clf.load()) is False

    def test_singleton_available(self):
        assert isinstance(memory_type_clf, MemoryTypeClassifier)


# ──────────────────────────────────────────────────────────────
# WP2 类型化衰减（AC §2）
# ──────────────────────────────────────────────────────────────

class TestTypeHalfLife:
    """_type_half_life 映射 + _memory_type_of 提取"""

    def test_preference_slow_event_fast_else_short(self):
        assert memory_service._type_half_life("preference") == 30.0
        assert memory_service._type_half_life("event") == 1.0
        assert memory_service._type_half_life("fact") == settings.memory_short_half_life
        assert memory_service._type_half_life("") == settings.memory_short_half_life
        assert memory_service._type_half_life("bogus") == settings.memory_short_half_life

    def test_memory_type_of_extracts_string_only(self):
        assert _memory_type_of(mock.MagicMock(type="event")) == "event"
        assert _memory_type_of(mock.MagicMock(type="PREFERENCE")) == "preference"
        assert _memory_type_of(mock.MagicMock(type="")) == ""
        # MagicMock 缺 type → 自动属性是真值 MagicMock（非 str）→ ""（零回归）
        assert _memory_type_of(mock.MagicMock()) == ""
        assert _memory_type_of(None) == ""


class TestEvolveRecallTypeDecay:
    """_evolve_recall 按 type 差异化半衰期：同 age 不同 type 衰减系数不同"""

    @staticmethod
    def _evolve(parents, memories, child_docs):
        async def run():
            with mock.patch("rag.memory.async_session_factory",
                            _fake_factory(_FakeSession(scalars=parents))):
                return await memory_service._evolve_recall(memories, child_docs, "42")
        return asyncio.run(run())

    def test_preference_decays_slower_than_event(self):
        now = datetime.now(timezone.utc)
        pref = _parent(1, "偏好记忆", type="preference", created_at=now - timedelta(days=5))
        evt = _parent(2, "事件记忆", type="event", created_at=now - timedelta(days=5))
        memories = [
            {"content": "偏好记忆", "score": 0.9, "title": "t", "created_at": ""},
            {"content": "事件记忆", "score": 0.9, "title": "t", "created_at": ""},
        ]
        child_docs = [
            {"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9},
            {"id": 2, "content": "子", "parent_id": 2, "hybrid_score": 0.9},
        ]
        out = self._evolve([pref, evt], memories, child_docs)
        scores = {m["content"]: m["score"] for m in out}
        # 同 age 5 天：preference(30d half-life) 远高于 event(1d half-life)
        assert scores["偏好记忆"] > 0.7     # 0.5^(5/30)≈0.89 → 0.9*0.89≈0.80
        assert scores["事件记忆"] < 0.1     # 0.5^5≈0.031 → 0.9*0.03≈0.028
        assert scores["偏好记忆"] > scores["事件记忆"]

    def test_legacy_no_type_uses_short_half_life(self):
        now = datetime.now(timezone.utc)
        legacy = _parent(1, "无类型记忆", created_at=now - timedelta(days=5))
        memories = [{"content": "无类型记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._evolve([legacy], memories, child_docs)
        # 存量无 type → memory_short_half_life=3：0.9 * 0.5^(5/3) ≈ 0.28（现状行为）
        assert 0.2 < out[0]["score"] < 0.35

    def test_switch_off_falls_back_to_global_half_life(self):
        now = datetime.now(timezone.utc)
        pref = _parent(1, "偏好记忆", type="preference", created_at=now - timedelta(days=5))
        memories = [{"content": "偏好记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        async def run():
            settings.memory_type_decay_enabled = False
            try:
                with mock.patch("rag.memory.async_session_factory",
                                _fake_factory(_FakeSession(scalars=[pref]))):
                    out = await memory_service._evolve_recall(memories, child_docs, "42")
            finally:
                settings.memory_type_decay_enabled = True
            return out
        out = asyncio.run(run())
        # 开关关 → 全局 half_life=3（即使 type=preference 也按 3 天，零回归）
        assert 0.2 < out[0]["score"] < 0.35


# ──────────────────────────────────────────────────────────────
# WP3 冷记忆降权（AC §3）
# ──────────────────────────────────────────────────────────────

class TestColdDecay:
    """_apply_cold_decay：久未召回 ×0.3-1.0 / 最近 ×1.0 / 存量不降权 / 刷新"""

    @staticmethod
    def _apply(parents, memories, child_docs, enabled=True):
        async def run():
            settings.memory_cold_decay_enabled = enabled
            try:
                with mock.patch("rag.memory.async_session_factory",
                                _fake_factory(_FakeSession(scalars=parents))):
                    return await memory_service._apply_cold_decay(
                        memories, child_docs, "42")
            finally:
                settings.memory_cold_decay_enabled = False
        return asyncio.run(run())

    def test_long_unrecalled_downgraded(self):
        now = datetime.now(timezone.utc)
        cold = _parent(1, "旧记忆", last_recalled_at=now - timedelta(days=90))
        memories = [{"content": "旧记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([cold], memories, child_docs)
        # 90 天 → factor = max(0.3, 1.0-(90-30)/100)=0.4 → 0.9*0.4=0.36
        assert out[0]["score"] == 0.36

    def test_recent_recall_keeps_full_score(self):
        now = datetime.now(timezone.utc)
        warm = _parent(1, "新记忆", last_recalled_at=now - timedelta(days=1))
        memories = [{"content": "新记忆", "score": 0.8, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([warm], memories, child_docs)
        assert out[0]["score"] == 0.8  # <30 天不降权

    def test_cold_factor_floor_03(self):
        now = datetime.now(timezone.utc)
        very_cold = _parent(1, "超旧记忆", last_recalled_at=now - timedelta(days=200))
        memories = [{"content": "超旧记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([very_cold], memories, child_docs)
        # 200 天 → 1.0-(200-30)/100<0 → 下限 0.3
        assert out[0]["score"] == 0.27

    def test_legacy_no_last_recalled_uses_created_at(self):
        now = datetime.now(timezone.utc)
        legacy = _parent(1, "存量记忆", created_at=now - timedelta(days=60),
                         last_recalled_at=None)
        memories = [{"content": "存量记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([legacy], memories, child_docs)
        # 无 last_recalled_at → 按 created_at 60 天 → factor=max(0.3,0.7)=0.7
        assert out[0]["score"] == 0.63

    def test_missing_times_keeps_full_score(self):
        parent = mock.MagicMock(id=1, content="无时间记忆", superseded=False)
        memories = [{"content": "无时间记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([parent], memories, child_docs)
        assert out[0]["score"] == 0.9  # 无参考时间 → 不降权（零回归）

    def test_naive_db_roundtrip_still_decays(self):
        # Review 修复锁定：PG TIMESTAMP 列无 tz——_refresh_last_recalled 写 aware UTC
        # 落库丢 tz、读回 naive（tzinfo=None）。naive 按 UTC 解释仍应正常降权；
        # 修复前 now(aware)-ref(naive) 抛 TypeError 被吞 → 恒 ×1.0（WP3 生产失效）。
        naive_90d = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
        cold = _parent(1, "naive旧记忆", last_recalled_at=naive_90d)
        memories = [{"content": "naive旧记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([cold], memories, child_docs)
        # 90 天 → factor = max(0.3, 1.0-(90-30)/100)=0.4 → 0.9*0.4=0.36
        assert out[0]["score"] == 0.36

    def test_switch_off_no_decay(self):
        now = datetime.now(timezone.utc)
        cold = _parent(1, "旧记忆", last_recalled_at=now - timedelta(days=90))
        memories = [{"content": "旧记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        out = self._apply([cold], memories, child_docs, enabled=False)
        assert out[0]["score"] == 0.9  # 开关关 → 完全不降权

    def test_db_failure_keeps_original(self):
        memories = [{"content": "旧记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        async def run():
            settings.memory_cold_decay_enabled = True
            try:
                with mock.patch("rag.memory.async_session_factory",
                                side_effect=RuntimeError("db down")):
                    out = await memory_service._apply_cold_decay(
                        memories, child_docs, "42")
            finally:
                settings.memory_cold_decay_enabled = False
            return out
        assert asyncio.run(run()) == memories  # 加载失败 → 保持原分（fail-open）

    def test_resort_by_new_score(self):
        now = datetime.now(timezone.utc)
        cold = _parent(1, "冷记忆", last_recalled_at=now - timedelta(days=90))
        warm = _parent(2, "热记忆", last_recalled_at=now - timedelta(days=1))
        memories = [
            {"content": "冷记忆", "score": 0.9, "title": "t", "created_at": ""},
            {"content": "热记忆", "score": 0.8, "title": "t", "created_at": ""},
        ]
        child_docs = [
            {"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9},
            {"id": 2, "content": "子", "parent_id": 2, "hybrid_score": 0.8},
        ]
        out = self._apply([cold, warm], memories, child_docs)
        # 冷记忆 0.36 < 热记忆 0.8 → 降权后热记忆排前
        assert [m["content"] for m in out] == ["热记忆", "冷记忆"]

    def test_refresh_last_recalled_fire_forget(self):
        now = datetime.now(timezone.utc)
        cold = _parent(1, "旧记忆", last_recalled_at=now - timedelta(days=90))
        memories = [{"content": "旧记忆", "score": 0.9, "title": "t", "created_at": ""}]
        child_docs = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        async def run():
            settings.memory_cold_decay_enabled = True
            try:
                with mock.patch("rag.memory.async_session_factory",
                                _fake_factory(_FakeSession(scalars=[cold]))):
                    with mock.patch.object(memory_service, "_refresh_last_recalled",
                                           new=mock.AsyncMock()) as refresh:
                        await memory_service._apply_cold_decay(memories, child_docs, "42")
                        # fire-and-forget：create_task 调度时即记录调用
                        return refresh
            finally:
                settings.memory_cold_decay_enabled = False
        refresh = asyncio.run(run())
        assert refresh.await_count == 1
        refresh.assert_awaited_with([1])


class TestRefreshLastRecalled:
    """_refresh_last_recalled：UPDATE last_recalled_at=now（fire-and-forget 降级语义）"""

    def test_refreshes_last_recalled_issues_update(self):
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(session)):
                await memory_service._refresh_last_recalled([1, 2])
        asyncio.run(run())
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    def test_db_failure_degrades(self):
        async def run():
            with mock.patch("rag.memory.async_session_factory",
                            side_effect=RuntimeError("db down")):
                await memory_service._refresh_last_recalled([1, 2])  # 不抛，仅日志降级
        asyncio.run(run())

    def test_empty_ids_noop(self):
        async def run():
            with mock.patch("rag.memory.async_session_factory") as fac:
                await memory_service._refresh_last_recalled([])
                fac.assert_not_called()
        asyncio.run(run())


class TestRecallIntegratesColdDecay:
    """recall 长期层集成：检索 → _apply_cold_decay → 动态 K 截断"""

    def test_recall_invokes_cold_decay(self):
        children = [{"id": 1, "content": "子", "parent_id": 1, "hybrid_score": 0.9}]
        memories = [{"content": "记忆", "score": 0.9, "title": "t", "created_at": ""}]
        async def run():
            with mock.patch("rag.memory.hybrid_retriever") as ret:
                ret.retrieve = mock.AsyncMock(return_value=children)
                with mock.patch("rag.memory.embedding_service") as emb:
                    emb.embed_text = mock.AsyncMock(return_value=[1.0, 0.0])
                    with mock.patch.object(memory_service, "_child_embeddings",
                                           new=mock.AsyncMock(return_value={1: [0.9, 0.0]})):
                        with mock.patch.object(memory_service, "_expand_to_parents",
                                               new=mock.AsyncMock(return_value=memories)):
                            with mock.patch.object(memory_service, "_apply_cold_decay",
                                                   new=mock.AsyncMock(side_effect=lambda m, d, i: m)) as cold:
                                await memory_service.recall("q", "42", top_k=5)
                                return cold
        cold = asyncio.run(run())
        cold.assert_awaited_once()


# ──────────────────────────────────────────────────────────────
# WP1/WP2 类型注入：save 写入 type（AC §2）
# ──────────────────────────────────────────────────────────────

class TestMemoryTypeSave:
    """save/save_short 将 memory_type 写入 Document（默认 fact 零回归）"""

    def test_save_writes_type_to_documents(self):
        fs = _FakeSession(scalar=0)
        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                with mock.patch("rag.memory.chunker") as chunker_mock:
                    with mock.patch("rag.memory.embedding_service") as emb_mock:
                        chunker_mock.chunk.return_value = _chunk_single("用户喜欢咖啡")
                        emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                        await memory_service.save("用户喜欢咖啡", "42", memory_type="preference")
        asyncio.run(run())
        assert {getattr(d, "type", None) for d in fs.added} == {"preference"}

    def test_save_default_type_fact(self):
        fs = _FakeSession(scalar=0)
        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                with mock.patch("rag.memory.chunker") as chunker_mock:
                    with mock.patch("rag.memory.embedding_service") as emb_mock:
                        chunker_mock.chunk.return_value = _chunk_single("用户是Java开发")
                        emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                        await memory_service.save("用户是Java开发", "42")
        asyncio.run(run())
        assert {getattr(d, "type", None) for d in fs.added} == {"fact"}

    def test_save_short_writes_type(self):
        fs = _FakeSession(scalar=0)
        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                with mock.patch("rag.memory.chunker") as chunker_mock:
                    with mock.patch("rag.memory.embedding_service") as emb_mock:
                        chunker_mock.chunk.return_value = _chunk_single("用户明天去北京")
                        emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                        await memory_service.save_short("用户明天去北京", "42", memory_type="event")
        asyncio.run(run())
        assert {getattr(d, "type", None) for d in fs.added} == {"event"}


class TestPersistMemoryTypeInjection:
    """_persist_memory 按 memory_type_mode 注入类型（engine 接线）"""

    @staticmethod
    def _run(mode, fact, expected_type):
        from rag.engine import rag_engine
        out = {}

        async def run():
            settings.memory_type_mode = mode
            try:
                with mock.patch("rag.engine.extract_facts",
                                new=mock.AsyncMock(return_value=[fact])) as extract:
                    with mock.patch("rag.engine.memory_service.save",
                                    new=mock.AsyncMock()) as save:
                        with mock.patch("rag.engine.memory_service.save_short",
                                        new=mock.AsyncMock()) as save_short:
                            await rag_engine._persist_memory("问题", "答案", "42", [])
                            out["save_kwargs"] = save.call_args_list[0].kwargs
                            out["short_kwargs"] = save_short.call_args_list[0].kwargs
            finally:
                settings.memory_type_mode = "none"
        asyncio.run(run())
        assert out["save_kwargs"].get("memory_type") == expected_type
        assert out["short_kwargs"].get("memory_type") == expected_type

    def test_mode_llm_injects_extracted_type(self):
        self._run("llm", {"content": "用户喜欢喝咖啡", "importance": 0.9, "type": "preference"},
                  "preference")

    def test_mode_none_defaults_fact(self):
        self._run("none", {"content": "用户明天去北京", "importance": 0.9, "type": "event"},
                  "fact")


# ──────────────────────────────────────────────────────────────
# WP4 矛盾检测：_judge_conflict 裁判切换（AC §4）
# ──────────────────────────────────────────────────────────────

class TestJudgeConflictDispatch:
    """_judge_conflict 按 memory_conflict_judge 选裁判（clf/nli + clf 失败回退 nli）"""

    def test_judge_nli_default_path(self):
        async def run():
            settings.memory_conflict_judge = "nli"
            try:
                with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                new=mock.AsyncMock(return_value="contradiction")):
                    return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "contradiction"

    def test_judge_clf_uses_classifier(self):
        async def run():
            settings.memory_conflict_judge = "clf"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="contradiction")) as clf_predict:
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock()) as nli_predict:
                            result = await memory_service._judge_conflict("旧", "新")
                            return result, clf_predict, nli_predict
            finally:
                settings.memory_conflict_judge = "nli"
        result, clf_predict, nli_predict = asyncio.run(run())
        assert result == "contradiction"
        clf_predict.assert_awaited_once_with("旧", "新")
        nli_predict.assert_not_awaited()  # clf 命中不调 NLI

    def test_judge_clf_failure_falls_back_to_nli(self):
        async def run():
            settings.memory_conflict_judge = "clf"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(side_effect=RuntimeError("clf down"))):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="entailment")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "entailment"

    def test_judge_clf_none_falls_back_to_nli(self):
        async def run():
            settings.memory_conflict_judge = "clf"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value=None)):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="neutral")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "neutral"

    def test_judge_clf_model_missing_falls_back_to_nli(self):
        async def run():
            settings.memory_conflict_judge = "clf"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=False)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock()) as clf_predict:
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="contradiction")):
                            result = await memory_service._judge_conflict("旧", "新")
                            return result, clf_predict
            finally:
                settings.memory_conflict_judge = "nli"
        result, clf_predict = asyncio.run(run())
        assert result == "contradiction"      # 模型缺失 → 回退 NLI
        clf_predict.assert_not_awaited()


class TestJudgeConflictDual:
    """_judge_conflict 双判共识（module-070）：dual_verdict 决策表 + 对称回退"""

    def test_judge_dual_both_contradiction(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="contradiction")) as clf_predict:
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="contradiction")) as nli_predict:
                            result = await memory_service._judge_conflict("旧", "新")
                            return result, clf_predict, nli_predict
            finally:
                settings.memory_conflict_judge = "nli"
        result, clf_predict, nli_predict = asyncio.run(run())
        assert result == "contradiction"          # 双确认才标 superseded
        clf_predict.assert_awaited_once_with("旧", "新")
        nli_predict.assert_awaited_once_with("旧", "新")

    def test_judge_dual_nli_contradiction_clf_non_conflict(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="non_conflict")):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="contradiction")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "conflict_hint"   # 单判矛盾 → 新旧并存

    def test_judge_dual_clf_contradiction_nli_neutral(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="contradiction")):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="neutral")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "conflict_hint"   # 单判矛盾（反方向）→ 并存

    def test_judge_dual_both_non_conflict_returns_nli_label(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="non_conflict")):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="entailment")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "entailment"      # 双方非矛盾 → nli 标签

    def test_judge_dual_clf_model_missing_uses_nli(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=False)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock()) as clf_predict:
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="contradiction")):
                            result = await memory_service._judge_conflict("旧", "新")
                            return result, clf_predict
            finally:
                settings.memory_conflict_judge = "nli"
        result, clf_predict = asyncio.run(run())
        assert result == "contradiction"              # clf 缺失 → nli 单判 = 现状零回归
        clf_predict.assert_not_awaited()

    def test_judge_dual_clf_predict_exception_uses_nli(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(side_effect=RuntimeError("clf down"))):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="neutral")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "neutral"        # clf 异常 → nli 单判

    def test_judge_dual_nli_none_uses_clf(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="contradiction")):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value=None)):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "contradiction"  # nli 不可用 → clf 单判（新增对称回退）

    def test_judge_dual_both_unavailable_returns_none(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value=None)):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value=None)):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) is None             # 双方不可用 → None（上层追加，零回归）

    def test_dual_verdict_pure_function_table(self):
        """dual_verdict 纯函数决策表全覆盖（含 None 组合）"""
        from rag.memory.memory import dual_verdict
        # 双方都出 verdict
        assert dual_verdict("contradiction", "contradiction") == "contradiction"
        assert dual_verdict("contradiction", "non_conflict") == "conflict_hint"
        assert dual_verdict("entailment", "contradiction") == "conflict_hint"
        assert dual_verdict("neutral", "contradiction") == "conflict_hint"
        assert dual_verdict("entailment", "non_conflict") == "entailment"
        assert dual_verdict("neutral", "non_conflict") == "neutral"
        # 一方 None → 另一方单判（对称回退）
        assert dual_verdict(None, "contradiction") == "contradiction"
        assert dual_verdict(None, "non_conflict") == "non_conflict"
        assert dual_verdict("contradiction", None) == "contradiction"
        assert dual_verdict("neutral", None) == "neutral"
        # 双方 None → None（旧行为）
        assert dual_verdict(None, None) is None


class TestMergeConflictHint:
    """_merge_duplicate 收到 conflict_hint（module-070 双判不一致）→ 追加拼接不标 superseded"""

    def test_conflict_hint_appends_and_keeps_superseded_false(self):
        parent = mock.MagicMock(id=5, title="t", content="用户喜欢咖啡",
                                mention_count=0, last_mentioned_at=None)
        duplicate = mock.MagicMock(id=6, parent_id=parent.id)
        session = mock.MagicMock()
        session.get = mock.AsyncMock(return_value=parent)
        session.commit = mock.AsyncMock()
        out = {}

        async def run():
            with mock.patch("rag.memory.async_session_factory", _fake_factory(session)):
                settings.memory_conflict_enabled = True
                try:
                    with mock.patch("rag.memory.memory.logger.info") as log_info:
                        with mock.patch.object(memory_service, "_judge_conflict",
                                               new=mock.AsyncMock(return_value="conflict_hint")):
                            out["result"] = await memory_service._merge_duplicate(
                                duplicate, "新内容")
                            out["log"] = log_info
                finally:
                    settings.memory_conflict_enabled = False

        asyncio.run(run())
        assert out["result"]["status"] == "updated"        # 追加拼接（库内条数不涨）
        assert "用户喜欢咖啡\n新内容" in parent.content     # 新旧并存
        assert parent.superseded is not True                # 不标 SUPERSEDED（保守不冤枉）
        # 日志分支：双判不一致提示
        assert any("记忆冲突提示（双判不一致）" in str(c)
                   for c in out["log"].call_args_list)


class TestMemoryConflictClassifier:
    """MemoryConflictClassifier：predict / predict_proba / 特征形状"""

    def test_predict_returns_contradiction(self):
        clf = MemoryConflictClassifier(model_path="unused", embedding_service=mock.MagicMock())
        clf._embedding_service.embed_text = mock.AsyncMock(return_value=[0.1, 0.2, 0.3])
        clf._model = mock.MagicMock()
        clf._model.predict_proba.return_value = [[0.8, 0.2]]
        clf._model.classes_ = ["contradiction", "non_conflict"]
        assert asyncio.run(clf.predict("用户喜欢咖啡", "用户讨厌咖啡")) == "contradiction"

    def test_predict_returns_non_conflict(self):
        clf = MemoryConflictClassifier(model_path="unused", embedding_service=mock.MagicMock())
        clf._embedding_service.embed_text = mock.AsyncMock(return_value=[0.1, 0.2, 0.3])
        clf._model = mock.MagicMock()
        clf._model.predict_proba.return_value = [[0.3, 0.7]]
        clf._model.classes_ = ["contradiction", "non_conflict"]
        assert asyncio.run(clf.predict("用户喜欢咖啡", "用户养了一只猫")) == "non_conflict"

    def test_predict_failure_returns_none(self):
        clf = MemoryConflictClassifier(model_path="unused", embedding_service=mock.MagicMock())
        clf._embedding_service.embed_text = mock.AsyncMock(side_effect=RuntimeError("embed down"))
        assert asyncio.run(clf.predict("旧", "新")) is None

    def test_predict_empty_input_returns_none(self):
        clf = MemoryConflictClassifier(model_path="unused", embedding_service=mock.MagicMock())
        assert asyncio.run(clf.predict("", "新")) is None
        assert asyncio.run(clf.predict("旧", "")) is None

    def test_feature_concat_shape(self):
        vec = MemoryConflictClassifier._feature([1.0, 0.0], [0.0, 1.0])
        # [a(2), b(2), a-b(2), |a-b|(2)] = 8
        assert len(vec) == 8
        assert vec[:2] == [1.0, 0.0]
        assert vec[2:4] == [0.0, 1.0]
        assert vec[4:6] == [1.0, -1.0]
        assert vec[6:8] == [1.0, 1.0]

    def test_load_missing_model_returns_false(self):
        clf = MemoryConflictClassifier(model_path="nonexistent.joblib",
                                       embedding_service=mock.MagicMock())
        assert asyncio.run(clf.load()) is False

    def test_singleton_available(self):
        assert isinstance(memory_conflict_clf, MemoryConflictClassifier)


# ──────────────────────────────────────────────────────────────
# WP1/WP4 评测基线一致性（AC §1/§4/§7）
# ──────────────────────────────────────────────────────────────

class TestTypeEvalBaseline:
    """类型评测集结构 + 指标纯函数 + 达标判定 + fixture"""

    def test_dataset_structure_valid(self):
        data = load_memory_type_dataset()
        assert len(data) >= 30
        for cls in MEMORY_TYPES:
            assert sum(1 for i in data if i["type"] == cls) >= 10

    def test_type_metrics_pure(self):
        m = type_metrics(
            ["preference", "fact", "event", "preference"],
            ["preference", "fact", "event", "fact"])
        assert m["accuracy"] == 0.75

    def test_gate_accuracy(self):
        assert GATE_TYPE_ACCURACY == 0.8
        assert gate_passed({"accuracy": 0.8}) is True
        assert gate_passed({"accuracy": 0.79}) is False

    def test_fixture_judge_deterministic(self):
        assert fixture_judge({"content": "用户明天去北京"}) == "event"
        assert fixture_judge({"content": "用户喜欢喝咖啡"}) == "preference"
        assert fixture_judge({"content": "用户是Java开发"}) == "fact"

    def test_train_dataset_balance_and_no_overlap(self):
        train = build_type_train()
        types = [s["type"] for s in train]
        assert len(train) >= 120
        for cls in MEMORY_TYPES:
            assert types.count(cls) >= 40
        eval_data = load_memory_type_dataset()
        eval_contents = {item["content"] for item in eval_data}
        assert not (eval_contents & {s["content"] for s in train}), "训练集与评测集零重叠"


class TestConflictEvalBaseline:
    """矛盾训练集结构（100+）+ 与评测集零重叠 + contradiction 指标"""

    def test_train_dataset_balance_and_no_overlap(self):
        from eval.datasets.memory_conflict_dataset import MEMORY_CONFLICT_DATASET
        train = build_conflict_train()
        labels = [s["label"] for s in train]
        assert len(train) >= 100
        assert labels.count("contradiction") >= 40
        assert labels.count("non_conflict") >= 40
        eval_texts = set()
        for item in MEMORY_CONFLICT_DATASET:
            eval_texts.add(item["premise"])
            eval_texts.add(item["hypothesis"])
        train_texts = {s for pair in train for s in (pair["premise"], pair["hypothesis"])}
        assert not (eval_texts & train_texts), "矛盾训练集与评测集零重叠"

    def test_contradiction_metrics_reused(self):
        from eval.datasets.memory_conflict_dataset import contradiction_metrics
        # clf 二分类输出（contradiction/non_conflict）与评测三分类可混算 contradiction P/R/F1
        m = contradiction_metrics(
            ["contradiction", "contradiction", "entailment"],
            ["contradiction", "non_conflict", "non_conflict"])
        assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 0
        assert m["precision"] == 1.0 and m["recall"] == 0.5


# ──────────────────────────────────────────────────────────────
# DDL 幂等 + 配置（AC §5/§6/§7）
# ──────────────────────────────────────────────────────────────

class TestDdlIdempotency:
    """documents 加列 DDL：IF NOT EXISTS + 默认值兜底存量"""

    def test_type_columns_ddl_idempotent(self):
        from src.database import MEMORY_TYPE_COLUMNS_DDL
        assert "ADD COLUMN IF NOT EXISTS type VARCHAR(16) NOT NULL DEFAULT 'fact'" in MEMORY_TYPE_COLUMNS_DDL
        assert "ADD COLUMN IF NOT EXISTS last_recalled_at TIMESTAMP" in MEMORY_TYPE_COLUMNS_DDL

    def test_ensure_memory_type_columns_executes_statements(self):
        from src.database import ensure_memory_type_columns
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        async def run():
            with mock.patch("src.database.async_session_factory", _fake_factory(session)):
                await ensure_memory_type_columns()
        asyncio.run(run())
        # 2 ALTER + 2 COMMENT = 4 条语句拆分执行
        assert session.execute.await_count == 4
        session.commit.assert_awaited_once()


class TestConfig062:
    """module-062 配置默认值（半衰期/冷降权参数/类型注入模式）"""

    def test_type_half_life_params(self):
        assert settings.memory_type_half_life_preference == 30.0
        assert settings.memory_type_half_life_event == 1.0
        assert settings.memory_type_decay_enabled is True  # 生产默认开（存量无 type 零回归）

    def test_cold_decay_params(self):
        assert settings.memory_cold_decay_days == 30
        assert settings.memory_cold_decay_min == 0.3

    def test_type_mode_conservative_default(self):
        # 测试环境由 conftest 钉住 none（类型注入不依赖真实分类器/LLM，hermetic）；
        # 生产默认 = WP1 实测 winner（clf，Accuracy 1.0000 与 LLM 同分，取零成本者）
        assert settings.memory_type_mode == "none"

    def test_conflict_enabled_default_off(self):
        assert settings.memory_conflict_enabled is False  # 不预设成功（Precision≥0.8 达标才启用）


class TestColdRefTime:
    """_cold_ref_time：last_recalled_at 优先 / created_at 兜底 / 皆无 None"""

    def test_prefers_last_recalled_at(self):
        now = datetime.now(timezone.utc)
        doc = mock.MagicMock(last_recalled_at=now, created_at=now - timedelta(days=10))
        assert _cold_ref_time(doc) == now

    def test_falls_back_to_created_at(self):
        now = datetime.now(timezone.utc)
        doc = mock.MagicMock(last_recalled_at=None, created_at=now)
        assert _cold_ref_time(doc) == now

    def test_magicmock_attrs_are_not_datetime(self):
        # MagicMock 自动属性是真值 MagicMock 非 datetime → None（零回归）
        assert _cold_ref_time(mock.MagicMock()) is None
        assert _cold_ref_time(None) is None

    def test_naive_ref_normalized_to_utc(self):
        # Review 修复锁定：PG TIMESTAMP 落库往返读回 naive（tzinfo=None）→ 按 UTC 解释
        naive = datetime(2026, 1, 1, 12, 0, 0)
        doc = mock.MagicMock(last_recalled_at=naive, created_at=naive)
        out = _cold_ref_time(doc)
        assert out.tzinfo is timezone.utc  # 恒为 tz-aware，_apply_cold_decay 减法不再抛 TypeError
        assert out.replace(tzinfo=None) == naive  # 墙钟时间不变，仅补时区


# ──────────────────────────────────────────────────────────────
# module-070 Tester 复验：dual_verdict 决策表逐行 + dual 分支补漏
# + eval 脚本 --judge dual + scores["judge"] 落库
# ──────────────────────────────────────────────────────────────

class TestDualVerdictDecisionTable:
    """dual_verdict 纯函数 7 行决策表逐行枚举（Tester 复验，对齐 plan §2 决策表）

    生产 _judge_conflict 与 eval dual_judge 均引用本函数（单一来源 AC-23）；
    本测试把 7 行决策表逐行断言，含 clf 侧非矛盾变体（entailment/neutral）
    与一方/双方不可用（None）的对称回退。
    """

    def test_row_by_row_decision_table(self):
        from rag.memory.memory import dual_verdict
        # R1 双确认才 superseded：双 contradiction → "contradiction"
        assert dual_verdict("contradiction", "contradiction") == "contradiction"
        # R2 nli 矛盾 + clf 非矛盾 → "conflict_hint"（新旧并存，不标 superseded）
        assert dual_verdict("contradiction", "non_conflict") == "conflict_hint"
        assert dual_verdict("contradiction", "entailment") == "conflict_hint"
        assert dual_verdict("contradiction", "neutral") == "conflict_hint"
        # R3 clf 矛盾 + nli 非矛盾 → "conflict_hint"（反方向）
        assert dual_verdict("entailment", "contradiction") == "conflict_hint"
        assert dual_verdict("neutral", "contradiction") == "conflict_hint"
        # R4 双方非矛盾 → nli 标签（module-046 追加拼接行为不变）
        assert dual_verdict("entailment", "non_conflict") == "entailment"
        assert dual_verdict("neutral", "non_conflict") == "neutral"
        assert dual_verdict("entailment", "entailment") == "entailment"
        assert dual_verdict("neutral", "neutral") == "neutral"
        # R5 nli 不可用（None/超时/异常）→ clf 单判（新增对称回退）
        assert dual_verdict(None, "contradiction") == "contradiction"
        assert dual_verdict(None, "non_conflict") == "non_conflict"
        # R6 clf 不可用（模型缺失/None/异常）→ nli 单判（= 现状 judge="nli" 零回归）
        assert dual_verdict("contradiction", None) == "contradiction"
        assert dual_verdict("entailment", None) == "entailment"
        assert dual_verdict("neutral", None) == "neutral"
        # R7 双方不可用 → None（上层追加，旧行为零回归）
        assert dual_verdict(None, None) is None


class TestJudgeConflictDualTester:
    """Tester 复验补充：_judge_conflict dual 分支降级路径补漏（nli 异常 / clf None）"""

    def test_judge_dual_nli_exception_uses_clf(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value="contradiction")) as clf_predict:
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(side_effect=RuntimeError("nli down"))):
                            result = await memory_service._judge_conflict("旧", "新")
                            return result, clf_predict
            finally:
                settings.memory_conflict_judge = "nli"
        result, clf_predict = asyncio.run(run())
        assert result == "contradiction"          # nli 异常 → clf 单判（对称回退）
        clf_predict.assert_awaited_once_with("旧", "新")

    def test_judge_dual_clf_predict_none_uses_nli(self):
        async def run():
            settings.memory_conflict_judge = "dual"
            try:
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                                new=mock.AsyncMock(return_value=True)):
                    with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                    new=mock.AsyncMock(return_value=None)):
                        with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                        new=mock.AsyncMock(return_value="contradiction")):
                            return await memory_service._judge_conflict("旧", "新")
            finally:
                settings.memory_conflict_judge = "nli"
        assert asyncio.run(run()) == "contradiction"  # clf predict None → nli 单判（warning 分支）


class TestEvalScriptDual:
    """module-070 eval 脚本：dual_judge 复用生产 dual_verdict + --judge dual + scores['judge'] 落库"""

    def test_dual_judge_both_contradiction(self):
        from eval.datasets.memory_conflict_dataset import dual_judge
        async def run():
            with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                            new=mock.AsyncMock(return_value=True)):
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                new=mock.AsyncMock(return_value="contradiction")) as clf_predict:
                    with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                    new=mock.AsyncMock(return_value="contradiction")) as nli_predict:
                        result = await dual_judge("旧", "新")
                        return result, clf_predict, nli_predict
        result, clf_predict, nli_predict = asyncio.run(run())
        assert result == "contradiction"
        clf_predict.assert_awaited_once_with("旧", "新")
        nli_predict.assert_awaited_once_with("旧", "新")

    def test_dual_judge_single_contradiction_maps_to_neutral(self):
        from eval.datasets.memory_conflict_dataset import dual_judge
        async def run():
            with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                            new=mock.AsyncMock(return_value=True)):
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                new=mock.AsyncMock(return_value="contradiction")):
                    with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                    new=mock.AsyncMock(return_value="neutral")):
                        return await dual_judge("旧", "新")
        # conflict_hint → neutral（run_eval VERDICTS 校验 + contradiction P/R 主指标等价）
        assert asyncio.run(run()) == "neutral"

    def test_dual_judge_clf_load_false_uses_nli(self):
        from eval.datasets.memory_conflict_dataset import dual_judge
        async def run():
            with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                            new=mock.AsyncMock(return_value=False)):
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                new=mock.AsyncMock()) as clf_predict:
                    with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                    new=mock.AsyncMock(return_value="contradiction")):
                        result = await dual_judge("旧", "新")
                        return result, clf_predict
        result, clf_predict = asyncio.run(run())
        assert result == "contradiction"          # clf 模型缺失 → nli 单判（fail-open 零回归）
        clf_predict.assert_not_awaited()

    def test_dual_judge_both_unavailable_returns_none(self):
        from eval.datasets.memory_conflict_dataset import dual_judge
        async def run():
            with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.load",
                            new=mock.AsyncMock(return_value=True)):
                with mock.patch("rag.memory.memory_conflict_clf.memory_conflict_clf.predict",
                                new=mock.AsyncMock(return_value=None)):
                    with mock.patch("rag.memory.nli_judge.nli_judge.predict",
                                    new=mock.AsyncMock(return_value=None)):
                        return await dual_judge("旧", "新")
        # 双方不可用 → None（run_eval 侧按 skip/neutral 计数，存量语义 AC-18）
        assert asyncio.run(run()) is None

    def test_main_judge_dual_scores_judge_persisted(self):
        import eval.datasets.memory_conflict_dataset as mod
        captured = {}

        async def fake_run_eval(judge=None, dataset=None, limit=None):
            captured["judge"] = judge
            return {"accuracy_3class": 0.5286, "precision": 0.9412, "recall": 0.4,
                    "f1": 0.5614, "tp": 16, "fp": 1, "fn": 24,
                    "dataset_size": 70, "evaluated": 70, "skipped": 0}, [], []

        async def fake_record(scores, per_question):
            captured["scores"] = dict(scores)
            return "c" * 40, 48

        with mock.patch.object(mod, "run_eval", new=fake_run_eval):
            with mock.patch.object(mod, "record_eval_run", new=fake_record):
                with mock.patch("sys.argv", ["memory_conflict_dataset", "--judge", "dual"]):
                    asyncio.run(mod.main())
        assert captured["judge"] is mod.dual_judge    # --judge dual → dual_judge 接线
        assert captured["scores"]["judge"] == "dual"  # scores['judge'] 落库区分三方案（AC-13）

    def test_main_judge_nli_clf_selection_and_invalid_rejected(self):
        import eval.datasets.memory_conflict_dataset as mod
        selected = {}

        async def fake_run_eval(judge=None, dataset=None, limit=None):
            selected["judge"] = judge
            return {"accuracy_3class": 0.0, "precision": 0.0, "recall": 0.0,
                    "f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
                    "dataset_size": 1, "evaluated": 1, "skipped": 0}, [], []

        async def fake_record(scores, per_question):
            return "c" * 40, 0

        with mock.patch.object(mod, "run_eval", new=fake_run_eval):
            with mock.patch.object(mod, "record_eval_run", new=fake_record):
                with mock.patch("sys.argv", ["memory_conflict_dataset", "--judge", "nli"]):
                    asyncio.run(mod.main())
                assert selected["judge"] is mod.real_judge   # --judge nli 存量行为不变
                with mock.patch("sys.argv", ["memory_conflict_dataset", "--judge", "clf"]):
                    asyncio.run(mod.main())
                assert selected["judge"] is mod.clf_judge    # --judge clf 存量行为不变
        # 非法 judge → argparse SystemExit（不静默，AC-16 同口径）
        exited = False
        with mock.patch("sys.argv", ["memory_conflict_dataset", "--judge", "wat"]):
            try:
                asyncio.run(mod.main())
            except SystemExit:
                exited = True
        assert exited


class TestConfig070:
    """module-070 配置：memory_conflict_judge 默认 dual（WP-A 数据决策）+ Literal 三值"""

    def test_conflict_judge_default_dual(self):
        # conftest autouse 钉住实例为 'nli'（hermetic）；生产默认读类字段——
        # WP-A 70 条真实跑分 dual Precision 0.9412（fp=1）三方案最高（changelog §1.4）
        from src.config import Settings
        assert Settings.model_fields["memory_conflict_judge"].default == "dual"

    def test_conflict_judge_literal_three_values(self):
        from src.config import Settings
        args = Settings.model_fields["memory_conflict_judge"].annotation.__args__
        assert args == ("clf", "nli", "dual")   # 非法值 pydantic 启动拒绝（AC-16）
