"""module-063 golden_multi_turn 评测脚本单元测试（eval/golden/golden_multi_turn.py）

覆盖：
- load_dataset: 结构校验（≥10 条、prev/follow_up 非空、expected_intent 合法）
- heuristic_rewrite: 把 prev 核心术语补进 follow_up（fixture 演示）
- heuristic_intent: 规则词 → casual/realtime；术语 → knowledge（fixture 演示）
- overlap_ratio: 检索重叠度计算
- compute_metrics: 三指标聚合（自包含/意图保持/检索提升）纯函数
- run_eval_fixture: fixture 模式可运行（不依赖 LLM/DB，检索提升待环境）

说明：
- 指标计算为纯函数，不依赖数据库，可被 pytest 直接收集。
- fixture 模式异步路径用 asyncio.run 执行（无 pytest-asyncio）。
"""
import asyncio

import pytest

from eval.golden import golden_multi_turn as gmt


class TestLoadDataset:
    """多轮追问对评测集结构校验"""

    def test_structure_valid_and_count(self):
        data = gmt.load_dataset()
        assert len(data) >= 10
        for item in data:
            assert item["prev"].strip()
            assert item["follow_up"].strip()
            assert item["expected_intent"] in ("knowledge", "casual_chat", "realtime")

    def test_all_expected_knowledge(self):
        # 多轮追问对全部是技术追问 → 结合前文后的正确意图全 knowledge
        data = gmt.load_dataset()
        assert all(d["expected_intent"] == "knowledge" for d in data)


class TestHeuristicRewrite:
    """fixture 启发式对话改写（确定性，不依赖 LLM）"""

    def test_appends_prev_term(self):
        rewritten = gmt.heuristic_rewrite(
            "什么是Java线程池？核心参数有哪些？", "为什么")
        assert "为什么" in rewritten
        assert "Java" in rewritten or "线程池" in rewritten
        assert rewritten != "为什么"  # 有补全 → 自包含（rewrite_changed=True）

    def test_no_term_returns_followup(self):
        # prev 无知识库术语（全功能词）→ 原样返回（改写失败语义）
        rewritten = gmt.heuristic_rewrite("为什么你这样说呢", "为什么")
        assert rewritten == "为什么"


class TestHeuristicIntent:
    """fixture 启发式意图（不依赖 LLM/DB）"""

    def test_rule_word_realtime(self):
        assert gmt.heuristic_intent("今天天气怎么样") == "realtime"
        assert gmt.heuristic_intent("现在几点了") == "realtime"

    def test_rule_word_casual(self):
        assert gmt.heuristic_intent("哈哈") == "casual_chat"
        assert gmt.heuristic_intent("你好呀") == "casual_chat"

    def test_kb_term_knowledge(self):
        assert gmt.heuristic_intent("为什么 Java线程池") == "knowledge"

    def test_no_feature_falls_back_casual(self):
        # 无规则词无术语（fixture 最简降级）→ casual（真实走完整路由）
        assert gmt.heuristic_intent("为什么") == "casual_chat"


class TestOverlapRatio:
    """检索重叠度"""

    def test_full_overlap(self):
        assert gmt.overlap_ratio(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_partial_overlap(self):
        assert gmt.overlap_ratio(["a", "b", "c"], ["a", "x", "y"]) == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert gmt.overlap_ratio(["a"], ["b"]) == 0.0

    def test_empty_input(self):
        assert gmt.overlap_ratio([], ["a"]) == 0.0
        assert gmt.overlap_ratio(["a"], []) == 0.0


class TestComputeMetrics:
    """三指标聚合（纯函数）"""

    @staticmethod
    def _sample():
        return [
            {"expected_intent": "knowledge", "raw_intent": "casual_chat",
             "routed_intent": "knowledge", "rewrite_changed": True,
             "raw_overlap": 0.1, "rewritten_overlap": 0.6},
            {"expected_intent": "knowledge", "raw_intent": "casual_chat",
             "routed_intent": "knowledge", "rewrite_changed": True,
             "raw_overlap": 0.2, "rewritten_overlap": 0.5},
        ]

    def test_metrics_aggregation(self):
        m = gmt.compute_metrics(self._sample())
        assert m["count"] == 2
        assert m["self_contained_ratio"] == 1.0      # 全改写成功
        assert m["raw_intent_ratio"] == 0.0          # 单句省略句全漏检
        assert m["intent_preserved_ratio"] == 1.0    # 多轮路由全保持
        assert m["raw_overlap"] == pytest.approx(0.15)
        assert m["rewritten_overlap"] == pytest.approx(0.55)
        assert m["retrieval_delta"] == pytest.approx(0.4)  # 检索提升

    def test_metrics_no_retrieval_marks_none(self):
        # fixture 无 DB 检索 → 检索相关指标 None（如实标注待环境）
        m = gmt.compute_metrics([{
            "expected_intent": "knowledge", "raw_intent": "knowledge",
            "routed_intent": "knowledge", "rewrite_changed": False,
            "raw_overlap": None, "rewritten_overlap": None,
        }])
        assert m["retrieval_delta"] is None
        assert m["self_contained_ratio"] == 0.0

    def test_empty_input(self):
        m = gmt.compute_metrics([])
        assert m["count"] == 0
        assert m["retrieval_delta"] is None


class TestRunEvalFixture:
    """fixture 模式端到端（不依赖 LLM/DB）"""

    def test_fixture_runs_and_scores(self):
        scores, per_question, skipped = asyncio.run(gmt.run_eval_fixture())
        assert scores["fixture"] is True
        assert scores["evaluated"] == len(gmt.load_dataset())
        assert scores["skipped"] == 0
        assert scores["count"] == len(gmt.load_dataset())
        # 三指标键齐全；检索相关待环境（None）
        assert "self_contained_ratio" in scores
        assert "intent_preserved_ratio" in scores
        assert scores["retrieval_delta"] is None
        # 每题明细结构
        q = per_question[0]
        assert q["follow_up"]
        assert q["routed_intent"] in ("knowledge", "casual_chat", "realtime")

    def test_fixture_rewrite_improves_routing_demo(self):
        """fixture 演示：省略句单句被启发式判闲聊，改写补全术语后判 knowledge"""
        prev = "什么是Java线程池？核心参数有哪些？"
        follow_up = "为什么"
        assert gmt.heuristic_intent(follow_up) == "casual_chat"  # 单句漏检
        rewritten = gmt.heuristic_rewrite(prev, follow_up)
        assert gmt.heuristic_intent(rewritten) == "knowledge"    # 改写后路由正确


class TestLoadDatasetValidation:
    """结构非法 → 抛 ValueError（防评测集损坏）"""

    def test_invalid_expected_intent(self, monkeypatch):
        bad = [{"prev": "A", "follow_up": "B", "expected_intent": "bogus"}]
        gmt.load_dataset()  # 真实集合法，先确认不抛（防御回归）
        monkeypatch.setattr(gmt, "MULTI_TURN_DATASET", bad * 12)
        with pytest.raises(ValueError):
            gmt.load_dataset()

    def test_empty_prev(self, monkeypatch):
        bad = [{"prev": "", "follow_up": "B", "expected_intent": "knowledge"}]
        monkeypatch.setattr(gmt, "MULTI_TURN_DATASET", bad * 12)
        with pytest.raises(ValueError):
            gmt.load_dataset()


class TestRecordEvalRun:
    """eval_runs 落库契约（打桩，不依赖数据库）——module-072 WP-C 快照两键"""

    def test_eval_runs_contract_snapshot_two_switches(self, monkeypatch):
        captured = {}

        async def _fake_save(eval_type, git_commit, config_snapshot, scores, per_question):
            captured.update({
                "eval_type": eval_type,
                "config_snapshot": config_snapshot,
            })
            return 42

        async def _fake_config():
            return {"top_k": "5"}

        monkeypatch.setattr(gmt, "get_git_commit", lambda: "abc123def")
        monkeypatch.setattr(gmt, "load_rag_config", _fake_config)
        monkeypatch.setattr(gmt, "save_eval_run", _fake_save)

        commit, saved_id = asyncio.run(gmt.record_eval_run(
            scores={"count": 12}, per_question=[]))
        assert commit == "abc123def"
        assert saved_id == 42
        assert captured["eval_type"] == "multi_turn"
        # module-072（WP-C）：快照补两开关字段（短路路由 off/on 四跑可区分；
        # 测试环境 conftest 钉住 contextual false、生产默认 query_rewrite false）
        assert captured["config_snapshot"] == {
            "top_k": "5",
            "query_rewrite_enabled": "False",
            "contextual_rewrite_enabled": "False",
        }

    def test_eval_runs_snapshot_reflects_runtime_switches(self, monkeypatch):
        """开关运行时置位 → 快照如实记录（非恒 False）"""
        from src.config import settings

        captured = {}

        async def _fake_save(eval_type, git_commit, config_snapshot, scores, per_question):
            captured["config_snapshot"] = config_snapshot
            return 42

        async def _fake_config():
            return {}

        monkeypatch.setattr(gmt, "save_eval_run", _fake_save)
        monkeypatch.setattr(gmt, "load_rag_config", _fake_config)
        monkeypatch.setattr(settings, "query_rewrite_enabled", True)
        monkeypatch.setattr(settings, "contextual_rewrite_enabled", True)

        asyncio.run(gmt.record_eval_run(scores={}, per_question=[]))
        assert captured["config_snapshot"]["query_rewrite_enabled"] == "True"
        assert captured["config_snapshot"]["contextual_rewrite_enabled"] == "True"
