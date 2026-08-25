"""module-043 L1 评估脚本单元测试（eval/golden_intent.py）

覆盖：
- compute_confusion_matrix: 混淆矩阵 / per-class 精确率·召回率·F1 / 准确率正确性
- compute_confusion_matrix: 空输入 / 预测出现未知类 边界
- load_intent_dataset: 结构校验（三类齐全、intent 合法、含边界易混样本）
- run_eval: 循环聚合端到端（注入 stub 分类器，不依赖 LLM）
- run_eval: 分类器异常 → 跳过记录，不中断
- record_eval_run: eval_runs 落库契约（eval_type='intent' + git_commit + 配置快照，打桩 save_eval_run）

说明：
- 指标计算为纯函数，不依赖数据库，可被 pytest 直接收集。
- 异步路径用 asyncio.run + stub 打桩执行，不依赖 pytest-asyncio / 数据库 / LLM。
"""
import asyncio

import pytest

from eval.golden import golden_intent


class TestComputeConfusionMatrix:
    """混淆矩阵 + per-class 指标计算"""

    def test_basic_matrix_and_metrics(self):
        labels = ["knowledge", "knowledge", "casual_chat", "casual_chat", "realtime"]
        preds = ["knowledge", "casual_chat", "casual_chat", "casual_chat", "realtime"]
        conf = golden_intent.compute_confusion_matrix(labels, preds)
        assert conf["accuracy"] == 0.8
        assert conf["matrix"]["knowledge"]["knowledge"] == 1
        assert conf["matrix"]["knowledge"]["casual_chat"] == 1
        assert conf["matrix"]["casual_chat"]["casual_chat"] == 2
        assert conf["matrix"]["realtime"]["realtime"] == 1
        # knowledge：精确率 1.0（预测为 knowledge 的 1 条全对），召回率 0.5（2 条只抓回 1 条）
        assert conf["per_class"]["knowledge"]["precision"] == 1.0
        assert conf["per_class"]["knowledge"]["recall"] == 0.5
        assert conf["per_class"]["knowledge"]["support"] == 2
        # casual_chat：精确率 2/3（预测 3 条对 2 条，round 4 位），召回率 1.0
        assert conf["per_class"]["casual_chat"]["precision"] == pytest.approx(2 / 3, abs=0.0001)
        assert conf["per_class"]["casual_chat"]["recall"] == 1.0
        assert conf["per_class"]["realtime"]["precision"] == 1.0
        assert conf["per_class"]["realtime"]["recall"] == 1.0

    def test_perfect_predictions(self):
        labels = ["knowledge", "casual_chat", "realtime"] * 2
        conf = golden_intent.compute_confusion_matrix(labels, list(labels))
        assert conf["accuracy"] == 1.0
        for c in ("knowledge", "casual_chat", "realtime"):
            assert conf["per_class"][c]["precision"] == 1.0
            assert conf["per_class"][c]["recall"] == 1.0

    def test_empty_inputs(self):
        # 空输入 → 全 0 指标，不崩溃
        conf = golden_intent.compute_confusion_matrix([], [])
        assert conf["accuracy"] == 0.0
        assert conf["per_class"] == {}
        assert conf["matrix"] == {}

    def test_unknown_predicted_class(self):
        # 预测出现未标注类 → 类并集入矩阵，不崩溃（router 白名单兜底通常不会出现，防御性验证）
        labels = ["knowledge", "casual_chat"]
        preds = ["knowledge", "unknown"]
        conf = golden_intent.compute_confusion_matrix(labels, preds)
        assert "unknown" in conf["classes"]
        assert conf["per_class"]["knowledge"]["precision"] == 1.0
        assert conf["per_class"]["casual_chat"]["recall"] == 0.0


class TestLoadIntentDataset:
    """intent 评测集结构校验"""

    def test_structure_valid_and_classes_complete(self):
        data = golden_intent.load_intent_dataset()
        assert len(data) >= 10
        assert {d["intent"] for d in data} == set(golden_intent.INTENT_CLASSES)
        for d in data:
            assert d["query"].strip()
            assert d["intent"] in golden_intent.INTENT_CLASSES

    def test_boundary_samples_present(self):
        # AC：含边界易混样本——"你们网站有什么功能"看似闲聊实为知识库
        data = golden_intent.load_intent_dataset()
        boundary = next(d for d in data if "你们网站有什么功能" in d["query"])
        assert boundary["intent"] == "knowledge"


class TestRunEval:
    """run_eval 循环聚合（注入 stub 分类器，不依赖 LLM）"""

    @staticmethod
    def _dataset():
        return [
            {"query": "什么是G1？", "intent": "knowledge"},
            {"query": "你好呀", "intent": "casual_chat"},
            {"query": "现在几点了？", "intent": "realtime"},
        ]

    def test_end_to_end_correct(self):
        async def _clf(query):
            return {"什么是G1？": "knowledge", "你好呀": "casual_chat", "现在几点了？": "realtime"}[query]

        scores, per_question, skipped = asyncio.run(
            golden_intent.run_eval(classifier=_clf, dataset=self._dataset())
        )
        assert scores["evaluated"] == 3
        assert scores["skipped"] == 0
        assert scores["accuracy"] == 1.0
        assert all(q["correct"] for q in per_question)

    def test_misclassification_recorded(self):
        # 分类器全判闲聊 → knowledge/realtime 漏检进混淆矩阵
        async def _clf(query):
            return "casual_chat"

        scores, per_question, skipped = asyncio.run(
            golden_intent.run_eval(classifier=_clf, dataset=self._dataset())
        )
        assert scores["accuracy"] == pytest.approx(1 / 3, abs=0.0001)
        assert scores["confusion_matrix"]["knowledge"]["casual_chat"] == 1
        assert scores["confusion_matrix"]["realtime"]["casual_chat"] == 1
        assert len([q for q in per_question if not q["correct"]]) == 2

    def test_classifier_error_skipped_not_crash(self):
        async def _clf(query):
            raise RuntimeError("llm down")

        scores, per_question, skipped = asyncio.run(
            golden_intent.run_eval(classifier=_clf, dataset=self._dataset())
        )
        assert scores["evaluated"] == 0
        assert scores["skipped"] == 3
        assert all(s["reason"].startswith("error:") for s in skipped)


class TestRunEvalShortcut:
    """module-072 WP-C：短路路由测量（query_rewrite_enabled 开启时的确定性信号）

    precise AND NOT rule_hits → knowledge（engine.chat 同款），零 LLM。
    per_question 打 reason 标记（与 engine.chat 字符串逐字一致）供过滤统计。
    """

    @staticmethod
    def _dataset():
        return [
            {"query": "什么是G1？", "intent": "knowledge"},
            {"query": "你好呀", "intent": "casual_chat"},
            {"query": "现在几点了？", "intent": "realtime"},
        ]

    def test_shortcut_applied_and_statistics(self, monkeypatch):
        from src.config import settings
        from unittest import mock

        monkeypatch.setattr(settings, "query_rewrite_enabled", True)

        async def _clf(query):
            return {"什么是G1？": "knowledge", "你好呀": "casual_chat", "现在几点了？": "realtime"}[query]

        # 前两题分诊命中术语（precise 且非规则词 → 短路 knowledge）；
        # "现在几点了？" 分诊 vague（不进短路，走 classify）
        async def _fake_triage(query):
            return "precise" if query != "现在几点了？" else "vague"

        with mock.patch("rag.retrieval.query_rewrite.triage",
                        new=mock.AsyncMock(side_effect=_fake_triage)):
            with mock.patch.object(golden_intent.router_agent, "_rule_hits",
                                   return_value=False):
                scores, per_question, _ = asyncio.run(
                    golden_intent.run_eval(classifier=_clf, dataset=self._dataset())
                )

        # "你好呀" 被短路误归 knowledge（casual 被术语句信号吞掉）→ 判对率 1/2，
        # 短路统计如实记录（这正是 WP-C 要暴露的风险面）
        assert scores["shortcut_fired"] == 2
        assert scores["shortcut_correct"] == 1
        assert scores["shortcut_accuracy"] == 0.5
        fired = [q for q in per_question if "分诊命中 FTS 术语" in q.get("reason", "")]
        assert len(fired) == 2
        assert all(q["predicted"] == "knowledge" for q in fired)
        # 非短路样本无 reason 标记，走 classify
        normal = [q for q in per_question if "reason" not in q]
        assert normal[0]["query"] == "现在几点了？"
        assert normal[0]["predicted"] == "realtime"

    def test_shortcut_disabled_by_default(self):
        # 默认关闭（query_rewrite_enabled=False）→ 全走 classify，短路统计为空
        async def _clf(query):
            return "knowledge"

        scores, per_question, _ = asyncio.run(
            golden_intent.run_eval(classifier=_clf, dataset=self._dataset())
        )
        assert scores["shortcut_fired"] == 0
        assert scores["shortcut_accuracy"] is None
        assert all("reason" not in q for q in per_question)


class TestRecordEvalRun:
    """eval_runs 落库契约（打桩，不依赖数据库）"""

    def test_eval_runs_contract(self, monkeypatch):
        captured = {}

        async def _fake_save(eval_type, git_commit, config_snapshot, scores, per_question):
            captured.update({
                "eval_type": eval_type,
                "git_commit": git_commit,
                "config_snapshot": config_snapshot,
                "scores": scores,
                "per_question": per_question,
            })
            return 42

        async def _fake_config():
            return {"top_k": "5"}

        monkeypatch.setattr(golden_intent, "get_git_commit", lambda: "abc123def")
        monkeypatch.setattr(golden_intent, "load_rag_config", _fake_config)
        monkeypatch.setattr(golden_intent, "save_eval_run", _fake_save)

        commit, saved_id = asyncio.run(golden_intent.record_eval_run(
            scores={"accuracy": 0.9, "classes": ["knowledge"]},
            per_question=[{"query": "q", "label": "knowledge", "predicted": "knowledge", "correct": True}],
        ))
        assert commit == "abc123def"
        assert saved_id == 42
        assert captured["eval_type"] == "intent"
        assert captured["git_commit"] == "abc123def"
        # module-056 Review 修复：快照补运行时 L4 开关字段（rag_config 表无此键，
        # 测试环境 conftest 钉住 False → 如实记录 "False"）
        # module-072（WP-C，plan 许可扩展）：快照补两开关字段（短路路由 off/on
        # 四跑可区分；conftest 钉住 False → 如实记录 "False"）
        assert captured["config_snapshot"] == {
            "top_k": "5",
            "intent_classifier_enabled": "False",
            "query_rewrite_enabled": "False",
            "contextual_rewrite_enabled": "False",
        }
        assert captured["scores"]["accuracy"] == 0.9

    def test_save_failure_returns_zero(self, monkeypatch):
        # save_eval_run 内部已捕获异常返回 0（与 golden_retrieval 同契约），record 原样透传
        async def _fake_save(**kwargs):
            return 0

        monkeypatch.setattr(golden_intent, "save_eval_run", _fake_save)
        commit, saved_id = asyncio.run(golden_intent.record_eval_run(scores={}, per_question=[]))
        assert saved_id == 0


class TestRunCompareClassifier:
    """--compare-classifier 的 L4 钉住（module-056 Review 修复）

    防自污染：module-056 起 intent_classifier_enabled 默认 true，若 LLM 侧
    不经钉住直接跑，router_agent 会静默走 L4 分类器路径，「LLM vs 分类器」
    退化为「分类器 vs 分类器」（双 1.0000 恒成立、对比失去意义）。
    """

    @staticmethod
    def _patch_side_effects(monkeypatch):
        """打桩 run_eval/打印/IntentClassifier，不依赖 DB 与 LLM"""
        from src.config import settings

        observed = []

        async def _fake_run_eval(classifier=None, dataset=None):
            observed.append(settings.intent_classifier_enabled)
            items = list(dataset)
            scores = {
                "dataset_size": len(items), "evaluated": len(items), "skipped": 0,
                "accuracy": 1.0, "confusion_matrix": {}, "per_class": {}, "classes": [],
            }
            per_q = [{"query": it["query"], "label": it["intent"],
                      "predicted": it["intent"], "correct": True} for it in items]
            return scores, per_q, []

        class _FakeClassifier:
            async def load(self):
                return True

            async def predict_proba(self, query):
                return {"knowledge": 0.9, "casual_chat": 0.05, "realtime": 0.05}

        monkeypatch.setattr(golden_intent, "run_eval", _fake_run_eval)
        monkeypatch.setattr(golden_intent, "print_report", lambda *a, **k: None)
        monkeypatch.setattr(golden_intent, "print_comparison", lambda *a, **k: None)
        monkeypatch.setattr("agent.intent_classifier.IntentClassifier",
                            lambda model_path=None, embedding_service=None: _FakeClassifier())
        return observed

    def test_llm_side_pins_classifier_disabled_and_restores(self, monkeypatch):
        """LLM 侧运行期间 L4 必须被钉住关闭，结束后恢复原开关值"""
        from src.config import settings

        # 模拟 module-056 默认启用态
        monkeypatch.setattr(settings, "intent_classifier_enabled", True)
        observed = self._patch_side_effects(monkeypatch)

        asyncio.run(golden_intent.run_compare_classifier(no_save=True))

        # 两次 run_eval（LLM 侧 + 分类器侧）均在钉住态运行
        assert observed and all(v is False for v in observed)
        # 结束后恢复调用前原值（默认启用态不被本脚本破坏）
        assert settings.intent_classifier_enabled is True
