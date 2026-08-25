"""module-044 层 0 充分性评测脚本单元测试（eval/golden_sufficiency.py）

覆盖：
- load_sufficiency_dataset: 结构校验（充分/不充分两类齐全、sufficient 为 bool）
- heuristic_judge: fixture 启发式判断器正确性（关键词命中/不命中/空文档）
- run_eval: 循环聚合端到端（注入 stub 判断器，不依赖 LLM/DB）
- run_eval: 判断器异常 → 跳过记录，不中断
- run_eval: insufficient_recall 重点项正确聚合（漏判不充分最致命）
- record_eval_run: eval_runs 落库契约（eval_type='sufficiency' + git_commit + 配置快照，打桩 save_eval_run）

说明：
- 指标计算复用 eval/golden_intent.py 的 compute_confusion_matrix（纯函数，已被 test_golden_intent 覆盖）。
- 异步路径用 asyncio.run + stub 打桩执行，不依赖 pytest-asyncio / 数据库 / LLM。
"""
import asyncio

import pytest

from eval.golden import golden_sufficiency


class TestLoadSufficiencyDataset:
    """充分性标注集结构校验"""

    def test_structure_valid_and_classes_complete(self):
        data = golden_sufficiency.load_sufficiency_dataset()
        assert len(data) >= 10
        assert len([d for d in data if d["sufficient"]]) >= 5
        assert len([d for d in data if not d["sufficient"]]) >= 5
        for d in data:
            assert d["question"].strip()
            assert d["documents"]
            assert isinstance(d["sufficient"], bool)

    def test_questions_borrowed_from_golden(self):
        # AC：标注集问题借 golden 集真实题目（"什么是G1垃圾收集器" 等）
        data = golden_sufficiency.load_sufficiency_dataset()
        questions = {d["question"] for d in data}
        assert "什么是G1垃圾收集器？它的核心创新是什么？" in questions
        assert "Kafka的ISR机制是如何保证消息可靠性的？" in questions
        assert "AQS (AbstractQueuedSynchronizer) 的工作原理是什么？ReentrantLock如何基于AQS实现？" in questions

    def test_insufficient_samples_include_unrelated_docs(self):
        # 不充分样本须含"完全不沾边"类型（问 A 检索到 B）——最典型漏判场景
        data = golden_sufficiency.load_sufficiency_dataset()
        unrelated = [
            d for d in data if not d["sufficient"] and "完全不沾边" in d.get("note", "")
        ]
        assert len(unrelated) >= 3


class TestHeuristicJudge:
    """fixture 启发式判断器（确定性，不依赖 LLM/DB）"""

    def test_keyword_hit_returns_sufficient(self):
        docs = [{"title": "t", "content": "G1 垃圾收集器使用 Region 分区机制"}]
        assert golden_sufficiency.heuristic_judge("q", docs, ["G1", "Region"]) is True

    def test_no_keyword_hit_returns_insufficient(self):
        docs = [{"title": "t", "content": "Kafka 的 ISR 机制保证消息可靠性"}]
        assert golden_sufficiency.heuristic_judge("q", docs, ["G1", "Region"]) is False

    def test_empty_documents_returns_insufficient(self):
        assert golden_sufficiency.heuristic_judge("q", [], ["G1"]) is False


class TestRunEval:
    """run_eval 循环聚合（注入 stub 判断器，不依赖 LLM/DB）"""

    @staticmethod
    def _dataset():
        return [
            {"question": "什么是G1？", "documents": [{"title": "t", "content": "G1"}],
             "sufficient": True, "category": "java_gc"},
            {"question": "什么是Kafka？", "documents": [{"title": "t", "content": "Kafka"}],
             "sufficient": True, "category": "kafka"},
            {"question": "ZGC的特点？", "documents": [{"title": "t", "content": "G1 文档"}],
             "sufficient": False, "category": "java_gc"},
        ]

    def test_end_to_end_correct(self):
        async def _judge(query, documents):
            return query != "ZGC的特点？"

        scores, per_question, skipped = asyncio.run(
            golden_sufficiency.run_eval(judge=_judge, dataset=self._dataset())
        )
        assert scores["evaluated"] == 3
        assert scores["skipped"] == 0
        assert scores["accuracy"] == 1.0
        assert scores["insufficient_recall"] == 1.0
        assert all(q["correct"] for q in per_question)

    def test_misclassification_recorded_and_recall_highlighted(self):
        # 判断器全判充分 → 不充分样本全部漏判（Recall=0），指标如实反映
        async def _judge(query, documents):
            return True

        scores, per_question, skipped = asyncio.run(
            golden_sufficiency.run_eval(judge=_judge, dataset=self._dataset())
        )
        assert scores["accuracy"] == pytest.approx(2 / 3, abs=0.0001)
        assert scores["confusion_matrix"]["insufficient"]["sufficient"] == 1
        # 重点项：漏判不充分 → insufficient Recall = 0.0（报告大字标出的指标）
        assert scores["insufficient_recall"] == 0.0
        assert scores["per_class"]["insufficient"]["recall"] == 0.0
        assert len([q for q in per_question if not q["correct"]]) == 1

    def test_judge_error_skipped_not_crash(self):
        async def _judge(query, documents):
            raise RuntimeError("llm down")

        scores, per_question, skipped = asyncio.run(
            golden_sufficiency.run_eval(judge=_judge, dataset=self._dataset())
        )
        assert scores["evaluated"] == 0
        assert scores["skipped"] == 3
        assert all(s["reason"].startswith("error:") for s in skipped)

    def test_insufficient_recall_absent_when_all_skipped(self):
        # 全部跳过时 insufficient_recall 取 0.0，不崩溃
        async def _judge(query, documents):
            raise RuntimeError("down")

        scores, per_question, skipped = asyncio.run(
            golden_sufficiency.run_eval(judge=_judge, dataset=self._dataset())
        )
        assert scores["insufficient_recall"] == 0.0


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

        monkeypatch.setattr(golden_sufficiency, "get_git_commit", lambda: "abc123def")
        monkeypatch.setattr(golden_sufficiency, "load_rag_config", _fake_config)
        monkeypatch.setattr(golden_sufficiency, "save_eval_run", _fake_save)

        commit, saved_id = asyncio.run(golden_sufficiency.record_eval_run(
            scores={"accuracy": 0.9, "insufficient_recall": 0.8},
            per_question=[{"question": "q", "label": True, "predicted": True, "correct": True}],
        ))
        assert commit == "abc123def"
        assert saved_id == 42
        assert captured["eval_type"] == "sufficiency"
        assert captured["git_commit"] == "abc123def"
        assert captured["config_snapshot"] == {"top_k": "5"}
        assert captured["scores"]["accuracy"] == 0.9
        assert captured["scores"]["insufficient_recall"] == 0.8

    def test_save_failure_returns_zero(self, monkeypatch):
        # save_eval_run 内部已捕获异常返回 0（与 golden_retrieval 同契约），record 原样透传
        async def _fake_save(**kwargs):
            return 0

        monkeypatch.setattr(golden_sufficiency, "save_eval_run", _fake_save)
        commit, saved_id = asyncio.run(golden_sufficiency.record_eval_run(scores={}, per_question=[]))
        assert saved_id == 0
