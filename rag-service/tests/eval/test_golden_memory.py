"""module-046 WP3 记忆提取评测脚本单元测试（eval/golden_memory.py）

覆盖（验收 §3 WP3 + 降级 §4）：
- load_memory_golden: 标注集结构校验（≥20 条、dialogue 非空、facts 字符串列表、
  "不应提取"样本齐全；非法结构报错）
- _dialogue_to_extract_inputs: dialogue → (query, answer, history) 映射
  （末轮 user/assistant；无 assistant 回答 → None）
- compute_prf / _match_sample: P/R/F1 计算正确性（含"不应提取"样本 → fp 计入，
  防过度提取）
- run_eval: 注入 stub 提取器端到端聚合；提取器异常 → 跳过不崩溃
- record_eval_run: eval_runs 落库契约（eval_type='memory_extraction'，打桩
  save_eval_run / get_git_commit / load_rag_config）
- fixture_extract: 关键词启发式确定性（不依赖 LLM/DB），无关键词返回空

实现说明：同步用例内 asyncio.run 执行（与套件同款模式）；stub 提取器注入，
不依赖真实 LLM / 数据库。
"""
import asyncio

import pytest

from eval.golden import golden_memory


class TestLoadMemoryGolden:
    """标注集结构校验"""

    def test_structure_valid_and_size(self):
        data = golden_memory.load_memory_golden()
        assert len(data) >= 20
        for d in data:
            assert d["dialogue"].strip()
            assert isinstance(d["facts"], list)
            assert all(isinstance(f, str) and f.strip() for f in d["facts"])

    def test_has_no_extract_samples(self):
        # "不应提取"样本（facts=[]）须齐全：防过度提取评测
        data = golden_memory.load_memory_golden()
        no_extract = [d for d in data if not d["facts"]]
        assert len(no_extract) >= 5
        # 通用知识问答（G1）与寒暄都在不应提取之列
        dialogues = "\n".join(d["dialogue"] for d in no_extract)
        assert "G1" in dialogues
        assert "你好" in dialogues

    def test_extract_samples_present(self):
        data = golden_memory.load_memory_golden()
        extract = [d for d in data if d["facts"]]
        assert len(extract) >= 15
        assert all(d["keywords"] for d in extract)  # fixture 关键词齐全

    def test_invalid_raises(self, monkeypatch):
        # 结构校验：缺 dialogue / 空 dialogue / facts 非字符串列表 / 全为应提取 → ValueError
        bad_missing_dialogue = [{"facts": ["x"]}]
        bad_empty_dialogue = [{"dialogue": "  ", "facts": ["x"]}]
        bad_facts_type = [{"dialogue": "用户: 你好\n助手: 你好", "facts": [123]}]
        bad_no_no_extract = [{"dialogue": "用户: 你好\n助手: 你好", "facts": ["x"]}] * 20
        for bad in (bad_missing_dialogue, bad_empty_dialogue, bad_facts_type,
                    bad_no_no_extract):
            with monkeypatch.context() as m:
                m.setattr(golden_memory, "MEMORY_GOLDEN_DATASET", bad)
                with pytest.raises(ValueError):
                    golden_memory.load_memory_golden()
        # 基线与 bad 独立：上下文退出后恢复，正常加载
        assert len(golden_memory.load_memory_golden()) >= 20


class TestDialogueMapping:
    """dialogue → extract_facts 输入映射"""

    def test_maps_last_user_and_assistant(self):
        dialogue = "用户: 你好\n助手: 你好！\n用户: 我喜欢简洁回答\n助手: 记住了"
        query, answer, history = golden_memory._dialogue_to_extract_inputs(dialogue)
        assert query == "我喜欢简洁回答"
        assert answer == "记住了"
        assert history == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]

    def test_no_assistant_answer_returns_none(self):
        # 对话未完成（无 assistant 回答）→ 无法调 extract_facts → None（跳过）
        assert golden_memory._dialogue_to_extract_inputs("用户: 你好") is None

    def test_empty_or_garbage_dialogue(self):
        assert golden_memory._dialogue_to_extract_inputs("") is None
        assert golden_memory._dialogue_to_extract_inputs("随便一句话") is None


class TestPrfMetrics:
    """P/R/F1 计算与单样本匹配"""

    def test_fact_match_containment_both_directions(self):
        # 归一化后互相包含任一方向即匹配（容忍措辞差异）
        assert golden_memory._fact_match("用户偏好简洁的回答风格", "偏好简洁")
        assert golden_memory._fact_match("简洁", "用户偏好简洁的回答风格")
        assert not golden_memory._fact_match("用户喜欢咖啡", "用户偏好简洁")

    def test_match_sample_basic(self):
        tp, fp, fn = golden_memory._match_sample(
            ["用户偏好简洁的回答风格", "用户喜欢摄影"], ["用户偏好简洁的回答风格", "用户是Java后端"])
        assert (tp, fp, fn) == (1, 1, 1)

    def test_match_sample_golden_greedy_single_use(self):
        # 两条预测同时包含同一条标注 → 该标注只计一次（贪心防重复计数）
        tp, fp, fn = golden_memory._match_sample(
            ["偏好简洁的回答风格", "简洁的回答"], ["偏好简洁"])
        assert (tp, fp, fn) == (1, 1, 0)

    def test_match_sample_over_extraction(self):
        # "不应提取"样本（空标注）被预测 → 全部 fp（过度提取惩罚）
        tp, fp, fn = golden_memory._match_sample(["用户喜欢咖啡"], [])
        assert (tp, fp, fn) == (0, 1, 0)

    def test_compute_prf_known_values(self):
        rows = [{"tp": 2, "fp": 1, "fn": 1}, {"tp": 1, "fp": 0, "fn": 0}]
        prf = golden_memory.compute_prf(rows)
        assert prf["tp"] == 3
        assert prf["fp"] == 1
        assert prf["fn"] == 1
        assert prf["precision"] == pytest.approx(3 / 4, abs=1e-4)
        assert prf["recall"] == pytest.approx(3 / 4, abs=1e-4)
        assert prf["f1"] == pytest.approx(0.75, abs=1e-4)

    def test_compute_prf_empty_rows(self):
        prf = golden_memory.compute_prf([])
        assert prf["precision"] == 0.0
        assert prf["recall"] == 0.0
        assert prf["f1"] == 0.0


class TestRunEval:
    """run_eval 循环聚合（注入 stub 提取器，不依赖 LLM/DB）"""

    @staticmethod
    def _dataset():
        return [
            {"dialogue": "用户: 我喜欢简洁回答\n助手: 记住了", "facts": ["用户偏好简洁"],
             "keywords": ["简洁"]},
            {"dialogue": "用户: 我在做 Java 后端\n助手: 了解", "facts": ["用户是 Java 后端"],
             "keywords": ["Java"]},
            {"dialogue": "用户: G1 是什么？\n助手: 一种垃圾收集器", "facts": [],
             "keywords": []},  # 不应提取
        ]

    def test_end_to_end_with_stub_extractor(self):
        async def _extract(item):
            return [f for f in item["facts"]]  # 完全命中标注

        scores, per_question, skipped = asyncio.run(
            golden_memory.run_eval(extractor=_extract, dataset=self._dataset())
        )
        assert scores["dataset_size"] == 3
        assert scores["evaluated"] == 3
        assert scores["skipped"] == 0
        assert scores["over_extraction_count"] == 0
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert all(q["tp"] >= 0 and q["fp"] == 0 and q["fn"] == 0 for q in per_question)

    def test_extractor_over_extracts_penalized(self):
        # 提取器在"不应提取"样本也输出 → fp 计入，precision 下降（防过度提取）
        async def _extract(item):
            return ["用户偏好简洁"]  # 只在样本 1 命中标注，其余全为 fp

        scores, per_question, skipped = asyncio.run(
            golden_memory.run_eval(extractor=_extract, dataset=self._dataset())
        )
        assert scores["over_extraction_count"] == 1  # G1（空标注）样本被预测
        assert scores["tp"] == 1
        assert scores["fp"] == 2  # 样本 2 无关 + 样本 3 过度提取
        assert scores["fn"] == 1  # 样本 2 漏提取
        assert scores["precision"] == pytest.approx(1 / 3, abs=1e-4)

    def test_extractor_error_skipped_not_crash(self):
        async def _extract(item):
            raise RuntimeError("llm down")

        scores, per_question, skipped = asyncio.run(
            golden_memory.run_eval(extractor=_extract, dataset=self._dataset())
        )
        assert scores["evaluated"] == 0
        assert scores["skipped"] == 3
        assert all(s["reason"].startswith("error:") for s in skipped)
        # 全跳过 → P/R 取 0.0，不崩溃
        assert scores["precision"] == 0.0
        assert scores["recall"] == 0.0


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
            return 7

        async def _fake_config():
            return {"memory_importance_threshold": "0.6"}

        monkeypatch.setattr(golden_memory, "get_git_commit", lambda: "abc123def")
        monkeypatch.setattr(golden_memory, "load_rag_config", _fake_config)
        monkeypatch.setattr(golden_memory, "save_eval_run", _fake_save)

        commit, saved_id = asyncio.run(golden_memory.record_eval_run(
            scores={"precision": 0.8, "recall": 0.9},
            per_question=[{"dialogue": "d", "tp": 1, "fp": 0, "fn": 0}],
        ))
        assert commit == "abc123def"
        assert saved_id == 7
        assert captured["eval_type"] == "memory_extraction"
        assert captured["git_commit"] == "abc123def"
        assert captured["config_snapshot"] == {"memory_importance_threshold": "0.6"}
        assert captured["scores"]["precision"] == 0.8
        assert captured["per_question"][0]["tp"] == 1

    def test_save_failure_returns_zero(self, monkeypatch):
        # save_eval_run 内部已捕获异常返回 0（与 golden_retrieval 同契约）
        async def _fake_save(**kwargs):
            return 0

        monkeypatch.setattr(golden_memory, "save_eval_run", _fake_save)
        commit, saved_id = asyncio.run(
            golden_memory.record_eval_run(scores={}, per_question=[]))
        assert saved_id == 0


class TestFixtureExtract:
    """fixture 关键词启发式（确定性，不依赖 LLM/DB）"""

    def test_keyword_hit_returns_sentence(self):
        item = {"dialogue": "用户: 我比较喜欢简洁的回答\n助手: 好的，记住了。",
                "keywords": ["简洁"]}
        out = golden_memory.fixture_extract(item)
        assert len(out) == 1
        assert "简洁" in out[0]

    def test_no_keywords_returns_empty(self):
        # "不应提取"样本无关键词 → fixture 下同样不提取
        item = {"dialogue": "用户: 你好\n助手: 你好！", "keywords": []}
        assert golden_memory.fixture_extract(item) == []

    def test_keyword_miss_returns_empty(self):
        item = {"dialogue": "用户: 我喜欢摄影\n助手: 好", "keywords": ["Java"]}
        assert golden_memory.fixture_extract(item) == []

    def test_deterministic_and_no_llm(self):
        item = {"dialogue": "用户: 我在做 Agentic RAG\n助手: 厉害。请记住我偏好美式咖啡。",
                "keywords": ["美式咖啡", "RAG"]}
        first = golden_memory.fixture_extract(item)
        second = golden_memory.fixture_extract(item)
        assert first == second  # 确定性（不依赖 LLM 输出）
        assert all("美式咖啡" in s or "RAG" in s for s in first)
