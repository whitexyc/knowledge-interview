"""
module-056 测试：人造意图训练集 + 训练/评测分离 + L4 回退
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

覆盖 AC §1（数据集结构/类别平衡/边界样本/训练评测分离）+ §5（分类器
加载/推理失败回退 LLM）。
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent.router import RouterAgent

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
DATASET_PATH = EVAL_DIR / "datasets" / "intent_train_dataset.json"

INTENT_CLASSES = ("knowledge", "casual_chat", "realtime")
E2E_BUG_QUERY = "G1垃圾收集器的核心创新是什么？"


class FakeLLM:
    """模拟 LLM 分类返回（JSON payload），与 test_intent_validation 同款"""

    def __init__(self, payload: str):
        self._payload = payload

    async def generate(self, prompt: str) -> str:
        return self._payload


class _StubClassifier:
    """可注入 agent.intent_classifier.IntentClassifier 的桩（load/predict 可控）"""

    def __init__(self, load_ok: bool = True, predict_error: bool = False):
        self._load_ok = load_ok
        self._predict_error = predict_error

    async def load(self):
        return self._load_ok

    async def predict_proba(self, query):
        if self._predict_error:
            raise RuntimeError("模型推理失败")
        return {"knowledge": 0.9, "casual_chat": 0.05, "realtime": 0.05}


# ─── AC §1 数据集结构 ───


def load_dataset() -> list[dict]:
    assert DATASET_PATH.exists(), f"数据集文件缺失: {DATASET_PATH}"
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


class TestDatasetStructure:
    """AC §1: intent_train_dataset.json 结构/类别平衡/边界样本/分离"""

    def test_json_list_of_query_intent_dicts(self):
        data = load_dataset()
        assert isinstance(data, list) and len(data) >= 300
        for item in data:
            assert isinstance(item, dict)
            assert isinstance(item.get("query"), str) and item["query"].strip()
            assert item.get("intent") in INTENT_CLASSES

    def test_class_balance_each_at_least_80(self):
        data = load_dataset()
        counts = {cls: sum(1 for s in data if s["intent"] == cls) for cls in INTENT_CLASSES}
        assert counts["knowledge"] >= 80
        assert counts["casual_chat"] >= 80
        assert counts["realtime"] >= 80

    def test_boundary_and_term_and_colloquial_counts(self):
        data = load_dataset()
        notes = [s.get("note", "") for s in data]
        assert sum(1 for n in notes if "边界易混" in n) >= 30
        assert notes.count("专有术语") >= 30
        assert notes.count("口语化") >= 20

    def test_e2e_bug_query_present(self):
        """E2E bug 类样本：G1 查询曾被 LLM 高置信误判 casual_chat（module-054/055）"""
        data = load_dataset()
        assert E2E_BUG_QUERY in {s["query"] for s in data}
        bug = next(s for s in data if s["query"] == E2E_BUG_QUERY)
        assert bug["intent"] == "knowledge"
        assert "E2E bug" in bug.get("note", "")

    def test_queries_unique(self):
        data = load_dataset()
        queries = [s["query"] for s in data]
        assert len(set(queries)) == len(queries)


# ─── AC §2 训练/评测分离 ───


class TestTrainEvalSeparation:
    """AC §2: golden_intent 100 条评测集不进入训练（防泄漏）"""

    def test_training_pipeline_loads_new_dataset_first(self):
        from eval.train.train_intent_classifier import load_training_samples

        samples = load_training_samples()
        assert len(samples) >= 300
        train_queries = {q for q, _ in samples}
        # 人造训练集全量进入训练（去重后）
        assert set(s["query"] for s in load_dataset()) <= train_queries
        # golden.json knowledge 天然样本一并进入（计划内，评测集 knowledge 题同源）
        golden = json.loads((EVAL_DIR / "golden" / "golden.json").read_text(encoding="utf-8"))
        assert all(item["question"] in train_queries for item in golden)

    def test_eval_set_casual_realtime_zero_leak_into_training(self):
        """评测集 casual/realtime 样本（训练源中不存在）零混入训练"""
        from eval.golden.golden_intent import INTENT_DATASET
        from eval.train.train_intent_classifier import load_training_samples

        eval_non_kb = {i["query"] for i in INTENT_DATASET if i["intent"] != "knowledge"}
        train_queries = {q for q, _ in load_training_samples()}
        assert eval_non_kb.isdisjoint(train_queries)

    def test_train_script_no_longer_reads_golden_intent(self):
        """训练脚本不再从 golden_intent 评测集取数（module-056 分离口径）"""
        import eval.train.train_intent_classifier as t
        assert not hasattr(t, "load_golden_intent_samples")
        assert not hasattr(t, "_BUILTIN_SAMPLES")


# ─── AC §5 L4 回退 LLM ───


class TestL4Fallback:
    """AC §5: 分类器加载失败/推理失败 → 回退 LLM 分类，零影响"""

    def _patch_classifier(self, monkeypatch, load_ok=True, predict_error=False):
        from src.config import settings

        monkeypatch.setattr(settings, "intent_classifier_enabled", True)
        monkeypatch.setattr(
            "agent.intent_classifier.IntentClassifier",
            lambda model_path=None, embedding_service=None:
            _StubClassifier(load_ok=load_ok, predict_error=predict_error))

    def test_config_enabled_load_failure_falls_back_to_llm(self, monkeypatch):
        """开关开启但模型加载失败 → 回退 LLM 分类（零影响）"""
        self._patch_classifier(monkeypatch, load_ok=False)
        agent = RouterAgent()
        payload = '{"intent": "knowledge", "confidence": 0.9, "reason": "知识"}'
        monkeypatch.setattr("llm.client.LLMFactory.get_client",
                            lambda *a, **k: FakeLLM(payload))
        result = asyncio.run(agent.classify("什么是GC"))
        assert result["intent"] == "knowledge"
        assert result["confidence"] == 0.9

    def test_config_enabled_predict_failure_falls_back_to_llm(self, monkeypatch):
        """开关开启、加载成功但推理抛错 → 回退 LLM 分类（零影响）"""
        self._patch_classifier(monkeypatch, load_ok=True, predict_error=True)
        agent = RouterAgent()
        payload = '{"intent": "knowledge", "confidence": 0.9, "reason": "知识"}'
        monkeypatch.setattr("llm.client.LLMFactory.get_client",
                            lambda *a, **k: FakeLLM(payload))
        result = asyncio.run(agent.classify("什么是GC"))
        assert result["intent"] == "knowledge"
        assert result["confidence"] == 0.9

    def test_config_enabled_classifier_used_no_llm_call(self, monkeypatch):
        """开关开启且分类器可用 → 分类器路径，不调用 LLM"""
        self._patch_classifier(monkeypatch, load_ok=True)
        agent = RouterAgent()
        monkeypatch.setattr("llm.client.LLMFactory.get_client",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("分类器可用时不应调用 LLM")))
        result = asyncio.run(agent.classify("什么是GC"))
        assert result["intent"] == "knowledge"
        assert "L4" in result["reason"]
