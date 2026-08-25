"""module-050 幻觉检测模型对比脚本单元测试（eval/compare_factcheck_models.py）

覆盖：
- build_pairs: 100 对、claim=问题、doc=中文句切拼接、label 映射（sufficient=True→1）
- _pre_chunk: 中文标点切句 + 换行连接（MiniCheck nltk 预切兼容）
- model_metrics: Accuracy/F1/Precision/Recall 计算正确（正类=supported）
- cohen_kappa: 两判定一致性（sklearn 口径）
- _require_model: 模型缺失报错清晰（指出缺失路径），不静默通过
- 单侧 --skip 参数可独立跑（函数级：加载函数各自独立）

说明：
- 不加载真实模型（模型加载留给 --limit 冒烟），只测数据构造与指标纯函数。
- 对齐 tests 现有模式：纯单元、不打真实 DB、不打 LLM。
"""
import os

import numpy as np
import pytest

from eval.benchmarks import compare_factcheck_models as cm


class TestBuildPairs:
    """数据构造：SUFFICIENCY_DATASET → (doc, claim, label) 对"""

    def test_count_and_label_mapping(self):
        pairs = cm.build_pairs()
        assert len(pairs) == 100
        # label=1 当且仅当人工标注充分（同一问题可重复出现，充分/不充分各一条）
        for p in pairs:
            assert p["label"] in (0, 1)
        g1 = [p for p in pairs if p["claim"] == "什么是G1垃圾收集器？它的核心创新是什么？"]
        assert len(g1) == 2
        assert sorted(p["label"] for p in g1) == [0, 1]  # 与标注集逐条映射

    def test_claim_is_question_doc_is_concatenated(self):
        pairs = cm.build_pairs()
        p = pairs[0]
        assert p["claim"] == "什么是G1垃圾收集器？它的核心创新是什么？"
        # doc 为两篇文档 content 拼接（含两篇的内容特征）
        assert "Region 分区" in p["doc"] or "调优参数" in p["doc"]
        assert "\n" in p["doc"]

    def test_chinese_sentence_chunked_with_newlines(self):
        # 中文句切：句号后断行——MiniCheck 内部 nltk 按英文标点切，预切后整行不被吞
        doc = "G1是JDK 9之后的默认垃圾收集器。核心设计是把堆划分为Region。"
        out = cm._pre_chunk(doc)
        assert out == "G1是JDK 9之后的默认垃圾收集器。\n核心设计是把堆划分为Region。"
        # 问号/叹号/分号同样切
        assert cm._pre_chunk("a？b！c；d。") == "a？\nb！\nc；\nd。"

    def test_empty_doc_returns_as_is(self):
        assert cm._pre_chunk("") == ""


class TestMetrics:
    """指标纯函数（不依赖模型）"""

    def test_model_metrics_perfect(self):
        labels = np.array([1, 1, 0, 0])
        probs = np.array([0.9, 0.8, 0.1, 0.2])
        m = cm.model_metrics(labels, probs)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_model_metrics_recall_emphasized(self):
        # 漏抓 supported（正类 Recall 低）——F1 正类=supported 口径
        labels = np.array([1, 1, 0, 0])
        probs = np.array([0.1, 0.1, 0.9, 0.9])  # 全反
        m = cm.model_metrics(labels, probs)
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_model_metrics_partial(self):
        labels = np.array([1, 1, 1, 0, 0])
        probs = np.array([0.9, 0.1, 0.9, 0.9, 0.1])  # 2TP 1FN 1FP 1TN
        m = cm.model_metrics(labels, probs)
        assert m["accuracy"] == pytest.approx(3 / 5)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (2 / 3 + 2 / 3))

    def test_cohen_kappa(self):
        a = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
        b = np.array([0.8, 0.2, 0.8, 0.2, 0.8])
        assert cm.cohen_kappa(a, b) == pytest.approx(1.0)  # 完全一致
        b2 = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        assert cm.cohen_kappa(a, b2) < 0.0  # 完全相反 → 负值


class TestRequireModel:
    """模型缺失报错（不加载真实模型，只测存在性检查）"""

    def test_missing_dir_raises_with_path(self, tmp_path):
        with pytest.raises(FileNotFoundError) as ei:
            cm._require_model(str(tmp_path / "nope"), ["config.json"])
        assert "nope" in str(ei.value)
        assert "config.json" in str(ei.value)

    def test_missing_file_raises(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError) as ei:
            cm._require_model(str(tmp_path), ["config.json", "model.safetensors"])
        assert "model.safetensors" in str(ei.value)

    def test_complete_dir_passes(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        # 不抛异常即通过
        cm._require_model(str(tmp_path), ["config.json", "model.safetensors"])

    def test_hf_cache_layout_detected(self, tmp_path):
        # MiniCheck 走 HF cache 布局（snapshots/<commit>/）——两层探测
        snap = tmp_path / "models--lytang--MiniCheck-RoBERTa-Large" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "pytorch_model.bin").write_text("x")
        (snap / "config.json").write_text("{}")
        cm._require_model(str(tmp_path), ["pytorch_model.bin", "config.json"])
