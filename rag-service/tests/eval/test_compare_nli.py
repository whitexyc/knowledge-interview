"""module-052 NLI 矛盾扫描前置对比脚本单元测试（eval/compare_nli_models.py）

覆盖：
- 三分类标注：100 对从 SUFFICIENCY_DATASET 派生（entailment 50 / neutral 50 /
  contradiction 0——本数据源无矛盾构造成分）、与 module-050 二值标注一致（一套两用）
- hhem_to_three_class：HHEM 连续分数 → 三态映射（≥0.7 entailment / 0.3-0.7 neutral /
  <0.3 contradiction，对齐 module-051 factcheck_judge 阈值）
- binarize：entailment vs 其他（NLI 与 HHEM 对齐口径）
- model_metrics：Cohen's kappa（三分类 + 二值化两口径）+ Accuracy（参考，注基线）
- mdeberta_score：mock 模型不加载真实模型（假 tokenizer/模型 + 真 torch softmax）
- _require_model：模型缺失报错清晰（指出缺失路径），不静默通过
- 降级：--skip 两侧后 main() 不加载任何模型直接退出

说明：
- 不加载真实模型（模型加载留给 --smoke / --limit 冒烟），只测数据构造、映射与指标纯函数。
- 对齐 tests 现有模式：纯单元、不打真实 DB、不打 LLM。
"""
import sys

import numpy as np
import pytest

from eval.benchmarks import compare_nli_models as nm
from eval.benchmarks.compare_factcheck_models import build_pairs


class TestThreeClassLabels:
    """三分类标注：一套两用（与 module-050 二值标注同源派生）"""

    def test_100_pairs_and_distribution(self):
        assert len(nm.THREE_CLASS_LABELS) == 100
        from collections import Counter
        dist = Counter(nm.THREE_CLASS_LABELS)
        assert dist["entailment"] == 50
        assert dist["neutral"] == 50
        assert dist["contradiction"] == 0  # 本数据源无矛盾构造成分（诚实边界）

    def test_labels_valid(self):
        assert set(nm.THREE_CLASS_LABELS) <= {"entailment", "neutral", "contradiction"}

    def test_consistent_with_module050_binary_labels(self):
        # 一套标注两用：sufficient=True ⟺ entailment（本批无 contradiction，映射无损）
        pairs = build_pairs()
        for p, label in zip(pairs, nm.THREE_CLASS_LABELS):
            if p["label"] == 1:
                assert label == "entailment"
            else:
                assert label == "neutral"


class TestHHEMMapping:
    """HHEM 连续分数 → 三态（对齐 module-051 factcheck_judge 阈值 0.7/0.3）"""

    def test_thresholds(self):
        scores = np.array([0.9, 0.7, 0.6999, 0.3, 0.2999, 0.0])
        out = nm.hhem_to_three_class(scores)
        assert list(out) == ["entailment", "entailment", "neutral",
                             "neutral", "contradiction", "contradiction"]

    def test_custom_thresholds(self):
        out = nm.hhem_to_three_class(np.array([0.8, 0.5, 0.2]), high=0.75, low=0.25)
        assert list(out) == ["entailment", "neutral", "contradiction"]

    def test_binarize_entailment_vs_other(self):
        labels = np.array(["entailment", "neutral", "contradiction", "entailment"])
        out = nm.binarize(labels)
        assert list(out) == [True, False, False, True]


class TestMetrics:
    """指标纯函数：kappa（主）+ Accuracy（参考）"""

    def test_perfect_agreement_kappa_1(self):
        human = ["entailment"] * 50 + ["neutral"] * 50
        pred = list(human)
        m = nm.model_metrics(human, pred)
        assert m["kappa_3class"] == pytest.approx(1.0)
        assert m["kappa_binary"] == pytest.approx(1.0)
        assert m["accuracy_3class"] == pytest.approx(1.0)
        assert m["accuracy_binary"] == pytest.approx(1.0)

    def test_complete_disagreement_kappa_negative(self):
        human = ["entailment"] * 50 + ["neutral"] * 50
        pred = ["neutral"] * 50 + ["entailment"] * 50  # 全反（三分类内互换）
        m = nm.model_metrics(human, pred)
        assert m["kappa_3class"] < 0.0
        assert m["accuracy_3class"] == pytest.approx(0.0)

    def test_chance_level_kappa_near_zero(self):
        # 与随机一致的判定 → kappa≈0（校正随机一致是选 kappa 而非 Acc 的原因）
        rng = np.random.RandomState(7)
        human = (rng.choice(["entailment", "neutral"], 100)).tolist()
        pred = (rng.choice(["entailment", "neutral"], 100)).tolist()
        m = nm.model_metrics(human, pred)
        assert abs(m["kappa_3class"]) < 0.15
        assert m["accuracy_3class"] == pytest.approx(np.mean(np.asarray(human)
                                                             == np.asarray(pred)))

    def test_model_predicting_extra_class_handled(self):
        # 模型预测出现人工没有的类（如 contradiction）→ kappa 不崩（union 类集）
        human = ["entailment", "entailment", "neutral", "neutral"]
        pred = ["entailment", "contradiction", "neutral", "entailment"]
        m = nm.model_metrics(human, pred)
        assert -1.0 <= m["kappa_3class"] <= 1.0
        assert m["accuracy_3class"] == pytest.approx(0.5)


class TestMdebertaScoreMock:
    """mdeberta_score 用 mock 模型（不加载真实模型），验证 argmax/id2label/截断参数"""

    def test_score_with_mock_model(self, monkeypatch):
        import torch
        from types import SimpleNamespace

        class FakeTokenizer:
            def __call__(self, *a, **kw):
                # 记录截断参数（对齐 README truncation 语义）
                assert kw.get("truncation") is True
                assert kw.get("max_length") == 512
                return {"input_ids": torch.zeros(2, 3, dtype=torch.long)}

        class FakeModel:
            config = SimpleNamespace(
                id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                label2id={"entailment": 0, "neutral": 1, "contradiction": 2})

            def __call__(self, **kw):
                return SimpleNamespace(logits=torch.tensor([[3.0, 0.0, 0.0],
                                                            [0.0, 3.0, 0.0]]))

        monkeypatch.setitem(nm._mdeberta, "tokenizer", FakeTokenizer())
        monkeypatch.setitem(nm._mdeberta, "model", FakeModel())
        monkeypatch.setitem(nm._mdeberta, "id2label",
                            {0: "entailment", 1: "neutral", 2: "contradiction"})
        labels, probs = nm.mdeberta_score(["d1", "d2"], ["c1", "c2"])
        assert list(labels) == [0, 1]
        assert probs.shape == (2, 3)
        assert probs[0].argmax() == 0


class TestRequireModel:
    """模型缺失报错（不加载真实模型，只测存在性检查）"""

    def test_missing_dir_raises_with_path(self, tmp_path):
        with pytest.raises(FileNotFoundError) as ei:
            nm._require_model(str(tmp_path / "nope"), ["config.json"])
        assert "nope" in str(ei.value)
        assert "config.json" in str(ei.value)

    def test_missing_file_raises(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError) as ei:
            nm._require_model(str(tmp_path), ["config.json", "model.safetensors"])
        assert "model.safetensors" in str(ei.value)

    def test_complete_dir_passes(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        nm._require_model(str(tmp_path), ["config.json", "model.safetensors"])


class TestDegradation:
    """降级：两侧 --skip 时 main() 不加载任何模型直接退出"""

    def test_main_skip_both_loads_nothing(self, capsys, monkeypatch):
        # 降级断言：两侧 --skip 时加载函数绝不被调用（被调用即 pytest.fail）
        monkeypatch.setattr(nm, "load_mdeberta",
                            lambda: pytest.fail("mDeBERTa 不应被加载"))
        monkeypatch.setattr(nm, "load_hhem",
                            lambda: pytest.fail("HHEM 不应被加载"))
        monkeypatch.setattr(sys, "argv", ["compare_nli_models",
                                          "--skip-mdeberta", "--skip-hhem", "--limit", "5"])
        nm.main()
        out = capsys.readouterr().out
        assert "两侧模型都被跳过" in out
