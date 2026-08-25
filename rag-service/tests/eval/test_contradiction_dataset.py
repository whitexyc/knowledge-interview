"""Module-054 矛盾样本集测试

覆盖（验收 §3/§7）：
- 样本数量：contradiction ≥ 30（两类：claim_vs_doc / internal_contradiction）
- 正例对照（entailment）与 neutral 对照存在
- JSON 结构：question/claim/doc/verdict 四键 + verdict 三分类合法
- golden_factcheck 兼容：to_factcheck_item 映射（question/documents/label +
  verdict→label 三态映射）与反向 from_factcheck_item 往返一致
- 标注指南落盘存在
"""
import os

import pytest

from eval.datasets.contradiction_dataset import (DATASET_PATH, load_contradiction_dataset,
                                        to_factcheck_item, from_factcheck_item)
from eval.datasets import contradiction_dataset as cd

GUIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(cd.__file__)),
                          "contradiction_annotation_guide.md")


@pytest.fixture(scope="module")
def samples():
    return load_contradiction_dataset()


class TestDatasetScale:
    """数量与分布：矛盾 ≥30 + 两类 + 正例对照"""

    def test_at_least_30_contradictions(self, samples):
        contradictions = [s for s in samples if s["verdict"] == "contradiction"]
        assert len(contradictions) >= 30

    def test_two_contradiction_types(self, samples):
        types = {s.get("contradiction_type") for s in samples
                 if s["verdict"] == "contradiction"}
        assert {"claim_vs_doc", "internal_contradiction"} <= types

    def test_positive_controls_present(self, samples):
        # 正例对照（一致样本）与 neutral 对照都在
        assert any(s["verdict"] == "entailment" for s in samples)
        assert any(s["verdict"] == "neutral" for s in samples)
        positives = sum(1 for s in samples if s["verdict"] == "entailment")
        contradictions = sum(1 for s in samples if s["verdict"] == "contradiction")
        assert positives >= contradictions // 4, "正例对照应与矛盾样本成比例"


class TestDatasetStructure:
    """JSON 结构：question/claim/doc/verdict 四键 + verdict 合法"""

    def test_required_keys(self, samples):
        for s in samples:
            for key in ("question", "claim", "doc", "verdict"):
                assert s.get(key, "").strip(), f"缺 {key}: {s.get('question', '')[:30]}"

    def test_verdict_valid(self, samples):
        assert {s["verdict"] for s in samples} <= {"entailment", "neutral", "contradiction"}

    def test_docs_are_real_content(self, samples):
        # doc 为真实知识库段落（非空、非占位符）
        for s in samples:
            assert len(s["doc"]) >= 30, f"doc 过短: {s.get('question', '')[:30]}"
            assert s["doc"].strip() not in ("", "[NO_DOCS]")

    def test_annotation_guide_written(self):
        assert os.path.isfile(GUIDE_PATH), "标注指南应落盘"
        with open(GUIDE_PATH, encoding="utf-8") as f:
            content = f.read()
        assert "矛盾" in content and "contradiction" in content


class TestGoldenFactcheckCompat:
    """与 golden_factcheck 结构兼容（question/documents/label）"""

    def test_to_factcheck_mapping(self, samples):
        for s in samples:
            fc = to_factcheck_item(s)
            assert fc["question"] == s["question"]
            assert len(fc["documents"]) == 1
            assert fc["documents"][0]["content"] == s["doc"]
            assert fc["documents"][0]["title"] == s.get("doc_title", "")
            expected = {"entailment": "supported", "neutral": "inferred",
                        "contradiction": "unsupported"}[s["verdict"]]
            assert fc["label"] == expected
            assert fc["label"] in ("supported", "inferred", "unsupported")

    def test_roundtrip(self, samples):
        for s in samples:
            rt = from_factcheck_item(to_factcheck_item(s))
            assert rt["verdict"] == s["verdict"]
            assert rt["doc"] == s["doc"]
            assert rt["question"] == s["question"]

    def test_dataset_json_file_exists(self):
        assert os.path.isfile(DATASET_PATH)
