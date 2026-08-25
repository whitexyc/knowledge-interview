"""Module-057 矛盾改进测试：句级拆解 + 阈值校准 + 样本扩充（WP-A1）

覆盖（验收 §8 tests/test_nli_improve.py）：
- 句切：中文/英文句末标点（。！？；!?）切子句；无 ≥2 子句回退整句（零回归）
- 低置信降级：max prob < 阈值 → neutral
- 聚合：任一子句/子句对 contradiction → contradiction（最严）；
  无矛盾有 entailment → entailment；全 neutral → neutral
- 拆解判定：整句形态 / 拆解形态（子句 vs doc + 子句两两互判双向）
- 阈值扫描：0.5-0.9 步长 0.05 逐阈值 kappa + 预测分布；最优阈值选取
- 样本集：contradiction ≥ 50、internal ≥ 20（含多句混合"前真后假"）、
  JSON 结构不变、constructed[:56] 保持 module-054 同集（同口径对比前提）

实现说明：
- scorer 全部注入假实现（不加载 mDeBERTa 模型），拆解/聚合/阈值逻辑纯函数单测
- 数据集校验走 eval/contradiction_dataset.py 真实 JSON
"""
import pytest

from eval.benchmarks.retest_nli import (SCAN_THRESHOLDS, aggregate_sub_judgments,
                             apply_threshold, collect_decomposed_predictions,
                             scan_thresholds, split_claim,
                             verdict_for_threshold)
from eval.datasets.contradiction_dataset import (DATASET_PATH, load_contradiction_dataset)


# ── 句切 ──

class TestSplitClaim:
    def test_split_on_chinese_punctuation(self):
        # 与 _pre_chunk 同语义：标点留在句尾（lookbehind 切分）
        assert split_claim("G1 是 JDK 9 之后的默认垃圾收集器。它自 JDK 9 起不再使用。") == \
            ["G1 是 JDK 9 之后的默认垃圾收集器。", "它自 JDK 9 起不再使用。"]

    def test_split_on_all_punctuation_kinds(self):
        parts = split_claim("前句成立！中句成立？后句也成立。")
        assert parts == ["前句成立！", "中句成立？", "后句也成立。"]

    def test_split_on_semicolon(self):
        assert split_claim("A 是这样；B 是那样；C 如此。") == ["A 是这样；", "B 是那样；", "C 如此。"]

    def test_no_split_falls_back_to_whole(self):
        # 无句末标点（单句）→ 回退整句（零回归）
        claim = "G1 垃圾收集器是 JDK 8 及之前的默认垃圾收集器，JDK 9 之后已被 CMS 取代"
        assert split_claim(claim) == [claim.strip()]

    def test_single_sentence_with_trailing_period(self):
        # 只有结尾一个句号 → 拆出 1 个子句 → 回退整句
        claim = "G1 垃圾收集器是 JDK 9 之后的默认垃圾收集器。"
        out = split_claim(claim)
        assert len(out) == 1
        assert out[0] == "G1 垃圾收集器是 JDK 9 之后的默认垃圾收集器。"

    def test_empty_claim(self):
        assert split_claim("") == [""]

    def test_newline_splits(self):
        assert split_claim("第一句。\n第二句。") == ["第一句。", "第二句。"]


# ── 低置信降级 ──

class TestApplyThreshold:
    def test_high_conf_keeps_pred(self):
        assert apply_threshold("contradiction", 0.85, 0.6) == "contradiction"
        assert apply_threshold("entailment", 0.72, 0.6) == "entailment"

    def test_low_conf_becomes_neutral(self):
        assert apply_threshold("contradiction", 0.55, 0.6) == "neutral"
        assert apply_threshold("entailment", 0.59, 0.6) == "neutral"

    def test_boundary_exact_equal_not_neutral(self):
        assert apply_threshold("neutral", 0.6, 0.6) == "neutral"

    def test_threshold_05_no_effect_on_decisive(self):
        # 0.5 阈值下高置信判定原样
        assert apply_threshold("contradiction", 0.99, 0.5) == "contradiction"


# ── 聚合（最严语义） ──

class TestAggregate:
    def test_any_sub_contradiction_wins(self):
        assert aggregate_sub_judgments(
            ["entailment", "contradiction"], []) == "contradiction"
        assert aggregate_sub_judgments(
            ["entailment", "neutral", "contradiction"], ["neutral"]) == "contradiction"

    def test_pair_contradiction_wins(self):
        # 子句对判 contradiction（内部矛盾信号）同样触发最严聚合
        assert aggregate_sub_judgments(
            ["entailment", "neutral"], ["neutral", "contradiction"]) == "contradiction"

    def test_no_contradiction_with_entailment(self):
        assert aggregate_sub_judgments(["entailment", "neutral"], []) == "entailment"
        assert aggregate_sub_judgments(
            ["neutral", "entailment"], ["neutral", "neutral"]) == "entailment"

    def test_all_neutral(self):
        assert aggregate_sub_judgments(["neutral", "neutral"], []) == "neutral"
        assert aggregate_sub_judgments(
            ["neutral"], ["neutral", "neutral"]) == "neutral"

    def test_pair_entailment_does_not_drive_entailment(self):
        # 子句对 entailment（子句间同义）不算"与文档一致"，最终判定由子句 vs doc 决定
        assert aggregate_sub_judgments(
            ["neutral", "neutral"], ["entailment"]) == "neutral"


# ── 拆解判定管线（假 scorer 注入） ──

def _fake_scorer(docs, claims):
    """假 scorer：返回 (标签, 置信度)；默认高置信 0.95"""
    out_labels, out_confs = [], []
    for doc, claim in zip(docs, claims):
        if "完全移除" in claim or "不再使用" in claim:
            label, conf = "contradiction", 0.95
        elif "默认垃圾收集器" in claim or "Region" in claim:
            label, conf = "entailment", 0.95
        else:
            label, conf = "neutral", 0.95
        out_labels.append(label)
        out_confs.append(conf)
    return out_labels, out_confs


def _sample(doc, claim, verdict):
    return {"question": "q", "claim": claim, "doc": doc, "verdict": verdict}


class TestCollectDecomposed:
    def test_single_sentence_uses_whole(self):
        s = _sample("G1 是 JDK 9 之后的默认垃圾收集器。", "G1 是 JDK 9 之后的默认垃圾收集器。",
                    "entailment")
        recs = collect_decomposed_predictions([s], scorer=_fake_scorer)
        assert "whole" in recs[0]
        assert recs[0]["parts"] == ["G1 是 JDK 9 之后的默认垃圾收集器。"]
        assert recs[0]["whole"][0] == "entailment"

    def test_multi_sentence_has_sub_and_pair(self):
        s = _sample(
            "G1 是 JDK 9 之后的默认垃圾收集器。",
            "G1 是 JDK 9 之后的默认垃圾收集器。它自 JDK 9 起不再使用。",
            "contradiction")
        recs = collect_decomposed_predictions([s], scorer=_fake_scorer)
        rec = recs[0]
        assert "whole" not in rec
        assert len(rec["parts"]) == 2
        assert len(rec["sub"]) == 2          # 每子句 vs doc
        assert len(rec["pair"]) == 2         # 双向：子句 0→1 和 1→0

    def test_pair_docs_claims_bidirectional(self):
        # 子句 i 作 doc、子句 j 作 claim；i≠j 双向
        s = _sample("doc 无关。", "甲。乙。", "neutral")
        recs = collect_decomposed_predictions([s], scorer=_fake_scorer)
        pair = recs[0]["pair"]
        assert len(pair) == 2

    def test_three_parts_pair_count(self):
        s = _sample("doc 无关。", "甲。乙。丙。", "neutral")
        recs = collect_decomposed_predictions([s], scorer=_fake_scorer)
        assert len(recs[0]["sub"]) == 3
        assert len(recs[0]["pair"]) == 6     # 3×2


class TestVerdictForThreshold:
    def test_whole_form(self):
        s = _sample("G1 是 JDK 9 之后的默认垃圾收集器。", "G1 是 JDK 9 之后的默认垃圾收集器。",
                    "entailment")
        rec = collect_decomposed_predictions([s], scorer=_fake_scorer)[0]
        verdict, subs, pairs = verdict_for_threshold(rec, 0.6)
        assert verdict == "entailment"
        assert subs == ["entailment"]
        assert pairs == []

    def test_decomposed_catches_second_sentence(self):
        # "前真后假"多句混合：后句与文档矛盾 → 最严聚合 contradiction
        s = _sample(
            "G1 是 JDK 9 之后的默认垃圾收集器。",
            "G1 是 JDK 9 之后的默认垃圾收集器。它自 JDK 9 起不再使用。",
            "contradiction")
        rec = collect_decomposed_predictions([s], scorer=_fake_scorer)[0]
        verdict, subs, pairs = verdict_for_threshold(rec, 0.6)
        assert verdict == "contradiction"
        assert "contradiction" in subs

    def test_low_confidence_sub_downgraded_to_neutral(self):
        # 低置信子句被降级为 neutral 后，聚合看剩余子句
        def low_conf_scorer(docs, claims):
            return (["entailment"] * len(docs), [0.4] * len(docs))
        s = _sample("doc 支持。", "甲句。乙句。", "neutral")
        rec = collect_decomposed_predictions([s], scorer=low_conf_scorer)[0]
        # 全子句低置信 → 全 neutral → 聚合 neutral
        assert verdict_for_threshold(rec, 0.6)[0] == "neutral"


# ── 阈值扫描 ──

class TestThresholdScan:
    def test_scan_covers_050_090_step_005(self):
        assert SCAN_THRESHOLDS == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    def test_scan_rows_per_threshold(self):
        samples = [
            _sample("G1 是 JDK 9 之后的默认垃圾收集器。",
                    "G1 是 JDK 9 之后的默认垃圾收集器。", "entailment"),
            _sample("G1 是 JDK 9 之后的默认垃圾收集器。",
                    "G1 是 JDK 9 之后的默认垃圾收集器。它自 JDK 9 起不再使用。",
                    "contradiction"),
            _sample("G1 是 JDK 9 之后的默认垃圾收集器。", "Kafka 的 ISR 机制。", "neutral"),
        ]
        records = collect_decomposed_predictions(samples, scorer=_fake_scorer)
        human = [s["verdict"] for s in samples]
        rows = scan_thresholds(records, human)
        assert len(rows) == len(SCAN_THRESHOLDS)
        for row in rows:
            assert row["threshold"] in SCAN_THRESHOLDS
            assert 0.0 <= row["kappa_3class"] <= 1.0
            assert (row["pred_entailment"] + row["pred_neutral"]
                    + row["pred_contradiction"]) == len(samples)

    def test_scan_best_threshold_selection(self):
        # 高置信判对的样本在任意阈值下 kappa 一致；阈值选择逻辑取 argmax kappa
        samples = [
            _sample("doc。", "甲。乙。", "contradiction"),
            _sample("doc。", "丙。", "neutral"),
        ]
        records = collect_decomposed_predictions(
            samples, scorer=lambda d, c: (["contradiction"] * len(d),
                                           [0.9] * len(d)))
        rows = scan_thresholds(records, [s["verdict"] for s in samples])
        best = max(rows, key=lambda r: r["kappa_3class"])
        assert best["threshold"] == 0.5  # 全部高置信，0.5 最低阈即最优（无降级）

    def test_scan_low_conf_shift_distribution(self):
        # 低置信 contradiction → neutral 会改变预测分布（threshold 越高 neutral 越多）
        samples = [_sample("doc 无关。", "甲句。", "contradiction")]
        records = collect_decomposed_predictions(
            samples, scorer=lambda d, c: (["contradiction"], [0.52]))
        human = ["contradiction"]
        rows = scan_thresholds(records, human)
        assert rows[0]["pred_contradiction"] == 1      # t=0.5: 0.52 ≥ 0.5 保留
        assert rows[-1]["pred_contradiction"] == 0     # t=0.9: 0.52 < 0.9 → neutral
        assert rows[-1]["pred_neutral"] == 1


# ── 样本集（WP-A1 扩充验收） ──

class TestExpandedDataset:
    def test_contradiction_at_least_50(self):
        samples = load_contradiction_dataset()
        contradictions = [s for s in samples if s["verdict"] == "contradiction"]
        assert len(contradictions) >= 50

    def test_internal_at_least_20_with_multi_sentence(self):
        samples = load_contradiction_dataset()
        internal = [s for s in samples if s.get("contradiction_type") == "internal_contradiction"]
        assert len(internal) >= 20
        # 多句混合"前真后假"样本必须存在（句号分隔 ≥2 子句）
        multi = [s for s in internal if len(split_claim(s["claim"])) >= 2]
        assert len(multi) >= 5

    def test_old_56_first_is_module054_same_set(self):
        # 同口径对比前提：constructed[:56] 保持 module-054 首版构成
        #（16 claim_vs_doc + 15 internal + 16 entailment + 9 neutral，顺序不变）
        samples = load_contradiction_dataset()
        first56 = samples[:56]
        assert sum(1 for s in first56 if s.get("contradiction_type") == "claim_vs_doc") == 16
        assert sum(1 for s in first56 if s.get("contradiction_type") == "internal_contradiction") == 15
        assert sum(1 for s in first56 if s["verdict"] == "entailment") == 16
        assert sum(1 for s in first56 if s["verdict"] == "neutral") == 9

    def test_json_structure_unchanged(self):
        samples = load_contradiction_dataset()
        for s in samples:
            for key in ("question", "claim", "doc", "verdict"):
                assert s.get(key, "").strip(), f"缺 {key}: {s.get('question', '')[:30]}"
            assert s["verdict"] in ("entailment", "neutral", "contradiction")
            assert s.get("part") == "constructed"

    def test_dataset_json_exists(self):
        import os
        assert os.path.isfile(DATASET_PATH)
