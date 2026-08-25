"""module-047 WP2 阈值扫描脚本单元测试（eval/threshold_scan.py）

覆盖：
- compute_prf: P/R/F1 计算（含分母为 0 降级）
- scan_scores: 通用阈值扫描（score<t 触发 / None 永不触发 / TP/FP/FN/准确率）
- recommend_threshold: 推荐选取（argmax F1 / min_recall 约束 / 无候选回退）
- l2_trigger_samples: L2 触发窗口与"应触发"定义（knowledge 被 LLM 漏判）
- l2_final_accuracy: 复现 router.classify L2 修正逻辑的最终意图准确率
- scan_l2 / scan_gate: 端到端扫描行结构
- 扫描区间常量: L2 0.20-0.80 步长 0.05 / 闸门 0.20-0.60 步长 0.05（plan 3.2）

说明：
- 全部为纯函数测试，不依赖 LLM / 数据库 / embedding（数据收集在脚本
  main 中，真实数字走实测；本文件只测扫描/推荐逻辑）。
"""
from eval.benchmarks import threshold_scan as ts


class TestComputePrf:
    """P/R/F1 计算（与 compute_confusion_matrix 同口径）"""

    def test_standard_case(self):
        prf = ts.compute_prf(tp=10, fp=2, fn=3)
        assert prf["precision"] == round(10 / 12, 4)
        assert prf["recall"] == round(10 / 13, 4)
        assert prf["f1"] == round(2 * 10 / (20 + 2 + 3), 4)

    def test_all_zero_returns_zeros(self):
        assert ts.compute_prf(tp=0, fp=0, fn=0) == {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    def test_no_negative_predictions(self):
        # tp+fp=0 → precision 0；tp+fn=0 → recall 0（不除零崩溃）
        assert ts.compute_prf(tp=0, fp=0, fn=5)["precision"] == 0.0
        assert ts.compute_prf(tp=0, fp=0, fn=5)["recall"] == 0.0
        assert ts.compute_prf(tp=0, fp=0, fn=5)["f1"] == 0.0


class TestScanScores:
    """通用阈值扫描核心逻辑（手工计算样例）"""

    @staticmethod
    def _samples():
        # A: 0.3 应触发 | B: 0.6 应触发 | C: 0.5 不应触发 | D: None 应触发 | E: 0.8 不应触发
        return [
            {"score": 0.3, "should": True},
            {"score": 0.6, "should": True},
            {"score": 0.5, "should": False},
            {"score": None, "should": True},
            {"score": 0.8, "should": False},
        ]

    def test_threshold_040_hand_computed(self):
        row = [r for r in ts.scan_scores(self._samples(), [0.40])][0]
        assert row["tp"] == 1      # A 触发且应触发
        assert row["fp"] == 0
        assert row["fn"] == 2      # B 未触发；D score=None 永不触发
        assert row["accuracy"] == round(3 / 5, 4)  # A✓ B✗ C✓ D✗ E✓

    def test_threshold_055_hand_computed(self):
        row = [r for r in ts.scan_scores(self._samples(), [0.55])][0]
        assert row["tp"] == 1
        assert row["fp"] == 1      # C(0.5<0.55) 误触发
        assert row["fn"] == 2      # B(0.6≥0.55) 漏触发；D None
        assert row["precision"] == 0.5
        assert row["recall"] == round(1 / 3, 4)
        assert row["f1"] == 0.4
        assert row["accuracy"] == round(2 / 5, 4)

    def test_threshold_070_hand_computed(self):
        row = [r for r in ts.scan_scores(self._samples(), [0.70])][0]
        assert row["tp"] == 2      # A、B 均触发
        assert row["fp"] == 1      # C 误触发
        assert row["fn"] == 1      # D None 永不触发
        assert row["precision"] == round(2 / 3, 4)
        assert row["recall"] == round(2 / 3, 4)
        assert row["f1"] == round(4 / 6, 4)
        assert row["accuracy"] == round(3 / 5, 4)

    def test_none_score_never_triggers(self):
        # score=None（intent==knowledge 无触发窗口 / confidence 缺失）→ 全阈值永不触发
        samples = [{"score": None, "should": True}]
        rows = ts.scan_scores(samples, [0.2, 0.5, 0.8])
        assert all(r["tp"] == 0 and r["fn"] == 1 and r["recall"] == 0.0 for r in rows)

    def test_empty_samples_no_crash(self):
        assert ts.scan_scores([], [0.2, 0.5]) == [
            {"threshold": 0.2, "tp": 0, "fp": 0, "fn": 0,
             "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
            {"threshold": 0.5, "tp": 0, "fp": 0, "fn": 0,
             "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
        ]


class TestRecommendThreshold:
    """推荐阈值选取（argmax F1，可选 recall 约束）"""

    def test_argmax_f1(self):
        rows = ts.scan_scores(
            [{"score": 0.1, "should": True},
             {"score": 0.3, "should": True},
             {"score": 0.5, "should": False}],
            [0.2, 0.4, 0.6],
        )
        rec = ts.recommend_threshold(rows)
        best = max(rows, key=lambda r: r["f1"])
        assert rec["threshold"] == best["threshold"]
        assert rec["f1"] == best["f1"]
        assert rec["fallback"] is False

    def test_min_recall_constraint(self):
        # 无约束时 F1 最大在 t=0.15（recall 仅 0.5）；约束 recall≥0.9 后改选 t=0.6
        rows = ts.scan_scores(
            [{"score": 0.1, "should": True},
             {"score": 0.45, "should": True},
             {"score": 0.2, "should": False},
             {"score": 0.5, "should": False}],
            [0.15, 0.3, 0.6],
        )
        unconstrained = ts.recommend_threshold(rows)
        assert unconstrained["threshold"] == 0.15   # argmax F1（与 t=0.6 并列取先者）
        assert unconstrained["recall"] == 0.5
        rec = ts.recommend_threshold(rows, min_recall=0.9)
        assert rec["threshold"] == 0.6              # 唯一 recall 达标候选
        assert rec["recall"] == 1.0
        assert rec["fallback"] is False

    def test_min_recall_no_candidate_falls_back(self):
        # score=None 的正样本永不触发 → 最大 recall 2/3 < 0.99 → 回退全局 F1 最大
        rows = ts.scan_scores(
            [{"score": 0.1, "should": True},
             {"score": 0.3, "should": True},
             {"score": 0.5, "should": False},
             {"score": None, "should": True}],
            [0.2, 0.4, 0.6],
        )
        rec = ts.recommend_threshold(rows, min_recall=0.99)
        assert rec["fallback"] is True
        assert rec["threshold"] == 0.4              # 全局 argmax F1
        assert rec["f1"] == max(rows, key=lambda r: r["f1"])["f1"]

    def test_empty_rows_returns_none(self):
        assert ts.recommend_threshold([]) is None


class TestL2TriggerSamples:
    """L2 触发窗口与"应触发"定义"""

    def test_score_mapping(self):
        records = [
            {"label": "knowledge", "raw_intent": "casual_chat", "confidence": 0.4,
             "confirmed": True},
            {"label": "knowledge", "raw_intent": "knowledge", "confidence": 0.9,
             "confirmed": False},
            {"label": "casual_chat", "raw_intent": "casual_chat", "confidence": 0.3,
             "confirmed": True},
        ]
        samples = ts.l2_trigger_samples(records)
        # raw_intent≠knowledge → score=confidence，可被触发
        assert samples[0]["score"] == 0.4
        assert samples[0]["should"] is True      # knowledge 被 LLM 漏判 → 应触发
        # raw_intent==knowledge → 无触发窗口（None），与 router 条件一致
        assert samples[1]["score"] is None
        assert samples[1]["should"] is False     # LLM 判对，无需触发
        # 非 knowledge 且 LLM 判对 → 不应触发（触发只会带来误修正风险）
        assert samples[2]["score"] == 0.3
        assert samples[2]["should"] is False

    def test_confidence_none_never_should_trigger_with_missing_score(self):
        samples = ts.l2_trigger_samples([
            {"label": "knowledge", "raw_intent": "realtime", "confidence": None,
             "confirmed": True},
        ])
        assert samples[0]["score"] is None
        assert samples[0]["should"] is True      # 应触发但无 confidence 信号 → 全阈值漏检


class TestL2FinalAccuracy:
    """复现 router.classify 的 L2 修正逻辑（低置信 + 确认信号 → 修正为 knowledge）"""

    def test_hand_computed_050(self):
        records = [
            {"label": "knowledge", "raw_intent": "casual_chat", "confidence": 0.4,
             "confirmed": True},    # 触发+确认 → knowledge ✓
            {"label": "casual_chat", "raw_intent": "casual_chat", "confidence": 0.4,
             "confirmed": True},    # 触发+误确认 → knowledge ✗（误修正代价）
            {"label": "knowledge", "raw_intent": "knowledge", "confidence": 0.9,
             "confirmed": True},    # 无触发窗口 → knowledge ✓
            {"label": "knowledge", "raw_intent": "realtime", "confidence": 0.6,
             "confirmed": True},    # 0.6 ≥ 0.5 不触发 → realtime ✗（漏检）
            {"label": "realtime", "raw_intent": "realtime", "confidence": 0.3,
             "confirmed": False},   # 触发但无确认信号 → realtime ✓
        ]
        assert ts.l2_final_accuracy(records, 0.5) == round(3 / 5, 4)

    def test_higher_threshold_catches_more(self):
        records = [
            {"label": "knowledge", "raw_intent": "realtime", "confidence": 0.6,
             "confirmed": True},
            {"label": "realtime", "raw_intent": "realtime", "confidence": 0.55,
             "confirmed": False},
        ]
        assert ts.l2_final_accuracy(records, 0.5) == round(1 / 2, 4)
        assert ts.l2_final_accuracy(records, 0.7) == 1.0  # 0.6<0.7 触发并确认修正

    def test_empty_records_zero(self):
        assert ts.l2_final_accuracy([], 0.5) == 0.0


class TestScanL2AndGate:
    """L2 / 闸门端到端扫描行结构"""

    def test_scan_l2_rows_have_final_accuracy(self):
        records = [
            {"label": "knowledge", "raw_intent": "casual_chat", "confidence": 0.4,
             "confirmed": True},
            {"label": "casual_chat", "raw_intent": "casual_chat", "confidence": 0.9,
             "confirmed": False},
        ]
        rows = ts.scan_l2(records, [0.2, 0.5, 0.8])
        assert len(rows) == 3
        assert all({"threshold", "tp", "fp", "fn", "precision", "recall",
                    "f1", "accuracy", "final_accuracy"} <= set(r) for r in rows)
        # t=0.2: 无触发（0.4≥0.2）→ 漏检 1
        assert rows[0]["tp"] == 0 and rows[0]["fn"] == 1
        # t=0.5: 触发+确认 → 修正正确
        assert rows[1]["tp"] == 1 and rows[1]["fn"] == 0
        assert rows[1]["final_accuracy"] == 1.0

    def test_scan_gate_hand_computed(self):
        records = [
            {"top1_abs": 0.3, "sufficient": False},   # 不充分，低分 → 应被闸门抓到
            {"top1_abs": 0.55, "sufficient": False},  # 不充分，中分 → t=0.6 才抓到
            {"top1_abs": 0.75, "sufficient": True},   # 充分，高分 → 正确放行
            {"top1_abs": 0.35, "sufficient": True},   # 充分，低分 → 闸门误杀
        ]
        rows = ts.scan_gate(records, [0.4, 0.6])
        # t=0.4: 抓 A、D → TP=1(A) FP=1(D) FN=1(B)
        assert rows[0]["tp"] == 1 and rows[0]["fp"] == 1 and rows[0]["fn"] == 1
        assert rows[0]["precision"] == 0.5 and rows[0]["recall"] == 0.5
        # t=0.6: 抓 A、B、D（C=0.75 不触发）→ TP=2 FP=1 FN=0（不充分全抓回，D 误杀）
        assert rows[1]["tp"] == 2 and rows[1]["fp"] == 1 and rows[1]["fn"] == 0
        assert rows[1]["recall"] == 1.0
        assert rows[1]["precision"] == round(2 / 3, 4)


class TestScanRanges:
    """扫描区间常量（plan 3.2 定死）"""

    def test_l2_range_020_to_080_step_005(self):
        assert ts.L2_THRESHOLDS[0] == 0.2
        assert ts.L2_THRESHOLDS[-1] == 0.8
        assert len(ts.L2_THRESHOLDS) == 13
        assert all(
            abs(b - a - 0.05) < 1e-9
            for a, b in zip(ts.L2_THRESHOLDS, ts.L2_THRESHOLDS[1:])
        )

    def test_gate_range_020_to_060_step_005(self):
        assert ts.GATE_THRESHOLDS[0] == 0.2
        assert ts.GATE_THRESHOLDS[-1] == 0.6
        assert len(ts.GATE_THRESHOLDS) == 9

    def test_empirical_values_are_current_production(self):
        # 经验值必须与生产代码当前取值一致（被校准对象）
        assert ts.EMPIRICAL_L2 == 0.5
        assert ts.EMPIRICAL_GATE == 0.4
