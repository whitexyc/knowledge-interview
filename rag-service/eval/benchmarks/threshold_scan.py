"""
阈值扫描脚本 — 数据驱动校准两个经验阈值（module-047 WP2）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景:
    两个阈值目前是经验值（ADR-0005 追问 2 注明）:
      ① L2 触发阈值   router._L2_CONFIDENCE_THRESHOLD = 0.5
         （intent≠knowledge 且 LLM confidence < t → 触发确定性信号确认）
      ② 硬闸门阈值     reflector._SUFFICIENCY_MIN_ABS_COSINE = 0.4
         （top-1 abs_cosine < t → 零 LLM 直接判不充分）
    本脚本用真实标注集 + 真实测量做阈值扫描，输出 P/R/F1 曲线、
    推荐阈值与经验值对比（数据驱动，一致/不一致都给理由）。

扫描 ①（L2 触发阈值，t ∈ [0.2, 0.8] 步长 0.05）:
    数据: eval.golden.golden_intent.INTENT_DATASET 100 条 ×
          真实 LLM 原始分类（intent + confidence，不走 L2）+
          真实确定性信号（router._deterministic_confirm，零 LLM:
          FTS 术语命中 / 图谱实体命中 / 规则表否决 / 保守降级）。
    模拟: 逐 t 复现 router.classify 的 L2 逻辑——
          triggered = (raw_intent≠knowledge) and (confidence<t)
          confirmed → 修正为 knowledge。
    Positive 定义（应触发）: label==knowledge 且 raw_intent≠knowledge——
          LLM 把知识库问题判成非 knowledge（漏检），触发确认是救回它的唯一路径。
    重点: 触发漏检召回（FN 即"该触发没触发"）。

扫描 ②（硬闸门阈值，t ∈ [0.2, 0.6] 步长 0.05）:
    数据: eval.golden.golden_sufficiency.SUFFICIENCY_DATASET 100 条 ×
          实测相似度: 用生产同款 bge-m3 本地嵌入（rag.retrieval.embeddings）计算
          question 与每条注入文档（扮演"检索到的文档"）的余弦，
          top-1 = 每条两篇文档的最大余弦（对应生产闸门读取的 top-1 abs_cosine）。
    模拟: 逐 t 的闸门单独分类器——top-1 < t → 判不充分，否则默认充分
          （闸门未触发时生产走 LLM 层；本扫描隔离闸门单独判别力，
          对未触发样本默认充分的假设在报告中明示）。
    Positive 定义: 不充分（漏判不充分 → 基于无关文档硬答，最致命）。

输出:
    P/R/F1 曲线表 + 推荐阈值（argmax F1，可用 --min-recall 约束）+ 经验值对比。

数据收集（真实测量，缓存到 .ua/ 便于复跑）:
    python -m eval.benchmarks.threshold_scan --collect-l2   # 只收集 L2 数据（LLM 100 次调用，~2-4 分钟）
    python -m eval.benchmarks.threshold_scan --skip-l2      # 不跑 L2（零 LLM），只跑闸门扫描
    python -m eval.benchmarks.threshold_scan                # 默认: 无缓存先收集再扫描
    python -m eval.benchmarks.threshold_scan --no-cache     # 不读不写缓存
    python -m eval.benchmarks.threshold_scan --cache PATH   # 自定义缓存路径
    python -m eval.benchmarks.threshold_scan --min-recall 0.9  # 推荐约束: recall ≥ 0.9 中选 F1 最大

降级:
    - LLM 单条失败重试 1 次，仍失败记 skipped（不中断）
    - embedding 不可用 → 闸门扫描标注"待环境"，不伪造数字
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from agent.router import RouterAgent, _PROMPT_TEMPLATE, router_agent
from eval.golden.golden_intent import load_intent_dataset
from eval.golden.golden_sufficiency import load_sufficiency_dataset
from llm.client import LLMFactory
from rag.retrieval.embeddings import embedding_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("threshold_scan")

# ── 扫描区间（plan 3.2 定死）──
L2_THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(13)]   # 0.20 ~ 0.80
GATE_THRESHOLDS = [round(0.2 + 0.05 * i, 2) for i in range(9)]  # 0.20 ~ 0.60

# 经验值（被校准对象）
EMPIRICAL_L2 = 0.5
EMPIRICAL_GATE = 0.4

# 默认缓存位置（ai_service/.ua/ 已 gitignore，实验数据不入库）
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".ua" / "m047_threshold_cache.json"

_CACHE_SCHEMA = 1


# ──────────────────────────────────────────────────────────────
# 纯函数：P/R/F1 + 通用阈值扫描 + 推荐选取（可单测）
# ──────────────────────────────────────────────────────────────

def compute_prf(tp: int, fp: int, fn: int) -> dict:
    """Precision/Recall/F1（分母为 0 → 0.0，与 compute_confusion_matrix 同口径）

    Args:
        tp: 真阳性（触发且应触发）
        fp: 假阳性（触发但不应触发）
        fn: 漏检（应触发但未触发）

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    denom_p = tp + fp
    denom_r = tp + fn
    denom_f = 2 * tp + fp + fn
    precision = tp / denom_p if denom_p else 0.0
    recall = tp / denom_r if denom_r else 0.0
    f1 = 2 * tp / denom_f if denom_f else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def scan_scores(samples: list[dict], thresholds: list[float]) -> list[dict]:
    """通用阈值扫描：score < t → 预测 positive，与应然标签算 P/R/F1 + 准确率

    Args:
        samples: [{"score": float|None, "should": bool}, ...]
                 score 为 None 表示无分数信号（永不触发，如 intent==knowledge
                 或 confidence 缺失——对齐 router 的置信度缺失不触发）
        thresholds: 扫描阈值列表（升序）

    Returns:
        逐阈值行: {"threshold", "tp", "fp", "fn", "precision", "recall",
                   "f1", "accuracy"}
    """
    rows = []
    n = len(samples)
    for t in thresholds:
        tp = fp = fn = correct = 0
        for s in samples:
            predicted = s["score"] is not None and s["score"] < t
            should = s["should"]
            if predicted and should:
                tp += 1
            elif predicted and not should:
                fp += 1
            elif not predicted and should:
                fn += 1
            if predicted == should:
                correct += 1
        rows.append({
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **compute_prf(tp, fp, fn),
            "accuracy": round(correct / n, 4) if n else 0.0,
        })
    return rows


def recommend_threshold(rows: list[dict], min_recall: float = 0.0) -> dict | None:
    """推荐阈值选取：recall ≥ min_recall 的候选中选 F1 最大

    无满足 recall 约束的候选 → 回退到全局 F1 最大（并在报告标注回退）。

    Args:
        rows: scan_scores 的输出行
        min_recall: 召回约束（0.0 = 纯 F1 最大）

    Returns:
        推荐行（含 "fallback": bool 标记）或 None（rows 为空）
    """
    if not rows:
        return None
    candidates = [r for r in rows if r["recall"] >= min_recall]
    if not candidates:
        best = max(rows, key=lambda r: r["f1"])
        return {**best, "fallback": True}
    best = max(candidates, key=lambda r: r["f1"])
    return {**best, "fallback": False}


# ── L2 触发扫描的样本构造与最终意图模拟 ──

def l2_trigger_samples(records: list[dict]) -> list[dict]:
    """L2 记录 → 通用扫描样本

    触发分数: raw_intent≠knowledge 时为 confidence（LLM 低置信信号），
              raw_intent==knowledge 时无触发窗口 → None（与 router 一致）;
    应触发:   label==knowledge 且 raw_intent≠knowledge（LLM 漏检，
              确认机制是救回它的唯一路径）。

    Args:
        records: [{"label", "raw_intent", "confidence", "confirmed"}, ...]

    Returns:
        [{"score": float|None, "should": bool, "confirmed": bool, "label": str,
          "raw_intent": str}, ...]
    """
    samples = []
    for r in records:
        raw = r["raw_intent"]
        score = r.get("confidence") if raw != "knowledge" else None
        should = r["label"] == "knowledge" and raw != "knowledge"
        samples.append({
            "score": score,
            "should": should,
            "confirmed": bool(r.get("confirmed", False)),
            "label": r["label"],
            "raw_intent": raw,
        })
    return samples


def l2_final_accuracy(records: list[dict], t: float) -> float:
    """阈值 t 下复现 router.classify 完整 L2 逻辑的最终意图准确率

    模拟: final = raw_intent；
          (raw_intent≠knowledge) and (confidence is not None) and
          (confidence < t) and confirmed → final = knowledge（宁多检不漏检）。
    与生产 router.py 的 L2 触发/修正逻辑逐行对齐。

    Args:
        records: [{"label", "raw_intent", "confidence", "confirmed"}, ...]
        t: 触发阈值

    Returns:
        最终意图判对比例（0.0-1.0）
    """
    if not records:
        return 0.0
    correct = 0
    for r in records:
        raw = r["raw_intent"]
        confidence = r.get("confidence")
        final = raw
        if (raw != "knowledge" and confidence is not None
                and confidence < t and r.get("confirmed", False)):
            final = "knowledge"
        if final == r["label"]:
            correct += 1
    return round(correct / len(records), 4)


def scan_l2(records: list[dict], thresholds: list[float]) -> list[dict]:
    """L2 触发阈值全扫描（触发 P/R/F1 + 最终意图准确率）"""
    samples = l2_trigger_samples(records)
    rows = scan_scores(samples, thresholds)
    for row in rows:
        row["final_accuracy"] = l2_final_accuracy(records, row["threshold"])
    return rows


# ── 闸门扫描的样本构造 ──

def gate_samples(records: list[dict]) -> list[dict]:
    """充分性记录 → 通用扫描样本

    Args:
        records: [{"top1_abs": float, "sufficient": bool}, ...]

    Returns:
        [{"score": top1_abs, "should": not sufficient}, ...]
    """
    return [
        {"score": r["top1_abs"], "should": not r["sufficient"]}
        for r in records
    ]


def scan_gate(records: list[dict], thresholds: list[float]) -> list[dict]:
    """硬闸门阈值全扫描（不充分类 P/R/F1）"""
    return scan_scores(gate_samples(records), thresholds)


# ──────────────────────────────────────────────────────────────
# 真实数据收集（LLM / 本地嵌入 / 确定性信号，带降级）
# ──────────────────────────────────────────────────────────────

async def _classify_raw(query: str) -> dict:
    """真实 LLM 原始分类（不走 L2）: 返回 intent/confidence/reason"""
    client = LLMFactory.get_client(None)
    response = await client.generate(_PROMPT_TEMPLATE.format(query=query))
    return RouterAgent._parse_response(response)


async def _collect_l2_record(item: dict) -> tuple[dict, dict]:
    """单条 L2 记录收集: 真实 LLM 原始分类 + 真实确定性信号

    失败重试 1 次，仍失败返回 (None, skipped)。

    Returns:
        (record, skipped) 二元组，恰有一个非空
        record: {"query", "label", "raw_intent", "confidence", "confirmed", "signal"}
    """
    query = item["query"]
    for attempt in range(2):
        try:
            raw = await _classify_raw(query)
            confirmed, signal = await router_agent._deterministic_confirm(query)
            return {
                "query": query,
                "label": item["intent"],
                "raw_intent": raw.get("intent", "knowledge"),
                "confidence": raw.get("confidence"),
                "confirmed": confirmed,
                "signal": signal,
            }, {}
        except Exception as e:
            if attempt == 0:
                logger.warning("L2 收集第 1 次失败，重试: %s — %s", query[:40], e)
            else:
                return {}, {"query": query, "label": item["intent"],
                            "reason": f"error: {e}"}
    return {}, {}


async def collect_l2_data(dataset: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """收集 L2 扫描数据（真实 LLM 分类 + 确定性信号，每条重试 1 次）"""
    items = dataset if dataset is not None else load_intent_dataset()
    records: list[dict] = []
    skipped: list[dict] = []
    for i, item in enumerate(items):
        record, skip = await _collect_l2_record(item)
        if record:
            records.append(record)
        else:
            skipped.append(skip)
        if (i + 1) % 25 == 0:
            logger.info("L2 收集进度: %d/%d", i + 1, len(items))
    return records, skipped


async def _measure_gate_record(item: dict) -> tuple[dict, dict]:
    """单条闸门记录收集: 实测 bge-m3 余弦（question vs 每条注入文档，取 top-1）

    Args:
        item: 充分性样本（question / documents / sufficient）

    Returns:
        (record, error) 二元组，恰有一个非空
        record: {"question", "sufficient", "top1_abs", "n_docs"}
    """
    question = item["question"]
    docs = item["documents"]
    if len(docs) < 2:
        return {}, {"question": question, "reason": "documents < 2"}
    try:
        texts = [question] + [d.get("content", "") for d in docs[:2]]
        vectors = await embedding_service.embed_documents(texts)
        if len(vectors) != 3:
            raise RuntimeError(f"embedding 数量异常: {len(vectors)}")
        q, d1, d2 = vectors
        top1 = max(
            sum(a * b for a, b in zip(q, d1)),
            sum(a * b for a, b in zip(q, d2)),
        )
        return {"question": question, "sufficient": bool(item["sufficient"]),
                "top1_abs": round(top1, 4), "n_docs": len(docs)}, {}
    except Exception as e:
        return {}, {"question": question, "reason": f"error: {e}"}


async def collect_gate_data(dataset: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """收集闸门扫描数据（本地 bge-m3 实测余弦，无 LLM）"""
    items = dataset if dataset is not None else load_sufficiency_dataset()
    records: list[dict] = []
    errors: list[dict] = []
    for i, item in enumerate(items):
        record, err = await _measure_gate_record(item)
        if record:
            records.append(record)
        else:
            errors.append(err)
        if (i + 1) % 25 == 0:
            logger.info("闸门数据收集进度: %d/%d", i + 1, len(items))
    return records, errors


# ──────────────────────────────────────────────────────────────
# 缓存
# ──────────────────────────────────────────────────────────────

def save_cache(path: Path, payload: dict) -> None:
    """写缓存（.ua/ 等 gitignore 目录，失败仅警告）"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.info("缓存已写入: %s", path)
    except Exception as e:
        logger.warning("缓存写入失败: %s", e)


def load_cache(path: Path) -> dict | None:
    """读缓存，结构不合法返回 None"""
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != _CACHE_SCHEMA:
            logger.warning("缓存 schema 不匹配（%s），忽略", data.get("schema_version"))
            return None
        return data
    except Exception as e:
        logger.warning("缓存读取失败: %s", e)
        return None


# ──────────────────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────────────────

def _percentiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    vals = sorted(values)
    n = len(vals)
    def pct(p: float) -> float:
        return vals[min(n - 1, int(p * n))]
    return (f"n={n} min={vals[0]:.3f} p25={pct(0.25):.3f} "
            f"median={pct(0.5):.3f} p75={pct(0.75):.3f} max={vals[-1]:.3f}")


def _print_curve(title: str, rows: list[dict], recommended: dict | None,
                 empirical: float, extra_key: str | None = None) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    header = (f"  {'t':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
              f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>9}")
    if extra_key:
        header += f" {'final_acc':>10}"
    print(header)
    for row in rows:
        line = (f"  {row['threshold']:>6.2f} {row['tp']:>4} {row['fp']:>4} {row['fn']:>4} "
                f"{row['precision']:>10.4f} {row['recall']:>8.4f} {row['f1']:>8.4f} "
                f"{row['accuracy']:>9.4f}")
        if extra_key:
            line += f" {row[extra_key]:>10.4f}"
        print(line)
    if recommended is None:
        print("  无数据，无法推荐")
        return
    fb = "（无候选满足 recall 约束，回退全局 F1 最大）" if recommended["fallback"] else ""
    print("-" * 78)
    print(f"  推荐阈值: {recommended['threshold']:.2f}  "
          f"(P={recommended['precision']:.4f} R={recommended['recall']:.4f} "
          f"F1={recommended['f1']:.4f}){fb}")
    print(f"  经验值:   {empirical:.2f}  →  "
          f"{'一致，经验值合理' if abs(recommended['threshold'] - empirical) < 1e-9 else '不一致，见结论说明'}")
    print("=" * 78)


def print_report(l2_records: list[dict], l2_skipped: list[dict],
                 gate_records: list[dict], gate_errors: list[dict],
                 min_recall: float) -> None:
    """打印全部扫描报告（含分布统计与结论说明）"""
    print("\n" + "=" * 78)
    print("Threshold Scan Report (module-047 WP2)")
    print("=" * 78)

    # ── L2 触发阈值 ──
    if l2_records:
        rows = scan_l2(l2_records, L2_THRESHOLDS)
        rec = recommend_threshold(rows, min_recall)
        print("\n数据: golden_intent %d 条（skipped %d）× 真实 LLM 原始分类 + 真实确定性信号"
              % (len(l2_records), len(l2_skipped)))
        confs = [r["confidence"] for r in l2_records if r.get("confidence") is not None]
        print(f"LLM confidence 分布（全部样本）: {_percentiles(confs)}")
        print(f"触发窗口样本（raw_intent≠knowledge）: "
              f"{sum(1 for s in l2_trigger_samples(l2_records) if s['score'] is not None)}")
        positives = sum(row["tp"] + row["fn"] for row in rows)
        if positives == 0:
            # 退化解：本数据集不存在"应触发"样本（LLM 原始分类未把任何
            # knowledge 漏判为非 knowledge）→ 阈值无法校准，经验值既无法
            # 证伪也无法确认，如实报告而非推荐一个无意义的 t。
            _print_curve(
                "① L2 触发阈值扫描（positive=应触发: knowledge 被 LLM 漏判为非 knowledge）",
                rows, None, EMPIRICAL_L2, extra_key="final_accuracy",
            )
            print(f"  结论: 数据集中应触发样本数为 0（LLM 原始分类 100% 判对）→ "
                  f"该数据上无校准需求；经验值 {EMPIRICAL_L2:.2f} 既无法证伪也无法确认，"
                  f"保持现状（零回归）。")
        else:
            _print_curve(
                "① L2 触发阈值扫描（positive=应触发: knowledge 被 LLM 漏判为非 knowledge）",
                rows, rec, EMPIRICAL_L2, extra_key="final_accuracy",
            )
            if rec:
                print(f"  结论: 触发漏检召回（FN 占比）在 t={rec['threshold']:.2f} 时为 "
                      f"{1 - rec['recall']:.2%}; "
                      f"经验值 {EMPIRICAL_L2:.2f} 处于扫描区间内"
                      f"{'（与推荐一致）' if abs(rec['threshold'] - EMPIRICAL_L2) < 1e-9 else '（与推荐不一致，理由见 changelog）'}")
    else:
        print("\n① L2 触发阈值扫描: 无数据（LLM 收集失败/被跳过）。"
              "可用 python -m eval.benchmarks.threshold_scan --collect-l2 收集。")

    # ── 硬闸门阈值 ──
    if gate_records:
        rows = scan_gate(gate_records, GATE_THRESHOLDS)
        rec = recommend_threshold(rows, min_recall)
        suff = [r["top1_abs"] for r in gate_records if r["sufficient"]]
        insuff = [r["top1_abs"] for r in gate_records if not r["sufficient"]]
        print("\n数据: golden_sufficiency %d 条（errors %d）× 实测 bge-m3 余弦（question vs 注入文档 top-1）"
              % (len(gate_records), len(gate_errors)))
        print(f"top-1 余弦分布  充分类: {_percentiles(suff)}")
        print(f"top-1 余弦分布 不充分类: {_percentiles(insuff)}")
        print("假设: 闸门未触发（score≥t）时默认判充分——本扫描隔离闸门单独判别力，"
              "生产剩余样本走 LLM 层。")
        _print_curve(
            "② 充分性硬闸门阈值扫描（positive=不充分，漏判最致命）",
            rows, rec, EMPIRICAL_GATE,
        )
        if rec:
            print(f"  结论: 不充分漏判率在 t={rec['threshold']:.2f} 时为 "
                  f"{1 - rec['recall']:.2%}; "
                  f"经验值 {EMPIRICAL_GATE:.2f} 处于扫描区间内"
                  f"{'（与推荐一致）' if abs(rec['threshold'] - EMPIRICAL_GATE) < 1e-9 else '（与推荐不一致，理由见 changelog）'}")
    else:
        print("\n② 充分性硬闸门阈值扫描: 无数据（embedding 不可用，标注'待环境'）。")

    if l2_skipped:
        print(f"\nL2 收集 skipped {len(l2_skipped)} 条（如实记录，不影响其余样本）")
    if gate_errors:
        print(f"闸门收集 errors {len(gate_errors)} 条（如实记录）")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

async def main() -> None:
    """脚本入口"""
    parser = argparse.ArgumentParser(description="阈值扫描: L2 触发阈值 + 充分性硬闸门（数据驱动校准）")
    parser.add_argument("--collect-l2", action="store_true",
                        help="只收集 L2 数据（真实 LLM 分类 + 确定性信号）写缓存，不扫描")
    parser.add_argument("--skip-l2", action="store_true",
                        help="不跑 L2（零 LLM 调用），只跑闸门扫描")
    parser.add_argument("--no-cache", action="store_true",
                        help="不读不写缓存（纯内存）")
    parser.add_argument("--cache", default=str(_DEFAULT_CACHE),
                        help="缓存文件路径（默认 .ua/m047_threshold_cache.json）")
    parser.add_argument("--min-recall", type=float, default=0.0,
                        help="推荐阈值约束: 在 recall ≥ 该值的候选中选 F1 最大（默认 0.0）")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cached = load_cache(cache_path) if not args.no_cache else None
    # 单一 payload 贯穿全流程，避免分次保存互相覆盖（先 L2 后闸门）
    payload: dict = {
        "schema_version": _CACHE_SCHEMA,
        "collected_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    if cached:
        payload.update(cached)

    l2_records: list[dict] = []
    l2_skipped: list[dict] = []
    if args.collect_l2:
        l2_records, l2_skipped = await collect_l2_data()
        payload["l2"] = l2_records
        payload["l2_skipped"] = l2_skipped
        if not args.no_cache:
            save_cache(cache_path, payload)
        print(f"L2 收集完成: {len(l2_records)} 条记录, {len(l2_skipped)} 条 skipped")
        return

    if not args.skip_l2:
        l2_records = (cached or {}).get("l2", []) if cached else []
        l2_skipped = (cached or {}).get("l2_skipped", []) if cached else []
        if not l2_records:
            logger.info("无 L2 缓存数据，开始真实收集（LLM ~100 次调用）...")
            l2_records, l2_skipped = await collect_l2_data()
            if not args.no_cache:
                payload["l2"] = l2_records
                payload["l2_skipped"] = l2_skipped
                save_cache(cache_path, payload)

    gate_records = (cached or {}).get("gate", []) if cached else []
    gate_errors: list[dict] = []
    if not gate_records:
        logger.info("无闸门缓存数据，开始实测（本地 bge-m3，无 LLM）...")
        try:
            gate_records, gate_errors = await collect_gate_data()
            if gate_records and not args.no_cache:
                payload["gate"] = gate_records
                payload["gate_errors"] = gate_errors
                save_cache(cache_path, payload)
        except Exception as e:
            logger.error("闸门数据收集失败（标注待环境）: %s", e)
            gate_records = []

    print_report(l2_records, l2_skipped, gate_records, gate_errors, args.min_recall)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
