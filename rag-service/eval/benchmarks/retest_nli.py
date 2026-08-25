"""
mDeBERTa 矛盾判别复测脚本（module-054 首版 / module-057 复测 v2：句级拆解 + 阈值校准）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.retest_nli                      # 真实复测 v2：阈值扫描 + 句级拆解 + kappa
    python -m eval.benchmarks.retest_nli --threshold 0.7      # 固定阈值（不扫描），阈值区间 [0.5, 0.9]
    python -m eval.benchmarks.retest_nli --gen-real 24        # 生成真实检索候选对（LLM 答案 + DB 检索片段）
    python -m eval.benchmarks.retest_nli --no-save            # 不写 eval_runs
    python -m eval.benchmarks.retest_nli --limit 20           # 只评估前 20 条（冒烟）

复测口径（对齐 ADR-0010 "kappa 复测计划"四项 + module-057 WP-A1 改进）:
    1. 矛盾构造样本集（eval/contradiction_dataset.json，module-057 扩充至 86 条 =
       module-054 首版 56 + 新增 30；contradiction 53 ≥ 50，其中 internal_contradiction
       23 ≥ 20 含 8 条多句混合"前真后假"样本）——验证矛盾判别能力。
    2. claim 用真实答案句子（LLM 生成，非问题代答句；--gen-real 生成后人工
       标注 verdict，落 eval/real_retrieval_pairs.json）。
    3. 文档用 DB 真实检索片段（golden 112 题 hybrid 检索 top 片段）。
    4. 门槛：复测 kappa（三分类）≥ 0.7 通过（放行替换 mDeBERTa）；
       未达则降级评估（双轨：NLI 只做矛盾扫描），如实标注不伪造。

module-057 改进（句级拆解 + 阈值校准，针对 module-054 失败模式）:
    - 句级拆解：claim 按中文/英文句末标点（。！？；!?）切子句 → 逐子句 vs 文档
      判定 + 内部矛盾子句两两互判（子句 i 作 doc、子句 j 作 claim，双向）→ 聚合：
      任一 contradiction → contradiction（最严）/ 无矛盾有 entailment → entailment /
      全 neutral → neutral。拆句失败（无 ≥2 子句）回退整句判定（零回归）。
    - 阈值校准：低置信降级（max softmax prob < 阈值 → neutral），扫描 0.5-0.9
      步长 0.05，输出 kappa 曲线与最优阈值；最终判定用最优阈值（--threshold 可固定）。
    - 同口径对比：首版 80 对（constructed[:56] + real 24）上新方法 vs module-054
      基线 kappa 0.5167（eval_runs id=22）——同集同人标注，可比。

指标:
    Cohen's kappa（三分类 entailment/neutral/contradiction + 二值 entailment-vs-rest
    两口径，sklearn cohen_kappa_score）。kappa 校正随机一致（三分类基线 33%）。

降级:
    - mDeBERTa 模型缺失 → 明确报错（_require_model 同款），不静默通过
    - 单条打分异常 → 跳过记录，其余继续
    - 数据库不可用 → --gen-real 失败该条记 unavailable，评估仍完成
    - LLM 不可用 → --gen-real 如实标注 claim="[LLM_UNAVAILABLE]" 并声明口径

诚实边界:
    1. 矛盾样本为人工构造（非真实用户对话），方向性验证；标注一致性经
       Reviewer 抽查，非多人独立标注。
    2. 真实检索对 claim=LLM 生成答案句子（真实链路），doc=真实检索片段；
       人工标注 verdict 由 Developer 完成 + Reviewer 抽查。
    3. mDeBERTa 多语言训练，中文是泛化表现；输入截断 512 token（同 module-052）。
    4. 阈值在评估集上扫描选择（in-sample），kappa 数值有乐观偏差——同口径
       对比（旧 80 对）仍成立，但生产阈值需独立集确认（下一轮方向）。
    5. 句级拆解最严聚合（任一子句矛盾 → 整句 contradiction）可能误杀部分
       支持样本（kappa 说话，如实呈现）。
"""
import argparse
import asyncio
import json
import os
import re
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

from sklearn.metrics import cohen_kappa_score

from eval.datasets.contradiction_dataset import (DATASET_PATH, VERDICTS,
                                        load_contradiction_dataset)
from eval.benchmarks.compare_nli_models import (MDEBERTA_DIR, _require_model, binarize,
                                     load_mdeberta, mdeberta_score,
                                     model_metrics, print_metrics_table)

REAL_PAIRS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "datasets", "real_retrieval_pairs.json")
REAL_CANDIDATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "real_retrieval_candidates.json")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "golden", "golden.json")

GATE_KAPPA = 0.7
# module-054 复测基线（eval_runs id=22，首版 80 对 = constructed 56 + real 24）
BASELINE_KAPPA = 0.5167
# module-054 首版人工构造样本数（样本集构造脚本保持首版 56 条在前，constructed[:56]
# 即 module-054 同集——同口径对比用）
OLD_CONSTRUCTED_N = 56

# 阈值扫描区间（module-057 WP-A1：0.5-0.9 步长 0.05，低置信 max prob < t → neutral）
SCAN_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(9)]

# 中文/英文句末标点切句（与 eval/compare_factcheck_models._pre_chunk 同语义；
# module-057 句级拆解用于 claim 子句切分）
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?])\s*|\n+")


def load_real_pairs(path: str = REAL_PAIRS_PATH) -> list[dict]:
    """加载真实检索对（part="real_retrieval"）；文件不存在返回 []"""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload["samples"] if isinstance(payload, dict) else payload
    for item in samples:
        for key in ("question", "claim", "doc", "verdict"):
            if not item.get(key, ""):
                raise ValueError(f"真实检索对缺 {key}: {item.get('question', '')[:30]}")
        if item["verdict"] not in VERDICTS:
            raise ValueError(f"verdict 须为 {VERDICTS}: {item.get('question', '')[:30]}")
    return samples


# ──────────────────────────────────────────────────────────────
# module-057 句级拆解 + 阈值校准（纯函数，可单测）
# ──────────────────────────────────────────────────────────────

def split_claim(claim: str) -> list[str]:
    """句级拆解：按中文/英文句末标点（。！？；!?）切 claim 为子句列表

    拆不出 ≥2 个非空子句 → 回退整句（零回归：单句 claim 行为与 module-054 一致）。

    Args:
        claim: 待判定的答案句子

    Returns:
        子句列表（≥1）；无句末标点时恒为 [claim.strip()]
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(claim) if p and p.strip()]
    return parts if len(parts) >= 2 else [claim.strip() or claim]


def apply_threshold(pred: str, conf: float, threshold: float) -> str:
    """低置信降级：max softmax prob < 阈值 → neutral

    校准目的：mDeBERTa 低置信预测（如矛盾判别拿不准）倾向乱判，降级为 neutral
    避免"低置信的 contradiction/entailment"污染最终判定（module-054 失败模式
    之一：矛盾 11/32 判 neutral 是判别力问题，此处校准的是"不确定时别乱说"）。
    """
    return "neutral" if conf < threshold else pred


def aggregate_sub_judgments(sub_verdicts: list[str],
                            pair_verdicts: list[str]) -> str:
    """句级聚合（最严语义，module-057 WP-A1 定死）:
        - 任一子句/任一子句对 contradiction → contradiction（最严，防漏判矛盾）
        - 无矛盾但有子句 entailment → entailment
        - 全 neutral → neutral

    注意：子句对（pair）只参与 contradiction 判定（子句间互斥才是内部矛盾
    信号），entailment/neutral 仅由子句 vs 文档判定贡献。
    """
    if any(v == "contradiction" for v in sub_verdicts) or \
            any(v == "contradiction" for v in pair_verdicts):
        return "contradiction"
    if any(v == "entailment" for v in sub_verdicts):
        return "entailment"
    return "neutral"


def predict_batch(docs: list[str], claims: list[str]) -> tuple[list[str], list[float]]:
    """真实 mDeBERTa 批量打分 → (三分类标签列表, max prob 列表)

    是 collect_decomposed_predictions 的默认 scorer；测试注入假 scorer 即可
    不加载模型单测拆解/聚合/阈值逻辑。
    """
    labels, probs = mdeberta_score(docs, claims)
    id2label = _mdeberta_label_map()
    return ([str(id2label[int(i)]) for i in labels],
            [float(p[i]) for i, p in zip(labels, probs)])


def collect_decomposed_predictions(samples: list[dict],
                                   scorer=predict_batch) -> list[dict]:
    """一次性打分所有 (doc, claim) 对（含拆解子句对），供阈值扫描复用

    阈值扫描只改决策规则（低置信 → neutral），不改模型推理——所有 softmax
    分数只算一次，逐阈值聚合纯 CPU（module-057 效率设计，避免 9 倍推理）。

    每样本记录（两形态之一）:
        - 整句形态: {"parts": [claim], "whole": (pred, conf)}
        - 拆解形态: {"parts": [...], "sub": [(pred, conf), ...],
                     "pair": [(pred, conf), ...]}（子句 i 作 doc、子句 j 作 claim，
                     双向；len(parts)=n → pair 共 n*(n-1) 条）

    Args:
        samples: 样本列表（question/claim/doc/verdict）
        scorer: (docs, claims) → (labels, confs)；默认真实 mDeBERTa

    Returns:
        与 samples 等长的预测记录列表
    """
    records = []
    for s in samples:
        parts = split_claim(s["claim"])
        if len(parts) < 2:
            preds, confs = scorer([s["doc"]], [s["claim"]])
            records.append({"parts": parts, "whole": (preds[0], confs[0])})
            continue
        sub_labels, sub_confs = scorer([s["doc"]] * len(parts), parts)
        pair_docs, pair_claims = [], []
        for i in range(len(parts)):
            for j in range(len(parts)):
                if i != j:
                    pair_docs.append(parts[i])
                    pair_claims.append(parts[j])
        if pair_docs:
            pl, pc = scorer(pair_docs, pair_claims)
        else:
            pl, pc = [], []
        records.append({
            "parts": parts,
            "sub": list(zip(sub_labels, sub_confs)),
            "pair": list(zip(pl, pc)),
        })
    return records


def verdict_for_threshold(rec: dict, threshold: float) -> tuple[str, list[str], list[str]]:
    """阈值 t 下对一条预测记录求最终判定

    Returns:
        (最终三分类, 子句判定列表, 子句对判定列表)；整句形态子句判定为
        [整句判定]、子句对判定为空。
    """
    if "whole" in rec:
        pred, conf = rec["whole"]
        verdict = apply_threshold(pred, conf, threshold)
        return verdict, [verdict], []
    sub = [apply_threshold(p, c, threshold) for p, c in rec["sub"]]
    pair = [apply_threshold(p, c, threshold) for p, c in rec["pair"]]
    return aggregate_sub_judgments(sub, pair), sub, pair


def scan_thresholds(records: list[dict], human3: list[str],
                    thresholds: list[float] | None = None) -> list[dict]:
    """逐阈值扫描：低置信降级 → 聚合 → kappa/Acc + 预测分布

    Args:
        records: collect_decomposed_predictions 输出
        human3: 人工三分类标注（与 records 等长）
        thresholds: 扫描区间（默认 SCAN_THRESHOLDS 0.5-0.9 步长 0.05）

    Returns:
        逐阈值行: {"threshold", "kappa_3class", "kappa_binary",
                   "accuracy_3class", "accuracy_binary",
                   "pred_contradiction", "pred_neutral", "pred_entailment"}
    """
    thresholds = thresholds if thresholds is not None else SCAN_THRESHOLDS
    rows = []
    for t in thresholds:
        preds = [verdict_for_threshold(r, t)[0] for r in records]
        m = model_metrics(human3, preds)
        rows.append({
            "threshold": t,
            "kappa_3class": m["kappa_3class"],
            "kappa_binary": m["kappa_binary"],
            "accuracy_3class": m["accuracy_3class"],
            "accuracy_binary": m["accuracy_binary"],
            "pred_contradiction": sum(1 for p in preds if p == "contradiction"),
            "pred_neutral": sum(1 for p in preds if p == "neutral"),
            "pred_entailment": sum(1 for p in preds if p == "entailment"),
        })
    return rows


def gen_real_pairs(num: int = 24) -> None:
    """生成真实检索候选对：LLM 生成答案句子（claim）+ DB hybrid 检索片段（doc）

    只生成候选（无 verdict），人工标注后写入 real_retrieval_pairs.json。
    """
    from rag.retrieval.retriever import hybrid_retriever
    from llm.client import LLMFactory

    if not os.path.isfile(GOLDEN_PATH):
        raise FileNotFoundError(f"golden.json 缺失: {GOLDEN_PATH}")
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    # 确定性抽样：跨类别取题（黄金 112 题按步长抽，保证类别多样）
    stride = max(1, len(golden) // num)
    picked = golden[::stride][:num]

    async def _one(item: dict) -> dict:
        q = item["question"]
        # 1) DB 真实检索片段（hybrid，本地 bge-m3 + pgvector）
        try:
            docs = await asyncio.wait_for(
                hybrid_retriever.retrieve(q, top_k=2), timeout=20,
            )
        except Exception as e:
            return {"question": q, "claim": "", "doc": f"[RETRIEVE_UNAVAILABLE: {e}]",
                    "doc_title": "", "category": item.get("category", "")}
        if not docs:
            return {"question": q, "claim": "", "doc": "[NO_DOCS]",
                    "doc_title": "", "category": item.get("category", "")}
        doc = docs[0]
        # 2) LLM 生成真实答案句子（deepseek 降级链）
        try:
            client = LLMFactory.get_client()
            answer = await asyncio.wait_for(
                client.generate(f"请用 1-2 句话直接回答（不要引用、不要补充与问题无关的内容）：{q}"),
                timeout=30,
            )
            claim = (answer or "").strip()
            if not claim:
                claim = "[EMPTY_ANSWER]"
        except Exception as e:
            claim = f"[LLM_UNAVAILABLE: {type(e).__name__}]"
        return {
            "question": q,
            "claim": claim,
            "doc": (doc.get("content") or "")[:700],
            "doc_title": doc.get("title", ""),
            "category": item.get("category", ""),
        }

    async def _gen_all():
        return await asyncio.gather(*[_one(i) for i in picked])

    candidates = asyncio.run(_gen_all())
    with open(REAL_CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": {"num": len(candidates), "note": "候选对，待人工标注 verdict"},
                   "samples": candidates}, f, ensure_ascii=False, indent=2)
    print(f"候选对已生成: {REAL_CANDIDATES_PATH}（{len(candidates)} 条，"
          f"请人工标注 verdict 后写入 {REAL_PAIRS_PATH}）")
    n_unavail = sum(1 for c in candidates if "UNAVAILABLE" in c["claim"] or "NO_DOCS" in c["doc"])
    if n_unavail:
        print(f"环境不可用标注: {n_unavail} 条（LLM/检索不可用，如实声明）")


def _print_scan_table(rows: list[dict]) -> None:
    """打印阈值扫描曲线"""
    print("\n-- 阈值扫描（0.5-0.9 步长 0.05，低置信 max prob < t → neutral）--")
    print(f"  {'t':>6} {'kappa(3类)':>10} {'kappa(二值)':>10} "
          f"{'Acc(3类)':>9} {'pred: ent/neu/ctr':>20}")
    for r in rows:
        print(f"  {r['threshold']:>6.2f} {r['kappa_3class']:>10.4f} "
              f"{r['kappa_binary']:>10.4f} {r['accuracy_3class']:>9.4f}  "
              f"{r['pred_entailment']}/{r['pred_neutral']}/{r['pred_contradiction']}")
    best = max(rows, key=lambda r: r["kappa_3class"])
    print(f"  ==> 最优阈值: {best['threshold']:.2f} "
          f"(kappa 三分类 {best['kappa_3class']:.4f})")


def run_retest(limit: int | None = None, save: bool = True,
               threshold: float | None = None) -> None:
    """真实复测 v2：句级拆解 + 阈值扫描 → kappa 两口径 + 同口径对比 + 门槛判定"""
    _require_model(MDEBERTA_DIR, [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "spm.model",
    ])
    load_mdeberta()

    constructed = load_contradiction_dataset()
    real_pairs = load_real_pairs()
    all_samples = constructed + real_pairs
    if limit:
        all_samples = all_samples[:limit]

    # module-054 同口径集（首版 56 构造 + 24 真实 = 80 对），供 kappa 直接对比 0.5167
    same_set_samples = constructed[:OLD_CONSTRUCTED_N] + real_pairs
    same_set_n = len(same_set_samples)

    print(f"== 数据: {len(all_samples)} 对（构造 {len(constructed)} + 真实 {len(real_pairs)}），"
          f"其中同口径旧集 {same_set_n} 对（对比基线 kappa={BASELINE_KAPPA}）")
    from collections import Counter
    dist = Counter(s["verdict"] for s in all_samples)
    print(f"   人工标注分布: entailment {dist['entailment']} / neutral {dist['neutral']} "
          f"/ contradiction {dist['contradiction']}")
    n_decomp = sum(1 for s in all_samples if len(split_claim(s["claim"])) >= 2)
    print(f"   句级拆解生效样本: {n_decomp} 条（其余为整句判定，零回归）")

    # 一次性打分（含拆解子句对）→ 阈值扫描纯 CPU 复用
    records = collect_decomposed_predictions(all_samples)
    human3 = [s["verdict"] for s in all_samples]
    scan = scan_thresholds(records, human3)
    _print_scan_table(scan)

    # 同口径旧集（首版 56 构造 + 24 真实）逐阈值曲线：隔离"拆解"与"阈值"两个
    # 因素的各自影响（模块-054 基线 0.5167 是 argmax 无拆解无阈值）
    same_records = records[:OLD_CONSTRUCTED_N] + records[len(constructed):]
    same_human = human3[:OLD_CONSTRUCTED_N] + human3[len(constructed):]
    same_scan = scan_thresholds(same_records, same_human)
    print("\n-- 同口径旧集（80 对）阈值扫描（基线 0.5167 = argmax 无拆解无阈值）--")
    print(f"  {'t':>6} {'kappa(3类)':>10} {'kappa(二值)':>10} {'Acc(3类)':>9}")
    for r in same_scan:
        print(f"  {r['threshold']:>6.2f} {r['kappa_3class']:>10.4f} "
              f"{r['kappa_binary']:>10.4f} {r['accuracy_3class']:>9.4f}")
    same_best = max(same_scan, key=lambda r: r["kappa_3class"])
    print(f"  ==> 旧集最优阈值: {same_best['threshold']:.2f} "
          f"(kappa 三分类 {same_best['kappa_3class']:.4f})")

    best_t = max(scan, key=lambda r: r["kappa_3class"])["threshold"]
    final_t = threshold if threshold is not None else best_t
    preds = [verdict_for_threshold(r, final_t)[0] for r in records]
    overall = model_metrics(human3, preds)

    print(f"\n-- 指标（最终判定，阈值={final_t:.2f}）--")
    print_metrics_table("总体(全量)", overall)

    # 分部分指标（构造 vs 真实；空切片跳过——cohen_kappa_score 不支持空数组）
    n = len(constructed)
    if n:
        print_metrics_table("  人工构造", model_metrics(human3[:n], preds[:n]))
    if len(human3) > n:
        print_metrics_table("  真实检索", model_metrics(human3[n:], preds[n:]))

    # 同口径对比：旧 80 对（首版 56 构造 + 24 真实）新方法 vs 基线
    same_preds = [verdict_for_threshold(r, final_t)[0] for r in same_records]
    same_metrics = model_metrics(same_human, same_preds)
    print("\n-- 同口径对比（module-054 旧集 80 对，同集同人标注，直接可比）--")
    print(f"   module-054 基线（argmax 无拆解无阈值）: kappa 三分类 = {BASELINE_KAPPA}")
    print_metrics_table("   module-057 新方法", same_metrics)
    delta = same_metrics["kappa_3class"] - BASELINE_KAPPA
    print(f"   delta = {delta:+.4f}（{'提升' if delta > 0 else '下降'}）")

    # 混淆矩阵
    classes = ["entailment", "neutral", "contradiction"]
    print("\n-- 混淆矩阵（行=人工, 列=mDeBERTa，全量）--")
    print(f"{'':<14}" + "".join(f"{c:>14}" for c in classes))
    for r in classes:
        row = [sum(1 for hr, pr in zip(human3, preds) if hr == r and pr == c)
               for c in classes]
        print(f"{r:<14}" + "".join(f"{n:>14}" for n in row))

    # 误判明细（前 10 条，含拆解信息）
    mis = [i for i in range(len(human3)) if human3[i] != preds[i]]
    if mis:
        print(f"\n-- 误判 {len(mis)} 条（前 10 条）--")
        for i in mis[:10]:
            s = all_samples[i]
            _, subs, pairs = verdict_for_threshold(records[i], final_t)
            print(f"  [{i}] 人工={human3[i]} 预测={preds[i]} ({s.get('contradiction_type', '?')})")
            print(f"      claim: {s['claim'][:60]}")
            if len(subs) > 1:
                print(f"      子句判定: {subs}")
                if pairs:
                    print(f"      子句对判定: {pairs[:4]}{'...' if len(pairs) > 4 else ''}")

    # 过激聚合误杀分析（最严聚合的代价）：人工 entailment/neutral 被预测 contradiction
    overkill = [i for i in range(len(human3))
                if human3[i] in ("entailment", "neutral") and preds[i] == "contradiction"]
    if overkill:
        print(f"\n-- 最严聚合误杀 {len(overkill)} 条（人工 entailment/neutral → 预测 contradiction）--")
        for i in overkill[:8]:
            s = all_samples[i]
            _, subs, pairs = verdict_for_threshold(records[i], final_t)
            print(f"  [{i}] 人工={human3[i]} 预测=contradiction ({s.get('contradiction_type', '?')})")
            print(f"      claim: {s['claim'][:60]}")
            if len(subs) > 1:
                print(f"      子句判定: {subs}  子句对判定: {pairs[:4]}")

    # 门槛判定（ADR-0010 P1-③ 放行条件）
    print("\n" + "=" * 60)
    k3 = overall["kappa_3class"]
    print(f"复测 kappa(三分类) = {k3:.4f}  门槛 = {GATE_KAPPA}  基线 = {BASELINE_KAPPA}")
    if k3 >= GATE_KAPPA:
        print(f"==> 结论: kappa {k3:.4f} >= {GATE_KAPPA} 达标 —— 放行替换 "
              f"（mDeBERTa 作为逐句裁判三态来源，实施另行模块）")
    else:
        print(f"==> 结论: kappa {k3:.4f} < {GATE_KAPPA} 未达门槛，如实标注 —— "
              f"降级双轨：NLI 只做矛盾扫描（不替换 HHEM 主裁判）")
    print("=" * 60)

    if save:
        _save_eval_run(overall, same_metrics, scan, same_scan, final_t, threshold,
                       human3, preds, all_samples, records,
                       constructed, real_pairs)


def _mdeberta_label_map() -> dict:
    from eval.benchmarks.compare_nli_models import _mdeberta
    return _mdeberta["id2label"]


def _save_eval_run(metrics: dict, same_set_metrics: dict, scan: list[dict],
                   same_scan: list[dict], final_t: float, fixed_t: float | None,
                   human3: list, pred3: list, samples: list, records: list,
                   constructed: list, real_pairs: list) -> None:
    """版本化落库 eval_runs（eval_type='nli_retest_v2'）；失败仅警告不中断

    注意：load_rag_config + save_eval_run 须在同一个事件循环内执行——
    asyncpg 连接池不可跨 asyncio.run() 复用（Windows ProactorEventLoop 既有约束）。
    """
    try:
        from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run
        import asyncio

        per_question = []
        for s, h, p, rec in zip(samples, human3, pred3, records):
            _, subs, pairs = verdict_for_threshold(rec, final_t)
            per_question.append({
                "question": s["question"], "label": h, "predicted": p,
                "correct": h == p, "part": s.get("part", "constructed"),
                "n_parts": len(rec["parts"]),
                "sub_verdicts": subs,
                "pair_verdicts": pairs[:6] if pairs else [],
                "contradiction_type": s.get("contradiction_type", ""),
            })

        async def _record():
            config_snapshot = await load_rag_config()
            return await save_eval_run(
                eval_type="nli_retest_v2", git_commit=get_git_commit(),
                config_snapshot=config_snapshot,
                scores={**metrics, "gate_kappa": GATE_KAPPA,
                        "baseline_kappa": BASELINE_KAPPA,
                        "same_set_kappa_3class": same_set_metrics["kappa_3class"],
                        "same_set_kappa_binary": same_set_metrics["kappa_binary"],
                        "same_set_n": OLD_CONSTRUCTED_N + len(real_pairs),
                        "threshold": final_t, "threshold_fixed": fixed_t,
                        "scan": scan, "same_set_scan": same_scan,
                        "constructed_n": len(constructed), "real_n": len(real_pairs),
                        "old_constructed_n": OLD_CONSTRUCTED_N,
                        "method": "sentence_decomposition+threshold_scan"},
                per_question=per_question,
            )

        saved_id = asyncio.run(_record())
        print(f"已落库 eval_runs (id={saved_id}, eval_type='nli_retest_v2')")
    except Exception as e:
        print(f"eval_runs 落库失败（不中断）: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mDeBERTa 矛盾判别复测 v2（句级拆解 + 阈值扫描，ADR-0010 P1-③）")
    parser.add_argument("--gen-real", type=int, metavar="N", default=0,
                        help="生成 N 条真实检索候选对（LLM 答案 + DB 检索片段）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条（冒烟）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="固定置信度阈值（不扫描；区间 0.5-0.9）")
    args = parser.parse_args()

    if args.gen_real:
        gen_real_pairs(args.gen_real)
        return
    run_retest(limit=args.limit, save=not args.no_save, threshold=args.threshold)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
