"""
Golden Query Rewrite 评测脚本 — 原始 query vs 改写 query 检索对比（module-049 WP3 / ADR-0009）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.golden.golden_query_rewrite              # 真实模式：LLM 改写 + DB 检索对比 + 落库
    python -m eval.golden.golden_query_rewrite --fixture    # fixture 模式：启发式分诊+改写，不依赖 LLM/DB（管线演示）
    python -m eval.golden.golden_query_rewrite --no-save    # 纯跑分，不写 eval_runs

指标定义（对齐 eval/golden_retrieval.py 的 compute_metrics）:
    Hit@k / Recall@k / MRR：对每道 golden 题分别跑"原始 query 检索"与
    "改写 query 检索"（k=5），输出整体 delta + 每题明细。正 delta 表示
    改写对召回有增益。每题附带保真余弦（改写 vs 原 query，嵌入可得时），
    供 rewrite_fidelity_threshold 阈值校准。

不充分题子集:
    golden 集本身无充分性标注，交叉引用 module-044 充分性标注集
    （SUFFICIENCY_DATASET，100 条含 sufficient 标签）——题目完全一致的
    题打上 sufficient 标签，单独聚合"不充分题"的改写增益（改写价值最大的
    场景：模糊 query 检索不到好结果）。

评测只度量不接线:
    本脚本不改变生产行为。生产改写链路在 rag/query_rewrite.py，本脚本
    直接调用其 llm_rewrite / fidelity_check 做对比检索。

降级策略:
    - LLM 改写失败/超时 → 该题记 skipped（rewrite_failed），不中断
    - 单题检索失败 → 跳过并记录错误，其余继续
    - 保真余弦不可得 → fidelity=None 如实记录（评测只度量，不拦截）
    - 数据库不可用 → 用 --fixture 模式演示管线，如实标注"待环境"
"""
import argparse
import asyncio
import logging
import sys

from agent.router import RouterAgent
from eval.golden.golden_retrieval import (
    compute_metrics, get_git_commit, load_rag_config, load_golden, save_eval_run,
)
from rag.retrieval import query_rewrite
from rag.retrieval.retriever import hybrid_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("golden_query_rewrite")

TOP_K = 5


def heuristic_triage(query: str) -> str:
    """fixture 启发式分诊：分词术语存在即"命中"（确定性，不依赖 DB）

    真实分诊（rag.retrieval.query_rewrite.triage）查 FTS 倒排需要数据库；fixture
    用 _kb_terms 非空近似"词表对得上"（module-020 语义的简化演示）。
    仅用于 fixture 模式演示管线，不代表真实分诊能力。
    """
    return "precise" if RouterAgent._kb_terms(query) else "vague"


def heuristic_rewrite(query: str) -> str:
    """fixture 启发式改写：核心术语空格拼接（确定性，不依赖 LLM）

    仅用于 fixture 模式演示管线，不代表 LLM 改写质量。
    """
    terms = RouterAgent._kb_terms(query)
    return " ".join(terms) if terms else query


def load_sufficiency_map() -> dict[str, bool]:
    """交叉引用 module-044 充分性标注集：题目完全一致的题 → 充分性标签

    Returns:
        {question: sufficient}；标注集加载失败/无匹配 → 空 dict（不阻塞）
    """
    try:
        from eval.golden.golden_sufficiency import load_sufficiency_dataset
        return {item["question"]: item["sufficient"]
                for item in load_sufficiency_dataset()}
    except Exception as e:
        logger.warning("充分性标注集交叉引用不可用: %s", e)
        return {}


async def _eval_question(item: dict, sufficiency_map: dict, top_k: int) -> tuple[dict, dict]:
    """单题评估：原始 query 检索 vs 改写 query 检索 + 指标对比

    Args:
        item: golden 题目（question / golden_docs / category）
        sufficiency_map: {question: sufficient} 交叉引用表
        top_k: 检索深度

    Returns:
        (evaluated, skipped) 二元组，二者恰有一个非空 dict
    """
    question = item["question"]
    category = item.get("category", "")
    golden_titles = item.get("golden_docs", [])
    if not golden_titles:
        return {}, {"question": question, "category": category, "reason": "no_gold_docs"}

    # ① 原始 query 检索
    try:
        orig_docs = await hybrid_retriever.retrieve(question, top_k=top_k)
    except Exception as e:
        return {}, {"question": question, "category": category, "reason": f"error: {e}"}
    orig_metrics = compute_metrics(
        [d.get("title", "") for d in orig_docs], golden_titles, top_k)

    # ② 改写 query 检索（LLM 改写失败 → skipped，不中断）
    rewritten = await query_rewrite.llm_rewrite(question)
    if rewritten is None:
        return {}, {"question": question, "category": category,
                    "reason": "rewrite_failed", "orig_metrics": orig_metrics}
    try:
        rw_docs = await hybrid_retriever.retrieve(rewritten, top_k=top_k)
    except Exception as e:
        return {}, {"question": question, "category": category, "reason": f"error: {e}"}
    rw_metrics = compute_metrics([d.get("title", "") for d in rw_docs], golden_titles, top_k)

    # 保真余弦（嵌入可得时；评测只度量，失败不拦截）
    fidelity = await query_rewrite.fidelity_check(question, rewritten)

    return {
        "question": question,
        "category": category,
        "golden_docs": golden_titles,
        "sufficient": sufficiency_map.get(question),  # None = 无交叉引用标注
        "rewritten_query": rewritten,
        "fidelity": round(fidelity, 4) if fidelity is not None else None,
        "orig": {"retrieved_titles": [d.get("title", "") for d in orig_docs], **orig_metrics},
        "rewritten": {"retrieved_titles": [d.get("title", "") for d in rw_docs], **rw_metrics},
        "delta": {k: round(rw_metrics[k] - orig_metrics[k], 4)
                  for k in ("hit_at_k", "recall_at_k", "mrr")},
    }, {}


def _aggregate(questions: list[dict], subset_filter=None) -> dict:
    """聚合整体/子集指标（按题平均；subset_filter 为 None 时用全部）"""
    subset = questions if subset_filter is None else [q for q in questions if subset_filter(q)]
    n = len(subset)
    if not n:
        return {"count": 0}
    agg = {"count": n}
    for side in ("orig", "rewritten"):
        agg[side] = {
            key: round(sum(q[side][key] for q in subset) / n, 4)
            for key in ("hit_at_k", "recall_at_k", "mrr")
        }
    agg["delta"] = {key: round(agg["rewritten"][key] - agg["orig"][key], 4)
                    for key in ("hit_at_k", "recall_at_k", "mrr")}
    agg["improved"] = sum(1 for q in subset if q["delta"]["mrr"] > 0)
    agg["worsened"] = sum(1 for q in subset if q["delta"]["mrr"] < 0)
    return agg


async def run_eval_real(top_k: int) -> tuple[dict, list[dict], list[dict]]:
    """真实模式：golden 112 题全量对比（LLM 改写 + DB 检索）"""
    golden = load_golden()
    sufficiency_map = load_sufficiency_map()
    per_question: list[dict] = []
    skipped: list[dict] = []

    for i, item in enumerate(golden):
        evaluated, skip = await _eval_question(item, sufficiency_map, top_k)
        if evaluated:
            per_question.append(evaluated)
            continue
        reason = skip["reason"]
        if reason == "no_gold_docs":
            logger.warning("[%d/%d] 跳过无 gold doc 题目: %s", i + 1, len(golden), item["question"][:40])
        elif reason == "rewrite_failed":
            logger.warning("[%d/%d] 改写失败跳过: %s", i + 1, len(golden), item["question"][:40])
        else:
            logger.error("[%d/%d] 题目检索失败: %s — %s", i + 1, len(golden), item["question"][:40], reason)
        skipped.append(skip)

    overall = _aggregate(per_question)
    insufficient = _aggregate(
        per_question, lambda q: q.get("sufficient") is False)
    per_category: dict[str, dict] = {}
    for q in per_question:
        cat = q["category"]
        bucket = per_category.setdefault(cat, [])
        bucket.append(q)
    per_category = {cat: _aggregate(items) for cat, items in per_category.items()}

    scores = {
        "dataset_size": len(golden),
        "evaluated": len(per_question),
        "skipped": len(skipped),
        "top_k": top_k,
        "fixture": False,
        "overall": overall,
        "insufficient_subset": insufficient,
        "per_category": per_category,
    }
    return scores, per_question, skipped


async def run_eval_fixture() -> tuple[dict, list[dict], list[dict]]:
    """fixture 模式：启发式分诊 + 启发式改写（不依赖 LLM/DB，管线演示）

    无 DB 检索 → 无 Recall 对比（如实标注待环境）；输出每题分诊/改写结果
    与整体统计，演示评测管线可运行。
    """
    golden = load_golden()
    per_question: list[dict] = []
    for item in golden:
        question = item["question"]
        mode = heuristic_triage(question)
        rewritten = heuristic_rewrite(question)
        per_question.append({
            "question": question,
            "category": item.get("category", ""),
            "triage": mode,
            "rewritten_query": rewritten,
            "rewrite_changed": rewritten != question,
        })
    # 注入人工泛词样例（不含专有术语）演示 vague 分诊分支：golden 112 题
    # 全部含技术术语 → heuristic_triage 全部返回 precise，vague 分支无法
    # 演示。以下人工样例不含专业术语，使 fixture 输出可同时看到 precise
    # 与 vague 两分支。不影响真实模式数据（run_eval_real 不执行此段）。
    _vague_demos = [
        {"question": "有没有什么好办法提高性能", "category": "fixture-demo-vague"},
        {"question": "系统老是崩怎么办", "category": "fixture-demo-vague"},
    ]
    for item in _vague_demos:
        question = item["question"]
        mode = heuristic_triage(question)
        rewritten = heuristic_rewrite(question)
        per_question.append({
            "question": question,
            "category": item["category"],
            "triage": mode,
            "rewritten_query": rewritten,
            "rewrite_changed": rewritten != question,
        })
    n_precise = sum(1 for q in per_question if q["triage"] == "precise")
    n_changed = sum(1 for q in per_question if q["rewrite_changed"])
    scores = {
        "dataset_size": len(golden),
        "evaluated": len(per_question),
        "skipped": 0,
        "top_k": TOP_K,
        "fixture": True,
        "precise_ratio": round(n_precise / len(per_question), 4),
        "rewrite_ratio": round(n_changed / len(per_question), 4),
    }
    return scores, per_question, []


async def record_eval_run(scores: dict, per_question: list[dict]) -> tuple[str, int]:
    """版本化落库：git_commit + rag_config 快照 + eval_type='query_rewrite'

    Returns:
        (commit, saved_id)；落库失败 saved_id=0（save_eval_run 内部已捕获并警告）
    """
    commit = get_git_commit()
    config_snapshot = await load_rag_config()
    saved_id = await save_eval_run(
        eval_type="query_rewrite",
        git_commit=commit,
        config_snapshot=config_snapshot,
        scores=scores,
        per_question=per_question,
    )
    return commit, saved_id


def print_report(scores: dict, per_question: list[dict], skipped: list[dict],
                 saved_id: int, commit: str) -> None:
    """打印评估报告：原始 vs 改写指标 + delta + 不充分题子集 + 每题明细"""
    print("\n" + "=" * 60)
    title = "Golden Query Rewrite Eval"
    if scores.get("fixture"):
        title += "  [fixture 模式：启发式分诊+改写，非真实指标；Recall 对比需真实模式（DB+LLM）]"
    print(title)
    print("=" * 60)
    print(f"Dataset: {scores['dataset_size']} questions | Evaluated: {scores['evaluated']} | Skipped: {scores['skipped']}")
    if scores.get("fixture"):
        print(f"fixture 启发式: precise(不分诊)={scores['precise_ratio']:.4f} | 改写变化率={scores['rewrite_ratio']:.4f}")
        print("（真实模式落库 eval_type='query_rewrite' 需 DB+LLM 环境，待环境补跑）")
        print("=" * 60)
        if saved_id:
            print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
        else:
            print("Not saved to eval_runs")
        print()
        return

    def _print_agg(label: str, agg: dict) -> None:
        print("-" * 60)
        print(f"{label} (n={agg['count']}):")
        print(f"  {'metric':<12}{'orig':>10}{'rewritten':>12}{'delta':>10}")
        for key in ("hit_at_k", "recall_at_k", "mrr"):
            print(f"  {key:<12}{agg['orig'][key]:>10.4f}"
                  f"{agg['rewritten'][key]:>12.4f}{agg['delta'][key]:>+10.4f}")
        print(f"  improved={agg['improved']}  worsened={agg['worsened']}")

    _print_agg("Overall", scores["overall"])
    ins = scores["insufficient_subset"]
    if ins.get("count"):
        _print_agg("Insufficient Subset (module-044 标注集交叉引用)", ins)
    else:
        print("-" * 60)
        print("Insufficient Subset: 无交叉引用标注题（golden 与充分性标注集题目未对齐），待标注对齐后补分析")

    print("-" * 60)
    print("Per-Category delta (mrr):")
    for cat, agg in scores.get("per_category", {}).items():
        if agg["count"]:
            print(f"  {cat:<18} n={agg['count']:<3} MRR: "
                  f"{agg['orig']['mrr']:.4f} -> {agg['rewritten']['mrr']:.4f} "
                  f"({agg['delta']['mrr']:+.4f})")
    if per_question:
        print("-" * 60)
        print(f"Per-Question (first 20, delta=rewritten-orig):")
        for q in per_question[:20]:
            suf = "" if q.get("sufficient") is None else ("不充分" if not q["sufficient"] else "充分")
            fid = f"{q['fidelity']:.2f}" if q.get("fidelity") is not None else "N/A"
            print(f"  dMRR={q['delta']['mrr']:+.2f} hit {q['orig']['hit_at_k']:.0f}->{q['rewritten']['hit_at_k']:.0f} "
                  f"fid={fid} {suf} | {q['question'][:36]}")
    if skipped:
        print("-" * 60)
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  [{s['reason'][:30]}] {s['question'][:50]}")
    print("=" * 60)
    if saved_id:
        print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to eval_runs")
    print()


async def main() -> None:
    """评测脚本入口"""
    parser = argparse.ArgumentParser(description="Golden query rewrite 评测：原始 vs 改写检索对比 + 版本化回归")
    parser.add_argument("--fixture", action="store_true",
                        help="fixture 模式：启发式分诊+改写（确定性，不依赖 LLM/DB），仅演示管线")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="检索深度 k（默认 5，0/负数自动回退 5）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    args = parser.parse_args()

    top_k = args.top_k if args.top_k and args.top_k > 0 else TOP_K
    load_golden()  # 先校验 golden 集（文件缺失/结构非法时立即报错退出）

    if args.fixture:
        scores, per_question, skipped = await run_eval_fixture()
    else:
        scores, per_question, skipped = await run_eval_real(top_k)

    saved_id = 0
    commit = ""
    if args.fixture:
        # fixture 模式不依赖 DB（无 LLM/无检索），强制跳过 eval_runs 落库，
        # 与 --no-save 语义等价——明确不依赖 DB 的定位
        print("[fixture] 强制跳过 eval_runs 落库（fixture 模式不依赖 DB）")
    elif not args.no_save:
        commit, saved_id = await record_eval_run(scores, per_question)
    print_report(scores, per_question, skipped, saved_id, commit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
