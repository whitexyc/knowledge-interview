"""
RRF k 扫描 + 图谱贡献归因脚本（module-057 WP-A4 / module-053 已知边界收敛）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景:
    module-053 固定 k=60（业界默认）启用三通道 RRF，未做 k 扫描；两通道 vs
    三通道 RRF 的图谱贡献量未量化（RRF 融合公式贡献 vs 图谱通道贡献混在一起）。
    本脚本补齐两个数字：① 本场景最优 k；② 图谱通道净增益（三通道 - 两通道）。

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.benchmark_rrf_k               # k=20-100 步长 10 + 拐点加密 + 两/三通道归因 + 落库
    python -m eval.benchmarks.benchmark_rrf_k --limit 10    # 冒烟（前 N 题）
    python -m eval.benchmarks.benchmark_rrf_k --no-save     # 不写 eval_runs
    python -m eval.benchmarks.benchmark_rrf_k --top-k 5     # 检索深度（默认 5）

效率设计:
    各通道候选（FTS/向量/图谱）与 k 无关——每题三通道只跑一次（图谱实体提取
    LLM 每题 1 次），逐 k 的 RRF 融合在纯 Python 内完成（公式确定性，无模型
    调用）。k 扫描 10 个值只付出 1 次检索成本。

指标:
    Hit@5 / Recall@5 / MRR（复用 eval.golden.golden_retrieval.compute_metrics +
    _aggregate，标题匹配同口径容忍层级前缀）。

归因口径:
    - 三通道 RRF = FTS + 向量 + 图谱 三路融合（rrf_constant_k 可变）
    - 两通道 RRF = FTS + 向量 两路融合（图谱通道置空，融合公式不变）
    - 图谱净增益 = 三通道 Hit@5 - 两通道 Hit@5（图谱通道的贡献，公式不变）
    - 对比基准：module-053/055 口径注明——本脚本全部为 RRF 公式口径；
      历史 hybrid（两通道 min-max 加权）数字 0.9565/0.9714 为另一融合公式，
      仅作参考不直接可比。

降级:
    - 单题通道失败 → 该路不参与融合（对齐 retriever 缺路降级语义）
    - 图谱实体提取 LLM 失败 → 图谱通道空（两/三通道退化一致）
    - 数据库不可用 → 标注"待环境"，不伪造数字
"""
import argparse
import asyncio
import logging
import sys

from src.database import async_session_factory
from rag.retrieval.retriever import hybrid_retriever
from eval.golden.golden_retrieval import (_aggregate, compute_metrics, get_git_commit,
                                   load_golden, load_rag_config, save_eval_run)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark_rrf_k")

TOP_K = 5
FETCH_K = TOP_K * 2  # 与 retriever.retrieve 同口径：top_k 的 2 倍候选

# k 扫描区间（module-057 WP-A4：20-100 步长 10；拐点处自动加密 ±5）
K_VALUES = list(range(20, 101, 10))


def fuse_rrf(channels: list[list[dict]], k: int,
             score_keys: list[str]) -> list[dict]:
    """纯函数 RRF 融合：score(d) = Σ 1/(k + rank_i(d))

    与 rag/retrieval/retriever.py::_fuse_rrf 公式一致（rank 为通道内 1-based
    排名，缺路不贡献）；独立实现便于逐 k 复用与单测（不依赖 settings）。

    Args:
        channels: 各通道候选列表（FTS/向量按 "score" 降序，图谱按
                  "hybrid_score" 降序）
        k: RRF 常数（业界默认 60）
        score_keys: 与 channels 对齐的通道排序键

    Returns:
        按 RRF 分降序的文档列表（含 doc 原始字段）
    """
    contrib: dict[int, float] = {}
    merged: dict[int, dict] = {}
    for results, score_key in zip(channels, score_keys):
        ordered = sorted(results, key=lambda d: d.get(score_key, 0.0), reverse=True)
        for rank, d in enumerate(ordered, start=1):
            doc_id = d["id"]
            contrib[doc_id] = contrib.get(doc_id, 0.0) + 1.0 / (k + rank)
            merged.setdefault(doc_id, d)
    return sorted(merged.values(), key=lambda d: contrib[d["id"]], reverse=True)


async def _collect_channels(query: str) -> tuple[list[dict], list[dict], list[dict]]:
    """单题三通道候选收集（每题一次；会话逻辑对齐 retriever._execute_fusion）

    Returns:
        (fts_results, vector_results, graph_results)；单路失败降级为空
        （对齐 retriever 缺路降级语义）
    """
    try:
        emb = await hybrid_retriever._embedding_service.embed_text(query)
    except Exception as e:
        logger.warning("查询向量化失败，向量路降级为空: %s", e)
        emb = None

    if emb is None:
        fts, vec = [], []
        try:
            async with async_session_factory() as fts_sess:
                fts = await hybrid_retriever._fts_search(query, FETCH_K, fts_sess)
        except Exception as e:
            logger.warning("全文检索失败，降级为空: %s", e)
    else:
        try:
            async with async_session_factory() as fts_sess, async_session_factory() as vec_sess:
                fts_task = hybrid_retriever._fts_search(query, FETCH_K, fts_sess)
                vec_task = hybrid_retriever._vector_search(emb, FETCH_K, vec_sess)
                fts, vec = await asyncio.gather(fts_task, vec_task, return_exceptions=True)
        except Exception as e:
            logger.warning("通道 session 创建失败，降级为空: %s", e)
            fts, vec = [], []
        if isinstance(fts, Exception):
            logger.warning("全文检索失败，该路不参与融合: %s", fts)
            fts = []
        if isinstance(vec, Exception):
            logger.warning("向量检索失败，该路不参与融合: %s", vec)
            vec = []

    graph = await hybrid_retriever._retrieve_graph_only(query, FETCH_K)
    return fts, vec, graph


def metrics_for_k(channels_list: list[tuple[list, list, list]],
                  golden: list[dict], k: int, with_graph: bool,
                  top_k: int) -> tuple[dict, list[dict]]:
    """k 与通道组合的聚合指标

    Args:
        channels_list: _collect_channels 逐题结果（与 golden 对齐）
        golden: golden 题（question/golden_docs/category）
        k: RRF 常数
        with_graph: True=三通道（含图谱）/ False=两通道（图谱置空）
        top_k: 检索深度

    Returns:
        (聚合 dict {overall, per_category}, 逐题明细)
    """
    questions = []
    for item, (fts, vec, graph) in zip(golden, channels_list):
        golden_titles = item.get("golden_docs", [])
        if not golden_titles:
            continue
        channels = [fts, vec]
        score_keys = ["score", "score"]
        if with_graph:
            channels.append(graph)
            score_keys.append("hybrid_score")
        fused = fuse_rrf(channels, k, score_keys)[:top_k]
        titles = [d.get("title", "") for d in fused]
        questions.append({
            "question": item["question"],
            "category": item.get("category", ""),
            **compute_metrics(titles, golden_titles, top_k),
        })
    return _aggregate(questions), questions


async def run_benchmark(limit: int | None = None, save: bool = True,
                        top_k: int = TOP_K) -> None:
    """k 扫描 + 两/三通道归因 + 最优 k 结论"""
    golden = load_golden()
    if limit:
        golden = golden[:limit]

    print("== 收集三通道候选（每题一次，含图谱实体提取 LLM；之后逐 k 纯 CPU 融合）==")
    channels_list: list[tuple] = []
    for i, item in enumerate(golden):
        channels_list.append(await _collect_channels(item["question"]))
        if (i + 1) % 20 == 0 or i + 1 == len(golden):
            logger.info("通道收集进度: %d/%d", i + 1, len(golden))
    n_graph_empty = sum(1 for _, _, g in channels_list if not g)
    print(f"   图谱通道空候选: {n_graph_empty}/{len(golden)} 题"
          + ("（实体提取失败或图无覆盖，如实标注）" if n_graph_empty else ""))

    # ── k 扫描主表（20-100 步长 10）──
    print("\n" + "=" * 78)
    print(f"RRF k 扫描（top_k={top_k}，golden {len(golden)} 题）")
    print("=" * 78)
    print(f"  {'k':>4} | {'三通道 Hit@5':>12} {'Recall@5':>10} {'MRR':>8} | "
          f"{'两通道 Hit@5':>12} {'图谱增益':>8} | {'注':>12}")
    print("-" * 78)
    rows: dict[int, dict] = {}
    all_k = list(K_VALUES)
    k_vals = all_k[:]
    while True:
        # 逐 k 计算（两/三通道），纯 CPU
        for k in k_vals:
            if k in rows:
                continue
            agg3, _ = metrics_for_k(channels_list, golden, k, with_graph=True, top_k=top_k)
            agg2, _ = metrics_for_k(channels_list, golden, k, with_graph=False, top_k=top_k)
            rows[k] = {
                "k": k,
                "three_hit": agg3["overall"]["hit_at_k"],
                "three_recall": agg3["overall"]["recall_at_k"],
                "three_mrr": agg3["overall"]["mrr"],
                "two_hit": agg2["overall"]["hit_at_k"],
                "graph_gain": round(agg3["overall"]["hit_at_k"]
                                    - agg2["overall"]["hit_at_k"], 4),
            }
        # 拐点加密：最优 k 的 ±5（若未扫过；保持 20-100 声明区间内）
        best_k = max(rows, key=lambda kk: rows[kk]["three_hit"])
        denser = [k for k in (best_k - 5, best_k + 5)
                  if 20 <= k <= 100 and k not in rows]
        if not denser:
            break
        k_vals = denser

    for k in sorted(rows):
        r = rows[k]
        note = "（最优）" if r["three_hit"] == max(
            v["three_hit"] for v in rows.values()) else ""
        print(f"  {r['k']:>4} | {r['three_hit']:>12.4f} {r['three_recall']:>10.4f} "
              f"{r['three_mrr']:>8.4f} | {r['two_hit']:>12.4f} "
              f"{r['graph_gain']:>+8.4f} | {note:>12}")

    best = max(rows.values(), key=lambda r: r["three_hit"])
    k60 = rows.get(60, {})
    print("-" * 78)
    print(f"  最优 k = {best['k']}（三通道 Hit@5={best['three_hit']:.4f}，"
          f"图谱增益 {best['graph_gain']:+.4f}）")
    if k60:
        k60_delta = best["three_hit"] - k60["three_hit"]
        cmp_note = "持平" if abs(k60_delta) < 1e-9 else f"差 {k60_delta:+.4f}"
        print(f"  k=60（module-053 业界默认）对比: Hit@5={k60['three_hit']:.4f} → {cmp_note}")
    print("  口径声明: 全部为 RRF 公式口径（三通道=含图谱 / 两通道=图谱置空）；"
          "历史 hybrid 两通道 min-max 加权数字（0.9565/0.9714）为另一融合公式，"
          "仅参考不直接可比。")
    print("=" * 78)

    if save:
        await _save_eval_run(rows, best, channels_list, golden, top_k)


async def _save_eval_run(rows: dict, best: dict, channels_list: list,
                         golden: list, top_k: int) -> None:
    """落库 eval_runs（eval_type='rrf_k_scan'）；失败仅警告"""
    try:
        config_snapshot = await load_rag_config()
        saved_id = await save_eval_run(
            eval_type="rrf_k_scan", git_commit=get_git_commit(),
            config_snapshot=config_snapshot,
            scores={
                "top_k": top_k,
                "dataset_size": len(golden),
                "k_values": sorted(rows),
                "curve": rows,
                "best_k": best["k"],
                "best_three_hit": best["three_hit"],
                "best_graph_gain": best["graph_gain"],
                "k60_three_hit": rows.get(60, {}).get("three_hit"),
                "fusion_formula": "rrf",
                "caliber_note": "三通道=含图谱/两通道=图谱置空；"
                                "历史 hybrid 数字为另一融合公式仅参考",
            },
            per_question=[],
        )
        print(f"已落库 eval_runs (id={saved_id}, eval_type='rrf_k_scan')")
    except Exception as e:
        print(f"eval_runs 落库失败（不中断）: {e}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="RRF k 扫描 + 图谱贡献归因（module-057 WP-A4）")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 题（冒烟）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="检索深度（默认 5，0/负数自动回退 5）")
    args = parser.parse_args()
    top_k = args.top_k if args.top_k and args.top_k > 0 else TOP_K
    await run_benchmark(limit=args.limit, save=not args.no_save, top_k=top_k)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
