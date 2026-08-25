"""
Golden 检索集评估脚本 — Hit@k / Recall@k / MRR + 单通道消融 + 版本化回归
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.golden.golden_retrieval                    # 组合模式（默认 hybrid）
    python -m eval.golden.golden_retrieval --mode fts_only    # 消融：仅全文检索
    python -m eval.golden.golden_retrieval --mode vector_only # 消融：仅向量检索
    python -m eval.golden.golden_retrieval --mode graph_only  # 消融：仅图检索
    python -m eval.golden.golden_retrieval --top-k 10
    python -m eval.golden.golden_retrieval --no-save          # 不写 eval_runs（纯跑分）
    python -m eval.golden.golden_retrieval --compare          # 对比最近两次运行的 delta
    python -m eval.golden.golden_retrieval --ablate           # 消融对比：graph_only vs hybrid 一键跑两边 + side-by-side delta（图谱贡献量）

三通道融合（module-053，环境变量切换，默认 hybrid 两通道零回归）:
    $env:PW_RETRIEVAL_FUSION_MODE='rrf';      python -m eval.golden.golden_retrieval   # 三通道 RRF（k=60）
    $env:PW_RETRIEVAL_FUSION_MODE='weighted'; python -m eval.golden.golden_retrieval   # 三通道加权（默认 0.3,0.6,0.1）
    $env:PW_RETRIEVAL_FUSION_WEIGHTS='0.25,0.5,0.25'  # 权重消融组
    每次运行的 fusion_mode/fusion_weights 写入 eval_runs scores，新旧数字对比须同口径。

指标定义:
    Hit@k      该题检索 top_k 结果中是否命中任意 golden doc（0/1，按题平均）
    Recall@k   命中的 golden doc 数 / golden doc 总数（按题平均）
    MRR        第一个命中 golden doc 的排位倒数，未命中为 0（按题平均）

版本化回归:
    每次运行记录 eval_runs 表（eval_type/git_commit/config_snapshot/scores/per_question），
    --compare 对比最近两次 retrieval 运行的整体指标 delta。

降级策略:
    - 某题无 gold doc → 跳过并记录（不崩溃，不计入指标）
    - 某题检索失败（如 embedding API 502）→ 该题跳过并记录错误，其余继续
    - 数据库不可用 → 分数记录失败打印警告，评估仍完成
"""
import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from src.config import settings
from src.database import async_session_factory
from rag.retrieval.retriever import RetrievalException, hybrid_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("golden_retrieval")

# 本文件所在目录（eval/）
EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden.json"

# eval_runs 表 DDL（module-019 §3.2，幂等）
EVAL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id            BIGSERIAL    PRIMARY KEY,
    eval_type     VARCHAR(20)  NOT NULL DEFAULT 'retrieval',
    git_commit    VARCHAR(64)  NOT NULL DEFAULT '',
    config_snapshot JSONB      NOT NULL DEFAULT '{}',
    scores        JSONB        NOT NULL DEFAULT '{}',
    per_question  JSONB        NOT NULL DEFAULT '[]',
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
COMMENT ON TABLE eval_runs IS '评估运行记录（版本化回归基准）';
COMMENT ON COLUMN eval_runs.git_commit IS '评估时的 git commit';
COMMENT ON COLUMN eval_runs.config_snapshot IS '评估时 rag_config 快照';
COMMENT ON COLUMN eval_runs.scores IS '整体指标分数';
COMMENT ON COLUMN eval_runs.per_question IS '每题明细';
"""


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """加载 golden 检索集，校验结构

    Args:
        path: golden.json 路径

    Returns:
        题目列表，每项含 question / golden_docs / category

    Raises:
        FileNotFoundError: golden.json 缺失时抛错退出
        ValueError: 结构非法（缺 question 或 golden_docs）
    """
    if not path.exists():
        raise FileNotFoundError(
            f"golden 集不存在: {path}，请先创建标注文件（eval/golden.json）"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 20:
        raise ValueError(f"golden 集结构非法：需为 list 且 ≥ 20 题，当前 {len(data) if isinstance(data, list) else 'N/A'}")
    for item in data:
        if "question" not in item or "golden_docs" not in item:
            raise ValueError(f"golden 集题目缺少 question/golden_docs 字段: {item.get('question', '')[:30]}")
    return data


def get_git_commit() -> str:
    """取当前 git commit hash（版本化回归用），失败返回空串"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2],  # 仓库根目录
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning("获取 git commit 失败: %s", e)
    return ""


async def load_rag_config() -> dict:
    """读取 rag_config 表作为配置快照，失败返回空 dict"""
    try:
        async with async_session_factory() as session:
            rows = await session.execute(text("SELECT config_key, config_value FROM rag_config"))
            return {r[0]: r[1] for r in rows}
    except Exception as e:
        logger.warning("读取 rag_config 失败，配置快照为空: %s", e)
        return {}


def compute_metrics(retrieved_titles: list[str], golden_titles: list[str], k: int) -> dict:
    """计算单题召回指标

    指标定义:
        hit_at_k    : 前 k 个检索结果中是否命中任意 golden doc → 1.0 / 0.0
        recall_at_k : 命中的 golden doc 数 / golden doc 总数（golden 为空时为 0.0）
        mrr         : 第一个命中 golden doc 的排位倒数（1-based），未命中为 0.0

    标题匹配（module-031 分块重建后）:
        golden_titles 是文档名（如 "1-G1垃圾收集器..."），而检索返回的是父块
        层级标题（如 "1-G1垃圾收集器... > 板块3 > 第一步"）。匹配需容忍层级前缀：
        golden "X" 匹配检索标题 "X" 或 "X > ..."（split 取最左段）。

    Args:
        retrieved_titles: 检索返回的文档标题列表（长度 ≤ top_k）
        golden_titles: 该题的 golden doc 标题列表
        k: 评估的截断深度（只考察前 k 位）

    Returns:
        {"hit_at_k": float, "recall_at_k": float, "mrr": float, "first_hit_rank": int}
        first_hit_rank 为 0 表示未命中。
    """

    def _golden_matches(golden: str, retrieved: str) -> bool:
        """golden 文档名是否匹配检索标题（容忍 'X > 小节' 层级前缀）"""
        if retrieved == golden:
            return True
        # 检索标题是层级路径时，取最左段（顶层文档名）比对
        root = retrieved.split(" > ", 1)[0]
        return root == golden

    golden_set = set(golden_titles)
    hits = [
        i for i, title in enumerate(retrieved_titles[:k])
        if any(_golden_matches(g, title) for g in golden_set)
    ]
    hit_at_k = 1.0 if hits else 0.0
    recalled = sum(
        1 for g in golden_set
        if any(_golden_matches(g, title) for title in retrieved_titles[:k])
    )
    recall_at_k = recalled / len(golden_set) if golden_set else 0.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0
    return {
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "first_hit_rank": (hits[0] + 1) if hits else 0,
    }


def _aggregate(questions: list[dict]) -> dict:
    """对多题明细聚合整体/分类指标（按题平均）"""
    n = len(questions)
    total = {"count": n}
    for key in ("hit_at_k", "recall_at_k", "mrr"):
        total[key] = round(
            sum(q[key] for q in questions) / n, 4
        ) if n else 0.0

    per_category: dict[str, dict] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        bucket = per_category.setdefault(cat, {"count": 0, "hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0})
        bucket["count"] += 1
        for key in ("hit_at_k", "recall_at_k", "mrr"):
            bucket[key] += q[key]
    for bucket in per_category.values():
        for key in ("hit_at_k", "recall_at_k", "mrr"):
            bucket[key] = round(bucket[key] / bucket["count"], 4)

    return {"overall": total, "per_category": per_category}


async def ensure_eval_runs_table() -> None:
    """幂等创建 eval_runs 表（数据库不可用时抛异常，由调用方处理）

    DDL 含多条语句（CREATE TABLE + COMMENT），asyncpg 不允许单条
    prepared statement 执行多条命令，因此按 ';' 拆分逐条执行。
    """
    statements = [s.strip() for s in EVAL_RUNS_DDL.split(";") if s.strip()]
    async with async_session_factory() as session:
        for stmt in statements:
            await session.execute(text(stmt))
        await session.commit()


async def save_eval_run(
    eval_type: str,
    git_commit: str,
    config_snapshot: dict,
    scores: dict,
    per_question: list[dict],
) -> int:
    """记录一次评估运行到 eval_runs 表

    Args:
        eval_type: 评估类型（retrieval / ragas）
        git_commit: 当前 git commit
        config_snapshot: rag_config 快照
        scores: 整体指标 dict
        per_question: 每题明细 list

    Returns:
        新记录 id；记录失败返回 0（打印警告，不中断评估）
    """
    try:
        await ensure_eval_runs_table()
        async with async_session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO eval_runs (eval_type, git_commit, config_snapshot, scores, per_question)
                    VALUES (:eval_type, :git_commit,
                            CAST(:config_snapshot AS jsonb),
                            CAST(:scores AS jsonb),
                            CAST(:per_question AS jsonb))
                    RETURNING id
                """),
                {
                    "eval_type": eval_type,
                    "git_commit": git_commit,
                    "config_snapshot": json.dumps(config_snapshot, ensure_ascii=False),
                    "scores": json.dumps(scores, ensure_ascii=False),
                    "per_question": json.dumps(per_question, ensure_ascii=False),
                },
            )
            await session.commit()
            row = result.fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        logger.warning("记录 eval_runs 失败（评估结果仍有效）: %s", e)
        return 0


async def compare_runs(limit: int = 2) -> None:
    """对比最近两次 retrieval 运行的指标 delta（版本化回归）

    Args:
        limit: 取最近多少次运行（默认 2）
    """
    try:
        async with async_session_factory() as session:
            rows = await session.execute(text("""
                SELECT id, eval_type, git_commit, scores, created_at
                FROM eval_runs
                WHERE eval_type = 'retrieval'
                ORDER BY id DESC
                LIMIT :limit
            """), {"limit": limit})
            # 使用 RowMapping 以支持按列名访问（text() 查询的普通 Row 不支持 ["col"]）
            runs = [dict(r) for r in rows.mappings()]
    except Exception as e:
        logger.error("查询 eval_runs 失败: %s", e)
        return

    if len(runs) < 2:
        print(f"\n[compare] 不足两次 retrieval 运行（当前 {len(runs)} 条），无法对比。")
        return

    def _as_dict(value) -> dict:
        """JSONB 列可能是已解析 dict 或 JSON 字符串，统一转 dict"""
        if isinstance(value, dict):
            return value
        return json.loads(value or "{}")

    newest, older = runs[0], runs[1]
    ns, os_ = _as_dict(newest["scores"]), _as_dict(older["scores"])
    print("\n" + "=" * 60)
    print("Regression Compare (recent vs previous)")
    print("=" * 60)
    print(f"  new  #{newest['id']}  commit={newest['git_commit'][:8]}  {newest['created_at']}"
          f"  fusion={ns.get('fusion_mode', 'hybrid')}")
    print(f"  prev #{older['id']}  commit={older['git_commit'][:8]}  {older['created_at']}"
          f"  fusion={os_.get('fusion_mode', 'hybrid')}")
    print("  ⚠️ 新旧对比须同口径（fusion_mode 一致），否则 delta 无意义")
    print("-" * 60)
    print(f"  {'metric':<12}{'new':>8}{'prev':>8}{'delta':>8}")
    for key in ("hit_at_k", "recall_at_k", "mrr"):
        new_v = ns.get(key, 0.0)
        old_v = os_.get(key, 0.0)
        print(f"  {key:<12}{new_v:>8.4f}{old_v:>8.4f}{new_v - old_v:>+8.4f}")
    print("=" * 60)
    print("  负 delta 表示本次相对上次回归，需检查检索配置变更。")


async def _eval_question(item: dict, mode: str, top_k: int) -> tuple[dict, dict]:
    """单题评估：检索 + 指标 + 降级/失败处理

    Args:
        item: golden 题目（question / golden_docs / category）
        mode: 检索模式（hybrid / vector_only / fts_only / graph_only）
        top_k: 该题的检索深度

    Returns:
        (evaluated, skipped) 二元组，二者恰有一个非空 dict：
        - evaluated 非空：该题已评估，含命中指标与降级标记
        - skipped 非空：该题跳过，reason 为跳过原因
    """
    question = item["question"]
    category = item.get("category", "")
    golden_titles = item.get("golden_docs", [])
    if not golden_titles:
        return {}, {"question": question, "category": category, "reason": "no_gold_docs"}

    degraded = False
    try:
        docs = await hybrid_retriever.retrieve(question, top_k=top_k, mode=mode)
    except RetrievalException as e:
        # 向量通道不可用（如 embedding API 502）时：
        # hybrid 降级为仅 FTS 继续评估，vector_only 则如实记录通道不可用
        if mode != "hybrid":
            return {}, {"question": question, "category": category, "reason": f"error: {e}"}
        logger.warning("向量通道不可用，降级为 FTS 评估: %s", e)
        degraded = True
        try:
            docs = await hybrid_retriever.retrieve(question, top_k=top_k, mode="fts_only")
        except Exception as fts_err:
            return {}, {"question": question, "category": category, "reason": f"error: {e}; fts_fallback: {fts_err}"}
    except Exception as e:
        return {}, {"question": question, "category": category, "reason": f"error: {e}"}

    retrieved_titles = [d.get("title", "") for d in docs]
    evaluated = {
        "question": question,
        "category": category,
        "golden_docs": golden_titles,
        "retrieved_titles": retrieved_titles,
        "degraded": degraded,
        **compute_metrics(retrieved_titles, golden_titles, top_k),
    }
    return evaluated, {}


async def run_eval(mode: str, top_k: int) -> tuple[dict, list[dict], list[dict]]:
    """执行一次检索评估

    Args:
        mode: 检索模式（hybrid / vector_only / fts_only / graph_only）
        top_k: 每题的检索深度

    Returns:
        (scores, per_question, skipped)
        - scores: 整体指标 + 按类别汇总
        - per_question: 每题明细（含命中情况）
        - skipped: 跳过题目记录（无 gold doc / 检索失败）
    """
    golden = load_golden()
    per_question: list[dict] = []
    skipped: list[dict] = []

    for i, item in enumerate(golden):
        evaluated, skip = await _eval_question(item, mode, top_k)
        if evaluated:
            per_question.append(evaluated)
            continue
        reason = skip["reason"]
        if reason == "no_gold_docs":
            logger.warning("[%d/%d] 跳过无 gold doc 题目: %s", i + 1, len(golden), item["question"][:40])
        elif "fts_fallback" in reason:
            logger.error("[%d/%d] 题目检索失败（含 FTS 回退）: %s — %s", i + 1, len(golden), item["question"][:40], reason)
        else:
            logger.error("[%d/%d] 题目检索失败: %s — %s", i + 1, len(golden), item["question"][:40], reason)
        skipped.append(skip)

    agg = _aggregate(per_question)
    scores = {
        **agg["overall"],
        "mode": mode,
        # module-053 口径声明：retrieval_fusion_mode 标注本次运行的融合模式
        #（hybrid=两通道 / rrf=三通道 RRF / weighted=三通道加权），新旧数字
        # 对比前必须同口径（评估路径直调 retriever，不含引擎层 round 0 图谱并行）
        "fusion_mode": settings.retrieval_fusion_mode,
        "fusion_weights": settings.retrieval_fusion_weights
        if settings.retrieval_fusion_mode == "weighted" else "",
        "top_k": top_k,
        "dataset_size": len(golden),
        "evaluated": len(per_question),
        "skipped": len(skipped),
        "per_category": agg["per_category"],
    }
    return scores, per_question, skipped


def print_report(mode: str, top_k: int, scores: dict, per_question: list[dict], skipped: list[dict], saved_id: int, commit: str) -> None:
    """打印评估报告到控制台"""
    print("\n" + "=" * 60)
    print("Golden Retrieval Eval")
    print("=" * 60)
    print(f"Dataset: {scores['dataset_size']} questions | Evaluated: {scores['evaluated']} | Skipped: {scores['skipped']}")
    print(f"Mode: {mode} | fusion_mode: {scores.get('fusion_mode', 'hybrid')}"
          + (f" | weights: {scores['fusion_weights']}" if scores.get("fusion_weights") else "")
          + f" | top_k: {top_k}")
    print("-" * 60)
    print(f"Hit@{top_k}:   {scores['hit_at_k']:.4f}")
    print(f"Recall@{top_k}: {scores['recall_at_k']:.4f}")
    print(f"MRR:      {scores['mrr']:.4f}")
    print("-" * 60)
    print("Per-Category:")
    for cat, info in scores.get("per_category", {}).items():
        print(f"  {cat:<18} n={info['count']:<3} Hit={info['hit_at_k']:.4f} Recall={info['recall_at_k']:.4f} MRR={info['mrr']:.4f}")

    if per_question:
        print("-" * 60)
        print("Per-Question (first 15):")
        for q in per_question[:15]:
            print(f"  {q['first_hit_rank']:>2} hit={q['hit_at_k']:.0f} recall={q['recall_at_k']:.2f} | {q['question'][:42]}")
    if skipped:
        print("-" * 60)
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  [{s['reason']}] {s['question'][:50]}")
    print("=" * 60)
    if saved_id:
        print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to eval_runs")
    print()


async def ablate(top_k: int) -> None:
    """消融实验：graph_only vs hybrid 对比，输出 side-by-side delta

    分别以 graph_only 和 hybrid 模式运行 golden 检索评估，
    对比两个模式的 Hit@k / Recall@k / MRR 及按类别的差值。
    """
    print("\n" + "=" * 60)
    print("Ablation: graph_only vs hybrid")
    print("=" * 60)

    # 校验 golden 集
    load_golden()

    results: dict[str, dict] = {}
    for mode in ("graph_only", "hybrid"):
        print(f"Running {mode}...")
        scores, _, _ = await run_eval(mode, top_k)
        results[mode] = scores
        print(f"  {mode}: Hit@{top_k}={scores['hit_at_k']:.4f}  "
              f"Recall@{top_k}={scores['recall_at_k']:.4f}  MRR={scores['mrr']:.4f}")

    graph = results["graph_only"]
    hybrid = results["hybrid"]

    print("\n" + "-" * 60)
    print(f"{'Metric':<16} {'graph_only':>12} {'hybrid':>12} {'delta':>12}")
    print("-" * 60)
    for key in ("hit_at_k", "recall_at_k", "mrr"):
        gv = graph[key]
        hv = hybrid[key]
        delta = hv - gv
        print(f"{key:<16} {gv:>12.4f} {hv:>12.4f} {delta:>+12.4f}")

    print("-" * 60)
    print("Per-Category Delta (hybrid - graph_only):")
    gcat = graph.get("per_category", {})
    hcat = hybrid.get("per_category", {})
    all_cats = sorted(set(gcat) | set(hcat))
    for cat in all_cats:
        gv = gcat.get(cat, {}).get("hit_at_k", 0.0)
        hv = hcat.get(cat, {}).get("hit_at_k", 0.0)
        print(f"  {cat:<18} Hit@{top_k}: {gv:.4f} → {hv:.4f}  ({hv-gv:+.4f})")
    print("=" * 60)
    print("  正 delta 表示 hybrid 优于 graph_only（图谱贡献量）。")
    print()


async def main() -> None:
    """评估脚本入口"""
    parser = argparse.ArgumentParser(description="Golden 检索集召回评估")
    parser.add_argument("--mode", default="hybrid",
                        choices=["hybrid", "vector_only", "fts_only", "graph_only"],
                        help="检索模式（默认 hybrid）")
    parser.add_argument("--top-k", type=int, default=5, help="检索深度 k（默认 5，0/负数自动回退 5）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    parser.add_argument("--compare", action="store_true", help="对比最近两次运行")
    parser.add_argument("--ablate", action="store_true",
                        help="消融实验：graph_only vs hybrid side-by-side 对比")
    args = parser.parse_args()

    if args.compare:
        await compare_runs()
        return

    top_k = args.top_k if args.top_k and args.top_k > 0 else 5

    if args.ablate:
        await ablate(top_k)
        return

    # 先校验 golden 集（文件缺失/结构非法时立即报错退出）
    load_golden()

    scores, per_question, skipped = await run_eval(args.mode, top_k)

    saved_id = 0
    commit = ""
    if not args.no_save:
        commit = get_git_commit()
        config_snapshot = await load_rag_config()
        saved_id = await save_eval_run(
            eval_type="retrieval",
            git_commit=commit,
            config_snapshot=config_snapshot,
            scores={k: v for k, v in scores.items() if k != "per_category"},
            per_question=per_question,
        )

    print_report(args.mode, top_k, scores, per_question, skipped, saved_id, commit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
