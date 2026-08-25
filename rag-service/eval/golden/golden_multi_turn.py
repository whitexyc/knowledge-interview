"""
Golden Multi-Turn 评测脚本 — 多轮追问对意图保持 + 检索提升（module-063 / ADR-0015）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.golden.golden_multi_turn              # 真实模式：LLM 对话改写 + 生产路由 + DB 检索对比 + 落库
    python -m eval.golden.golden_multi_turn --fixture    # fixture 模式：启发式改写+启发式意图，不依赖 LLM/DB（管线演示）
    python -m eval.golden.golden_multi_turn --no-save    # 纯跑分，不写 eval_runs

指标定义（zenvanriel 改写质量三指标，多轮追问题型）:
    自包含清晰度（self_contained_ratio）: 对话改写把省略句/指代句补全成
        可独立理解 query 的比例（改写成功且 != 原 follow_up；失败/回退记 0）
    意图保持（intent_preserved_ratio）: 生产多轮路由
        router_agent.classify(follow_up, history=[prev]) 的 intent 与标注意图
        一致的比例（对照 raw_intent_ratio = 单句路由基线——展示省略句漏检）
    检索提升（retrieval_delta）: 改写后检索与"上一轮完整问题检索"（相关锚点）
        的重叠度增量 = mean(overlap(rewrite, prev) - overlap(raw, prev))，
        正 = 改写把省略句对齐回主题文档（无 golden 标注，用 prev 检索作锚点，
        如实声明代理口径）

评测只度量不接线:
    本脚本不改变生产行为。真实模式对话改写用生产 contextual_rewrite
    （module-072 起：rag/retrieval/query_rewrite.py 分诊式改写链的上下文
    分支单一来源——triage + LLM 改写 + 保真门控，含 10s 超时；自包含
    记 0 语义 = 改写失败/保真被拒/句子已自包含（triage precise）均算
    失败）；生产多轮路由走 router_agent.classify(query, history)——本脚本
    直接调用生产路由测量。query_rewrite_enabled 开启时 _classify 先走引擎
    短路路由语义（分诊命中 FTS 术语且非规则词 → knowledge，engine.chat
    同款确定性信号）。

降级策略:
    - 对话改写失败/超时 → 该对按原 follow_up 参与意图/检索（自包含记 0），不中断
    - 单对检索失败 → 跳过并记录错误，其余继续
    - 数据库不可用 → 用 --fixture 模式演示管线，如实标注"待环境"
"""
import argparse
import asyncio
import logging
import sys

from agent.router import RouterAgent
from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run
from rag.retrieval.query_rewrite import contextual_rewrite
from rag.retrieval.retriever import hybrid_retriever
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("golden_multi_turn")

TOP_K = 5

# 多轮追问对评测集（module-063 / ADR-0015，≥10 条）：
# prev = 上一轮知识库完整问题；follow_up = 省略句/指代句（单句无特征会被误判）；
# expected_intent = 结合前文后的正确意图（全部 knowledge：追问都是技术问题）。
MULTI_TURN_DATASET: list[dict] = [
    {"prev": "什么是Java线程池？核心参数有哪些？", "follow_up": "为什么", "expected_intent": "knowledge"},
    {"prev": "G1垃圾收集器和CMS有什么区别？", "follow_up": "那CMS呢", "expected_intent": "knowledge"},
    {"prev": "Redis持久化机制RDB和AOF有什么区别？", "follow_up": "它们各自的适用场景呢", "expected_intent": "knowledge"},
    {"prev": "HashMap和ConcurrentHashMap底层实现有什么区别？", "follow_up": "为什么ConcurrentHashMap线程安全", "expected_intent": "knowledge"},
    {"prev": "Spring事务失效的场景有哪些？", "follow_up": "怎么解决呢", "expected_intent": "knowledge"},
    {"prev": "分布式锁怎么实现？Redis和Zookeeper怎么选？", "follow_up": "它怎么保证原子性", "expected_intent": "knowledge"},
    {"prev": "MySQL索引为什么选B+树？", "follow_up": "那为什么不选红黑树", "expected_intent": "knowledge"},
    {"prev": "JVM内存区域怎么划分？堆和栈有什么区别？", "follow_up": "栈里的内容是什么", "expected_intent": "knowledge"},
    {"prev": "Netty的Reactor线程模型怎么工作？", "follow_up": "它是单线程还是多线程", "expected_intent": "knowledge"},
    {"prev": "Kafka消息队列怎么保证消息不丢失？", "follow_up": "消费者怎么确认", "expected_intent": "knowledge"},
    {"prev": "CAS和synchronized分别适合什么场景？", "follow_up": "CAS的缺点呢", "expected_intent": "knowledge"},
    {"prev": "ThreadLocal原理是什么？为什么有内存泄漏问题？", "follow_up": "怎么避免泄漏", "expected_intent": "knowledge"},
]


def load_dataset() -> list[dict]:
    """加载多轮追问对评测集，校验结构

    Returns:
        样本列表，每项含 prev / follow_up / expected_intent

    Raises:
        ValueError: 样本 < 10、prev/follow_up 为空或 expected_intent 非法
    """
    data = MULTI_TURN_DATASET
    if len(data) < 10:
        raise ValueError(f"多轮评测集过小：需 ≥ 10 条，当前 {len(data)}")
    for item in data:
        if not item.get("prev", "").strip():
            raise ValueError(f"多轮评测集存在空 prev: {item}")
        if not item.get("follow_up", "").strip():
            raise ValueError(f"多轮评测集存在空 follow_up: {item}")
        if item.get("expected_intent") not in ("knowledge", "casual_chat", "realtime"):
            raise ValueError(f"expected_intent 非法: {item}")
    return data


def heuristic_rewrite(prev: str, follow_up: str) -> str:
    """fixture 启发式对话改写：把 prev 的核心术语补进 follow_up（确定性）

    仅用于 fixture 模式演示管线，不代表 LLM 改写质量。prev 无术语 → 原样
    返回（改写失败语义，与真实模式回退一致）。

    Args:
        prev: 上一轮完整问题
        follow_up: 当前省略句

    Returns:
        改写后 query（无术语可用时 = follow_up 原样）
    """
    terms = RouterAgent._kb_terms(prev)
    if terms:
        return f"{follow_up} {terms[0]}"
    return follow_up


def heuristic_intent(query: str) -> str:
    """fixture 启发式意图：规则词 → 闲聊/实时；有知识库术语 → knowledge

    仅用于 fixture 模式演示管线，不代表真实路由能力（真实走
    router_agent.classify 含 L4/L2/LLM）。顺序：先规则表（闲聊/实时特征词），
    再 FTS 术语特征（_kb_terms 非空近似"词表对得上"）。

    Args:
        query: 用户问题

    Returns:
        意图标签（knowledge/casual_chat/realtime）
    """
    from agent.router import _RULE_TABLE
    q = query.lower()
    # 简化规则表：实时词（时间/天气）优先，闲聊词次之（真实 _rule_hits 只返回
    # 是否命中，不区分类别；fixture 用词表子集近似演示）
    if any(w in q for w in ("几点", "天气", "气温", "星期", "几号", "温度")):
        return "realtime"
    if any(w in q for w in _RULE_TABLE):
        return "casual_chat"
    if RouterAgent._kb_terms(query):
        return "knowledge"
    return "casual_chat"


def overlap_ratio(a_titles: list[str], b_titles: list[str]) -> float:
    """检索重叠度：a 结果与 b 结果在 top_k 窗口的标题命中比例（0-1）

    Args:
        a_titles: 一组检索结果的标题列表（已按序）
        b_titles: 另一组检索结果的标题列表

    Returns:
        |set(a) ∩ set(b)| / min(len(a), len(b), 1)；空输入 → 0.0
    """
    if not a_titles or not b_titles:
        return 0.0
    return len(set(a_titles) & set(b_titles)) / min(len(a_titles), len(b_titles))


def compute_metrics(per_question: list[dict]) -> dict:
    """三指标聚合（纯函数，可单测）

    Args:
        per_question: 每题明细（见 _eval_question）

    Returns:
        {
          "count", "self_contained_ratio"（自包含清晰度）,
          "raw_intent_ratio"（单句路由基线对照）,
          "intent_preserved_ratio"（意图保持）,
          "retrieval_delta"（检索提升，改写 vs 原重叠度均值差）,
          "raw_overlap", "rewritten_overlap"
        }
    """
    n = len(per_question)
    if not n:
        return {"count": 0, "self_contained_ratio": 0.0, "raw_intent_ratio": 0.0,
                "intent_preserved_ratio": 0.0, "retrieval_delta": None,
                "raw_overlap": None, "rewritten_overlap": None}
    self_contained = sum(
        1 for q in per_question if q.get("rewrite_changed") is True)
    raw_preserved = sum(
        1 for q in per_question if q.get("raw_intent") == q.get("expected_intent"))
    preserved = sum(
        1 for q in per_question if q.get("routed_intent") == q.get("expected_intent"))
    # 检索提升：改写成功且检索可得才计入（fixture 无 DB → 全 None 如实标注）
    pairs_with_retrieval = [q for q in per_question
                            if q.get("raw_overlap") is not None
                            and q.get("rewritten_overlap") is not None]
    raw_overlap = None
    rewritten_overlap = None
    retrieval_delta = None
    if pairs_with_retrieval:
        raw_overlap = round(sum(q["raw_overlap"] for q in pairs_with_retrieval)
                            / len(pairs_with_retrieval), 4)
        rewritten_overlap = round(
            sum(q["rewritten_overlap"] for q in pairs_with_retrieval)
            / len(pairs_with_retrieval), 4)
        retrieval_delta = round(rewritten_overlap - raw_overlap, 4)
    return {
        "count": n,
        "self_contained_ratio": round(self_contained / n, 4),
        "raw_intent_ratio": round(raw_preserved / n, 4),
        "intent_preserved_ratio": round(preserved / n, 4),
        "retrieval_delta": retrieval_delta,
        "raw_overlap": raw_overlap,
        "rewritten_overlap": rewritten_overlap,
    }


async def _eval_question(item: dict, top_k: int,
                         fixture: bool) -> tuple[dict, dict]:
    """单对评估：对话改写 → 单句/多轮/改写后路由 → 检索重叠对比

    Args:
        item: 多轮追问对（prev / follow_up / expected_intent）
        top_k: 检索深度
        fixture: 是否 fixture 模式（启发式改写+意图，不依赖 LLM/DB）

    Returns:
        (evaluated, skipped) 二元组，二者恰有一个非空 dict
    """
    prev = item["prev"]
    follow_up = item["follow_up"]
    expected = item["expected_intent"]

    # ① 对话改写（fixture 用启发式；真实用 LLM）
    if fixture:
        rewritten = heuristic_rewrite(prev, follow_up)
    else:
        rewritten = await contextual_rewrite(prev, follow_up)
    rewrite_changed = bool(rewritten and rewritten != follow_up)

    # ② 意图：单句基线 + 生产多轮路由 + 改写后意图
    try:
        raw_intent = (await _classify(follow_up, history=None,
                                      fixture=fixture)).get("intent", "knowledge")
    except Exception as e:
        return {}, {"prev": prev, "follow_up": follow_up,
                    "reason": f"error: {e}"}
    try:
        routed = await _classify(follow_up, history=[
            {"role": "user", "content": prev}], fixture=fixture)
        routed_intent = routed.get("intent", "knowledge")
    except Exception as e:
        return {}, {"prev": prev, "follow_up": follow_up,
                    "reason": f"error: {e}"}
    rewrite_intent = None
    if rewrite_changed:
        try:
            rewrite_intent = (await _classify(rewritten, history=None,
                                              fixture=fixture)).get("intent", "knowledge")
        except Exception as e:
            logger.warning("改写后意图分类失败: %s", e)

    # ③ 检索重叠（真实模式；fixture 无 DB → None 如实标注）
    raw_overlap = None
    rewritten_overlap = None
    if not fixture:
        try:
            prev_docs = await hybrid_retriever.retrieve(prev, top_k=top_k)
            raw_docs = await hybrid_retriever.retrieve(follow_up, top_k=top_k)
            prev_titles = [d.get("title", "") for d in prev_docs]
            raw_overlap = overlap_ratio(
                [d.get("title", "") for d in raw_docs], prev_titles)
            if rewrite_changed:
                rw_docs = await hybrid_retriever.retrieve(rewritten, top_k=top_k)
                rewritten_overlap = overlap_ratio(
                    [d.get("title", "") for d in rw_docs], prev_titles)
        except Exception as e:
            return {}, {"prev": prev, "follow_up": follow_up,
                        "reason": f"error: {e}"}

    return {
        "prev": prev,
        "follow_up": follow_up,
        "expected_intent": expected,
        "rewritten_query": rewritten,
        "rewrite_changed": rewrite_changed,
        "raw_intent": raw_intent,
        "routed_intent": routed_intent,
        "rewrite_intent": rewrite_intent,
        "raw_overlap": raw_overlap,
        "rewritten_overlap": rewritten_overlap,
    }, {}


async def _classify(query: str, history, fixture: bool):
    """路由封装：fixture 用启发式意图（不依赖 LLM/DB）；真实用生产 router

    module-072（WP-C）：query_rewrite_enabled 开启且非 fixture 时先走引擎
    短路路由语义（分诊命中 FTS 术语 precise 且非闲聊/实时规则词 → 短路
    knowledge，engine.chat 同款确定性信号，reason 与生产逐字一致）。

    Args:
        query: 用户问题
        history: 对话历史（None = 单句）
        fixture: 是否 fixture 模式

    Returns:
        {"intent": str, "confidence": float, "reason": str}（router 契约）
    """
    if fixture:
        return {"intent": heuristic_intent(query), "confidence": 0.0,
                "reason": "fixture 启发式意图"}
    from agent.router import router_agent
    if settings.query_rewrite_enabled:
        from rag.retrieval.query_rewrite import triage
        if (await triage(query) == "precise"
                and not router_agent._rule_hits(query)):
            return {"intent": "knowledge", "confidence": 0.0,
                    "reason": "分诊命中 FTS 术语，短路 knowledge"}
    return await router_agent.classify(query, history=history)


def _aggregate(per_question: list[dict], skipped: list[dict],
               fixture: bool) -> dict:
    """聚合 scores"""
    scores = compute_metrics(per_question)
    scores.update({
        "dataset_size": len(load_dataset()),
        "evaluated": len(per_question),
        "skipped": len(skipped),
        "top_k": TOP_K,
        "fixture": fixture,
    })
    return scores


async def run_eval_fixture() -> tuple[dict, list[dict], list[dict]]:
    """fixture 模式：启发式改写 + 启发式意图（不依赖 LLM/DB，管线演示）

    无 DB 检索 → 检索提升三指标 raw/rewritten_overlap 与 retrieval_delta
    为 None（如实标注待环境）；自包含/意图保持可量化（纯启发式）。
    """
    per_question: list[dict] = []
    skipped: list[dict] = []
    for item in load_dataset():
        evaluated, skip = await _eval_question(item, TOP_K, fixture=True)
        if evaluated:
            per_question.append(evaluated)
        else:
            skipped.append(skip)
    scores = _aggregate(per_question, skipped, fixture=True)
    return scores, per_question, skipped


async def run_eval_real(top_k: int) -> tuple[dict, list[dict], list[dict]]:
    """真实模式：LLM 对话改写 + 生产多轮路由 + DB 检索重叠对比"""
    per_question: list[dict] = []
    skipped: list[dict] = []
    for i, item in enumerate(load_dataset()):
        evaluated, skip = await _eval_question(item, top_k, fixture=False)
        if evaluated:
            per_question.append(evaluated)
            continue
        logger.error("[%d/%d] 评估失败: %s — %s", i + 1, len(load_dataset()),
                     item["follow_up"][:40], skip["reason"])
        skipped.append(skip)
    scores = _aggregate(per_question, skipped, fixture=False)
    return scores, per_question, skipped


async def record_eval_run(scores: dict, per_question: list[dict]) -> tuple[str, int]:
    """版本化落库：git_commit + rag_config 快照 + eval_type='multi_turn'

    Returns:
        (commit, saved_id)；落库失败 saved_id=0（save_eval_run 内部已捕获并警告）
    """
    commit = get_git_commit()
    config_snapshot = await load_rag_config()
    # module-072（WP-C）：补运行时开关快照（rag_config 表无此键），使
    # eval_runs 可回溯本次评估的两开关启用态（off/on 四跑可区分）
    config_snapshot["query_rewrite_enabled"] = str(settings.query_rewrite_enabled)
    config_snapshot["contextual_rewrite_enabled"] = str(settings.contextual_rewrite_enabled)
    saved_id = await save_eval_run(
        eval_type="multi_turn",
        git_commit=commit,
        config_snapshot=config_snapshot,
        scores=scores,
        per_question=per_question,
    )
    return commit, saved_id


def print_report(scores: dict, per_question: list[dict], skipped: list[dict],
                 saved_id: int, commit: str) -> None:
    """打印评估报告：三指标 + 每题明细"""
    print("\n" + "=" * 60)
    title = "Golden Multi-Turn Eval"
    if scores.get("fixture"):
        title += "  [fixture 模式：启发式改写+意图，非真实指标；检索对比需真实模式（DB+LLM）]"
    print(title)
    print("=" * 60)
    print(f"Dataset: {scores['dataset_size']} pairs | Evaluated: {scores['evaluated']} | Skipped: {scores['skipped']}")
    print("-" * 60)
    print(f"自包含清晰度 self_contained_ratio : {scores['self_contained_ratio']:.4f}"
          "（改写把省略句补全的比例）")
    print(f"意图保持   intent_preserved_ratio : {scores['intent_preserved_ratio']:.4f}"
          "（多轮路由 intent==标注）")
    print(f"  └ 对照单句路由 raw_intent_ratio  : {scores['raw_intent_ratio']:.4f}"
          "（省略句单句漏检基线）")
    if scores.get("retrieval_delta") is not None:
        print(f"检索提升   retrieval_delta        : {scores['retrieval_delta']:+.4f}"
              f"（raw_overlap={scores['raw_overlap']:.4f} → "
              f"rewritten_overlap={scores['rewritten_overlap']:.4f}）")
    else:
        print("检索提升   retrieval_delta        : 待环境（fixture 无 DB 检索）")
    print("-" * 60)
    if per_question:
        print("Per-Pair:")
        for q in per_question[:20]:
            rw = f"{q['rewritten_query'][:24]}" if q.get("rewrite_changed") else "-"
            ov = ""
            if q.get("raw_overlap") is not None:
                ov = f" overlap {q['raw_overlap']:.2f}->{q['rewritten_overlap'] if q['rewritten_overlap'] is not None else 0.0:.2f}"
            print(f"  {q['expected_intent']:<10} raw={q['raw_intent']:<12} "
                  f"routed={q['routed_intent']:<12} 改写={rw}{ov} | {q['follow_up'][:24]}")
    if skipped:
        print("-" * 60)
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  [{s['reason'][:30]}] {s['follow_up'][:50]}")
    print("=" * 60)
    if saved_id:
        print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to eval_runs")
    print()


async def main() -> None:
    """评测脚本入口"""
    parser = argparse.ArgumentParser(description="Golden 多轮追问评测：三指标 + 版本化回归")
    parser.add_argument("--fixture", action="store_true",
                        help="fixture 模式：启发式改写+意图（确定性，不依赖 LLM/DB），仅演示管线")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="检索深度 k（默认 5，0/负数自动回退 5）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    args = parser.parse_args()

    top_k = args.top_k if args.top_k and args.top_k > 0 else TOP_K
    load_dataset()  # 先校验评测集（结构非法时立即报错退出）

    if args.fixture:
        scores, per_question, skipped = await run_eval_fixture()
    else:
        scores, per_question, skipped = await run_eval_real(top_k)

    saved_id = 0
    commit = ""
    if args.fixture:
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
