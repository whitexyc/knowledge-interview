"""
分诊式 Query 改写（module-049 / ADR-0009）— 检索前主动增强
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置（前置增强，不改检索核心）：
  用户 Query → [分诊: FTS 术语命中?]
                  ├─ 命中（精确）──→ 直接检索（零成本，不走改写）
                  └─ 不命中（模糊）─→ [LLM 改写] → [保真预检: 余弦 < 0.6 → 回退原话]
                                        → [并行检索: 原 query + 改写 query]
                                        → [择优: 改写 top-1 abs_cosine > 原 → 用改写，否则回退原]
                  → [HybridRetriever] → [Reflector 反思兜底（保留，事后充分性检查）] → ...

设计决策：
  1. 分诊判据是"词表对得上"（检索质量信号），不调 LLM/生成——复用
     agent.router 的 FTS 术语命中（jieba 分词 + 功能词过滤 + search_tokens
     倒排，毫秒级零成本）。分诊失败（DB 异常）→ 保守默认"模糊"走改写路径
     （宁多检不漏检）。
  2. 保真预检用本地 bge-m3 嵌入（1024 维）算改写 vs 原 query 余弦：
     改写跑偏（低于 rewrite_fidelity_threshold）→ 直接用原 query 检索，
     省一次并行检索；预检失败（嵌入不可用）→ 跳过预检直接并行，让择优兜底。
  3. 并行检索用 asyncio.gather(return_exceptions=True)，单路失败降级为
     另一路（对齐 round 0 模式）；双路失败 → 空结果走现有无结果降级。
  4. 择优判据：改写检索 top-1 abs_cosine > 原检索 → 用改写结果；否则回退
     原结果；相等/缺失/异常 → 回退原（保守，防合并噪声）。abs_cosine
     缺失的文档按 0 处理。
  5. 降级哲学：改写链路任何一环失败 = 回退原 query，行为与现状完全一致
     （零回归）。HyDE 与反思 check_sufficiency 是既有环节，保留不删——
     本模块只把"改写时机"提前。

接入形态（engine.py）：
  - chat 主路径：prepare() 全管线（分诊+改写+保真+并行择优），返回
    (检索 query, round 0 择优文档, 决策信息)，round 0 直接用择优文档。
  - 流式/_retrieve 路径：prepare_query() 查询级（分诊+改写+保真门控，不
    并行——round 0 已有向量+图并行与 HyDE 扩展，叠加并行成本翻倍且语义
    重叠），改写通过保真后作为 HyDE 扩展的基础 query（改写与 HyDE 正交）。
"""
import asyncio
import logging

from agent.router import fts_term_hit
from llm.client import LLMFactory
from rag.retrieval.embeddings import embedding_service
from src.config import settings

logger = logging.getLogger(__name__)

# LLM 改写 prompt：把口语化/模糊问题改写成适合知识库检索的精确查询。
# 与反思（check_sufficiency 内嵌 rewritten_query）不同——本改写发生在检索
# 之前、不依赖任何文档，目标是"搜索词更精准"，独立封装（module-049 WP2①）。
_REWRITE_PROMPT = """你是技术知识库的检索助手。用户的问题可能口语化、含糊，请把它改写成适合在知识库中检索的精确查询。

要求：
1. 保留核心专有术语原样（如 G1、CMS、AQS、Redis、Kafka 等，不翻译不改写）
2. 去掉口语化表达（"有没有什么好办法""是不是""怎么搞"等）与疑问语气
3. 改写为简洁的关键词组合，但保持语义完整、可独立检索
4. 只返回改写后的查询文本，不要任何解释或其他文字

用户问题: {query}

改写后的查询:"""

# 改写超时（秒）：与 HyDE 扩展一致，超时回退原 query
_REWRITE_TIMEOUT = 10

# 上下文改写 prompt（module-072，自 golden_multi_turn.py 迁移，单一来源）：
# 把多轮对话中的省略句/指代句（"为什么""那它呢"）结合上一轮完整问题补全成
# 可独立检索的自包含问题。与 _REWRITE_PROMPT 的区别：输入含"上一轮问题"段
#（主题锚点），输出是自包含问题而非关键词组合。
_CONTEXTUAL_REWRITE_PROMPT = """你是技术知识库的检索助手。用户在多轮对话中说了省略句/指代句（如"为什么""那它呢"），
请结合上一轮问题，把它改写成可独立检索的自包含问题。

上一轮问题: {prev}
当前省略句: {query}

改写后的自包含问题:"""


async def triage(query: str) -> str:
    """静态分诊：FTS 术语命中 → 精确 query 直接检索，不走改写

    判据是"词表对得上"（检索质量信号），零 LLM、零生成。分诊失败
    （DB 异常/超时）→ 保守默认 "vague" 走改写路径，不中断链路
    （宁多检不漏检，对齐 ADR-0009 降级）。

    Args:
        query: 用户问题

    Returns:
        "precise"（FTS 术语命中，精确 query 直接检索）/"vague"（模糊 query 走改写）
    """
    try:
        hit = await fts_term_hit(query)
    except Exception as e:
        logger.warning("分诊失败（保守默认模糊走改写）: %s", e)
        return "vague"
    return "precise" if hit else "vague"


def extract_prev(history: list | None) -> str | None:
    """从对话历史取最近一条 user 消息 content 作上下文改写的前文（module-072）

    history 为空/无 user 消息 → None（不走上下文改写）。取最近一条
    （模块-063 classify 同款"取最近"语义）；content 非字符串（如多模态
    数组）或空白 → 跳过该条继续向前找。

    Args:
        history: 对话历史（[{role, content}, ...]）；None/空 → None

    Returns:
        最近一条 user 消息 content；无 → None
    """
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


async def llm_rewrite(query: str, prev: str | None = None) -> str | None:
    """LLM 改写：把模糊 query 改写为精确检索查询（独立封装）

    prev 非空 → 上下文改写（module-072：结合上一轮问题把省略句补全成
    自包含 query，_CONTEXTUAL_REWRITE_PROMPT）；prev 为空/None → 走
    module-049 原 _REWRITE_PROMPT（逐字零回归）。

    失败/超时/空结果 → 返回 None（调用方回退原 query，链路不中断，
    与 HyDE 失败降级同哲学）。改写文本与原 query 相同也视为无效
    （调用方按 None 处理）。

    Args:
        query: 用户原始查询（prev 非空时 = 当前省略句）
        prev: 上一轮用户完整问题（多轮上下文改写用；None = 单轮普通改写）

    Returns:
        改写后的查询文本；失败/超时/空 → None
    """
    if prev:
        prompt = _CONTEXTUAL_REWRITE_PROMPT.format(prev=prev, query=query)
    else:
        prompt = _REWRITE_PROMPT.format(query=query)
    try:
        client = LLMFactory.get_client("fallback", temperature=0.1)
        rewritten = await asyncio.wait_for(client.generate(prompt), timeout=_REWRITE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Query 改写超时 (%ds)，回退原 query: %s", _REWRITE_TIMEOUT, query[:50])
        return None
    except Exception as e:
        logger.warning("Query 改写失败，回退原 query: %s", e)
        return None
    rewritten = (rewritten or "").strip()
    if not rewritten:
        logger.warning("Query 改写返回空，回退原 query: %s", query[:50])
        return None
    if rewritten == query:
        logger.info("Query 改写无变化，按无效处理: %s", query[:50])
        return None
    return rewritten


async def fidelity_check(original: str, rewritten: str) -> float | None:
    """保真预检：改写 vs 原 query 的余弦相似度（本地 bge-m3，归一化后点积）

    同义改写余弦 ≈ 0.88（module-035 实测口径），0.6 阈值拦"跑偏"改写。
    嵌入失败 → 返回 None（调用方跳过预检直接并行，让择优兜底）。

    Args:
        original: 原 query
        rewritten: 改写 query

    Returns:
        余弦相似度；嵌入不可用/失败 → None
    """
    try:
        vectors = await embedding_service.embed_documents([original, rewritten])
    except Exception as e:
        logger.warning("保真预检嵌入失败，跳过预检: %s", e)
        return None
    if len(vectors) != 2:
        logger.warning("保真预检嵌入数量异常 (%d)，跳过预检", len(vectors))
        return None
    # 嵌入已 L2 归一化（embeddings.py _normalize），点积即余弦
    try:
        return sum(a * b for a, b in zip(vectors[0], vectors[1]))
    except Exception as e:
        logger.warning("保真预检余弦计算异常，跳过预检: %s", e)
        return None


def select_better(orig_docs: list[dict], rewritten_docs: list[dict]) -> tuple[list[dict], bool]:
    """择优：改写检索 top-1 绝对余弦 > 原检索 → 用改写结果；否则回退原

    保守原则（防合并噪声）：相等/缺失/异常 → 回退原结果。abs_cosine
    缺失的文档按 0 处理（module-045 口径，`d.get("abs_cosine") or 0.0`）。

    Args:
        orig_docs: 原 query 检索结果
        rewritten_docs: 改写 query 检索结果

    Returns:
        (docs, used_rewrite)：docs 为择优后的文档；used_rewrite=True 表示
        采用改写结果（false 表示回退原结果）
    """
    orig_top1 = orig_docs[0].get("abs_cosine") or 0.0 if orig_docs else 0.0
    rewritten_top1 = rewritten_docs[0].get("abs_cosine") or 0.0 if rewritten_docs else 0.0
    if rewritten_docs and rewritten_top1 > orig_top1:
        return rewritten_docs, True
    return orig_docs, False


async def prepare(query: str, retrieve_fn,
                  history: list | None = None) -> tuple[str, list[dict] | None, dict]:
    """分诊式改写全管线（chat 主路径）：分诊 → 改写 → 保真 → 并行检索择优

    Args:
        query: 用户原始查询
        retrieve_fn: 异步检索函数 (q: str) -> list[dict]，调用方注入
            （engine 传入 hybrid_retriever.retrieve + 超时包装，top_k 由
            retrieve_fn 闭包自行控制；测试传 stub）
        history: 对话历史（module-072 上下文改写：非空时取最近一条 user
            消息作 prev，把省略句补全成自包含 query；None = module-049
            原单轮改写逐字零回归）

    Returns:
        (search_query, round0_docs, info)
        - search_query: 后续检索使用的 query（原话或改写择优结果）
        - round0_docs: 非 None 时 = 并行择优结果，调用方应直接用作
          round 0 检索结果（不再重复检索）；None = 未做并行（调用方
          按原流程检索 search_query）
        - info: 决策信息 dict（mode: precise/rewrite_fallback/fidelity_reject/
          parallel；used_rewrite、orig/rewrite top-1 abs_cosine、fidelity 等）
    """
    mode = await triage(query)
    if mode == "precise":
        # 精确 query 直接检索：不调 LLM、不并行，链路零增量
        return query, None, {"mode": "precise"}

    prev = extract_prev(history)
    rewritten = await llm_rewrite(query, prev=prev)
    if rewritten is None:
        # LLM 改写失败/超时/无变化 → 回退原 query（零回归）
        return query, None, {"mode": "rewrite_fallback"}

    # 保真锚点（module-072）：上下文改写（prev 非空）用 f"{prev} {query}"
    # 拼接双锚（主题+原句：防 LLM 漂移到无关话题 + 防丢失原句意图）；裸省略
    # 句（如"为什么"3 字）作锚无信息量，会系统性误杀上下文改写
    anchor = f"{prev} {query}" if prev else query
    fidelity = await fidelity_check(anchor, rewritten)
    if fidelity is not None and fidelity < settings.rewrite_fidelity_threshold:
        # 保真预检未过：改写跑偏 → 直接用原 query 检索（省一次并行检索）
        logger.info("改写保真未过: cos=%.3f < %.1f，回退原 query: %s",
                    fidelity, settings.rewrite_fidelity_threshold, query[:50])
        return query, None, {"mode": "fidelity_reject", "fidelity": fidelity}

    # 保真通过（或预检失败跳过）→ 并行检索 + 择优
    orig_docs, rewritten_docs = await asyncio.gather(
        retrieve_fn(query), retrieve_fn(rewritten), return_exceptions=True,
    )
    if isinstance(orig_docs, Exception):
        logger.warning("原 query 并行检索失败，降级为仅改写结果: %s", orig_docs)
        orig_docs = []
    if isinstance(rewritten_docs, Exception):
        logger.warning("改写 query 并行检索失败，降级为仅原结果: %s", rewritten_docs)
        rewritten_docs = []

    docs, used_rewrite = select_better(orig_docs, rewritten_docs)
    orig_top1 = orig_docs[0].get("abs_cosine") or 0.0 if orig_docs else 0.0
    rewritten_top1 = rewritten_docs[0].get("abs_cosine") or 0.0 if rewritten_docs else 0.0
    logger.info("改写并行择优: used_rewrite=%s, orig_top1=%.3f, rewrite_top1=%.3f, query=%s",
                used_rewrite, orig_top1, rewritten_top1, query[:50])
    return (rewritten if used_rewrite else query), docs, {
        "mode": "parallel",
        "used_rewrite": used_rewrite,
        "orig_top1_abs": orig_top1,
        "rewrite_top1_abs": rewritten_top1,
        "fidelity": fidelity,
        "rewritten": rewritten,
    }


async def prepare_query(query: str, history: list | None = None) -> tuple[str, dict]:
    """查询级分诊式改写（流式/_retrieve 路径）：分诊 → 改写 → 保真门控

    不做并行检索择优：round 0 已有向量+图并行与 HyDE 扩展（module-024），
    叠加并行检索成本翻倍且与 HyDE 语义重叠——改写通过保真预检后作为
    检索基础 query，HyDE 在其上继续扩展（改写与 HyDE 正交）。

    Args:
        query: 用户原始查询
        history: 对话历史（module-072 上下文改写：非空时取最近一条 user
            消息作 prev；None = module-049 原单轮改写逐字零回归）

    Returns:
        (base_query, info)：base_query 为后续检索的基础 query；
        info.mode: precise / rewrite_fallback / fidelity_reject / rewrite_accepted
    """
    mode = await triage(query)
    if mode == "precise":
        return query, {"mode": "precise"}

    prev = extract_prev(history)
    rewritten = await llm_rewrite(query, prev=prev)
    if rewritten is None:
        return query, {"mode": "rewrite_fallback"}

    # 保真锚点同 prepare：上下文改写用 f"{prev} {query}" 拼接双锚
    anchor = f"{prev} {query}" if prev else query
    fidelity = await fidelity_check(anchor, rewritten)
    if fidelity is None or fidelity < settings.rewrite_fidelity_threshold:
        # 保真未过或预检不可得：本路径无并行择优兜底，保守回退原 query
        #（改写链路任何一环失败 = 回退原 query）
        return query, {"mode": "fidelity_reject", "fidelity": fidelity}
    logger.info("改写保真通过: cos=%.3f，采用改写 query: %s",
                fidelity, rewritten[:50])
    return rewritten, {"mode": "rewrite_accepted", "fidelity": fidelity, "rewritten": rewritten}


async def contextual_rewrite(prev: str, follow_up: str) -> str | None:
    """对话上下文化改写（module-072 单一来源）：省略句补全成自包含 query

    生产 prepare 链的上下文分支封装（golden_multi_turn 真实模式调用本函数，
    防 eval 与生产漂移——原 eval-only 实现已删除）：
      triage（FTS 术语命中 = 句子已自包含 → None 不改写）→ llm_rewrite
      （prev 分支）→ 保真门控（锚点 = f"{prev} {follow_up}" 拼接双锚；
      未过/预检失败 → None）。
    失败/超时/无变化/保真未过 → None（调用方回退原句，链路不中断）。

    Args:
        prev: 上一轮完整问题
        follow_up: 当前省略句

    Returns:
        改写后自包含 query；失败/保真未过 → None
    """
    if await triage(follow_up) == "precise":
        # 术语命中 = 句子已自包含（如"那CMS呢"），直接检索即可，不改写
        #（precise 零 LLM 语义，module-049）
        return None
    rewritten = await llm_rewrite(follow_up, prev=prev)
    if rewritten is None:
        return None
    fidelity = await fidelity_check(f"{prev} {follow_up}", rewritten)
    if fidelity is None or fidelity < settings.rewrite_fidelity_threshold:
        logger.info("上下文改写保真未过（cos=%s < %.1f），回退原句: %s",
                    "不可得" if fidelity is None else f"{fidelity:.3f}",
                    settings.rewrite_fidelity_threshold, follow_up[:50])
        return None
    return rewritten
