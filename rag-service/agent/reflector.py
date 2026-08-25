"""
自我反思与纠错 (Self-Reflection) — RAG 链路质量控制
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置：
  Rerank (Top 5) → [Reflector.check_sufficiency] 检查是否充分
                       ├─ 充分 → [Reflector.generate_answer] 生成答案
                       └─ 不充分 → 改写 Query → 二次检索 → 合并文档 → 生成答案

为什么需要自我反思？
  这是 Agentic RAG 的核心特性。传统 RAG 是"检索一次就回答"，
  如果检索结果不相关，答案就错了。自我反思让 LLM 自己检查
  检索结果是否足够回答用户问题，不够就改写 query 再试一次。

  这是"Self-RAG" (https://arxiv.org/abs/2310.11511) 思想的具体实现。
  虽然不是完整的 Self-RAG（没有训练专门的 reflection token），
  但通过 prompt engineering 实现了类似的效果。

设计决策：
  1. 反思和生成走同一 fallback 降级链（消除单点，不再硬编码 deepseek）。
     反思用低温度（0.1）保证结构化 JSON 稳定，生成保持默认 0.7（module-026）。
     反思任务不需要专门的"评估模型"，通用 LLM 通过 prompt 即可胜任。

  2. 反思 prompt 要求返回结构化 JSON，而不是自由文本。
     便于下游程序化判断（if sufficient → generate else 二次检索）。

  3. 二次检索的结果与原始结果合并（而不是替换）。
     因为改写后的 query 可能丢失部分原始意图，保留原始结果做互补。

  4. 生成 prompt 包含历史对话（history 参数），支持多轮追问。
     这是后来加的特性，最初 generate_answer 只接受当前 query。
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from llm.client import LLMFactory
from src.config import settings

logger = logging.getLogger(__name__)

# ── 层 1 分数/数量硬闸门阈值（ADR-0005，module-044）──
# 分数阈值读配置 settings.sufficiency_gate_threshold（module-048：默认 0.55，
# module-047 实测数据结论——0.4 漏判 60% 不充分、0.55 F1=0.98 切在分布间隙上缘）
_SUFFICIENCY_MIN_DOCS = 2          # 文档数 < 2 → 直接判不充分（零 LLM）

# ── HHEM 交叉对数上限（module-055，E2E 实测数据支撑，见 _judge_by_hhem）──
# E2E 实测（module-054 收尾）：5 claims × 3 docs = 15 对在服务负载下耗时
# 12s+ 贴近 15s 超时哲学，HHEM 超时 → LLM 判分降级又 15s 超时 → 级联超时
# verified_claims=0；本机实测 15 对冷启动（含模型加载）≈9s、热推理 0.11s/对。
# 上限后典型 5 claims × 2 docs = 10 对（冷 ≈6s），20s 预算下 3 倍余量。
_MAX_HHEM_DOCS = 2                # 每 claim 最多打分的文档数（按相关度取前 N）
_MAX_HHEM_CLAIMS = 8              # 最多打分的 claims 数（防超长答案拆句爆炸）
# 每对文档文本截断上限（module-055 实测）：verify 传入的是父块全文（≤4000
# 字符 ≈ 2000+ token），超 HHEM 512 token 上限（transformers 报 indexing
# error）且拖慢单对推理；截断到 500 字符后单对耗时显著下降。实测（E2E G1
# 问题 10 对）：截断后 verdict 与全文口径一致（claims 证据通常位于文档头部）。
_MAX_HHEM_DOC_CHARS = 500

# 反思 prompt：判断检索结果是否充分
# 要求 LLM 输出 JSON，包含 sufficient（是否充分）和 rewritten_query（改写后的查询）。
# 如果不充分，rewritten_query 会被用于二次检索。
# module-044（ADR-0005 层 2）强化：CoT 信息点比对（先列所需信息点，再逐点比对
# 文档覆盖，再下结论）+ few-shot 充分/不充分正反例。返回 JSON 结构不变（向后兼容）。
_CHECK_PROMPT = """你是一个严格的答案质量检查员，倾向于使用已有文档。
只有在现有文档完全无法回答问题时才判定不充分。

判断步骤（严格按顺序执行，先比对后下结论）：
1. 列出回答该问题需要的信息点（如关键概念、原理、数据、对比维度等），至少 2 个
2. 逐点比对上面对应的文档编号，标记"已覆盖 / 部分覆盖 / 未覆盖"
3. 根据覆盖情况综合下结论：关键信息点缺失或文档与问题完全不相关 → 不充分；否则充分

规则（严格遵守）：
1. 如果文档内容与问题部分相关、间接相关、或能提供部分信息 → sufficient=true
2. 即使文档没有直接给出答案，但只要包含相关的背景知识 → sufficient=true
3. 只有文档内容与问题完全无关（完全不沾边）才 → sufficient=false
4. 默认倾向 sufficient=true，宁可使用不完美的文档也不要空跑二次检索

示例 1（充分）：
用户问题: "Java 线程池的核心参数有哪些？"
文档摘要: "[1] 线程池核心参数包括核心线程数、最大线程数、队列容量等"
信息点比对: "核心参数列举" → [1] 已覆盖 → 可直接回答
返回: {{"sufficient": true, "reason": "文档[1]直接覆盖线程池核心参数信息点"}}

示例 2（不充分）：
用户问题: "G1 GC 的停顿时间预测模型是怎样的？"
文档摘要: "[1] G1 GC 是面向服务端应用的垃圾回收器，基于 Region 划分堆内存"
信息点比对: "停顿时间预测模型" → [1] 未覆盖（仅介绍 G1 基本概念）→ 无法回答
返回: {{"sufficient": false, "reason": "文档仅介绍 G1 基本概念，未覆盖停顿时间预测模型", "rewritten_query": "G1 GC 停顿时间预测模型"}}

如果文档信息充分，返回: {{"sufficient": true, "reason": "..."}}
如果文档信息不充分，返回: {{"sufficient": false, "reason": "...", "rewritten_query": "改写的搜索关键词"}}

只返回 JSON，不要其他文字。

用户问题: {query}

检索到的文档摘要:
{docs_summary}"""

# 验证 prompt（module-051 拆分，ADR-0010 P0-②）：只负责把答案拆成独立陈述句。
# verdict/evidence 由 HHEM 专职裁判判定（同 LLM 验证自己输出的同源问题，
# 换专职裁判解决）；LLM 不再判分（原全量版本保留为 _VERIFY_LLM_PROMPT 供降级链使用）。
_VERIFY_PROMPT = """你是 RAG 系统的答案拆解器。把以下答案拆成独立的陈述句（claims）。

## 待拆解答案
{answer}

## 任务
1. 把答案拆成独立的陈述句（claims），每条 1-2 句话
2. 只输出 claims 文本数组，不要其他文字、不要 JSON 对象

格式：["claim 1", "claim 2", ...]"""

# 验证 prompt 全量版（module-039 原版，module-051 保留供降级链使用）：
# 拆句 + 判 verdict + 填 evidence 一步完成。HHEM 不可用 / 开关 "llm" 时走此路径，
# 行为与 module-039 完全一致（降级路径行为不漂移）。
_VERIFY_LLM_PROMPT = """你是 RAG 系统的答案验证专家。检查以下答案是否被检索文档支持。

## 检索文档
{docs_text}

## 待验证答案
{answer}

## 任务
1. 把答案拆成独立的陈述句（claims），每条 1-2 句话
2. 对每条陈述判断：
   - "supported": 文档中有直接文字依据
   - "inferred": 没有直接文字，但可以从文档合理推断
   - "unsupported": 文档中找不到依据
3. 对 supported/inferred，填写 evidence 字段（关联文档编号，如 "[1]"）
4. unsupported 的 evidence 填 "N/A"
5. 只返回 JSON 数组，不要其他文字

格式：[{{"claim": "...", "verdict": "supported|inferred|unsupported", "evidence": "[1]"}}]"""

# 生成 prompt：基于检索文档生成回答
# 要求 LLM 用 [1][2] 格式标注引用来源，这是 RAG 答案"可溯源"的关键。
# sections（= 历史对话段 + 记忆段）是可选的，由 generate_answer 方法根据
# 传入的 history / memory 参数填充；两者均为空时 sections 为空串，
# 模板保留 {sections} 后的换行（对齐旧版 {history_section}\n 结构）。
# module-058（WP-B）：区块顺序改为 sections → 检索到的文档 → 用户问题——
# docs 前移为前缀缓存铺路（LLM API 对 prompt 开头重复前缀自动打折，
# 前提 = 前缀逐字一致；同 docs 重复生成时最受益）。query 标签格式不变，
# sections 内容/格式一字不改（仅调换区块顺序，存量测试零漂移）。
_GENERATE_PROMPT = """你是一个知识库问答助手。基于检索到的文档回答用户问题。

要求：
1. 引用文档原文进行回答，用 [1][2] 标注引用来源
2. 如果文档信息不足以回答问题，如实告知
3. 回答后附带引用文档列表

{sections}
检索到的文档:
{docs_detail}

用户问题: {query}

回答："""


class Reflector:
    """自我反思与 Query 改写

    职责：
    1. check_sufficiency: 检查检索结果是否充分，不充分时生成改写 query
    2. generate_answer: 基于检索文档 + 对话历史生成最终回答

    注意：这两个方法都调用 LLM，但用的是 generate（单轮生成）而非 chat。
    因为虽然逻辑上它需要"理解上下文"，但底层实现是拼接 prompt 而不是
    传 messages 数组。这是有意的设计选择，让 prompt 的组装更可控。
    """

    def __init__(self, provider: Optional[str] = None):
        # module-026：反思/生成走 fallback 降级链（消除单点，不再硬编码 deepseek）。
        # 反思用低温度 0.1 保证结构化 JSON 稳定；生成保持默认 0.7，不受反思低温度影响。
        self._provider = provider or "fallback"
        self._reflection_temperature = 0.1  # 结构化 JSON 判断需确定性
        self._generation_temperature = 0.7  # 生成保持创造性
        # module-044 自洽性检查第二温度：与反思温度 0.1 不同（结果多样化的依据）
        self._self_check_temperature = 0.7

    async def check_sufficiency(self, query: str, documents: list[dict],
                                prompt: Optional[str] = None) -> dict:
        """检查检索结果是否充分

        如果 LLM 判断检索结果不够回答问题，会返回一个 rewritten_query，
        这个 query 会被 engine.py 用于二次检索。

        module-044（ADR-0005 层 1+2+3）重构：
        - 层 1 分数/数量硬闸门（零 LLM）：文档数 < 2 或 top-1 abs_cosine < 0.4
          → 直接判不充分 + rewritten_query=query，不调 LLM
        - 层 2 prompt 强化：CoT 信息点比对 + few-shot 正反例（见 _CHECK_PROMPT）
        - 层 3 多信号融合：分数达标（或字段缺失）才问 LLM 判模糊地带；
          LLM 判不充分 → 尊重语义走 rewritten_query（不因分数高强制充分）
        - 自洽性检查（配置开关 PW_SUFFICIENCY_SELF_CHECK_ENABLED，默认关）：
          开启时同 query 两温度各判一次，不一致 → 保守判充分（防漏检）
        - 闸门/LLM 异常 → 默认充分（防死循环，保持"默认充分"哲学）

        module-055（ADR-0011 第一步）：
        - prompt 参数可注入变体（变体测试只度量不替换生产 prompt）；None 时
          使用模块默认 _CHECK_PROMPT（零回归）。自洽性检查第二判同用注入变体
          （同一变体两温度，保证对比口径一致）。

        Args:
            query: 原始查询
            documents: 检索到的文档列表
            prompt: 反思 prompt 变体（含 {query}/{docs_summary} 占位符），
                None = 默认 _CHECK_PROMPT

        Returns:
            充分时: {"sufficient": true, "reason": "..."}
            不充分时: {"sufficient": false, "reason": "...", "rewritten_query": "改写后的查询"}
        """
        if not documents:
            return {"sufficient": False, "reason": "未检索到任何文档",
                     "rewritten_query": query}

        # ── 层 1 数量闸门（零 LLM）──
        if len(documents) < _SUFFICIENCY_MIN_DOCS:
            return {"sufficient": False,
                    "reason": f"文档数不足 {_SUFFICIENCY_MIN_DOCS}",
                    "rewritten_query": query}

        # ── 层 1 分数闸门（零 LLM）──
        # abs_cosine：module-043 在 retriever 归一化前存档，engine 精排后 docs
        # 仍含该字段（rerank 不删字段）；仅 FTS 命中文档无该字段 → None 跳过
        # 闸门走 LLM（不误杀）。阈值读配置（module-048：默认 0.55）。
        top1_abs = documents[0].get("abs_cosine", None)
        if top1_abs is not None:
            gate = settings.sufficiency_gate_threshold
            try:
                if float(top1_abs) < gate:
                    return {"sufficient": False,
                            "reason": f"top-1 绝对余弦 {float(top1_abs):.3f} < "
                                      f"{gate}",
                            "rewritten_query": query}
            except (TypeError, ValueError):
                # 异常值不误杀：跳过闸门走 LLM 判断
                logger.warning("abs_cosine 异常值 %r，跳过分数闸门走 LLM", top1_abs)

        # ── 层 3 多信号融合：分数达标（或字段缺失）→ LLM 判模糊地带 ──
        try:
            # 只传前 5 个文档摘要给 LLM 检查，避免超出上下文窗口
            docs_summary = "\n".join(
                f"- [{i + 1}] {d.get('title', '')}: {d.get('content', '')[:200]}"
                for i, d in enumerate(documents[:5])
            )
            client = LLMFactory.get_client(
                self._provider, temperature=self._reflection_temperature,
            )
            # module-055: prompt 变体注入（默认 _CHECK_PROMPT 零回归）
            check_prompt = prompt if prompt is not None else _CHECK_PROMPT
            formatted = check_prompt.format(query=query, docs_summary=docs_summary)
            response = await client.generate(formatted)
            result = self._parse_check(response)

            # ── 层 2 自洽性检查（配置开关，默认关 → 零额外调用）──
            if settings.sufficiency_self_check_enabled:
                second_client = LLMFactory.get_client(
                    self._provider, temperature=self._self_check_temperature,
                )
                response2 = await second_client.generate(formatted)
                result2 = self._parse_check(response2)
                if result.get("sufficient") != result2.get("sufficient"):
                    logger.warning(
                        "自洽性检查不一致（温度 %.1f vs %.1f），保守判充分",
                        self._reflection_temperature,
                        self._self_check_temperature,
                    )
                    return {"sufficient": True,
                            "reason": "自洽性检查两次判断不一致，保守判充分"}

            logger.info("反思结果: sufficient=%s, rewritten=%s",
                        result.get("sufficient"),
                        result.get("rewritten_query", "无"))
            return result
        except Exception as e:
            # 反思失败时默认"充分"，避免反复重试导致无限循环
            logger.warning("反思检查失败，默认充分: %s", e)
            return {"sufficient": True, "reason": f"反思检查异常，默认通过: {e}"}

    async def generate_answer(
        self,
        query: str,
        documents: list[dict],
        history: Optional[list[dict]] = None,
        memory: str = "",
        scratchpad: Optional[list[str]] = None,
    ) -> str:
        """基于文档生成带引用的回答

        支持传入对话历史（history），保证多轮对话的上下文连贯性。
        例如用户先问"G1 GC的核心创新是什么"，再问"它和CMS有什么区别"，
        第二问的 prompt 中会包含第一问的对话记录，LLM 能理解"它"的指代。

        memory（module-023）：跨会话长期记忆片段，命中时以"历史记忆: ..."
        拼入生成 prompt；为空时不生成记忆段，行为与之前完全一致（零回归）。

        scratchpad（module-041）：Agent note_to_self 工具记录的工作笔记列表，
        非空时以"[工作笔记]"段拼入 prompt；为空时零回归。

        Args:
            query: 用户当前问题
            documents: 检索到的文档列表
            history: 历史对话列表，每项 {"role": str, "content": str}
            memory: 长期记忆文本片段（无记忆时为空字符串）
            scratchpad: Agent 工作笔记列表（无笔记时为 None 或空列表）
        """
        if not documents:
            return "抱歉，未检索到相关信息。"

        try:
            # 构造历史对话上下文
            # 只取最近 6 条消息（约 3 组问答），避免 prompt 过长
            history_section = ""
            if history:
                lines = []
                for msg in history:
                    role = "用户" if msg.get("role") == "user" else "AI助手"
                    lines.append(f"{role}: {msg.get('content', '')}")
                if lines:
                    history_section = "历史对话:\n" + "\n".join(lines[-6:]) + "\n"

            # 组装文档详情，每条带 [N] 引用编号
            docs_detail = "\n\n".join(
                f"[{i + 1}] {d.get('title', '')}\n来源: {d.get('source', '')}\n内容: {d.get('content', '')}"
                for i, d in enumerate(documents)
            )
            client = LLMFactory.get_client(
                self._provider, temperature=self._generation_temperature,
            )
            # module-041: 构造 scratchpad 段落（Agent 工作笔记）
            scratchpad_section = ""
            if scratchpad:
                lines = [f"  {i+1}. {n}" for i, n in enumerate(scratchpad)]
                scratchpad_section = f"\n[工作笔记 - Agent 推理过程中的关键发现]\n" + "\n".join(lines) + "\n"
            # 合并历史段、scratchpad 段与记忆段：三者为空时 sections=""（module-058
            # WP-B 区块顺序已定稿：sections → docs → query，无多余内容）
            sections = history_section + scratchpad_section + (f"{memory}\n" if memory else "")
            prompt = _GENERATE_PROMPT.format(
                query=query,
                docs_detail=docs_detail,
                sections=sections,
            )
            response = await client.generate(prompt)
            return response
        except Exception as e:
            logger.error("答案生成失败: %s", e)
            return "抱歉，回答生成时遇到问题，请稍后重试。"

    async def generate_answer_stream(
        self,
        query: str,
        documents: list[dict],
        history: Optional[list[dict]] = None,
        memory: str = "",
        scratchpad: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成答案，逐 token 产出

        与 generate_answer 逻辑相同，但使用 astream 替代 ainvoke。
        前置步骤（检索、反思）已完成，只流式传输 LLM 生成部分。
        memory（module-023）默认空串，不改变流式路径原有行为（零回归）。

        scratchpad（module-041）：Agent note_to_self 工具记录的工作笔记列表，
        非空时以"[工作笔记]"段拼入 prompt；为空时零回归。
        """
        if not documents:
            yield "抱歉，未检索到相关信息。"
            return

        try:
            history_section = ""
            if history:
                lines = []
                for msg in history:
                    role = "用户" if msg.get("role") == "user" else "AI助手"
                    lines.append(f"{role}: {msg.get('content', '')}")
                if lines:
                    history_section = "历史对话:\n" + "\n".join(lines[-6:]) + "\n"

            docs_detail = "\n\n".join(
                f"[{i + 1}] {d.get('title', '')}\n来源: {d.get('source', '')}\n内容: {d.get('content', '')}"
                for i, d in enumerate(documents)
            )

            client = LLMFactory.get_client(
                self._provider, temperature=self._generation_temperature,
            )
            # module-041: 构造 scratchpad 段落（Agent 工作笔记）
            scratchpad_section = ""
            if scratchpad:
                lines = [f"  {i+1}. {n}" for i, n in enumerate(scratchpad)]
                scratchpad_section = f"\n[工作笔记 - Agent 推理过程中的关键发现]\n" + "\n".join(lines) + "\n"
            # 合并历史段、scratchpad 段与记忆段：三者为空时 sections=""（module-058
            # WP-B 区块顺序已定稿：sections → docs → query，无多余内容）
            sections = history_section + scratchpad_section + (f"{memory}\n" if memory else "")
            prompt = _GENERATE_PROMPT.format(
                query=query,
                docs_detail=docs_detail,
                sections=sections,
            )
            async for token in client.generate_stream(prompt):
                yield token
        except Exception as e:
            logger.error("流式答案生成失败: %s", e)
            yield "抱歉，回答生成时遇到问题，请稍后重试。"

    async def verify_answer(self, answer: str, docs: list[dict]) -> dict:
        """逐句验证答案是否被检索文档支持（证据链幻觉检测，module-039）

        module-051（ADR-0010 P0-②）拆分：LLM 只负责拆句，verdict/evidence
        改由 HHEM-2.1-Open 专职裁判判定（解决同源验证 + 引用号伪验证 + 降低成本）。

        链路：
            1. LLM 拆句（_VERIFY_PROMPT 纯拆句，15s 超时）
            2. HHEM 判分（每 claim 对每文档打分 → max 分映射三态，evidence=max 分文档号）
            3. 降级链：HHEM 不可用 → LLM 全量判分（_VERIFY_LLM_PROMPT，行为与
               module-039 一致）→ LLM 也失败 → 空 claims；开关 verify_judge_model="llm"
                → 完全不加载 HHEM 直走旧逻辑（零回归开关）

        Args:
            answer: LLM 生成的答案文本（含 [N] 引用标记）
            docs: 检索到的文档列表（含 id/title/content）

        Returns:
            {
                "claims": [{"claim": str, "verdict": str, "evidence": str}, ...],
                "overall_confidence": float (0.0-1.0),
                "total_claims": int,
                "supported": int, "inferred": int, "unsupported": int,
            }
            验证失败或无文档时返回空 claims（claims=[], overall_confidence=0.0,
            total_claims=0, supported=0, inferred=0, unsupported=0）
        """
        empty_result = {
            "claims": [],
            "overall_confidence": 0.0,
            "total_claims": 0,
            "supported": 0,
            "inferred": 0,
            "unsupported": 0,
        }
        if not docs:
            return empty_result
        if not answer or not answer.strip():
            return empty_result

        try:
            # 组装完整文档内容（非截断——降级路径 LLM 判分需要全文上下文判断依据）
            docs_text = "\n\n".join(
                f"[{i + 1}] {d.get('title', '')}\n来源: {d.get('source', '')}\n内容: {d.get('content', '')}"
                for i, d in enumerate(docs)
            )
            doc_count = len(docs)

            if settings.verify_judge_model == "hhem":
                # ── HHEM 模式：LLM 拆句 → HHEM 判分（module-051 主路径）──
                client = LLMFactory.get_client(self._provider, temperature=0)
                prompt = _VERIFY_PROMPT.format(answer=answer)
                response = await asyncio.wait_for(client.generate(prompt), timeout=15)
                claims = self._parse_claims(response)
                if claims:
                    judged = await self._judge_by_hhem(claims, docs)
                    if judged is None:
                        # HHEM 不可用（缺失/加载失败/推理异常）→ 回退 LLM 判分
                        claims = await self._judge_by_llm(answer, docs_text)
                    else:
                        claims = judged
                # claims 为空：HHEM 不调用，返回空结果（现有降级哲学）
            else:
                # ── 开关 "llm"：完全不加载 HHEM，直走旧逻辑（零回归开关）──
                claims = await self._judge_by_llm(answer, docs_text)

            # 校验 evidence 引用号：越界则降级为 unsupported
            #（HHEM 路径 max 分来源文档天然不越界，此为兼容旧 claims 结构；
            #  LLM 路径防引用号编造）
            for c in claims:
                evidence = c.get("evidence", "")
                ref_match = None
                if evidence and evidence.startswith("[") and evidence.endswith("]"):
                    try:
                        ref_match = int(evidence[1:-1])
                    except ValueError:
                        pass
                if ref_match is not None and (ref_match < 1 or ref_match > doc_count):
                    c["verdict"] = "unsupported"
                    c["evidence"] = "N/A"

            supported = sum(1 for c in claims if c.get("verdict") == "supported")
            inferred = sum(1 for c in claims if c.get("verdict") == "inferred")
            unsupported = sum(1 for c in claims if c.get("verdict") == "unsupported")
            total = len(claims)
            overall_confidence = 1.0 - (unsupported / total) if total > 0 else 0.0

            logger.info(
                "验证完成: total=%d, supported=%d, inferred=%d, unsupported=%d, confidence=%.2f",
                total, supported, inferred, unsupported, overall_confidence,
            )
            return {
                "claims": claims,
                "overall_confidence": round(overall_confidence, 4),
                "total_claims": total,
                "supported": supported,
                "inferred": inferred,
                "unsupported": unsupported,
            }
        except asyncio.TimeoutError:
            logger.warning("verify_answer 超时 (15s)，返回空 claims")
            return empty_result
        except Exception as e:
            logger.warning("verify_answer 失败，返回空 claims: %s", e)
            return empty_result

    async def _judge_by_hhem(self, claims: list[dict], docs: list[dict]) -> Optional[list[dict]]:
        """HHEM 专职裁判：每 claim 对每篇文档打分，max 分映射三态（module-051）

        映射（阈值读配置，ADR-0010 P0-②）：
            max_score ≥ verify_hhem_threshold_high (0.7) → supported
            verify_hhem_threshold_low (0.3) ≤ max_score < 0.7 → inferred
            max_score < 0.3 → unsupported
        evidence = max 分对应文档号（1-based，与现结构一致）；unsupported 填 "N/A"。

        module-055 交叉对数上限（E2E 实测数据支撑，见 changelog）：
            - docs 每 claim 上限 _MAX_HHEM_DOCS（按传入顺序取前 N——文档已按
              相关度排序，最相关文档承载证据的概率最高；丢弃尾部文档的代价是
              该 claim 证据可能只存在于尾部 → verdict 从严，保守方向）
            - claims 上限 _MAX_HHEM_CLAIMS（防超长答案拆句爆炸，正常答案
              3-8 条不触达）
        上限在降级链之前（旧格式兼容判断之后）生效。

        Args:
            claims: LLM 拆句结果 [{"claim": str}, ...]
            docs: 检索到的文档列表

        Returns:
            带 verdict/evidence 的 claims；HHEM 不可用（缺失/推理异常）→ None
            （由 verify_answer 降级 LLM 判分）
        """
        try:
            # 兼容旧格式（module-039 存量语义 + 防双重判定）：LLM 未听新 prompt 指令、
            # 返回了带 verdict 的旧结构 claims → 视为已预判结果直接采用，不再由 HHEM
            # 重复判定（证据号越界校验由 verify_answer 统一兜底）
            if any("verdict" in c for c in claims):
                return claims

            from rag.retrieval.factcheck_judge import hhem_judge

            # module-055：交叉对数上限（实测 15 对冷启动 ≈9s、E2E 负载下 12s+
            # 贴近 15s 超时哲学致级联超时 → verified_claims=0；上限后典型 10 对）
            claims = claims[:_MAX_HHEM_CLAIMS]
            docs = docs[:_MAX_HHEM_DOCS]

            high = settings.verify_hhem_threshold_high
            low = settings.verify_hhem_threshold_low
            # module-055: 父块全文截断（防超 512 token 上限 + 提速，见常量注释）
            doc_texts = [(d.get("content") or "")[:_MAX_HHEM_DOC_CHARS]
                         for d in docs]
            n_docs = len(doc_texts)
            claim_texts = [c["claim"] for c in claims]
            # 交叉构造 (doc, claim) 对：先固定 claim 再遍历 docs（每 claim 段长度 = n_docs）
            flat_docs = [t for _ in claim_texts for t in doc_texts]
            flat_claims = [c for c in claim_texts for _ in doc_texts]
            scores = await hhem_judge.predict(flat_docs, flat_claims)
            if scores is None:
                return None
            if len(scores) != len(flat_claims):
                logger.warning("HHEM 返回分数数量异常（%d vs %d），降级 LLM 判分",
                               len(scores), len(flat_claims))
                return None

            judged = []
            for j, c in enumerate(claims):
                seg = scores[j * n_docs:(j + 1) * n_docs]
                best_idx = max(range(n_docs), key=lambda i: seg[i])
                max_score = seg[best_idx]
                if max_score >= high:
                    verdict, evidence = "supported", f"[{best_idx + 1}]"
                elif max_score >= low:
                    verdict, evidence = "inferred", f"[{best_idx + 1}]"
                else:
                    verdict, evidence = "unsupported", "N/A"
                judged.append({**c, "verdict": verdict, "evidence": evidence})
            return judged
        except Exception as e:
            logger.warning("HHEM 判定失败，降级 LLM 判分: %s", e)
            return None

    async def _judge_by_llm(self, answer: str, docs_text: str) -> list[dict]:
        """LLM 判分降级路径（module-051：HHEM 不可用 / 开关 "llm" 时使用）

        复用旧全量 prompt _VERIFY_LLM_PROMPT（拆句 + 判分 + evidence 一步完成），
        返回结构与 module-039 完全一致（降级路径行为不漂移）。
        evidence 引用号越界校验由 verify_answer 统一兜底。

        Args:
            answer: LLM 生成的答案文本
            docs_text: 完整文档上下文（"[N] 标题\n内容" 拼接）

        Returns:
            带 verdict/evidence 的 claims（可能为空）

        Raises:
            asyncio.TimeoutError / Exception: 由 verify_answer 外层兜底返回空 claims
        """
        client = LLMFactory.get_client(self._provider, temperature=0)
        prompt = _VERIFY_LLM_PROMPT.format(docs_text=docs_text, answer=answer)
        response = await asyncio.wait_for(client.generate(prompt), timeout=15)
        return self._parse_verification(response)

    @staticmethod
    def _parse_verification(response: str) -> list[dict]:
        """解析 LLM 返回的验证 JSON 数组

        与 _parse_check 类似，处理 LLM 输出中的非 JSON 杂质。
        """
        try:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(response[start:end + 1])
                if isinstance(parsed, list):
                    return [
                        {
                            "claim": item.get("claim", ""),
                            "verdict": item.get("verdict", "unsupported"),
                            "evidence": item.get("evidence", "N/A"),
                        }
                        for item in parsed
                        if isinstance(item, dict) and item.get("claim")
                    ]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("解析验证结果失败: %s", e)
        return []

    @staticmethod
    def _parse_claims(response: str) -> list[dict]:
        """解析 LLM 拆句 JSON 数组（module-051：纯拆句，不判 verdict）

        输出 [{"claim": str}, ...]（verdict/evidence 由 HHEM 判定后填充）。
        与 _parse_verification 类似处理非 JSON 杂质；容忍 dict 项（取 claim 字段）
        与空白过滤；dict 项若已带 verdict/evidence（LLM 未听指令返回旧格式），
        原样保留（缺 evidence 时补默认 "N/A"）——_judge_by_hhem 据此判定为
        预判结果直接采用（兼容 module-039 存量语义，防双重判定）。
        """
        try:
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(response[start:end + 1])
                if isinstance(parsed, list):
                    claims = []
                    for item in parsed:
                        text = item if isinstance(item, str) else (
                            item.get("claim", "") if isinstance(item, dict) else "")
                        text = text.strip()
                        if text:
                            claim = {"claim": text}
                            if isinstance(item, dict):
                                if "verdict" in item:
                                    claim["verdict"] = item.get("verdict", "unsupported")
                                    # 旧格式可能缺 evidence 键：默认 "N/A"（对齐
                                    # _parse_verification），前端 parseEvidenceRef 不抛错
                                    claim["evidence"] = item.get("evidence", "N/A")
                                if "evidence" in item:
                                    claim["evidence"] = item.get("evidence", "N/A")
                            claims.append(claim)
                    return claims
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("解析拆句结果失败: %s", e)
        return []

    @staticmethod
    def _parse_check(response: str) -> dict:
        """解析 LLM 返回的检查结果 JSON

        与 router.py 的 _parse_response 类似，处理 LLM 输出中的
        非 JSON 杂质（markdown 包裹、多余文字等）。
        """
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                result = json.loads(response[start:end + 1])
                sufficient = bool(result.get("sufficient", True))
                output = {"sufficient": sufficient, "reason": result.get("reason", "")}
                if not sufficient:
                    output["rewritten_query"] = result.get("rewritten_query", "")
                return output
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("解析反思结果失败: %s", e)
        return {"sufficient": True, "reason": "解析失败，默认充分"}


# 全局单例
reflector = Reflector()
