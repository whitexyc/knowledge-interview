"""
意图识别路由 (Router Agent) — RAG 链路第一关
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置：
  用户 Query → [Router Agent] 分类
                  ├─ knowledge  ──→ 知识库检索路径（主力路径）
                  ├─ casual_chat ──→ 直接 LLM 回答（跳过检索）
                  └─ realtime    ──→ 🔜 实时数据（未实现）

为什么需要意图路由？
  如果没有路由，每个问题都会走"检索→反思→生成"全链路。
  对于"你好"这种闲聊，检索知识库不仅浪费算力，还可能返回无关内容
  污染 LLM 的上下文。路由让系统只在必要时进行检索。

设计决策：
  LLM-as-Classifier：用 LLM 做分类而不是传统 ML 分类器。
  原因是：
  1. 传统分类器需要标注数据训练，维护成本高
  2. LLM zero-shot 即可分类，且能给出推理理由
  3. 新增类别无需重新训练，只需修改 prompt

  保守策略：当 LLM 分类失败或超时时，默认返回 "knowledge" 意图。
  宁可多检索一次，也不要漏检。这是"安全优先"的设计。

多轮对话（module-063 / ADR-0015）：
  单句识别会漏掉省略句/指代句（"为什么""那图谱呢"）——本类升级为会话级
  路由：classify(query, history) 取最近 4-6 轮，LLM prompt 拼对话上下文判断；
  短句（去语气词后 <6 字符且无 FTS/图谱/规则特征）继承上一轮 intent（规则层
  零 LLM，_short_inherit）；上轮工具轨迹含知识检索/生成 → 短 query 强制
  knowledge（_KB_TOOL_NAMES）。空 history 行为与改动前逐字一致（零回归）。

L2 前置校验（module-043 / ADR-0003 修订版，module-055 扩展触发）：
  LLM 判定 intent≠knowledge（无论置信高低，module-055：低置信限制在
  module-054 E2E 暴露缺口——LLM 高置信误判 casual_chat 同样漏检）时，用
  **确定性信号**确认是否涉及知识库——与 LLM 完全无关（同源复核已否决，
  红线：确认路径零 LLM）：
    ① FTS 术语命中：jieba 分词 query → documents.search_tokens 倒排匹配
       （复用 module-020 中文 FTS 通道），命中 ≥1 知识库专有术语 → 确认。
       术语需有专有术语特征（module-055：golden 扫描实测 20 个噪声词——
       今天/问题/怎么样/最近等知识库文档常见词——补入 _FUNCTION_STOPWORDS，
       消除误确认，扫描 50 条闲聊/实时样本误确认 0 条）
    ② 图谱实体命中：图谱 Entity 名称出现在 query 中 → 确认（Cypher 拉实体
       名后 Python 子串匹配，全程无 LLM——不走 graph_extractor，其依赖 LLM）。
       实测判别力最强：golden 50 条非 knowledge 样本 0 误命中
    ③ 规则表：明确闲聊/实时特征词（"几点""天气""你是谁"），命中 → 保持原判
       （否决确认信号，防止"现在""天气"等常见词在知识库文档中的巧合命中误转）。
       module-055：提前到信号查询前短路（无条件触发后规则表命中零 DB 开销）
  任何异常 → 保守 knowledge（宁多检不漏检）。

L4 分类器（module-043 / ADR-0003，module-056 达标启用）：
  bge-m3 冻结特征 + 逻辑回归头（intent_classifier.py）可插拔注入
  （构造器注入 / 配置开关惰性加载）；module-056 起默认启用（L4 为决策
  主体），模型缺失/加载/推理失败一律回退 LLM 分类，零影响；
  PW_INTENT_CLASSIFIER_ENABLED=false 保持纯 LLM 路径。
"""
import json
import logging
from typing import Optional

from llm.client import LLMFactory
from src.config import settings

logger = logging.getLogger(__name__)

# ── L2 前置校验配置（module-043 / ADR-0003 修订版，module-055 扩展） ──
# 触发条件：intent≠knowledge 无条件触发（module-055：原"且 LLM 低置信"
# 限制在 module-054 E2E 暴露缺口——LLM 高置信误判 casual_chat 直接漏检；
# 确定性信号便宜且精确（golden 50 条非 knowledge 样本误确认 0），
# 规则表否决闲聊/实时特征词，任何异常保守 knowledge，扩展零风险）。
# 常量保留以记录历史触发口径（低置信曾被用作"不放心"信号）。
_L2_CONFIDENCE_THRESHOLD = 0.5

# 规则表：明确闲聊/实时特征词，命中任一 → 保持原判（不修正为 knowledge）。
# 只收录几乎不可能出现在知识库问题中的词——"时间/温度"等词会误伤
# "停顿时间模型""温度监控"类知识库问题，不收录。
_RULE_TABLE = (
    # 实时：时间类（"几点" 无歧义，不会出现在知识库问题中）
    "几点", "几点了", "几点钟", "现在几点", "几号", "今天几号", "星期几",
    "今天星期", "周几", "今天周",
    # 实时：天气类
    "天气", "气温", "下雨", "下雪", "晴天", "阴天", "台风", "刮风",
    # 闲聊：身份/问候/寒暄
    # module-045 WP2a: 移除"你能做什么/你会什么"——golden 边界样本
    # "你能做什么？这个系统能帮我解决什么问题？" 标注 knowledge（问系统能力
    # 而非闲聊），规则表命中会否决 FTS/图谱确认信号（rule_veto），误伤边界样本
    "你是谁", "你叫什么", "你多大了", "介绍一下你自己",
    "你好", "您好", "嗨", "哈喽", "hello", "hi ", "在吗", "在不在", "再见", "拜拜",
    "谢谢", "感谢", "晚安", "早安", "辛苦了", "哈哈", "嗯嗯", "好的好的",
)

# 中文语气词规则表（module-063 / WP-B 短句意图继承）：
# 先做最常用 8 个（task-brief §八.6），不追求全。去语气词后再判短句长度——
# "为什么呀"→"为什么"（否则语气词干扰长度判定）。"请问"亦在
# _FUNCTION_STOPWORDS（FTS 术语过滤），此处是"短句长度判定前去语气词"，
# 语义不同（一个防 FTS 误命中、一个防长度误判）。
_PARTICLE_WORDS = ("哦", "呢", "呀", "啦", "请问", "那个", "嘛", "吧")

# 工具历史信号（module-063 / WP-D）：上一轮 tool_calls 含下述任一工具 →
# 本轮短 query 强制 knowledge（工具轨迹是意图的强信号，确定性零 token）。
_KB_TOOL_NAMES = ("search_knowledge", "generate_answer")

# 短句继承递归深度上限（module-063 / WP-B）：省略句链式继承（"为什么"前
# 又是"为什么"）逐层回退路由上一轮，防无限递归。
_INHERIT_MAX_DEPTH = 3

# 高频功能词/代词/疑问词：不计入 FTS 术语命中。"什么""怎么""区别"等词在
# 知识库文档中广泛存在，命中无判别力；只保留有专有术语特征的词参与确认。
# module-055 数据驱动扩充（2026-08-12 golden 扫描实测）：L2 无条件触发后，
# 用 _deterministic_confirm 对 golden 50 条闲聊/实时样本逐条模拟，实测 20 条
# 被下述噪声词 FTS 命中误确认（今天/问题/怎么样/最近…在知识库文档中普遍
# 存在），补入后扫描误确认归零。只补"知识库文档常见但无判别力"的词，
# 不收录技术术语（G1/JVM/Region 等保留参与确认）。
_FUNCTION_STOPWORDS = frozenset((
    "什么", "怎么", "为什么", "哪些", "哪个", "如何", "请问", "知道", "可以",
    "是不是", "区别", "关系", "原理", "作用", "特点", "介绍", "了解", "解释",
    "说下", "讲讲", "情况", "时候", "什么时候", "然后", "这个", "那个", "这样",
    "那样", "我们", "你们", "他们", "咱们", "自己", "现在", "东西", "回事",
    "究竟", "到底", "为什么", "是", "的", "了", "吗", "呢", "吧", "啊", "呀",
    "哦", "喔", "嗯", "哈", "喂", "你", "我", "他", "她", "它", "您", "有",
    "没", "在", "和", "与", "及", "就", "都", "也", "很", "太",
    # module-055 噪声词（golden 扫描实测两轮，见上注释；第二轮补入
    # 剩余误确认样本的命中术语——简历类文档含闲聊词（电影/人民币等），
    # FTS 信号固有噪声，停用词表是数据驱动的防护）：
    "今天", "明天", "最近", "怎么样", "问题", "名字", "一下", "随便", "厉害",
    "没关系", "没错", "明白", "换个", "上午", "下午", "今年", "距离", "外面",
    "新闻", "股市", "行情", "汇率", "流行", "周末", "心情", "工作",
    "早上好", "不错", "聊聊", "好累", "下次", "注意", "猜猜", "干嘛", "话题",
    "还是", "几年", "还有", "几天", "人民币", "多少", "热门", "电影",
))

# 意图分类的 prompt 模板
# 设计要点：
# 1. 要求 LLM 只返回 JSON，纯文本会破坏下游解析
# 2. 给出 3 个明确的类别定义，每个带具体例子
# 3. 要求返回 confidence 分数，便于下游做阈值判断
# 4. 要求返回 reason，便于调试和审计
_PROMPT_TEMPLATE = """你是一个问题分类器。判断用户问题的意图，只返回 JSON。

类别定义：
- knowledge: 询问知识库中的信息、文档内容、专业知识等（需要检索）
- casual_chat: 日常聊天、问候、寒暄等（不需要检索）
- realtime: 查询实时数据、当前时间、天气等（需要实时数据源）

用户问题: {query}

返回格式（只返回 JSON，不要其他文字）:
{{"intent": "knowledge|casual_chat|realtime", "confidence": 0.0-1.0, "reason": "简短原因"}}"""

# 多轮对话上下文块（module-063 / WP-A）：有 history 时拼在 _PROMPT_TEMPLATE
# 之后。强调"省略句/指代句结合上下文判断、含义完整句按字面判断"——防止多轮
# 闲聊被错误归因到 knowledge（话题漂移防护的 LLM 侧补充；规则层由
# _deterministic_confirm 的 rule_veto 兜底）。
_MULTITURN_CONTEXT = """

对话历史（最近 {n} 轮）:
{history}

注意：如果当前问题是省略句/指代句（如"为什么""那它呢"，缺少主语或指代前文），
请结合最近对话上下文判断意图；如果问题本身含义完整（即使是短句），按字面判断，
不要因为前文是知识库问题就强行归为 knowledge。
"""


async def fts_term_hit(query: str) -> bool:
    """FTS 术语命中（模块级公开函数，module-049 分诊复用；L2 确认同源）

    语义与 RouterAgent._fts_term_hit 完全一致（L2 逻辑单一来源，不复制）：
    jieba 分词（_kb_terms：过滤功能词/单字）→ 逐术语查 documents.search_tokens
    FTS 倒排（plainto_tsquery @@，大小写不敏感）→ 任一命中即 True。
    只查知识库文档（排除 memory:% 记忆文档）。

    Args:
        query: 用户问题

    Returns:
        命中 ≥1 知识库专有术语 → True
    """
    terms = RouterAgent._kb_terms(query)
    if not terms:
        return False
    from sqlalchemy import text
    from src.database import async_session_factory
    async with async_session_factory() as session:
        for term in terms:
            row = await session.execute(text("""
                SELECT 1 FROM documents
                WHERE search_tokens IS NOT NULL
                  AND parent_id IS NOT NULL
                  AND (source IS NULL OR source NOT LIKE 'memory:%')
                  AND to_tsvector('simple', search_tokens)
                      @@ plainto_tsquery('simple', :term)
                LIMIT 1
            """), {"term": term})
            if row.scalar_one_or_none() is not None:
                logger.info("FTS 术语命中: term=%s, query=%s", term, query[:50])
                return True
    return False


class RouterAgent:
    """意图识别路由器

    使用 LLM 对用户问题进行 zero-shot 分类（L4 分类器启用时替换决策主体）。
    实例化时可指定 provider，默认使用 settings.llm_provider。
    module-043：可注入 L4 意图分类器（intent_classifier，bge-m3+逻辑回归）；
    module-056 起默认启用（L4 为决策主体，失败回退 LLM）；LLM 路径结果
    走 L2 确定性信号确认（见模块 docstring）。
    """

    def __init__(self, provider: Optional[str] = None,
                 intent_classifier: Optional[object] = None):
        self._provider = provider  # None = 用默认 provider
        # L4 可插拔分类器：显式注入优先（测试/定制）；None 时若配置开关开启
        # 则惰性加载一次，失败回退 LLM（零影响）
        self._intent_classifier = intent_classifier
        self._classifier_tried = intent_classifier is not None

    async def _get_classifier(self):
        """L4 分类器获取：注入优先；未注入且开关开启 → 惰性加载一次

        Returns:
            可用的分类器（有 predict_proba(query) -> dict[str, float]），
            不可用返回 None（回退 LLM 分类）
        """
        if self._intent_classifier is None and not self._classifier_tried:
            self._classifier_tried = True
            if settings.intent_classifier_enabled:
                try:
                    from agent.intent_classifier import IntentClassifier
                    clf = IntentClassifier()
                    if await clf.load():
                        self._intent_classifier = clf
                        logger.info("L4 意图分类器已加载: %s", clf.model_path)
                except Exception as e:
                    logger.warning("L4 分类器加载失败，回退 LLM 分类: %s", e)
        return self._intent_classifier

    async def classify(self, query: str, history: Optional[list] = None,
                       tool_history: Optional[list] = None) -> dict:
        """判断问题意图（module-063：会话级路由 + 短句意图继承）

        内部使用 LLM.generate() 发送 prompt 给 LLM，让 LLM 返回 JSON。
        使用 generate（单轮）而不是 chat（多轮），因为分类不需要上下文。
        module-043 增强：
          - L4 分类器可用时用它替换 LLM 决策主体（校准概率，无 LLM 调用）
          - LLM 判定 intent≠knowledge（module-055 起无条件，不再限低置信）时
            走 L2 确定性信号确认（FTS 术语/图谱实体/规则表），命中 → 修正为
            knowledge（module-054 E2E 实测：LLM 高置信误判 casual_chat 也会
            漏检，低置信限制不可靠）
        module-063（ADR-0015）增强：
          - 会话级路由：history 取最近 4-6 轮（内部 history[-6:]），LLM prompt
            拼对话上下文（省略句/指代句结合上下文判断）；空/None history →
            行为与改动前逐字一致（零回归）
          - 短句意图继承（WP-B，规则层零 LLM）：去语气词后长度 <6 且
            _deterministic_confirm 无新特征 → 继承上一轮 intent（从 history
            最近一条 user 消息推演，无状态）
          - 工具历史信号（WP-D）：tool_history 含 search_knowledge/generate_answer
            → 短 query 强制 knowledge（轨迹不可得 None → 跳过）

        Args:
            query: 用户问题
            history: 最近对话历史消息列表（[{"role","content"}, ...]），可选
            tool_history: 上一轮工具调用名列表（agent 轨迹，不可得传 None）

        Returns:
            {"intent": str, "confidence": float, "reason": str}
            异常时默认返回 knowledge（保守策略）
        """
        if not query or not query.strip():
            return {"intent": "knowledge", "confidence": 0.0, "reason": "空查询，默认走知识库"}

        # 路由只用最近 4-6 轮（task-brief §八.5：历史不全塞，更早轮次几乎
        # 不影响指代还费 token）；空 history → 与现状逐字一致零回归。
        history = list(history or [])[-6:]

        # ── WP-B/WP-D 规则层短路（零 LLM）：短句继承 / 工具信号 ──
        inherited = await self._short_inherit(query, history, tool_history)
        if inherited is not None:
            return inherited

        return await self._classify_core(query, history)

    async def _classify_core(self, query: str, history: list) -> dict:
        """核心分类：L4 分类器路径 → LLM 路径 + L2 确定性信号确认

        module-063（WP-A）：history 非空时 LLM prompt 拼对话上下文块
        （_build_prompt）；L4 分类器在 settings.intent_classifier_multi_turn
        开启时拼接最近一轮 user query 向量（2048 维，需多轮重训模型——当前
        落盘 intent_clf.joblib 为单 query 1024 维训练，拼接会触发 sklearn
        维度不匹配抛异常 → 捕获回退 LLM，fail-open 零回归）。
        （2026-08-16 架构评估：多轮拼接已降级不做——能力已被 LLM 路径
        （省略句 12/12 全对）+ WP-B 规则层覆盖，性价比不足，见 METRICS
        待办区 #8；代码 fail-open 保留不删，开关恒 false）

        Args:
            query: 用户问题
            history: 最近对话历史（可能为空列表）

        Returns:
            {"intent": str, "confidence": float, "reason": str}
        """
        # ── L4 分类器路径（module-043）：可插拔注入，失败回退 LLM ──
        classifier = await self._get_classifier()
        if classifier is not None:
            try:
                if settings.intent_classifier_multi_turn:
                    last_content, _ = self._last_user_turn(history)
                    if last_content is not None:
                        # 拼接最近一轮 user query 向量（2048 维，训练时同构）
                        probs = await classifier.predict_proba(
                            query.strip(), prev_user_query=last_content)
                    else:
                        probs = await classifier.predict_proba(query.strip())
                else:
                    probs = await classifier.predict_proba(query.strip())
                intent = max(probs, key=probs.get)
                # module-045 WP2d: L4 分类器返回的 intent 过白名单——非法值
                # 归 knowledge（与 LLM 路径 _parse_response 口径一致，防模型
                # 类别外漂移导致路由落入未知分支）
                if intent not in ("knowledge", "casual_chat", "realtime"):
                    intent = "knowledge"
                # module-063（ADR-0015）：L4 路径同样走 L2 确定性信号确认
                #（与 LLM 路径同款安全网）——L4 单句分类无历史上下文，多轮
                # 省略句（如"怎么解决呢"）可能被误判 casual/realtime（eval/
                # golden_multi_turn 实测暴露）；确定性信号（FTS/图谱命中）修正
                # 为 knowledge（零 LLM，红线不变；module-055 已证信号精确——
                # golden 50 条非 knowledge 样本误确认 0）
                if intent != "knowledge":
                    confirmed, signal = await self._deterministic_confirm(query.strip())
                    if confirmed:
                        logger.info("L2 信号确认(%s)，L4 intent 修正为 knowledge: query=%s",
                                    signal, query[:50])
                        intent = "knowledge"
                # module-048 WP5: probs 缺键防御——白名单修正为 knowledge 后
                # 该键可能不存在（真实分类器缺 knowledge 键），回退默认置信度
                # 0.0，不抛 KeyError（缺键时语义等同于"该意图无置信度"）；
                # L2 修正为 knowledge 后置信度取 knowledge 概率（可能非最高分）
                confidence = probs.get(intent, 0.0)
                logger.info("意图识别(L4): query=%s, intent=%s, confidence=%.2f",
                            query[:50], intent, confidence)
                return {"intent": intent, "confidence": round(confidence, 4),
                        "reason": f"L4 classifier {probs}"}
            except Exception as e:
                logger.warning("L4 分类器推理失败，回退 LLM 分类: %s", e)

        try:
            client = LLMFactory.get_client(self._provider)
            prompt = self._build_prompt(query.strip(), history)
            response = await client.generate(prompt)
            result = self._parse_response(response)

            # ── L2 前置校验（module-043 / ADR-0003 修订版，module-055 扩展）──
            # 触发：intent≠knowledge 无条件触发（module-055：原"且低置信"
            # 限制在 module-054 E2E 暴露缺口——LLM 高置信误判 casual_chat
            # 直接漏检；确定性信号便宜且精确，规则表否决闲聊/实时特征词，
            # 任何异常保守 knowledge，扩展零风险）。确认动作是确定性信号
            # （_deterministic_confirm），与 LLM 完全无关（红线：零 LLM）。
            if result.get("intent") != "knowledge":
                confirmed, signal = await self._deterministic_confirm(query.strip())
                if confirmed:
                    logger.info("L2 信号确认(%s)，intent 修正为 knowledge: query=%s",
                                signal, query[:50])
                    original_reason = result.get("reason", "")
                    result["intent"] = "knowledge"
                    result["reason"] = f"L2 信号确认({signal})，宁多检不漏检" + (
                        f" | 原判: {original_reason}" if original_reason else "")
                else:
                    logger.info("L2 无确认信号(%s)，保持原判 %s: query=%s",
                                signal, result.get("intent"), query[:50])
            logger.info("意图识别: query=%s, intent=%s, confidence=%.2f",
                        query[:50], result.get("intent"), result.get("confidence", 0))
            return result
        except Exception as e:
            # 任何异常都保守地返回 knowledge
            logger.warning("意图识别失败，默认走知识库: %s", e)
            return {"intent": "knowledge", "confidence": 0.0, "reason": f"LLM 分类失败，保守路由: {e}"}

    # ── 多轮意图路由（module-063 / ADR-0015）：短句继承 + 工具信号 + 上下文 ──

    async def _short_inherit(self, query: str, history: list,
                             tool_history: Optional[list]) -> dict | None:
        """WP-B 短句意图继承 + WP-D 工具历史信号（规则层零 LLM，LLM/分类器前短路）

        条件（全部满足才继承）：
          ① history 非空（空历史不继承——单轮短 query 正常路由）
          ② 去除语气词后长度 < 6 字符
          ③ _deterministic_confirm 无新特征（FTS 术语/图谱实体未命中、规则表
             未命中）——有特征必须正常路由（防话题漂移，"今天天气"靠规则表
             rule_veto 挡住不继承）
        满足则继承上一轮 intent：路由 history 最近一条 user 消息（其之前的历史
        递归，省略句可链式继承；深度上限 _INHERIT_MAX_DEPTH 防无限递归）。
        WP-D 工具信号优先：上一轮 tool_calls 含 search_knowledge/generate_answer
        → 短 query 强制 knowledge（工具轨迹不可得 tool_history=None → 跳过）。

        Args:
            query: 用户当前问题
            history: 最近对话历史（已按 [-6:] 截断）
            tool_history: 上一轮工具调用名列表（不可得 None）

        Returns:
            继承结果 dict（intent/confidence/reason）；不继承返回 None（正常路由）
        """
        if not history:
            return None
        stripped = self._strip_particles(query)
        if len(stripped) >= 6:
            return None

        # WP-D 工具历史信号：上轮走知识检索/生成 → 强制 knowledge
        if tool_history:
            if any(t in _KB_TOOL_NAMES for t in tool_history):
                return {"intent": "knowledge", "confidence": 0.0,
                        "reason": "工具历史信号：上轮走知识检索/生成，短 query 强制 knowledge"}

        # 无新特征判定（复用 _deterministic_confirm，FTS/图谱/规则表）
        try:
            confirmed, signal = await self._deterministic_confirm(stripped)
        except Exception:
            return None  # 保守：异常不继承，走正常路由
        if confirmed or signal == "rule_veto":
            return None  # 有特征（术语命中/规则词）→ 正常路由（防话题漂移）

        # 继承上一轮 intent：路由 history 最近一条 user 消息（无状态，从
        # history 推演；其之前的历史允许链式继承）
        last_content, prev_history = self._last_user_turn(history)
        if last_content is None:
            return None
        prev = await self._classify_prev(last_content, prev_history, depth=1)
        if prev is None:
            return None
        intent = prev.get("intent", "knowledge")
        return {"intent": intent, "confidence": prev.get("confidence", 0.0),
                "reason": f"短句意图继承（上一轮 intent={intent}）"}

    async def _classify_prev(self, query: str, history: list, depth: int) -> dict | None:
        """递归路由上一轮 user 消息（省略句链式继承的上一层）

        Args:
            query: 上一轮 user 消息内容
            history: 上一轮之前的历史
            depth: 当前递归深度（从 1 起；>= _INHERIT_MAX_DEPTH 返回 None 防无限递归）

        Returns:
            上一轮的路由结果 dict；递归超限/异常返回 None（调用方回退正常路由）
        """
        if depth >= _INHERIT_MAX_DEPTH:
            return None
        history = list(history or [])[-6:]
        inherited = await self._short_inherit(query, history, None)
        if inherited is not None:
            return inherited
        return await self._classify_core(query, history)

    def _build_prompt(self, query: str, history: list) -> str:
        """构造分类 prompt：空 history 用原模板（逐字一致）；有 history 拼上下文块

        module-063（WP-A）：上下文块只放最近 4 轮（task-brief：路由用最后 4-6 轮，
        上下文块够用不费 token），每条截断 300 字符防超长。

        Args:
            query: 用户问题
            history: 最近对话历史

        Returns:
            完整分类 prompt
        """
        prompt = _PROMPT_TEMPLATE.format(query=query)
        if not history:
            return prompt
        lines = []
        for msg in history[-4:]:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = str(msg.get("content", ""))[:300]
            lines.append(f"{role}: {content}")
        return prompt + _MULTITURN_CONTEXT.format(
            n=min(len(history), 4), history="\n".join(lines))

    @staticmethod
    def _strip_particles(query: str) -> str:
        """去除中文语气词（哦/呢/呀/啦/请问/那个/嘛/吧，规则表可配）

        短句继承前先去语气词："为什么呀"→"为什么"、"那图谱呢"→"那图谱"，
        否则语气词干扰长度判定（"为什么呀"去除前 4 字 <6 本已达标，但"今天
        天气呀"不去除会被误判为短句走继承）。

        Args:
            query: 用户问题

        Returns:
            去语气词后的 query（去空白）
        """
        q = query.strip()
        for p in _PARTICLE_WORDS:
            q = q.replace(p, "")
        return q.strip()

    @staticmethod
    def _last_user_turn(history: list) -> tuple[str | None, list]:
        """提取最近一条 user 消息及其之前的历史（无状态，从 history 推演）

        Args:
            history: 会话历史消息列表（[{"role","content"}, ...]）

        Returns:
            (last_user_content, history_before_last_user)；
            无 user 消息 → (None, None)
        """
        if not history:
            return None, None
        for i in range(len(history) - 1, -1, -1):
            msg = history[i] or {}
            if msg.get("role") == "user" and str(msg.get("content", "")).strip():
                return str(msg["content"]).strip(), history[:i]
        return None, None

    # ── L2 确定性信号确认（module-043 / ADR-0003 修订版，红线：零 LLM） ──

    async def _deterministic_confirm(self, query: str) -> tuple[bool, str]:
        """L2 确定性信号确认 — 与 LLM 完全无关（确认路径零 LLM 调用）

        信号（按优先级，FTS/图谱任一命中即确认；规则表命中保持原判）：
          ① 规则表（_rule_hits）→ 保持原判（否决 FTS/图谱的巧合命中）。
             module-055 提前到首位：L2 无条件触发后，闲聊/实时特征词命中
             直接短路零 DB 查询（原 FTS 先行属无效开销，结果不变）
          ② FTS 术语命中（_fts_term_hit）→ confirmed
          ③ 图谱实体命中（_graph_entity_hit）→ confirmed
        任何异常 → 保守 knowledge（宁多检不漏检，AC 场景 4）。

        Args:
            query: 用户问题原文

        Returns:
            (confirmed, signal)
            - confirmed=True → 修正为 knowledge
            - signal: fts_term / graph_entity / rule_veto / no_signal /
                      error_conservative（可观测：写入日志与 reason）
        """
        try:
            if self._rule_hits(query):
                # 规则表：明确闲聊/实时特征词 → 保持原判（module-055 提前短路）
                return False, "rule_veto"
            fts_hit = await self._fts_term_hit(query)
            graph_hit = await self._graph_entity_hit(query) if not fts_hit else False
            if fts_hit:
                return True, "fts_term"
            if graph_hit:
                return True, "graph_entity"
            return False, "no_signal"
        except Exception as e:
            # 信号查询失败 → 保守 knowledge（宁多检不漏检）
            logger.warning("L2 确定性信号确认异常，保守 knowledge: %s", e)
            return True, "error_conservative"

    @staticmethod
    def _kb_terms(query: str) -> list[str]:
        """从 query 提取用于 FTS 术语命中的词元（过滤功能词/单字）

        只保留长度 ≥2 且不在 _FUNCTION_STOPWORDS 中的 jieba 词元——
        "什么""区别"等词在知识库文档中广泛存在，命中无判别力。

        Args:
            query: 用户问题

        Returns:
            候选术语列表；纯闲聊（如"你好呀"）可能为空
        """
        from rag.retrieval.text_tokenizer import tokenize
        return [
            tok for tok in tokenize(query).split()
            if len(tok) >= 2 and tok not in _FUNCTION_STOPWORDS
        ]

    async def _fts_term_hit(self, query: str) -> bool:
        """① FTS 术语命中：任一知识库专有术语出现在倒排索引（search_tokens）

        module-049：实现提取为模块级函数 fts_term_hit（L2 确认与分诊共用，
        逻辑单一来源），本方法委托之，L2 确认语义不变。

        Args:
            query: 用户问题

        Returns:
            命中 ≥1 知识库专有术语 → True
        """
        return await fts_term_hit(query)

    async def _graph_entity_hit(self, query: str) -> bool:
        """② 图谱实体命中：图谱 Entity 名称（≥2 字符）出现在 query 中

        确定性实现（红线：不调 LLM）：Cypher 拉取实体名列表 → Python 子串
        匹配。不走 graph_extractor——其 extract_from_query 依赖 LLM，确认
        路径禁用。实体名是知识库专有名词（如 GC/G1/JVM），子串匹配即可。

        Args:
            query: 用户问题

        Returns:
            query 包含任一实体名 → True；图谱不可用抛异常（由调用方保守降级）
        """
        import json
        from sqlalchemy import text
        from src.database import async_session_factory
        from rag.graph.graph_store import GRAPH_NAME
        async with async_session_factory() as session:
            await session.execute(text("LOAD 'age'"))
            await session.execute(text('SET search_path = ag_catalog, "$user", public'))
            rows = (await session.execute(text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $$
                    MATCH (e:Entity) RETURN e.name LIMIT 200
                $$) AS (name agtype)
            """))).fetchall()
        for row in rows:
            if row[0] is None:
                continue
            try:
                name = json.loads(str(row[0]))
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(name, str) and len(name) >= 2 and name in query:
                logger.info("L2 图谱实体命中: entity=%s, query=%s", name, query[:50])
                return True
        return False

    @staticmethod
    def _rule_hits(query: str) -> bool:
        """③ 规则表命中：明确闲聊/实时特征词出现在 query 中 → 保持原判

        Args:
            query: 用户问题

        Returns:
            query 含任一规则词 → True
        """
        q = query.lower()
        return any(word.lower() in q for word in _RULE_TABLE)

    @staticmethod
    def _parse_response(response: str) -> dict:
        """解析 LLM 返回的 JSON

        LLM 可能返回带 markdown 包裹的 JSON（```json...```），
        也可能返回纯 JSON。这里先提取 {} 块再解析。

        为什么不用 json.loads 直接解析？
        因为 LLM 有时会在 JSON 前后加多余文字（如"好的，这是分类结果:"），
        直接解析会失败。提取 JSON 块的方式更鲁棒。
        """
        try:
            # 尝试提取 JSON 块：找到第一个 { 和最后一个 }
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                json_str = response[start:end + 1]
                result = json.loads(json_str)
                intent = result.get("intent", "knowledge")
                # 校验 intent 值是否合法，防止 LLM 胡编乱造
                if intent not in ("knowledge", "casual_chat", "realtime"):
                    intent = "knowledge"
                return {
                    "intent": intent,
                    "confidence": float(result.get("confidence", 0.5)),
                    "reason": result.get("reason", ""),
                }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("解析 LLM 响应失败: %s", e)

        return {"intent": "knowledge", "confidence": 0.0, "reason": "解析失败，默认走知识库"}


# 全局单例
router_agent = RouterAgent()
