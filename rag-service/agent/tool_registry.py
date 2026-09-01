"""
Agent 工具注册表 — ToolRegistry（module-028）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

把固定 RAG 流水线升级为 Agentic ReAct 循环的第一步：把现有检索/图/记忆/生成
方法包装成带 name/description/args_schema 的工具，供 LLM 通过 function calling
自主调度。

设计要点：
  1. 注册表无状态（只存工具定义），执行时通过 AgentTool.run(args, ctx)
     注入会话上下文 ctx（query/identity/history/累积 docs/记忆），
     故全局单例 registry 可被多会话并发复用，无共享可变状态。
  2. 工具失败由 AgentTool.run 统一捕获返回空串（降级哲学），
     LLM 自行判断是继续检索还是如实告知用户。
  3. 内置 10 个工具：
     search_knowledge / search_fts / search_vector / search_graph /
     extract_entities / recall_memory / generate_answer / verify_answer /
     re_search / note_to_self
"""
import asyncio
import hashlib
import json
import logging
from typing import Callable, Optional

from jsonschema import ValidationError
from jsonschema import validate as _js_validate


from src.config import settings
from rag.engine import rag_engine
from rag.retrieval.retriever import hybrid_retriever
from rag.graph.graph_store import graph_store
from rag.graph.graph_extractor import graph_extractor
from agent.reflector import reflector

logger = logging.getLogger(__name__)

# module-073：异常自动重试排除清单——generate_answer / verify_answer 不重试
# （15s 超时是常态，重试无意义且翻倍墙钟）；其余工具（只读检索类 + note_to_self）
# 异常自动重试 1 次。排除清单比白名单简单：未来新工具默认继承重试。
_NO_RETRY_TOOLS = {"generate_answer", "verify_answer"}

# module-083（WP-B）：幂等启用清单——只读检索 7 工具（用户确认口径）。
# generate_answer / verify_answer / note_to_self 排除：每次调用语义不同
# （生成/验证结果随 docs 变化）或已有内容级去重（module-041 scratchpad）。
_IDEMPOTENT_TOOLS = {
    "search_knowledge", "search_fts", "search_vector", "search_graph",
    "extract_entities", "recall_memory", "re_search",
}


# ─── module-083 工具治理辅助（校验 / 幂等 / 审批，均模块级便于测试 monkeypatch） ───


def _fingerprint(name: str, args: dict) -> Optional[str]:
    """幂等指纹 —— sha256(name + "|" + args 规范化 JSON)（module-083 WP-B）

    sort_keys=True 保证参数键序无关（{"a":1,"b":2} 与 {"b":2,"a":1} 同一指纹）。
    args 非 JSON 序列化（罕见——LLM 参数来自 tool_calls 必为 JSON，理论不可达）
    → 返回 None 跳过幂等直接执行（防御，不阻断工具链路）。
    """
    try:
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return None
    return hashlib.sha256(f"{name}|{payload}".encode("utf-8")).hexdigest()


def _schema_error(name: str, args: dict, schema: Optional[dict]) -> Optional[str]:
    """校验工具参数是否符合 args_schema（module-083 WP-A），违规返回提示

    jsonschema 校验**已提供的参数**：schema 的 required 置空（浅拷贝不动原
    对象）——工具实现里 query 等"缺省回退 ctx.query"是设计契约（存量
    run({}, None) 与 MCP 外部调用依赖），强制 required 会把"静默回退"变
    "报错拒绝"。非 dict args → "参数应为 object"。jsonschema 自身异常
    （版本差异等）→ fail-open：warning + 放行执行（依赖层异常不能拖垮
    工具链路）。
    """
    if not isinstance(args, dict):
        return f"(工具 {name} 参数错误: 参数应为 object)"
    if not schema:
        return None
    try:
        _js_validate(args, {**schema, "required": []})
    except ValidationError as e:
        return f"(工具 {name} 参数错误: {e.message})"
    except Exception as e:
        logger.warning("工具 %s schema 校验异常（fail-open 放行）: %s", name, e)
    return None


async def _approval_allowed(name: str) -> bool:
    """高风险工具审批放行判定（module-083 WP-D）——工具级

    存在 status='approved' 记录（最近一条）即放行；DB 异常 → fail-closed
    拒绝（审批类工具属可能副作用类，宁拒勿放，安全侧；与 tool_call_logs
    观测路径 fail-open 语义严格区分）。仅 approval="required" 工具调用
    （auto 短路零 DB 开销）。
    """
    from sqlalchemy import text
    from src.database import async_session_factory
    try:
        async with async_session_factory() as session:
            row = (await session.execute(
                text("SELECT 1 FROM approval_requests WHERE tool_name=:n "
                     "AND status='approved' ORDER BY decided_at DESC LIMIT 1"),
                {"n": name},
            )).first()
        return row is not None
    except Exception as e:
        logger.warning("审批放行查询失败（fail-closed 拒绝执行）: %s", e)
        return False


async def _request_approval(name: str, args: dict, requester: str) -> None:
    """插入一条审批申请（同 tool_name 已有 pending 不重复插入，module-083 WP-D）

    落库失败仅日志告警（fail-open，观测/工作流路径语义：申请失败不阻断 LLM
    循环——执行已被审批闸拒绝、提示已返回）。SQL 全参数化防注入。
    """
    from sqlalchemy import text
    from src.database import async_session_factory
    try:
        args_json = json.dumps(args, ensure_ascii=False)
    except TypeError:  # 防御：非 JSON 序列化参数 → 兜底 {}
        args_json = "{}"
    try:
        async with async_session_factory() as session:
            exists = (await session.execute(
                text("SELECT 1 FROM approval_requests WHERE tool_name=:n "
                     "AND status='pending' LIMIT 1"),
                {"n": name},
            )).first()
            if exists is not None:
                return  # 同工具已有 pending 申请 → 不重复插入
            await session.execute(
                text("INSERT INTO approval_requests (tool_name, args, status, requester) "
                     "VALUES (:n, CAST(:args AS jsonb), 'pending', :r)"),
                {"n": name, "args": args_json, "r": requester},
            )
            await session.commit()
    except Exception as e:
        logger.warning("审批申请落库失败（fail-open，不影响循环）: %s", e)

class AgentTool:
    """单个 Agent 工具

    Attributes:
        name: 工具名（LLM 通过该名调用）
        description: 工具用途描述（指导 LLM 何时使用）
        args_schema: JSON Schema（OpenAI function parameters 格式）
        func: 执行函数，签名 async def func(ctx, args) -> str
        group: 所属执行阶段集合（module-058 / ADR-0012 方案 A）——
            "retrieval" / "generation"，双组工具 ["retrieval","generation"]；
            空集合 = 未分组，全阶段可见（向后兼容）
        timeout: 单次执行超时秒数（module-083 WP-C；缺省 settings.tool_default_timeout）
        approval: 审批模式（module-083 WP-D）："auto" 直接执行 / "required" 需人工审批
    """

    def __init__(self, name: str, description: str, args_schema: dict,
                 func: Callable, group: Optional[list] = None,
                 timeout: Optional[float] = None, approval: str = "auto"):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func
        # module-058：阶段归组（检索组 7 / 生成组 4，re_search 双组）。
        # 只影响暴露逻辑（to_llm_schemas 过滤），工具本身行为一字不改。
        self.group: set[str] = set(group) if group else set()
        # module-083 WP-C：工具级超时——config 是新工具默认值来源（现有 10
        # 工具不传 → 全 settings.tool_default_timeout=15.0，零行为变化）
        self.timeout: float = timeout if timeout is not None else settings.tool_default_timeout
        # module-083 WP-D：审批模式——默认 "auto" 短路零 DB 开销；"required"
        # 为 module-084 外部 MCP 工具（可能有副作用）的人工审批闸预留
        self.approval: str = approval

    async def run(self, args: dict, ctx) -> str:
        """执行工具；失败返回空结果，LLM 判断继续/放弃（module-028 降级哲学）

        module-073：异常（非超时）自动重试 1 次同一 func（同参数同 ctx）——
        只读检索类（search_*/extract_entities/recall_memory/re_search）异常多为
        瞬时抖动（429/网络闪断），重试大概率成功；note_to_self 重试安全依赖
        WP-A 去重拦双写；generate_answer/verify_answer 不重试（_NO_RETRY_TOOLS，
        15s 超时是常态）。**超时不重试**：超时=慢不是抖动（LLM 生成/rerank 慢），
        重试不修复根因只把单工具墙钟翻倍到 30s，且超时是预算围栏语义（module-042）。
        TimeoutError 分支必须先于重试分支判断（存量超时测试精确文案兼容前提）。
        重试发生在 run 内部 → 对 react_loop 完全不可见：不增加 tool_count /
        phase_count / 消息历史（tool 结果消息每 call 一条）。

        module-083：执行前 _precheck 三闸（审批 → schema 校验 → 幂等拦截），
        任一拦截返回提示文本（喂回 LLM，不真执行、不进重试分支）；执行成功后
        记幂等指纹（失败返回空串 / 超时提示不记 → 同参可重放，与 073 重试自洽）。

        Args:
            args: LLM 传入的工具参数（已由 args_schema 描述）
            ctx: ReAct 循环的会话上下文（见 react.ReactContext；存量测试形态
                可为 None，_precheck/记指纹均 getattr 短路）

        Returns:
            工具结果文本；执行失败返回 ""
        """
        pre = await self._precheck(args, ctx)
        if pre is not None:
            return pre
        result = await self._execute(args, ctx)
        self._record_fingerprint(args, ctx, result)
        return result

    async def _precheck(self, args: dict, ctx) -> Optional[str]:
        """执行前守门三闸：审批 → schema 校验 → 幂等拦截（module-083）

        总序（规划 §7）：审批闸（仅 approval="required" 工具，auto 短路零 DB
        开销）→ 参数校验（WP-A，置空 required 保留"缺省回退"契约）→ 幂等
        拦截（WP-B，同参二次只读检索返回提示）。任一拦截返回提示文本喂回 LLM
        （不进 073 重试分支——重试在 _execute 内）；None = 放行。
        """
        if settings.tool_approval_enabled and self.approval == "required" \
                and not await _approval_allowed(self.name):
            requester = getattr(ctx, "identity", "")
            await _request_approval(self.name, args, requester)
            return f"(工具 {self.name} 需人工审批，调用申请已提交)"
        err = _schema_error(self.name, args, self.args_schema)
        if err is not None:
            return err
        if settings.tool_idempotency_enabled:
            fp_set = getattr(ctx, "executed_fingerprints", None)
            if fp_set is not None and self.name in _IDEMPOTENT_TOOLS:
                fp = _fingerprint(self.name, args)
                if fp is not None and fp in fp_set:
                    return "(该调用已执行过，结果见上文)"
        return None

    async def _execute(self, args: dict, ctx) -> str:
        """工具执行主体（module-073 重试语义原样搬入；WP-C 超时参数化）

        首试 wait_for(self.timeout)；超时（永不重试）返回精确文案
        "(工具 X 执行超时)"、失败返回 ""——存量测试逐字断言，一字不改；
        异常（非超时）且开关开且不在排除清单 → 同 func 同参自动重试 1 次。
        """
        try:
            return await asyncio.wait_for(self.func(ctx, args), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.warning("工具 %s 超时 (%ss)", self.name, self.timeout)
            return f"(工具 {self.name} 执行超时)"
        except Exception as e:
            if settings.tool_auto_retry and self.name not in _NO_RETRY_TOOLS:
                logger.warning("工具 %s 首次失败，自动重试: %s", self.name, e)
                try:
                    return await asyncio.wait_for(self.func(ctx, args), timeout=self.timeout)
                except asyncio.TimeoutError:
                    logger.warning("工具 %s 重试超时 (%ss)", self.name, self.timeout)
                    return f"(工具 {self.name} 执行超时)"
                except Exception as e2:
                    logger.warning("工具 %s 重试仍失败，返回空: %s", self.name, e2)
                    return ""
            logger.warning("工具 %s 执行失败，返回空: %s", self.name, e)
            return ""

    def _record_fingerprint(self, args: dict, ctx, result: str) -> None:
        """成功执行后记幂等指纹（module-083 WP-B）

        成功 = 非空结果且非超时提示——超时返回文本非空但不是执行结果，不记
        （同参可重放）；失败（空串）不记；073 重试成功记 1 次。超时文案耦合
        说明：以本类精确文案判定"未执行成功"，与 react.py _EMPTY_RESULT_MARKERS
        红线文本耦合先例对齐（超时文案是存量测试红线，一字不改）。
        """
        if not settings.tool_idempotency_enabled or self.name not in _IDEMPOTENT_TOOLS:
            return
        fp_set = getattr(ctx, "executed_fingerprints", None)
        if fp_set is None:
            return
        if not result or result == f"(工具 {self.name} 执行超时)":
            return
        fp = _fingerprint(self.name, args)
        if fp is None:
            return
        fp_set.add(fp)


    def to_openai_schema(self) -> dict:
        """转成 OpenAI function calling 的 tool schema（ChatOpenAI.bind_tools 用）"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema,
            },
        }


class ToolRegistry:
    """工具注册表：注册 / 查询 / 序列化

    注册表只保存工具定义（name/description/args_schema/func），
    不持有任何请求级状态；执行时经 run(args, ctx) 注入会话上下文，
    因此全局单例可跨请求复用（并发安全）。
    """

    def __init__(self):
        self._tools: dict[str, AgentTool] = {}

    def register(self, name: str, description: str, args_schema: dict,
                 func: Callable, group: Optional[list] = None,
                 timeout: Optional[float] = None,
                 approval: str = "auto") -> "ToolRegistry":
        """注册一个工具（同名覆盖，便于测试替换）

        module-058（ADR-0012 方案 A）：group 标注阶段归属（"retrieval" /
        "generation"，双组传 ["retrieval","generation"]）；None = 未分组，
        全阶段可见（向后兼容，测试自定义工具不受影响）。

        module-083：timeout / approval 透传给 AgentTool（缺省 None / "auto" →
        15.0 秒 / 直接执行，现有 10 工具零行为变化）。
        """
        self._tools[name] = AgentTool(name, description, args_schema, func,
                                      group=group, timeout=timeout,
                                      approval=approval)
        return self


    def get(self, name: str) -> Optional[AgentTool]:
        """按名字取工具，未注册返回 None"""
        return self._tools.get(name)

    def list_tools(self) -> list[AgentTool]:
        """返回全部已注册工具（注册序）"""
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """返回全部工具名"""
        return [t.name for t in self._tools.values()]

    def to_llm_schemas(self, group: Optional[str] = None) -> list[dict]:
        """序列化为 OpenAI function calling 的 tool schema 列表

        module-058（ADR-0012 方案 A）阶段切分：
        - group=None：返回全量（向后兼容，`test_agent_tools.py` len==10 不挂）
        - group="retrieval"/"generation"：只返回该阶段可见工具（未分组工具
          恒全阶段可见）——省 schema token + 结构性防误调
        """
        tools = self._tools.values()
        if group is not None:
            tools = [t for t in tools if not t.group or group in t.group]
        return [t.to_openai_schema() for t in tools]


# ─── 检索结果格式化 ───


def _format_docs(docs: list[dict], limit: int = 5) -> str:
    """把检索结果格式化为 LLM 可读文本（标题 + 分数 + 内容截断）

    Args:
        docs: 检索结果列表（含 id/title/content/hybrid_score 等）
        limit: 最多展示条数

    Returns:
        格式化文本；无结果返回 "（无检索结果）"
    """
    if not docs:
        return "（无检索结果）"
    parts = []
    for i, d in enumerate(docs[:limit], start=1):
        score = round(float(d.get("hybrid_score", d.get("score", 0.0))), 3)
        content = (d.get("content") or "")[:400]
        parts.append(f"[{i}] {d.get('title', '')} (score={score})\n{content}")
    if len(docs) > limit:
        parts.append(f"……共 {len(docs)} 条结果，已展示前 {limit} 条")
    return "\n\n".join(parts)


# ─── 内置工具实现（func 签名: async def (ctx, args) -> str） ───


async def _search_knowledge(ctx, args: dict) -> str:
    """混合检索：FTS 关键词 + 向量语义融合，默认首选"""
    query = args.get("query") or ctx.query
    top_k = int(args.get("top_k", 5))
    docs = await hybrid_retriever.retrieve(query, top_k=top_k, mode="hybrid")
    ctx.add_docs(docs)
    return _format_docs(docs)


async def _search_fts(ctx, args: dict) -> str:
    """仅全文检索：精确关键词匹配（专有名词/代码/精确术语）"""
    query = args.get("query") or ctx.query
    top_k = int(args.get("top_k", 5))
    docs = await hybrid_retriever.retrieve(query, top_k=top_k, mode="fts_only")
    ctx.add_docs(docs)
    return _format_docs(docs)


async def _search_vector(ctx, args: dict) -> str:
    """仅向量检索：语义相似度匹配（概念性/同义表述查询）"""
    query = args.get("query") or ctx.query
    top_k = int(args.get("top_k", 5))
    docs = await hybrid_retriever.retrieve(query, top_k=top_k, mode="vector_only")
    ctx.add_docs(docs)
    return _format_docs(docs)


async def _search_graph(ctx, args: dict) -> str:
    """知识图谱检索：提取实体 → 沿实体关系图遍历返回关联文档"""
    query = args.get("query") or ctx.query
    top_k = int(args.get("top_k", 5))
    entities = await graph_extractor.extract_from_query(query)
    if not entities:
        return "（图检索：未提取到实体）"
    docs = await graph_store.search_related(entities, top_k=top_k)
    ctx.add_docs(docs)
    return _format_docs(docs)


async def _extract_entities(ctx, args: dict) -> str:
    """从查询/文本提取技术实体名称列表（JSON）"""
    query = args.get("query") or ctx.query
    entities = await graph_extractor.extract_from_query(query)
    return json.dumps({"entities": entities}, ensure_ascii=False)


async def _recall_memory(ctx, args: dict) -> str:
    """召回该用户的跨会话长期记忆（按身份隔离；无记忆返回提示）"""
    query = args.get("query") or ctx.query
    top_k = int(args.get("top_k", 3))
    text = await rag_engine._recall_memory(query, ctx.identity, top_k=top_k)
    if text:
        ctx.memory = text  # 供 generate_answer 工具拼入生成 prompt
    return text or "（无相关历史记忆）"


async def _generate_answer(ctx, args: dict) -> str:
    """基于本次已累积检索到的文档生成带引用标注的最终答案"""
    if not ctx.docs:
        return "（尚未检索到文档，请先调用 search_knowledge 等检索工具）"
    query = args.get("query") or ctx.query
    return await reflector.generate_answer(
        query, ctx.docs, history=ctx.history, memory=ctx.memory,
        scratchpad=ctx.scratchpad,
    )


async def _verify_answer(ctx, args: dict) -> str:
    """逐句验证已生成答案是否被检索文档支持，标注可信度（module-039）"""
    answer = args.get("answer")
    if not answer:
        return "（未提供答案文本，无法验证）"
    if not ctx.docs:
        return "（无检索文档，无法验证答案可信度；请先检索）"
    result = await reflector.verify_answer(answer, ctx.docs)
    if not result.get("claims"):
        return "（验证失败，无法判定答案可信度）"
    lines = []
    for c in result["claims"]:
        verdict_icon = {"supported": "✓", "inferred": "~", "unsupported": "✗"}.get(
            c.get("verdict", ""), "?"
        )
        lines.append(f"[{verdict_icon}] {c.get('verdict')}: {c.get('claim')} (证据: {c.get('evidence')})")
    lines.append(f"\n整体置信度: {result.get('overall_confidence', 0):.0%}")
    lines.append(f"supported={result.get('supported', 0)} inferred={result.get('inferred', 0)} unsupported={result.get('unsupported', 0)}")
    return "\n".join(lines)


async def _re_search(ctx, args: dict) -> str:
    """检索不足 → 改写 query 重检 → 新结果累积到 ctx.docs（module-040）

    流程：
      1. check_sufficiency 判断当前 ctx.docs 是否充分
      2. 不充分 → 用 rewritten_query 重新混合检索
      3. 新结果按 id 去重累积到 ctx.docs

    降级：
      - 无 ctx.docs → 提示先检索
      - check_sufficiency 返回充分 → 提示无需重检
      - 改写后仍无结果 → 提示知识库无相关内容
      - check_sufficiency 自身失败（LLM 异常）→ reflector 内部默认充分
    """
    if not ctx.docs:
        return "（尚未检索到文档，请先调用 search_knowledge 等检索工具）"
    query = args.get("query") or ctx.query
    result = await reflector.check_sufficiency(query, ctx.docs)
    if result.get("sufficient"):
        return "（当前检索结果已充分，无需重检）"
    rewritten = result.get("rewritten_query", query)
    # module-073：同改写 query 守卫（防 LLM 拿同一改写反复调 re_search 空转）——
    # 完全一致比较；在 check_sufficiency 之后拦截（rewritten 只能由它产出 + 它
    # 重新评估充分性，ctx.docs 可能已增长），拦截的是"重检索 + 文档格式化"大头。
    # 不做输入 query 级预拦截（完全免 LLM）：文档变化后（如已调其他检索工具）
    # 同 query 合法重评会被误拦。sufficient 分支提前返回不更新守卫字段。
    if rewritten == ctx.last_research_query:
        return "已按该改写重检过，无新结果"
    ctx.last_research_query = rewritten
    docs = await hybrid_retriever.retrieve(rewritten, top_k=5, mode="hybrid")
    ctx.add_docs(docs)
    if not docs:
        return f"改写查询 '{rewritten}' 后仍无结果，知识库可能无相关内容"
    return f"改写查询 '{rewritten}' → 检索到 {len(docs)} 篇文档：\n" + _format_docs(docs)


async def _note_to_self(ctx, args: dict) -> str:
    """记录中间发现或推理结论到工作笔记（module-041）"""
    note = args.get("note", "")
    if not note or not note.strip():
        return "（未提供笔记内容）"
    note = note.strip()[:500]  # 截断过长笔记（module-073：比较点取截断后的值，两次相同超长 note 仍判重复）
    if not ctx.add_note(note):  # module-073：完全一致去重，重复不追加
        return "笔记已存在（未重复记录）"
    return f"已记录笔记 ({len(ctx.scratchpad)}): {note[:200]}"


# ─── 内置工具注册 ───

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "检索关键词（缺省用原始问题）"},
        "top_k": {"type": "integer", "description": "返回数量，默认 5"},
    },
    "required": ["query"],
}

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "要提取实体的文本"}},
    "required": ["query"],
}

_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "记忆检索查询"},
        "top_k": {"type": "integer", "description": "返回条数，默认 3"},
    },
    "required": ["query"],
}

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "要回答的问题"}},
    "required": ["query"],
}

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "原始用户问题"},
        "answer": {"type": "string", "description": "待验证的答案文本（通常由 generate_answer 产出）"},
    },
    "required": ["answer"],
}

_RE_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "原始用户问题，缺省用 ctx.query"},
    },
}

_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "description": "要记录的笔记内容"},
    },
    "required": ["note"],
}


def register_builtin_tools(reg: Optional[ToolRegistry] = None) -> ToolRegistry:
    """注册 10 个内置工具到注册表（默认全局 registry）

    module-058（ADR-0012 方案 A）阶段归组（只动暴露逻辑，name/description/
    args_schema 一字不改）：
      - 检索组 7：search_knowledge / search_fts / search_vector /
        search_graph / extract_entities / recall_memory / re_search
      - 生成组 4：generate_answer / verify_answer / note_to_self / re_search
        （re_search 双组：初次检索不足 + 生成后验证不充分两个时机都要用）
    归组依据见 specs/adr/0012-tool-governance.md。

    Args:
        reg: 目标注册表（测试可传入独立实例），None 用全局 registry

    Returns:
        注册完成后的注册表
    """
    reg = reg or registry
    reg.register(
        "search_knowledge",
        "混合检索：同时使用全文关键词与语义向量在知识库中检索相关文档，默认首选。",
        _SEARCH_SCHEMA, _search_knowledge, group=["retrieval"],
    )
    reg.register(
        "search_fts",
        "全文检索：按精确关键词匹配知识库文档（适合专有名词、代码、精确术语）。",
        _SEARCH_SCHEMA, _search_fts, group=["retrieval"],
    )
    reg.register(
        "search_vector",
        "向量检索：按语义相似度检索知识库文档（适合概念性、同义表述查询）。",
        _SEARCH_SCHEMA, _search_vector, group=["retrieval"],
    )
    reg.register(
        "search_graph",
        "知识图谱检索：从查询中提取实体，沿实体关系图遍历返回关联文档。",
        _SEARCH_SCHEMA, _search_graph, group=["retrieval"],
    )
    reg.register(
        "extract_entities",
        "从查询/文本中提取技术实体名称列表（返回 JSON）。",
        _ENTITY_SCHEMA, _extract_entities, group=["retrieval"],
    )
    reg.register(
        "recall_memory",
        "召回该用户的跨会话长期记忆（历史问答沉淀，按用户隔离）。",
        _MEMORY_SCHEMA, _recall_memory, group=["retrieval"],
    )
    reg.register(
        "generate_answer",
        "基于本次已检索到的全部文档生成带引用标注的最终答案。",
        _ANSWER_SCHEMA, _generate_answer, group=["generation"],
    )
    reg.register(
        "verify_answer",
        "逐句验证已生成的答案是否被检索文档支持，标注每句的可信度（supported/inferred/unsupported），返回置信度。",
        _VERIFY_SCHEMA, _verify_answer, group=["generation"],
    )
    reg.register(
        "re_search",
        "检索不足时自动改写查询重检：检查已有文档是否充分，不充分则用改写后的查询重新混合检索，新结果累积到已有文档。",
        _RE_SEARCH_SCHEMA, _re_search, group=["retrieval", "generation"],
    )
    reg.register(
        "note_to_self",
        "记录中间发现或推理结论到工作笔记（草稿纸），后续轮次可参考。",
        _NOTE_SCHEMA, _note_to_self, group=["generation"],
    )
    return reg


# 全局单例 — 无状态定义容器，多会话并发安全
registry = ToolRegistry()
register_builtin_tools()
