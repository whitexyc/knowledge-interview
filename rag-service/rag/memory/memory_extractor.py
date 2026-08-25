"""
长期记忆事实提取器（module-033）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在对话结束后异步调用，从 (query, answer, 最近历史) 提取"值得长期记住"的事实，
作为自动写入长期记忆的输入。只在 knowledge 路径触发（闲聊/实时由调用方跳过）。

设计决策：
  1. 单次 LLM 调用，输出结构化 JSON {"facts": [{"content", "importance"}]}
  2. importance 过滤 + content 空过滤（阈值 settings.memory_importance_threshold）
  3. 失败/超时一律返回 []（降级不影响对话，与 graph_extractor 同款哲学）
"""
import asyncio
import json
import logging

from src.config import settings
from llm.client import LLMFactory

logger = logging.getLogger(__name__)

# 记忆类型契约（module-062 P2）：preference 偏好（慢衰减）/ fact 事实（中衰减）/
# event 带时间临时事件（快衰减）。LLM 判型与分类模型共用同一标签集。
MEMORY_TYPES = ("preference", "fact", "event")

# 提取 prompt：明确"值得长期记住"的标准（偏好/事实/任务状态），
# importance 为 0-1 数字（>= 0.6 才值得记住，阈值由配置控制）。
# module-062：每条事实加 type（preference/fact/event）——按记忆类型差异化衰减
#（ADR-0007 P2，A-MAC 参考：偏好慢衰减、事件快过期）。格式向后兼容：无 type
# 容错默认 fact（存量调用方取 content/importance 不受影响）。
_EXTRACT_PROMPT = """你是一个长期记忆管理员。从下面的对话中提取"值得长期记住"的事实，用于跨会话记忆。

值得长期记住的标准：
- 用户的偏好、习惯、兴趣（如"偏好简洁的回答风格"）→ type=preference
- 关于用户的客观事实（职业、技能、背景、计划等）→ type=fact
- 长期任务的状态或进展、带时间的临时事件（如"下周去北京"）→ type=event

不值得记住的：
- 一次性问答、临时闲聊
- 与用户无关的通用知识（如检索到的文档内容本身）

type 取值：
- preference：用户的喜好/习惯/偏好（长期有效，慢衰减）
- fact：客观事实（较稳定，中衰减）
- event：带时间的一次性/临时事件（迅速过期，快衰减）
无法判断类型时默认 fact。

用户问题: {query}
助手回答: {answer}
最近对话历史: {history}

只返回 JSON，不要其他文字，格式如下：
{{"facts": [{{"content": "事实内容", "importance": 0.8, "type": "preference"}}, ...]}}

importance 表示该事实对长期记忆的重要性，范围 0-1，低于 0.6 的不应出现。
JSON:"""

# LLM 提取超时（秒）：超时降级返回 []，不阻塞对话
_EXTRACT_TIMEOUT_SECONDS = 10


def _format_history(history: list[dict] | None) -> str:
    """把最近对话历史格式化为 prompt 文本（最多取最近 6 条）

    Args:
        history: 对话消息列表 [{"role": str, "content": str}, ...]

    Returns:
        形如 "用户: ...\n助手: ..." 的文本；无历史时返回"（无）"
    """
    if not history:
        return "（无）"
    lines = []
    for msg in history[-6:]:
        role = (msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户: {content}")
        elif role == "assistant":
            lines.append(f"助手: {content}")
    return "\n".join(lines) if lines else "（无）"


def _parse_json(raw: str) -> dict:
    """解析 LLM 输出的 JSON，多级回退（与 graph_extractor 同款策略）

    Args:
        raw: LLM 原始输出文本

    Returns:
        解析后的 dict；失败返回 {}
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        pass
    return {}


async def extract_facts(
    query: str, answer: str, history: list[dict] | None = None,
) -> list[dict]:
    """从对话提取值得长期记住的事实

    流程：LLM 一次调用 → 解析 {"facts": [...]} → 过滤 importance<阈值
    或空 content → 返回。失败/超时/answer 为空 → 返回 []（降级，不影响对话）。

    Args:
        query: 用户问题
        answer: 助手回答（空则不提取）
        history: 最近对话历史（可选）

    Returns:
        [{"content": str, "importance": float, "type": str}, ...]
        每条 importance >= settings.memory_importance_threshold 且 content 非空；
        type ∈ {preference, fact, event}，缺失/非法 → "fact"（中性兜底，fail-open）
    """
    if not answer or not answer.strip():
        return []
    prompt = _EXTRACT_PROMPT.format(
        query=query or "",
        answer=answer,
        history=_format_history(history or []),
    )
    try:
        client = LLMFactory.get_client()
        raw = await asyncio.wait_for(
            client.generate(prompt), timeout=_EXTRACT_TIMEOUT_SECONDS,
        )
        data = _parse_json(raw)
        facts = []
        for item in data.get("facts", []):
            content = str(item.get("content") or "").strip()
            importance = item.get("importance", 0.0)
            try:
                importance = float(importance)
            except (TypeError, ValueError):
                importance = 0.0
            # 过滤：importance < 阈值 或 content 空 → 丢弃
            if content and importance >= settings.memory_importance_threshold:
                mtype = str(item.get("type") or "").strip().lower()
                if mtype not in MEMORY_TYPES:
                    mtype = "fact"  # 缺失/非法类型 → 默认 fact（中性兜底）
                facts.append({
                    "content": content,
                    "importance": round(importance, 3),
                    "type": mtype,
                })
        logger.info("长期记忆事实提取: query=%s, facts=%d", query[:40], len(facts))
        return facts
    except asyncio.TimeoutError:
        logger.warning("长期记忆事实提取超时 (%.0fs)，降级返回空", _EXTRACT_TIMEOUT_SECONDS)
        return []
    except Exception as e:
        logger.warning("长期记忆事实提取失败，降级返回空: %s", e)
        return []
