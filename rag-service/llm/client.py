"""
LLM 多供应商适配层 — RAG 链路的推理引擎
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置：
  RAG 链路中所有需要 LLM 推理的地方都经过这个适配层：
    - Router Agent: 意图分类
    - Reflector: 反思检查 + 答案生成
    - Casual Chat: 直接闲聊

  本文件不直接参与 RAG 流水线编排，但为流水线中的多个环节提供"大脑"。

设计决策：
  1. 为什么用 LangChain？
     因为 LangChain 提供了统一的 ChatModel 接口，切换供应商只需
     换一个类（ChatAnthropic → ChatOpenAI），不用改调用代码。
     如果没有 LangChain，我们需要为每个供应商写一个 HTTP 客户端。

  2. 为什么用工厂模式？
     RAG 链路中有多处调用 LLM（路由、反思、生成），如果每个地方都
     自己实例化客户端，配置变更时（如切换供应商）需要改多处代码。
     工厂模式集中管理 LLM 实例，配置变更只需改一处。

  3. 为什么全异步（async/await）？
     LLM API 调用是 I/O 密集型操作，同步调用会阻塞事件循环。
     异步化让服务器在等待 API 响应时能处理其他请求，提高吞吐量。

  4. 为什么用实例缓存（_instances dict）？
     LLM 客户端初始化可能涉及网络连接（如 websocket），
     重复创建销毁浪费资源。缓存复用同一个客户端实例。
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI

from src.config import settings
from src import observability

logger = logging.getLogger(__name__)


def _extract_usage(response) -> Optional[tuple]:
    """从 LLM 响应提取 token usage（兼容各家格式），无 usage 返回 None

    支持的形态：
      - OpenAI 兼容：response.usage.prompt_tokens / completion_tokens
        （langchain ChatOpenAI 的 response_metadata["token_usage"]）
      - Anthropic：response.usage.input_tokens / output_tokens

    module-058（WP-C）：token 用量采集（fallback 链各供应商），无 usage 记
    跳过不中断（由调用方 _record_usage 处理 None）。

    Args:
        response: 供应商原始响应（对象或 dict）

    Returns:
        (prompt_tokens, completion_tokens) 或 None
    """
    try:
        prompt = completion = None
        if response is None:
            return None
        # OpenAI SDK ChatCompletion：raw.usage.prompt_tokens
        usage = getattr(response, "usage", None)
        if usage is not None and not isinstance(usage, dict):
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
        if prompt is None and completion is None:
            # langchain：response_metadata["token_usage"]（dict）
            meta = getattr(response, "response_metadata", None) or {}
            if isinstance(meta, dict):
                tu = meta.get("token_usage") or {}
                if isinstance(tu, dict):
                    prompt = tu.get("prompt_tokens", prompt)
                    completion = tu.get("completion_tokens", completion)
        if prompt is None and completion is None and usage is not None:
            # Anthropic：usage.input_tokens / output_tokens
            prompt = getattr(usage, "input_tokens", None)
            completion = getattr(usage, "output_tokens", None)
        prompt = int(prompt) if prompt is not None else None
        completion = int(completion) if completion is not None else None
        if prompt is None and completion is None:
            return None
        return (prompt, completion)
    except (TypeError, ValueError):
        # 异常值（如测试 MagicMock）静默跳过，不中断主链路
        return None


def _record_usage(label: str, response) -> None:
    """采集本次调用的 token usage 并累积到请求上下文（无 usage 静默跳过）

    module-058（WP-C）：各供应商响应返回处调用；流式（generate_stream）不
    采集——SSE 逐 token 场景供应商通常不返回 usage（口径见 changelog）。
    """
    usage = _extract_usage(response)
    if usage is not None:
        observability.record_usage(label, usage[0], usage[1])


class LLMException(Exception):
    """LLM 调用异常

    包装供应商特定的异常信息，统一异常接口。
    provider 字段告诉调用方是哪个供应商出了问题。
    cause 字段保留原始异常链，方便排查问题。
    """

    def __init__(self, provider: str, message: str, cause: Optional[Exception] = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")
        self.__cause__ = cause


class LLMClient(ABC):
    """LLM 客户端抽象基类（异步）

    定义 generate（单轮文本生成）、chat（多轮对话）和 generate_stream（流式生成）三个接口。
    所有供应商客户端都必须实现这些方法。

    generate vs chat 的区别：
    - generate: 接收字符串 prompt，返回字符串（内部转成单轮 messages）
    - chat: 接收 messages 列表，支持多轮对话（system/user/assistant）

    generate_stream vs generate 的区别：
    - generate_stream: 异步生成器，逐 token 产出文本片段
    - generate: 等待完整响应后一次性返回
    """

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """单轮文本生成"""
        ...

    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """多轮对话"""
        ...

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """多轮对话 + 工具调用（module-028 ReAct 循环用）

        DeepSeek thinking 模式（如 deepseek-v4-flash）要求把上一轮 assistant 消息的
        reasoning_content 原样回传，否则 400 报错（bind_tools 会丢弃该字段），
        因此对 ChatOpenAI 系（deepseek/qwen/zhipu/modelscope）改走底层 OpenAI
        兼容客户端（async_client.create），并返回原始 assistant 消息供循环追加；
        Claude（ChatAnthropic）无此问题，走 bind_tools。

        本方法在基类提供默认实现（依赖 self._llm），子类中仅有 FallbackClient
        覆写（它没有 self._llm，需遍历降级链）；其余子类直接继承。

        Args:
            messages: 对话消息（OpenAI dict 格式）
            tools: OpenAI function calling 格式的工具 schema 列表

        Returns:
            {"content": str, "tool_calls": [{"id", "name", "args"}, ...],
             "message": dict}
            message 为原始 assistant 消息 dict（含 reasoning_content / tool_calls），
            供 ReAct 循环原样追加到消息历史、下一轮回传。
            tool_calls 为空列表表示模型直接输出答案（不调用工具）

        Raises:
            LLMException: 工具调用失败（降级链内各供应商由调用方捕获切换）
        """
        if isinstance(self._llm, ChatOpenAI):
            return await self._chat_with_tools_openai(messages, tools)
        return await self._chat_with_tools_bind(messages, tools)

    def _provider_label(self) -> str:
        """当前客户端供应商标签（chat_with_tools token 用量按供应商归属）

        module-058（WP-C）Review 修复（MAJOR-2）：基类 chat_with_tools 不
        感知具体供应商，旧实现恒标 "llm" 导致工具调用轮次用量无法按供应商
        归属（fallback 链切换混在同一桶）。按实现类映射：
        DeepSeekClient → "deepseek"；_ModelScopeBaseClient 系 → self._label
        （qwen/zhipu/modelscope）；ClaudeClient（bind_tools 路径）→ "claude"。
        """
        if isinstance(self, DeepSeekClient):
            return "deepseek"
        if isinstance(self, _ModelScopeBaseClient):
            return self._label
        if isinstance(self, ClaudeClient):
            return "claude"
        return "llm"

    async def _chat_with_tools_openai(
        self, messages: list[dict], tools: list[dict],
    ) -> dict:
        """ChatOpenAI 系（deepseek/qwen/zhipu）工具调用：底层客户端直连

        保留 reasoning_content（thinking 模式回传要求）与原始 tool_calls
        （arguments 字符串保持模型原样），返回 message 供循环回传。
        """
        try:
            raw = await self._llm.async_client.create(
                model=self._llm.model_name,
                messages=messages,
                tools=tools,
                temperature=self._llm.temperature,
            )
        except Exception as e:
            logger.error("LLM 工具调用失败: %s", e)
            raise LLMException("llm", "工具调用服务暂不可用", cause=e)
        if not raw.choices or not raw.choices[0].message:
            raise LLMException("llm", "工具调用返回为空")

        # module-058（WP-C）：token 用量采集（OpenAI SDK usage 字段），
        # 标签按供应商（_provider_label：deepseek/qwen/zhipu/modelscope）
        _record_usage(self._provider_label(), raw)
        msg = raw.choices[0].message
        content = msg.content or ""
        assistant: dict = {"role": "assistant", "content": content}
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            assistant["reasoning_content"] = reasoning
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments or "{}"}}
                for tc in msg.tool_calls
            ]
        return {"content": content, "tool_calls": tool_calls, "message": assistant}

    async def _chat_with_tools_bind(
        self, messages: list[dict], tools: list[dict],
    ) -> dict:
        """Claude 等非 ChatOpenAI 供应商工具调用：LangChain bind_tools"""
        try:
            llm = self._llm.bind_tools(tools)
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.error("LLM 工具调用失败: %s", e)
            raise LLMException("llm", "工具调用服务暂不可用", cause=e)

        # module-058（WP-C）：token 用量采集（langchain response_metadata /
        # anthropic usage 字段），标签按供应商（_provider_label：claude）
        _record_usage(self._provider_label(), response)
        content = response.content or ""
        if isinstance(content, list):  # Claude 多模态内容块，提取文本段
            content = "".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        tool_calls = []
        assistant: dict = {"role": "assistant", "content": content}
        if response.tool_calls:
            for tc in response.tool_calls:
                args = tc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": args,
                })
            assistant["tool_calls"] = [
                {"id": tc.get("id", ""), "type": "function",
                 "function": {"name": tc.get("name", ""),
                              "arguments": json.dumps(tc.get("args") or {},
                                                       ensure_ascii=False)}}
                for tc in response.tool_calls
            ]
        return {"content": content, "tool_calls": tool_calls, "message": assistant}

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """流式文本生成，逐 token 产出

        Args:
            prompt: 输入文本

        Yields:
            文本片段，每次 yield 一个或几个 token
        """
        ...
        if False:  # pragma: no cover 让生成器成为真正的 async generator
            yield ""


class ClaudeClient(LLMClient):
    """Claude API 客户端（异步）

    通过 LangChain 的 ChatAnthropic 封装调用 Claude API。
    用于需要 Claude 推理能力的高质量生成场景。
    """

    def __init__(self, temperature: float = 0.7):
        if not settings.claude_api_key:
            raise LLMException("claude", "CLAUDE_API_KEY 未配置")
        # 懒加载：仅 claude provider 分支使用；langchain-anthropic 0.x 与 langchain-core 0.2 无兼容版本，
        # 模块级 import 会让 deepseek 等 OpenAI 兼容链路白白背负启动硬依赖（2026-08-25 环境恢复时修）。
        from langchain_anthropic import ChatAnthropic
        self._llm = ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.claude_api_key,
            temperature=temperature,  # 默认 0.7；反思等结构化任务可传低温度（module-026）
            timeout=120,              # RAG 全链路多次 LLM 调用，设 120s 避免过早超时
        )

    async def generate(self, prompt: str) -> str:
        logger.info("Claude generate, model=%s, prompt_len=%d", settings.claude_model, len(prompt))
        try:
            # ainvoke 是 LangChain 的异步调用方法
            response = await self._llm.ainvoke(prompt)
            _record_usage("claude", response)
            return response.content
        except Exception as e:
            logger.error("Claude 调用失败: %s", e)
            raise LLMException("claude", "Claude 服务暂不可用", cause=e)

    async def chat(self, messages: list[dict]) -> str:
        try:
            response = await self._llm.ainvoke(messages)
            _record_usage("claude", response)
            return response.content
        except Exception as e:
            logger.error("Claude chat 失败: %s", e)
            raise LLMException("claude", "Claude 对话服务暂不可用", cause=e)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        logger.info("Claude stream, model=%s", settings.claude_model)
        try:
            # 使用 astream_events 替代 astream，获得更精细的 token 级别事件
            async for event in self._llm.astream_events(prompt, version="v1"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield chunk.content
        except Exception as e:
            logger.error("Claude 流式调用失败: %s", e)
            raise LLMException("claude", "Claude 流式服务暂不可用", cause=e)


class DeepSeekClient(LLMClient):
    """DeepSeek API 客户端（兼容 OpenAI SDK，异步）

    DeepSeek 的 API 兼容 OpenAI 格式，所以用 LangChain 的 ChatOpenAI 封装。
    这是项目的默认 LLM 供应商。
    """

    def __init__(self, temperature: float = 0.7):
        if not settings.deepseek_api_key:
            raise LLMException("deepseek", "DEEPSEEK_API_KEY 未配置")
        self._llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,  # 默认 0.7；反思等结构化任务可传低温度（module-026）
            timeout=120,   # RAG 全链路多次 LLM 调用，设 120s 避免过早超时
        )

    async def generate(self, prompt: str) -> str:
        logger.info("DeepSeek generate, model=%s, prompt_len=%d", settings.deepseek_model, len(prompt))
        try:
            response = await self._llm.ainvoke(prompt)
            _record_usage("deepseek", response)
            return response.content
        except Exception as e:
            logger.error("DeepSeek 调用失败: %s", e)
            raise LLMException("deepseek", "DeepSeek 服务暂不可用", cause=e)

    async def chat(self, messages: list[dict]) -> str:
        try:
            response = await self._llm.ainvoke(messages)
            _record_usage("deepseek", response)
            return response.content
        except Exception as e:
            logger.error("DeepSeek chat 失败: %s", e)
            raise LLMException("deepseek", "DeepSeek 对话服务暂不可用", cause=e)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        logger.info("DeepSeek stream, model=%s", settings.deepseek_model)
        try:
            async for chunk in self._llm.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("DeepSeek 流式调用失败: %s", e)
            raise LLMException("deepseek", "DeepSeek 流式服务暂不可用", cause=e)


class _ModelScopeBaseClient(LLMClient):
    """ModelScope API 基类（OpenAI 兼容）

    Qwen / Zhipu GLM / DeepSeek(V4-Pro) 都走同一个 ModelScope API 端点，
    只是 model 参数不同。这个基类通过构造函数参数区分。
    """

    def __init__(self, model: str, label: str, temperature: float = 0.7):
        if not settings.modelscope_api_key:
            raise LLMException(label, "MODELSCOPE_API_KEY 未配置")
        self._label = label
        self._model = model
        self._llm = ChatOpenAI(
            model=model,
            api_key=settings.modelscope_api_key,
            base_url=settings.modelscope_base_url,
            temperature=temperature,  # 默认 0.7；反思等结构化任务可传低温度（module-026）
            timeout=120,
        )

    async def generate(self, prompt: str) -> str:
        logger.info("%s generate, model=%s", self._label, self._model)
        try:
            response = await self._llm.ainvoke(prompt)
            _record_usage(self._label, response)
            return response.content
        except Exception as e:
            logger.error("%s 调用失败: %s", self._label, e)
            raise LLMException(self._label, f"{self._label} 服务暂不可用", cause=e)

    async def chat(self, messages: list[dict]) -> str:
        try:
            response = await self._llm.ainvoke(messages)
            _record_usage(self._label, response)
            return response.content
        except Exception as e:
            logger.error("%s chat 失败: %s", self._label, e)
            raise LLMException(self._label, f"{self._label} 对话服务暂不可用", cause=e)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        logger.info("%s stream, model=%s", self._label, self._model)
        try:
            async for chunk in self._llm.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("%s 流式调用失败: %s", self._label, e)
            raise LLMException(self._label, f"{self._label} 流式服务暂不可用", cause=e)


class QwenClient(_ModelScopeBaseClient):
    """Qwen 客户端（ModelScope API，默认首选）"""

    def __init__(self, temperature: float = 0.7):
        super().__init__(model=settings.qwen_model, label="qwen", temperature=temperature)


class ZhipuClient(_ModelScopeBaseClient):
    """ZhipuAI GLM 客户端（ModelScope API，Qwen 降级备用）"""

    def __init__(self, temperature: float = 0.7):
        super().__init__(model=settings.zhipu_model, label="zhipu", temperature=temperature)


class ModelScopeClient(_ModelScopeBaseClient):
    """ModelScope 客户端（保留兼容旧配置）"""

    def __init__(self, temperature: float = 0.7):
        super().__init__(model=settings.modelscope_model, label="modelscope", temperature=temperature)


class FallbackClient(LLMClient):
    """降级链客户端：按顺序尝试多个供应商，失败自动切换

    降级链: qwen → zhipu → deepseek（由 PW_FALLBACK_CHAIN 配置）
    当 Qwen 次数用完或不可用时，自动切换到 ZhipuAI GLM，
    两个都不可用时回退到 DeepSeek。

    temperature 透传给链上各供应商（module-026：反思低温度贯穿降级链）。
    """

    def __init__(self, chain: list[str], temperature: float = 0.7):
        if not chain:
            raise LLMException("fallback", "降级链为空")
        self._chain = chain
        self._temperature = temperature
        logger.info("Fallback 降级链: %s", " → ".join(chain))

    async def generate(self, prompt: str) -> str:
        last_error = None
        for provider in self._chain:
            try:
                client = LLMFactory.get_client(provider, temperature=self._temperature)
                return await client.generate(prompt)
            except Exception as e:
                last_error = e
                logger.warning("降级链 %s 失败，尝试下一个: %s", provider, e)
        raise last_error or LLMException("fallback", "降级链所有供应商均失败")

    async def chat(self, messages: list[dict]) -> str:
        last_error = None
        for provider in self._chain:
            try:
                client = LLMFactory.get_client(provider, temperature=self._temperature)
                return await client.chat(messages)
            except Exception as e:
                last_error = e
                logger.warning("降级链 %s chat 失败，尝试下一个: %s", provider, e)
        raise last_error or LLMException("fallback", "降级链所有供应商 chat 均失败")

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """多轮对话 + 工具调用（遍历降级链，module-028）

        与 chat 同款降级语义：逐供应商尝试，成功即返回，全部失败抛 LLMException。
        """
        last_error = None
        for provider in self._chain:
            try:
                client = LLMFactory.get_client(provider, temperature=self._temperature)
                return await client.chat_with_tools(messages, tools)
            except Exception as e:
                last_error = e
                logger.warning("降级链 %s 工具调用失败，尝试下一个: %s", provider, e)
        raise last_error or LLMException("fallback", "降级链所有供应商工具调用均失败")

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        last_error = None
        for provider in self._chain:
            try:
                client = LLMFactory.get_client(provider, temperature=self._temperature)
                async for chunk in client.generate_stream(prompt):
                    yield chunk
                return  # 成功，退出
            except Exception as e:
                last_error = e
                logger.warning("降级链 %s stream 失败，尝试下一个: %s", provider, e)
        raise last_error or LLMException("fallback", "降级链所有供应商流式均失败")


class LLMFactory:
    """LLM 客户端工厂，根据配置返回对应实例

    工厂模式的好处：
    1. 调用方不需要知道具体客户端类的存在
    2. 实例缓存（_instances）避免重复创建
    3. 切换供应商只需改配置文件中的 llm_provider

    使用示例：
        client = LLMFactory.get_client()           # 默认供应商（fallback 降级链）
        client = LLMFactory.get_client("qwen")     # 指定 Qwen
        client = LLMFactory.get_client("zhipu")    # 指定 ZhipuAI GLM
        client = LLMFactory.get_client("modelscope")  # 指定 ModelScope
    """

    _instances: dict[tuple[str, float], LLMClient] = {}

    # 运行时降级链（module-029）：由 GET/PUT /ai/llm/chain 或启动时从 Redis
    # 加载；None 表示未覆盖，get_fallback_chain 回退到配置默认。
    _fallback_chain: Optional[list[str]] = None

    # 降级链白名单：链上的每一项必须是可实例化的单供应商（不允许嵌套 fallback）
    SUPPORTED_PROVIDERS = {"claude", "deepseek", "qwen", "zhipu", "modelscope"}

    @classmethod
    def validate_chain(cls, chain: list) -> list[str]:
        """校验降级链合法性：非空、全为支持供应商、无重复

        Args:
            chain: 待校验的供应商列表

        Returns:
            规范化后的供应商列表（去除空白、统一小写）

        Raises:
            ValueError: 链为空 / 含未知供应商 / 供应商重复 / 元素非字符串
        """
        if not isinstance(chain, list) or not chain:
            raise ValueError("降级链不能为空")
        cleaned: list[str] = []
        for p in chain:
            if not isinstance(p, str):
                raise ValueError(f"非法供应商类型: {p!r}")
            p = p.strip().lower()
            if p not in cls.SUPPORTED_PROVIDERS:
                raise ValueError(f"不支持的供应商: {p}")
            if p in cleaned:
                raise ValueError(f"供应商重复: {p}")
            cleaned.append(p)
        return cleaned

    @classmethod
    def set_fallback_chain(cls, chain: list[str]) -> None:
        """设置运行时降级链（跨请求生效，无需重启服务）

        注意：调用方需自行先 clear_cache() 使已缓存的 FallbackClient 失效，
        否则已存在的实例仍持有旧链。

        Args:
            chain: 校验通过的供应商列表
        """
        cls._fallback_chain = list(chain)
        logger.info("运行时降级链已设置: %s", " → ".join(chain))

    @classmethod
    def get_fallback_chain(cls) -> list[str]:
        """获取当前降级链（运行时覆盖优先，否则配置默认）

        Returns:
            供应商列表（如 ["qwen", "zhipu", "deepseek"]）
        """
        if cls._fallback_chain:
            return list(cls._fallback_chain)
        return [p.strip() for p in settings.fallback_chain.split(",") if p.strip()]

    @classmethod
    def get_client(cls, provider: Optional[str] = None,
                   temperature: Optional[float] = None) -> LLMClient:
        """获取 LLM 客户端实例（支持按温度创建）

        实例化后缓存，后续调用直接返回缓存的实例。
        如果调用方没有指定 provider，使用配置文件中的默认供应商。

        temperature：覆盖默认温度。None 用默认 0.7（与历史行为一致）；
        传 0.1 等低温度用于反思等结构化 JSON 任务（module-026）。
        实例按 (provider, temperature) 缓存，不同温度各建独立实例，
        不影响其他调用方使用的默认温度实例。

        fallback（module-029）：链来源为运行时链（Redis 持久化）优先，
        否则配置默认；clear_cache 后按新链重建。

        Args:
            provider: 供应商（claude/deepseek/qwen/zhipu/modelscope/fallback）
            temperature: 生成温度（None=默认 0.7）

        Returns:
            LLM 客户端实例

        Raises:
            ValueError: 不支持的供应商或 fallback 链为空
        """
        provider = provider or settings.llm_provider
        temp = 0.7 if temperature is None else temperature
        key = (provider, temp)
        if key not in cls._instances:
            if provider == "claude":
                cls._instances[key] = ClaudeClient(temperature=temp)
            elif provider == "deepseek":
                cls._instances[key] = DeepSeekClient(temperature=temp)
            elif provider == "modelscope":
                cls._instances[key] = ModelScopeClient(temperature=temp)
            elif provider == "qwen":
                cls._instances[key] = QwenClient(temperature=temp)
            elif provider == "zhipu":
                cls._instances[key] = ZhipuClient(temperature=temp)
            elif provider == "fallback":
                chain = cls.get_fallback_chain()
                if not chain:
                    raise ValueError("PW_FALLBACK_CHAIN 为空，无法创建降级客户端")
                cls._instances[key] = FallbackClient(chain, temperature=temp)
            else:
                raise ValueError(f"不支持的 LLM 供应商: {provider}")
        return cls._instances[key]

    @classmethod
    def clear_cache(cls):
        """清理客户端缓存

        在切换供应商或调整降级链时调用（module-029 调序后重建 FallbackClient）。
        清理后下次 get_client 按最新链/配置重建全部实例。
        """
        cls._instances.clear()
