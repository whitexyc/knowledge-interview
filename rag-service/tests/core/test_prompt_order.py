"""module-058 WP-B：_GENERATE_PROMPT 区块顺序测试

覆盖（验收 §1 功能验收）：
- 模板区块顺序：{sections} → 检索到的文档: {docs_detail} → 用户问题: {query}
  （docs 前移、query 最后，为前缀缓存铺路）
- 格式化后 prompt：docs_detail 内容在 query 之前
- query 标签格式不变（"用户问题: {query}" 一字不改）
- sections 内容/格式不变（历史/记忆/工作笔记段仍由 generate_answer 拼入，
  存量 scratchpad 用例已覆盖内容断言，本文件只断言区块顺序与标签）
- 空 sections 零漂移：无占位符残留、无多余空行

实现说明：mock 打桩 LLMFactory.get_client，asyncio.run 执行（沿用既有模式）。
"""
import asyncio
from unittest import mock

from agent.reflector import _GENERATE_PROMPT


class TestGeneratePromptOrder:
    """_GENERATE_PROMPT 区块顺序（docs 前移、query 最后）"""

    def test_template_block_order_docs_before_query(self):
        """模板区块顺序：sections → 检索到的文档 → 用户问题 → 回答"""
        idx_sections = _GENERATE_PROMPT.index("{sections}")
        idx_docs = _GENERATE_PROMPT.index("检索到的文档:")
        idx_query = _GENERATE_PROMPT.index("用户问题:")
        idx_answer = _GENERATE_PROMPT.index("回答：")
        assert idx_sections < idx_docs < idx_query < idx_answer

    def test_query_label_format_unchanged(self):
        """query 标签格式不变（"用户问题: {query}" 一字不改）"""
        assert "用户问题: {query}" in _GENERATE_PROMPT

    def test_docs_label_format_unchanged(self):
        """docs 标签格式不变（"检索到的文档:\n{docs_detail}" 结构保留）"""
        assert "检索到的文档:\n{docs_detail}" in _GENERATE_PROMPT

    def test_formatted_prompt_docs_before_query(self):
        """格式化后 prompt：docs 内容在 query 之前（前缀缓存的实际效果面）"""
        async def run():
            from agent.reflector import Reflector

            r = Reflector()
            docs = [{"id": 1, "title": "线程池基础", "source": "test",
                     "content": "线程池核心参数包括核心线程数。"}]
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value="答案")
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                await r.generate_answer("线程池有哪些参数？", docs)
            return client.generate.call_args[0][0]

        prompt = asyncio.run(run())
        assert prompt.index("线程池核心参数") < prompt.index("用户问题: 线程池有哪些参数？")

    def test_empty_sections_zero_drift(self):
        """空 sections（无历史/记忆/工作笔记）：无占位符残留、无多余内容"""
        prompt = _GENERATE_PROMPT.format(query="q", docs_detail="D", sections="")
        assert "{sections}" not in prompt
        assert "{query}" not in prompt
        assert "{docs_detail}" not in prompt
        assert "用户问题: q" in prompt
        assert "检索到的文档:\nD" in prompt

    def test_sections_still_prepended_in_generate_answer(self):
        """历史/记忆段仍拼在 prompt 最前（sections 内容拼接逻辑未变）"""
        async def run():
            from agent.reflector import Reflector

            r = Reflector()
            docs = [{"id": 1, "title": "t", "source": "s", "content": "内容"}]
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value="答案")
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                await r.generate_answer(
                    "q", docs, history=[{"role": "user", "content": "历史问题"}],
                    memory="历史记忆:\n- 某条记忆",
                )
            return client.generate.call_args[0][0]

        prompt = asyncio.run(run())
        # sections（历史对话/记忆）在最前，docs 次之，query 最后
        assert prompt.index("历史问题") < prompt.index("检索到的文档:") < prompt.index("用户问题: q")
