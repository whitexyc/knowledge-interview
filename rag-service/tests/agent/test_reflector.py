"""Module-039 证据链幻觉检测 Reflector 单元测试

覆盖（验收 §5.1）：
- verify_answer 正常返回测试（supported 文档 + 预期 claims）
- verify_answer 空文档降级测试（空 docs → 返回空 claims）
- verify_answer 异常降级测试（LLM 错误 → 返回空 claims，不抛异常）
- _parse_verification JSON 解析健壮性
- evidence 引用号越界降级

实现说明：
- 用 mock 打桩 LLMFactory.get_client，不依赖真实 LLM
- 同步用例内 asyncio.run 执行，不依赖 pytest-asyncio（沿用既有模式）
"""
import asyncio
import json
from unittest import mock

import pytest

from agent.reflector import Reflector


class TestVerifyAnswer:
    """Reflector.verify_answer 证据链验证"""

    @staticmethod
    def _sample_docs():
        return [
            {"id": 1, "title": "线程池基础", "source": "test",
             "content": "线程池核心参数包括核心线程数、最大线程数、队列容量。"},
            {"id": 2, "title": "线程池配置", "source": "test",
             "content": "最大线程数根据CPU密集型任务设置为核心数的2倍。"},
        ]

    @staticmethod
    def _valid_json_response():
        return json.dumps([
            {"claim": "线程池核心参数包括核心线程数、最大线程数、队列容量",
             "verdict": "supported", "evidence": "[1]"},
            {"claim": "最大线程数根据CPU密集型任务设置为核心数的2倍",
             "verdict": "supported", "evidence": "[2]"},
            {"claim": "建议使用无界队列避免任务丢失",
             "verdict": "unsupported", "evidence": "N/A"},
        ], ensure_ascii=False)

    def test_verify_answer_returns_claims(self):
        """正常路径：LLM 返回合法 JSON → 完整验证结果含 claims / overall_confidence"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value=self._valid_json_response())
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.verify_answer(
                    "线程池核心参数包括核心线程数、最大线程数、队列容量[1]。"
                    "最大线程数根据CPU密集型任务设置[2]。建议使用无界队列。",
                    docs,
                )
            return result

        result = asyncio.run(run())
        assert result["total_claims"] == 3
        assert result["supported"] == 2
        assert result["inferred"] == 0
        assert result["unsupported"] == 1
        assert result["overall_confidence"] == pytest.approx(0.6667, abs=0.01)
        assert result["claims"][0]["verdict"] == "supported"
        assert result["claims"][0]["evidence"] == "[1]"
        assert result["claims"][2]["verdict"] == "unsupported"
        assert result["claims"][2]["evidence"] == "N/A"

    def test_verify_answer_empty_docs(self):
        """空文档降级：docs 为空 → 返回空 claims（零回归）"""
        async def run():
            r = Reflector()
            result = await r.verify_answer("任意答案", [])
            return result

        result = asyncio.run(run())
        assert result["claims"] == []
        assert result["total_claims"] == 0
        assert result["supported"] == 0
        assert result["inferred"] == 0
        assert result["unsupported"] == 0
        assert result["overall_confidence"] == 0.0

    def test_verify_answer_handles_llm_error(self):
        """LLM 调用异常 → 返回空 claims，不抛异常"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(side_effect=RuntimeError("LLM 服务不可用"))
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.verify_answer("线程池很强大[1]", docs)
            return result

        result = asyncio.run(run())
        assert result["claims"] == []
        assert result["overall_confidence"] == 0.0

    def test_verify_answer_empty_answer_text(self):
        """空答案文本 → 返回空 claims"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            result = await r.verify_answer("", docs)
            return result

        result = asyncio.run(run())
        assert result["claims"] == []
        assert result["total_claims"] == 0

    def test_verify_answer_evidence_out_of_bounds(self):
        """evidence 引用号越界 → 对应的 claim verdict 降级为 unsupported"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()  # 只有 2 篇文档
            # LLM 返回的证据引用号 [5] 超出 docs 数量 → 应为 unsupported
            response = json.dumps([
                {"claim": "c1", "verdict": "supported", "evidence": "[1]"},
                {"claim": "c2", "verdict": "supported", "evidence": "[5]"},
            ], ensure_ascii=False)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value=response)
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.verify_answer("a[1] b[5]", docs)
            return result

        result = asyncio.run(run())
        assert result["total_claims"] == 2
        # c1 保持 supported；c2 被降级为 unsupported（证据号越界）
        assert result["claims"][0]["verdict"] == "supported"
        assert result["claims"][1]["verdict"] == "unsupported"
        assert result["claims"][1]["evidence"] == "N/A"
        assert result["supported"] == 1
        assert result["unsupported"] == 1

    def test_verify_answer_all_supported(self):
        """全部 supported 的正常答案 → overall_confidence == 1.0"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            response = json.dumps([
                {"claim": "全部正确", "verdict": "supported", "evidence": "[1]"},
            ], ensure_ascii=False)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value=response)
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.verify_answer("全部正确[1]", docs)
            return result

        result = asyncio.run(run())
        assert result["supported"] == 1
        assert result["unsupported"] == 0
        assert result["overall_confidence"] == 1.0


class TestGenerateAnswerWithScratchpad:
    """module-041: generate_answer 读取 scratchpad 工作笔记

    覆盖验收 4.1:
    - scratchpad 非空时 generate_answer prompt 注入工作笔记段
    - 空 scratchpad 零回归（prompt 不含工作笔记段）
    """

    @staticmethod
    def _sample_docs():
        return [
            {"id": 1, "title": "线程池基础", "source": "test",
             "content": "线程池核心参数包括核心线程数、最大线程数、队列容量。"},
        ]

    def test_generate_answer_includes_scratchpad(self):
        """scratchpad 非空时 generate_answer prompt 注入工作笔记段"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value="含笔记的答案")
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.generate_answer(
                    "测试问题", docs,
                    scratchpad=["发现1", "发现2"],
                )
                prompt_arg = client.generate.call_args[0][0]
            return result, prompt_arg

        result, prompt = asyncio.run(run())
        assert result == "含笔记的答案"
        assert "[工作笔记" in prompt
        assert "发现1" in prompt
        assert "发现2" in prompt

    def test_generate_answer_no_scratchpad_zero_regression(self):
        """空 scratchpad 时 generate_answer 零回归（prompt 不含工作笔记段）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value="正常答案")
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.generate_answer("测试问题", docs)
                prompt_arg = client.generate.call_args[0][0]
            return result, prompt_arg

        result, prompt = asyncio.run(run())
        assert result == "正常答案"
        assert "[工作笔记" not in prompt


class TestCheckSufficiencyGates:
    """module-044: check_sufficiency 层 1 分数/数量硬闸门 + 层 2 prompt 强化 + 层 3 多信号融合

    覆盖（验收 §1/§2/§5/§7，ADR-0005 层 1-3）：
    - 分数闸门：top-1 abs_cosine < 0.55 → 直接不充分 + rewritten_query，零 LLM 调用
    - 分数闸门 module-048：0.45 在旧阈值 0.4 下会漏判进 LLM → 新阈值直接判不充分
    - 数量闸门：文档数 < 2 → 直接不充分 + rewritten_query，零 LLM 调用
    - 达标走 LLM：≥0.55 → 才进 LLM 判模糊地带；LLM 判不充分 → 尊重语义走 rewritten_query
    - prompt 结构：_CHECK_PROMPT 含 few-shot 正反例 + CoT 信息点比对步骤
    - 自洽性检查：默认关（单次调用，零额外）；开启时两温度各判一次、不一致 → 保守充分
    - 降级：abs_cosine 缺失 → 走 LLM（不误杀）；LLM 异常 → 默认充分（防死循环）

    实现说明：与既有用例同风格——mock 打桩 LLMFactory.get_client，
    asyncio.run 执行，不依赖 pytest-asyncio。阈值读配置
    settings.sufficiency_gate_threshold（module-048 默认 0.55）。
    """

    @staticmethod
    def _sample_docs(top1_abs: float = 0.7):
        """两篇带 abs_cosine 的文档（与 retriever 归一化前存档口径一致）"""
        return [
            {"id": 1, "title": "线程池基础",
             "content": "线程池核心参数包括核心线程数、最大线程数、队列容量。",
             "abs_cosine": top1_abs},
            {"id": 2, "title": "线程池配置",
             "content": "最大线程数根据CPU密集型任务设置为核心数的2倍。",
             "abs_cosine": top1_abs - 0.1},
        ]

    def test_gate_top1_score_below_threshold_no_llm(self):
        """层 1 分数闸门：top-1 abs_cosine=0.25 < 0.55 → 直接不充分，零 LLM 调用"""
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.25)
            with mock.patch("llm.client.LLMFactory.get_client") as mock_get:
                result = await r.check_sufficiency("线程池参数", docs)
            return result, mock_get

        result, mock_get = asyncio.run(run())
        assert result["sufficient"] is False
        assert result["rewritten_query"] == "线程池参数"
        assert "0.55" in result["reason"]
        mock_get.assert_not_called()  # 零 LLM 调用断言

    def test_gate_threshold_0_55_catches_0_45(self):
        """module-048 AC 场景 4：top-1 abs_cosine=0.45 → 直接判不充分（零 LLM）

        旧阈值 0.4 下 0.45 会漏判进 LLM（module-047 实测漏判 60% 不充分）；
        0.55 切在分布间隙上缘（充分 min 0.490 / 不充分 max 0.550）。
        """
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.45)
            with mock.patch("llm.client.LLMFactory.get_client") as mock_get:
                result = await r.check_sufficiency("线程池参数", docs)
            return result, mock_get

        result, mock_get = asyncio.run(run())
        assert result["sufficient"] is False
        assert result["rewritten_query"] == "线程池参数"
        assert "0.55" in result["reason"]
        mock_get.assert_not_called()  # 硬闸门零 LLM 调用断言

    def test_gate_fewer_than_two_docs_no_llm(self):
        """层 1 数量闸门：只有 1 篇文档 → 直接不充分，零 LLM 调用"""
        async def run():
            r = Reflector()
            docs = [self._sample_docs()[0]]
            with mock.patch("llm.client.LLMFactory.get_client") as mock_get:
                result = await r.check_sufficiency("线程池参数", docs)
            return result, mock_get

        result, mock_get = asyncio.run(run())
        assert result["sufficient"] is False
        assert result["rewritten_query"] == "线程池参数"
        mock_get.assert_not_called()

    def test_score_passes_gate_goes_llm(self):
        """分数达标（0.7 ≥ 0.55）→ 才进 LLM 判模糊地带；LLM 判充分 → sufficient=true（零回归）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.7)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "文档覆盖核心参数"}')
            with mock.patch("llm.client.LLMFactory.get_client",
                            return_value=client) as mock_get:
                result = await r.check_sufficiency("线程池参数", docs)
            return result, mock_get, client

        result, mock_get, client = asyncio.run(run())
        assert result["sufficient"] is True
        mock_get.assert_called_once()  # 自洽性默认关：恰好一次 LLM 调用（零额外）
        client.generate.assert_called_once()

    def test_llm_insufficient_respected_high_score(self):
        """层 3 语义尊重：分数高但 LLM 判不充分 → 尊重语义走 rewritten_query（不强制充分）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.7)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(return_value=json.dumps(
                {"sufficient": False, "reason": "缺停顿时间预测模型",
                 "rewritten_query": "G1 GC 停顿时间预测模型"}, ensure_ascii=False))
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.check_sufficiency("G1 GC 停顿时间模型", docs)
            return result

        result = asyncio.run(run())
        assert result["sufficient"] is False
        assert result["rewritten_query"] == "G1 GC 停顿时间预测模型"

    def test_prompt_has_few_shot_and_cot(self):
        """层 2 prompt 强化：_CHECK_PROMPT 含 few-shot 正反例 + CoT 信息点比对步骤"""
        from agent.reflector import _CHECK_PROMPT
        # few-shot：充分/不充分正反例各 ≥1
        assert "示例 1" in _CHECK_PROMPT and "示例 2" in _CHECK_PROMPT
        # CoT：先列所需信息点，再逐点比对文档覆盖，再下结论
        assert "信息点" in _CHECK_PROMPT
        assert "判断步骤" in _CHECK_PROMPT
        # 返回 JSON 结构不变（向后兼容）
        assert '"sufficient": true' in _CHECK_PROMPT
        assert '"sufficient": false' in _CHECK_PROMPT

    def test_self_check_enabled_consistent_uses_result(self):
        """自洽性开启：两次判断一致 → 采用判断结果（两温度各判一次）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.7)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                side_effect=['{"sufficient": true, "reason": "ok"}',
                             '{"sufficient": true, "reason": "ok2"}'])
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                with mock.patch(
                        "agent.reflector.settings.sufficiency_self_check_enabled",
                        True):
                    result = await r.check_sufficiency("线程池参数", docs)
            return result, client

        result, client = asyncio.run(run())
        assert result["sufficient"] is True
        assert client.generate.call_count == 2  # 两温度各判一次

    def test_self_check_enabled_disagree_conservative_sufficient(self):
        """自洽性开启：两次判断不一致 → 保守判充分（防漏检，防死循环哲学）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.7)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                side_effect=['{"sufficient": true, "reason": "ok"}',
                             '{"sufficient": false, "reason": "no", "rewritten_query": "q"}'])
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                with mock.patch(
                        "agent.reflector.settings.sufficiency_self_check_enabled",
                        True):
                    result = await r.check_sufficiency("线程池参数", docs)
            return result, client

        result, client = asyncio.run(run())
        assert result["sufficient"] is True
        assert client.generate.call_count == 2

    def test_degrade_missing_abs_cosine_goes_llm(self):
        """降级：abs_cosine 字段缺失（如仅 FTS 命中）→ 不误杀，继续走 LLM"""
        async def run():
            r = Reflector()
            docs = [
                {"id": 1, "title": "T1", "content": "内容1"},
                {"id": 2, "title": "T2", "content": "内容2"},
            ]  # 无 abs_cosine 字段
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "ok"}')
            with mock.patch("llm.client.LLMFactory.get_client",
                            return_value=client) as mock_get:
                result = await r.check_sufficiency("线程池参数", docs)
            return result, mock_get

        result, mock_get = asyncio.run(run())
        assert result["sufficient"] is True
        mock_get.assert_called_once()

    def test_degrade_llm_exception_conservative_sufficient(self):
        """降级：LLM 异常 → 默认充分（防死循环，现有行为保留）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs(top1_abs=0.7)
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(side_effect=RuntimeError("LLM 不可用"))
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                result = await r.check_sufficiency("线程池参数", docs)
            return result

        result = asyncio.run(run())
        assert result["sufficient"] is True
        assert "默认通过" in result["reason"]


class TestParseVerification:
    """_parse_verification JSON 解析健壮性"""

    def test_valid_json_array(self):
        """合法 JSON 数组 → 正确解析为 claims"""
        response = '[{"claim": "c1", "verdict": "supported", "evidence": "[1]"}]'
        claims = Reflector._parse_verification(response)
        assert len(claims) == 1
        assert claims[0]["claim"] == "c1"
        assert claims[0]["verdict"] == "supported"

    def test_markdown_wrapped_json(self):
        """LLM 在 markdown 代码块中包裹 JSON → 成功解析"""
        response = '```json\n[{"claim": "c1", "verdict": "inferred", "evidence": "[1]"}]\n```'
        claims = Reflector._parse_verification(response)
        assert len(claims) == 1
        assert claims[0]["verdict"] == "inferred"

    def test_extra_text_before_json(self):
        """LLM 在 JSON 前加解释文字 → 仍成功提取"""
        response = '以下是验证结果：\n[{"claim": "c1", "verdict": "supported", "evidence": "[1]"}]'
        claims = Reflector._parse_verification(response)
        assert len(claims) == 1
        assert claims[0]["verdict"] == "supported"

    def test_invalid_json_returns_empty(self):
        """完全无法解析的响应 → 返回空列表"""
        claims = Reflector._parse_verification("纯文本无JSON结构！")
        assert claims == []

    def test_empty_claims_filtered_out(self):
        """JSON 中空 claim 条目被过滤"""
        response = '[{"claim": "", "verdict": "supported", "evidence": "[1]"}, {"claim": "有效", "verdict": "inferred", "evidence": "[2]"}]'
        claims = Reflector._parse_verification(response)
        assert len(claims) == 1
        assert claims[0]["claim"] == "有效"

    def test_missing_verdict_defaults_to_unsupported(self):
        """缺少 verdict 字段 → 默认 unsupported"""
        response = '[{"claim": "c1", "evidence": "[1]"}]'
        claims = Reflector._parse_verification(response)
        assert len(claims) == 1
        assert claims[0]["verdict"] == "unsupported"

    def test_missing_evidence_defaults_to_na(self):
        """缺少 evidence 字段 → 默认 N/A"""
        response = '[{"claim": "c1", "verdict": "supported"}]'
        claims = Reflector._parse_verification(response)
        assert len(claims) == 1
        assert claims[0]["evidence"] == "N/A"
