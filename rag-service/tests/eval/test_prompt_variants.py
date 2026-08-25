"""
module-055 / ADR-0011 第一步：Prompt 变体测试 + reflector prompt 可注入参数
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

覆盖（AC §1 / §8）：
- check_sufficiency(prompt=...) 参数注入：自定义 prompt 传给 LLM、默认 _CHECK_PROMPT
  零回归、自洽性检查第二判同用注入变体（对比口径一致）
- eval/prompt_variants.py：变体定义（占位符齐全校验 / 未知变体报错 / 默认全量）、
  对比表生成（Accuracy / insufficient Recall / kappa / 耗时）、CLI 参数解析
- 只度量不替换：生产默认 prompt 恒为 baseline，变体不影响默认行为
"""
import asyncio
import json
from unittest import mock

import pytest

from agent.reflector import Reflector, _CHECK_PROMPT


class TestCheckSufficiencyPromptInjection:
    """reflector prompt 常量可注入参数（默认不变零回归）"""

    @staticmethod
    def _sample_docs(top1_abs: float = 0.7):
        """两篇带 abs_cosine 的文档（与 test_reflector 同款，越过层 1 闸门）"""
        return [
            {"id": 1, "title": "线程池基础",
             "content": "线程池核心参数包括核心线程数、最大线程数、队列容量。",
             "abs_cosine": top1_abs},
            {"id": 2, "title": "线程池配置",
             "content": "最大线程数根据CPU密集型任务设置为核心数的2倍。",
             "abs_cosine": top1_abs - 0.1},
        ]

    def test_custom_prompt_passed_to_llm(self):
        """注入变体 prompt → client.generate 收到的是自定义 prompt 格式化结果"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            custom = "自定义检查: {query} | 摘要: {docs_summary}"
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "ok"}')
            with mock.patch("llm.client.LLMFactory.get_client",
                            return_value=client) as mock_get:
                result = await r.check_sufficiency("线程池参数", docs, prompt=custom)
            return result, mock_get, client

        result, mock_get, client = asyncio.run(run())
        assert result["sufficient"] is True
        mock_get.assert_called_once()
        args = client.generate.call_args
        assert "自定义检查: 线程池参数" in args[0][0]
        assert "摘要:" in args[0][0]
        assert "判断步骤" not in args[0][0]  # 非默认 prompt

    def test_default_prompt_zero_regression(self):
        """prompt=None → 用模块默认 _CHECK_PROMPT（逐字节格式化，零回归）"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            client = mock.MagicMock()
            client.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "ok"}')
            with mock.patch("llm.client.LLMFactory.get_client", return_value=client):
                await r.check_sufficiency("线程池参数", docs)
            return client

        client = asyncio.run(run())
        sent = client.generate.call_args[0][0]
        expected = _CHECK_PROMPT.format(
            query="线程池参数",
            docs_summary=(
                "- [1] 线程池基础: 线程池核心参数包括核心线程数、最大线程数、队列容量。\n"
                "- [2] 线程池配置: 最大线程数根据CPU密集型任务设置为核心数的2倍。"),
        )
        assert sent == expected  # 与生产 prompt 完全一致

    def test_injection_respects_self_check_second_call(self):
        """自洽性开启时第二判（不同温度）同用注入变体——对比口径一致"""
        async def run():
            r = Reflector()
            docs = self._sample_docs()
            custom = "变体X: {query} | {docs_summary}"
            client1 = mock.MagicMock()
            client1.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "ok"}')
            client2 = mock.MagicMock()
            client2.generate = mock.AsyncMock(
                return_value='{"sufficient": true, "reason": "ok2"}')
            with mock.patch("llm.client.LLMFactory.get_client",
                            side_effect=[client1, client2]):
                with mock.patch("src.config.settings.sufficiency_self_check_enabled", True):
                    result = await r.check_sufficiency("线程池参数", docs, prompt=custom)
            return result, client1, client2

        result, client1, client2 = asyncio.run(run())
        assert result["sufficient"] is True
        for cli in (client1, client2):
            assert "变体X: 线程池参数" in cli.generate.call_args[0][0]


class TestPromptVariantsModule:
    """eval/prompt_variants.py：变体定义 / 指标 / 对比表 / CLI"""

    def test_load_variants_default_all_with_placeholders(self):
        from eval.benchmarks.prompt_variants import load_variants, VARIANT_BUILDERS
        variants = load_variants(None)
        assert set(variants) == set(VARIANT_BUILDERS)
        assert "baseline" in variants
        for name, text in variants.items():
            assert "{query}" in text and "{docs_summary}" in text

    def test_baseline_is_production_prompt(self):
        """只度量不替换：baseline 变体 = 生产默认 _CHECK_PROMPT"""
        from eval.benchmarks.prompt_variants import load_variants
        variants = load_variants(["baseline"])
        assert variants["baseline"] == _CHECK_PROMPT

    def test_load_unknown_variant_raises(self):
        from eval.benchmarks.prompt_variants import load_variants
        with pytest.raises(ValueError, match="未知变体"):
            load_variants(["not_exist"])

    def test_load_selected_subset(self):
        from eval.benchmarks.prompt_variants import load_variants
        variants = load_variants(["baseline", "v_brief"])
        assert set(variants) == {"baseline", "v_brief"}

    def test_compute_variant_metrics(self):
        from eval.benchmarks.prompt_variants import compute_variant_metrics
        m = compute_variant_metrics(
            labels=[True, True, False, False],
            predictions=[True, True, False, True],  # 1 个不充分漏判
        )
        assert m["accuracy"] == 0.75
        assert m["insufficient_recall"] == 0.5
        assert m["kappa"] == 0.5  # 2x2 完全一致/随机混合

    def test_compute_metrics_single_class_kappa_zero(self):
        """单类样本 kappa 无意义 → 0.0（如实标注，不抛错）"""
        from eval.benchmarks.prompt_variants import compute_variant_metrics
        m = compute_variant_metrics(labels=[True, True], predictions=[True, True])
        assert m["accuracy"] == 1.0
        assert m["kappa"] == 0.0

    def test_run_variant_fixture_produces_metrics(self):
        """fixture 模式（零 LLM 零 DB）跑通 run_variant → 指标结构齐全"""
        from eval.benchmarks.prompt_variants import run_variant
        dataset = [
            {"question": "G1 核心创新", "documents": [{"content": "Region 分区"}],
             "sufficient": True, "keywords": ["Region"]},
            {"question": "停顿模型", "documents": [{"content": "G1 基本概念"}],
             "sufficient": False, "keywords": ["停顿"]},
        ]
        result = asyncio.run(run_variant(
            "baseline", "p {query} {docs_summary}", dataset,
            use_heuristic=True, limit=None))
        assert result["name"] == "baseline"
        assert result["evaluated"] == 2
        assert result["skipped"] == 0
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["insufficient_recall"] <= 1.0
        assert result["per_question"][0]["variant"] == "baseline"

    def test_cli_parsing(self):
        """CLI：--variant 逗号分隔 / --limit / --save / --fixture"""
        from eval.benchmarks.prompt_variants import main as main_fn
        with mock.patch("sys.argv",
                        ["eval.benchmarks.prompt_variants", "--variant", "baseline,v_brief",
                         "--limit", "3", "--no-save", "--fixture"]):
            with mock.patch("eval.benchmarks.prompt_variants.run_variant") as mock_run:
                mock_run.return_value = {
                    "name": "x", "note": "", "elapsed": 1.0, "evaluated": 3,
                    "skipped": 0, "accuracy": 1.0, "insufficient_recall": 0.0,
                    "kappa": 0.0, "per_question": [],
                }
                asyncio.run(main_fn())
        assert mock_run.call_count == 2  # 仅所选 2 个变体
        assert mock_run.call_args.kwargs["limit"] == 3
        assert mock_run.call_args.kwargs["use_heuristic"] is True

    def test_print_comparison_output(self, capsys):
        """对比表输出包含表头与每变体行"""
        from eval.benchmarks.prompt_variants import print_comparison
        print_comparison([
            {"name": "baseline", "note": "默认", "elapsed": 1.5, "evaluated": 10,
             "skipped": 0, "accuracy": 0.9, "insufficient_recall": 0.8, "kappa": 0.7},
        ], fixture=False)
        out = capsys.readouterr().out
        assert "Prompt Variant Comparison" in out
        assert "baseline" in out
        assert "kappa" in out
