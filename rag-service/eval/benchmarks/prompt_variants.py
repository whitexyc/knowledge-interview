"""
Prompt 变体测试脚本 — 同任务不同 prompt 消融对比（ADR-0011 第一步，module-055）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.prompt_variants                        # 全部变体跑 golden_sufficiency，不落库
    python -m eval.benchmarks.prompt_variants --variant baseline,v_brief --limit 20   # 选变体 + 限量
    python -m eval.benchmarks.prompt_variants --no-save              # 显式不落库（默认行为，CLI 对齐 golden 系）
    python -m eval.benchmarks.prompt_variants --save                 # 每变体落 eval_runs（eval_type='prompt_variant'）
    python -m eval.benchmarks.prompt_variants --fixture              # 启发式判断器（管线演示，零 LLM 零 DB）

设计（ADR-0011 第一步：变体测试）：
    - 变体 = 同任务（check_sufficiency 反思）不同 prompt 文本，逐个跑同一
      golden 评测集 → 对比表（Accuracy / insufficient Recall / kappa / 耗时）
    - 只度量不替换：生产 prompt（agent.reflector._CHECK_PROMPT）恒为
      baseline 变体，变体经 check_sufficiency(prompt=...) 注入，不改任何默认
      行为（零回归，reflector 参数注入测试见 tests/test_prompt_variants.py）
    - 指标对齐 golden_sufficiency（层 0 口径）：Accuracy + insufficient
      Recall（漏判"不充分"最致命）+ kappa（cohen_kappa_score，两态）
    - 降级：单条判断失败 → 跳过记录；LLM/DB 不可用 → 用 --fixture 演示管线

指标定义（对齐 ADR-0005 层 0 / module-047 真实 baseline 口径）:
    Accuracy              全部样本判对比例
    insufficient_recall   不充分类 Recall（漏判不充分 → 基于无关文档硬答，最致命）
    kappa                 Cohen's kappa（两态充分性标签的一致性）
"""
import argparse
import asyncio
import logging
import time

from sklearn.metrics import cohen_kappa_score

from eval.golden.golden_sufficiency import (
    label_str,
    load_sufficiency_dataset,
    run_eval,
    heuristic_judge,
)
from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("prompt_variants")


# ── 变体定义（module-055 / ADR-0011 第一步）──
# baseline 恒为生产默认 _CHECK_PROMPT（只度量不替换）；其余为作者构造的同任务
# 异质 prompt（风格/倾向/结构不同维度），均含 {query}/{docs_summary} 占位符。
def _load_baseline() -> str:
    from agent.reflector import _CHECK_PROMPT
    return _CHECK_PROMPT


def _variant_brief() -> str:
    """简洁版：无 CoT 信息点比对，直接判断（测 CoT 步骤的增量价值）"""
    return """你是答案质量检查员。判断检索文档能否回答用户问题。

用户问题: {query}

检索到的文档摘要:
{docs_summary}

规则：
1. 文档与问题部分相关、间接相关 → 充分
2. 仅完全无关才判不充分

返回 JSON：{{"sufficient": true|false, "reason": "..."}}
不充分时附加 "rewritten_query": "改写的搜索关键词"。只返回 JSON。"""


def _variant_strict() -> str:
    """严格版：要求关键信息点显式覆盖才充分（测宽松/严格倾向）"""
    return """你是严格的答案质量检查员。

用户问题: {query}

检索到的文档摘要:
{docs_summary}

判断步骤：
1. 列出回答该问题必需的 2 个以上关键信息点
2. 逐点核对文档是否**显式覆盖**（仅间接提及不算覆盖）
3. 任一关键信息点未显式覆盖 → 不充分

返回 JSON：{{"sufficient": true|false, "reason": "..."}}
不充分时附加 "rewritten_query": "改写的搜索关键词"。只返回 JSON。"""


def _variant_fewshot() -> str:
    """few-shot 精简版：正反例 + 短规则，无 CoT 步骤（测 few-shot 单维度）"""
    return """你是答案质量检查员。判断检索文档能否回答用户问题。

用户问题: {query}

检索到的文档摘要:
{docs_summary}

示例 1（充分）：
问题: "Java 线程池的核心参数有哪些？"
文档: "[1] 线程池核心参数包括核心线程数、最大线程数、队列容量等"
返回: {{"sufficient": true, "reason": "文档[1]直接覆盖核心参数"}}

示例 2（不充分）：
问题: "G1 GC 的停顿时间预测模型是怎样的？"
文档: "[1] G1 GC 是面向服务端应用的垃圾回收器，基于 Region 划分堆内存"
返回: {{"sufficient": false, "reason": "仅介绍 G1 基本概念，未覆盖停顿时间预测模型", "rewritten_query": "G1 GC 停顿时间预测模型"}}

规则：部分相关/背景知识 → 充分；完全无关 → 不充分。只返回 JSON。"""


def _variant_conservative() -> str:
    """保守倾向版：默认充分哲学显式化（与生产哲学同向、措辞不同）"""
    return """你是答案质量检查员，默认倾向使用已有文档。

用户问题: {query}

检索到的文档摘要:
{docs_summary}

除非文档与问题完全不相关，一律判充分（宁可使用不完美的文档也不要空跑二次检索）。
完全不相关时才判不充分。

返回 JSON：{{"sufficient": true|false, "reason": "..."}}
不充分时附加 "rewritten_query": "改写的搜索关键词"。只返回 JSON。"""


VARIANT_BUILDERS: dict[str, callable] = {
    "baseline": _load_baseline,       # 生产默认（只度量不替换）
    "v_brief": _variant_brief,        # 简洁版（无 CoT）
    "v_strict": _variant_strict,      # 严格版（显式覆盖才充分）
    "v_fewshot": _variant_fewshot,    # few-shot 精简版（无 CoT 步骤）
    "v_conservative": _variant_conservative,  # 保守倾向版
}

# 变体说明（对比表表头注释用）
VARIANT_NOTES: dict[str, str] = {
    "baseline": "生产默认 _CHECK_PROMPT（CoT 信息点比对 + few-shot）",
    "v_brief": "无 CoT，直接判断（测 CoT 增量）",
    "v_strict": "关键信息点须显式覆盖（严格倾向）",
    "v_fewshot": "few-shot 正反例 + 短规则，无 CoT（测 few-shot 单维度）",
    "v_conservative": "默认充分哲学显式化（保守倾向）",
}


def load_variants(names: list[str] | None) -> dict[str, str]:
    """加载指定变体 prompt 文本（默认全部）

    Args:
        names: 变体名列表（None = 全部）

    Returns:
        {name: prompt_text}；未知变体名抛 ValueError
    """
    selected = names if names else list(VARIANT_BUILDERS)
    variants = {}
    for name in selected:
        if name not in VARIANT_BUILDERS:
            raise ValueError(
                f"未知变体: {name}，可选: {','.join(VARIANT_BUILDERS)}")
        text = VARIANT_BUILDERS[name]()
        # 校验占位符齐全（LLM 调用前 fail-fast，防线上才发现 format 失败）
        if "{query}" not in text or "{docs_summary}" not in text:
            raise ValueError(f"变体 {name} 缺少 {{{{query}}}}/{{{{docs_summary}}}} 占位符")
        variants[name] = text
    return variants


def compute_variant_metrics(labels: list[bool], predictions: list[bool]) -> dict:
    """计算单变体指标：Accuracy / insufficient Recall / kappa

    Args:
        labels: 真实充分性（bool 列表）
        predictions: 预测充分性（bool 列表）

    Returns:
        {"accuracy": float, "insufficient_recall": float, "kappa": float}
        样本过少（<2 或单类）时 kappa 记 0.0（无意义，如实标注）
    """
    n = len(labels)
    accuracy = round(sum(1 for l, p in zip(labels, predictions) if l == p) / n, 4) if n else 0.0
    ins_total = sum(1 for l in labels if not l)
    ins_hit = sum(1 for l, p in zip(labels, predictions) if not l and not p)
    ins_recall = round(ins_hit / ins_total, 4) if ins_total else 0.0
    kappa = 0.0
    try:
        if n >= 2 and len(set(labels)) == 2:
            kappa = round(float(cohen_kappa_score(
                [label_str(l) for l in labels],
                [label_str(p) for p in predictions],
            )), 4)
    except ValueError:
        kappa = 0.0
    return {"accuracy": accuracy, "insufficient_recall": ins_recall, "kappa": kappa}


async def run_variant(name: str, prompt: str, dataset: list[dict],
                      use_heuristic: bool, limit: int | None) -> dict:
    """跑单个变体的评测（golden_sufficiency 同口径 run_eval）

    Args:
        name: 变体名
        prompt: 变体 prompt 文本
        dataset: 评测样本（None 用全量）
        use_heuristic: True 用启发式判断器（fixture 模式，零 LLM）
        limit: 限量跑前 N 条（None = 全量）

    Returns:
        {"name", "note", "elapsed", "evaluated", "skipped", "accuracy",
         "insufficient_recall", "kappa", "per_question"}
    """
    from agent.reflector import Reflector

    items = dataset[:limit] if limit is not None else dataset

    if use_heuristic:
        async def judge(query, documents):
            item = next(i for i in dataset if i["question"] == query)
            return heuristic_judge(query, documents, item.get("keywords", []))
    else:
        reflector = Reflector()

        async def judge(query, documents):
            # 变体注入：prompt 参数仅影响 LLM 层 3 判断，层 1 硬闸门（数量/分数）
            # 与降级哲学不变（对齐生产行为）
            result = await reflector.check_sufficiency(query, documents, prompt=prompt)
            return bool(result.get("sufficient", True))

    t0 = time.perf_counter()
    scores, per_question, skipped = await run_eval(judge=judge, dataset=items)
    elapsed = round(time.perf_counter() - t0, 1)

    metrics = compute_variant_metrics(
        [q["label"] for q in per_question],
        [q["predicted"] for q in per_question],
    )
    for q in per_question:
        q["variant"] = name
    return {
        "name": name,
        "note": VARIANT_NOTES.get(name, ""),
        "elapsed": elapsed,
        "evaluated": scores["evaluated"],
        "skipped": scores["skipped"],
        "accuracy": metrics["accuracy"],
        "insufficient_recall": metrics["insufficient_recall"],
        "kappa": metrics["kappa"],
        "per_question": per_question,
    }


def print_comparison(results: list[dict], fixture: bool) -> None:
    """打印变体对比表：Accuracy / insufficient Recall / kappa / 耗时"""
    print("\n" + "=" * 78)
    print("Prompt Variant Comparison (check_sufficiency)" + ("  [fixture 启发式，非真实指标]" if fixture else ""))
    print("=" * 78)
    header = f"{'variant':<15}{'acc':>8}{'ins_recall':>13}{'kappa':>9}{'eval':>8}{'skip':>7}{'elapsed':>10}"
    print(header)
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<15}{r['accuracy']:>8.4f}{r['insufficient_recall']:>13.4f}"
              f"{r['kappa']:>9.4f}{r['evaluated']:>8}{r['skipped']:>7}{r['elapsed']:>9.1f}s")
    print("-" * 78)
    print("说明（只度量不替换生产 prompt）:")
    for r in results:
        print(f"  {r['name']:<15} {r['note']}")
    print("=" * 78)


async def save_variant_runs(results: list[dict]) -> list[int]:
    """每变体落 eval_runs（eval_type='prompt_variant'）

    Returns:
        落库成功的 id 列表（失败项被 save_eval_run 内部捕获，id=0）
    """
    commit = get_git_commit()
    config_snapshot = await load_rag_config()
    ids = []
    for r in results:
        scores = {
            "variant": r["name"],
            "note": r["note"],
            "accuracy": r["accuracy"],
            "insufficient_recall": r["insufficient_recall"],
            "kappa": r["kappa"],
            "elapsed": r["elapsed"],
            "evaluated": r["evaluated"],
            "skipped": r["skipped"],
        }
        saved_id = await save_eval_run(
            eval_type="prompt_variant",
            git_commit=commit,
            config_snapshot=config_snapshot,
            scores=scores,
            per_question=r["per_question"],
        )
        ids.append(saved_id)
        logger.info("变体 %s 已落库: id=%s", r["name"], saved_id)
    return ids


async def main() -> None:
    """变体测试入口"""
    parser = argparse.ArgumentParser(
        description="Prompt 变体测试：N 变体 × golden 评测 × 对比表（ADR-0011 第一步）")
    parser.add_argument("--variant", default="", help="逗号分隔变体名（默认全部，如 baseline,v_brief）")
    parser.add_argument("--save", action="store_true", help="每变体落 eval_runs（eval_type='prompt_variant'）")
    # --no-save 为显式默认（与 golden 各脚本 CLI 对齐，无操作）
    parser.add_argument("--no-save", action="store_true", help="不落库（默认行为，显式声明用）")
    parser.add_argument("--limit", type=int, default=None, help="每变体限量跑前 N 条（控制 LLM 成本）")
    parser.add_argument("--fixture", action="store_true", help="启发式判断器（管线演示，零 LLM 零 DB）")
    args = parser.parse_args()

    names = [n.strip() for n in args.variant.split(",") if n.strip()] or None
    variants = load_variants(names)
    dataset = load_sufficiency_dataset()

    results = []
    for name, prompt in variants.items():
        logger.info("跑变体 %s ...", name)
        results.append(await run_variant(
            name, prompt, dataset, use_heuristic=args.fixture, limit=args.limit))

    print_comparison(results, fixture=args.fixture)

    if args.save:
        ids = await save_variant_runs(results)
        print(f"Saved to eval_runs (ids={ids}, commit={get_git_commit()[:8]})")
    else:
        print("Not saved to eval_runs（--save 可落库，eval_type='prompt_variant'）")


if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
