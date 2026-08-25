"""
RAGAS 评估脚本 — RAG 链路量化评测
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法:
    cd ai_service && python -m eval.golden.evaluate

流程:
    1. 加载 eval/dataset.json 测试集（30 题，6 类别）
    2. 逐题调用 rag_engine._retrieve() + reflector.generate_answer()
    3. 构建 RAGAS Dataset，运行 4 项指标评估
    4. 输出控制台报告 + eval/results.json

指标:
    - faithfulness: 答案是否忠实于提供的上下文
    - answer_relevancy: 答案与问题的相关性
    - context_precision: 检索到的上下文中相关部分的比例
    - context_recall: 检索到的上下文是否覆盖了 ground_truth

Judge LLM 配置:
    使用项目默认 DeepSeek 作为评估 LLM（temperature=0 保证一致性）。
"""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from rag.engine import rag_engine
from agent.reflector import reflector
from src.config import settings
from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ragas_eval")

# 本文件所在目录（eval/）
EVAL_DIR = Path(__file__).resolve().parent


def _build_judge_llm():
    """构建 RAGAS Judge LLM（DeepSeek, temperature=0）"""
    if not settings.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY 未配置，评估可能失败")
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=120,
    )


async def run_single(item: dict) -> dict:
    """对单条测试用例执行检索+生成

    Args:
        item: {"question": str, "ground_truth": str, "category": str}

    Returns:
        {"question", "answer", "contexts", "ground_truth", "category"}
    """
    question = item["question"]
    try:
        docs = await rag_engine._retrieve(question)
        answer = await reflector.generate_answer(question, docs)
        return {
            "question": question,
            "answer": answer,
            "contexts": [d.get("content", "") for d in docs],
            "ground_truth": item["ground_truth"],
            "category": item["category"],
        }
    except Exception as e:
        logger.error("题目执行失败 [%s]: %s", question[:40], e)
        return {
            "question": question,
            "answer": f"执行失败: {e}",
            "contexts": [],
            "ground_truth": item["ground_truth"],
            "category": item["category"],
        }


async def main():
    """评估主流程"""
    dataset_path = EVAL_DIR / "dataset.json"
    if not dataset_path.exists():
        logger.error("数据集文件不存在: %s", dataset_path)
        sys.exit(1)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    logger.info("加载数据集: %d 条问题", len(raw))

    # ── 1. 逐题执行检索+生成 ──
    logger.info("开始逐题评估...")
    start_time = time.time()
    results = []
    for i, item in enumerate(raw):
        logger.info("[%d/%d] %s", i + 1, len(raw), item["question"][:60])
        result = await run_single(item)
        results.append(result)

    elapsed = time.time() - start_time
    logger.info("评估完成，耗时: %.1f 秒", elapsed)

    # ── 2. 构建 RAGAS Dataset ──
    ds = Dataset.from_list(results)

    # ── 3. 运行 RAGAS 评估 ──
    judge_llm = _build_judge_llm()
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    logger.info("运行 RAGAS 评估 (4 项指标)...")

    try:
        scores = evaluate(ds, metrics=metrics, llm=judge_llm)
    except Exception as e:
        logger.error("RAGAS 评估失败: %s", e)
        scores = {"error": str(e)}

    # ── 4. 按类别汇总 ──
    per_category = {}
    for item, result in zip(raw, results):
        cat = item["category"]
        if cat not in per_category:
            per_category[cat] = {"count": 0}
        per_category[cat]["count"] += 1

    # 从 scores 中提取总体均值
    summary = {}
    if isinstance(scores, dict) and "error" not in scores:
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                summary[key] = round(value, 4)
            else:
                summary[key] = value

    # ── 5. 组装完整报告 ──
    report = {
        "metadata": {
            "dataset_size": len(raw),
            "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
            "judge_llm": f"{settings.deepseek_model} (temperature=0)",
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": summary,
        "per_category": per_category,
        "per_question": [
            {
                "question": r["question"],
                "answer": r["answer"][:300],
                "category": r["category"],
                "context_count": len(r["contexts"]),
            }
            for r in results
        ],
    }

    # ── 6. 输出 ──
    # 控制台报告
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Report — M13")
    print("=" * 60)
    print(f"Dataset: {len(raw)} questions, {len(per_category)} categories")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Judge LLM: {settings.deepseek_model} (temperature=0)")
    print("-" * 60)
    print("Metrics Summary:")
    for name, val in summary.items():
        print(f"  {name}: {val}")
    print("-" * 60)
    print("Per-Category Breakdown:")
    for cat, info in per_category.items():
        print(f"  {cat}: {info['count']} questions")
    print("=" * 60)

    # JSON 报告
    output_path = EVAL_DIR / "results.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("报告已保存到: %s", output_path)

    # ── 7. 版本化记录（module-019：注入 eval_runs） ──
    # 复用 golden_retrieval 的落库逻辑，eval_type='ragas'，
    # 使 RAGAS 生成侧评估结果也能与检索评估一起做版本化回归对比。
    saved_id = await save_eval_run(
        eval_type="ragas",
        git_commit=get_git_commit(),
        config_snapshot=await load_rag_config(),
        scores={
            "summary": summary,
            "elapsed_seconds": round(elapsed, 1),
            "dataset_size": len(raw),
        },
        per_question=report["per_question"],
    )
    if saved_id:
        logger.info("RAGAS 评估已记录到 eval_runs (id=%s)", saved_id)
    else:
        logger.warning("RAGAS 评估记录失败（不影响报告输出）")


if __name__ == "__main__":
    asyncio.run(main())
