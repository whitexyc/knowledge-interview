"""
Chat 对话 Faithfulness 评估 — LLM-as-Judge（module-038）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

从 session_memory 持久化数据中抽取用户→助手对话对，逐条评估：

- faithfulness（答案忠实度）：答案是否基于检索上下文（0-1）
- relevancy（答案相关性）：答案是否切合问题（0-1）

用法（ai_service 目录下）:
    python -m eval.golden.faithfulness                  # 默认抽 50 条（优先 session_memory）
    python -m eval.golden.faithfulness --sample 30      # 抽 30 条
    python -m eval.golden.faithfulness --dataset        # 强制使用 dataset.json 问题（生成新答案）

    python -m eval.golden.faithfulness --no-save        # 不落库
    python -m eval.golden.faithfulness --sample 5 --dataset --no-save # 仅抽 5 条，降级 dataset.json，且不落库
Judge LLM: 使用项目默认 DeepSeek（temperature=0 保证一致性）。
"""
import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import select, func

from src.config import settings
from src.database import async_session_factory
from llm.client import DeepSeekClient
from rag.engine import rag_engine
from rag.models import Document
from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("faithfulness_eval")

EVAL_DIR = Path(__file__).resolve().parent

FAITHFULNESS_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system for faithfulness.

QUESTION: {question}

ANSWER (to evaluate): {answer}

RETRIEVED CONTEXT (documents the system had access to):
{context}

Task: Rate how faithful the ANSWER is to the RETRIEVED CONTEXT on a scale of 0.0 to 1.0.

Scoring guide:
- 1.0: Answer is fully grounded in the context — every claim is supported by the provided documents.
- 0.7-0.9: Answer is mostly grounded, with minor embellishments or paraphrasing not contradicting context.
- 0.4-0.6: Answer is partially grounded but contains some unsupported or speculative claims.
- 0.1-0.3: Answer is mostly unsupported, with brief mentions of context amidst hallucination.
- 0.0: Answer completely contradicts the context or is entirely hallucinated.

Return ONLY a single float number. Do not include any other text."""

RELEVANCY_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system for answer relevancy.

QUESTION: {question}

ANSWER (to evaluate): {answer}

Task: Rate how relevant the ANSWER is to the QUESTION on a scale of 0.0 to 1.0.

Scoring guide:
- 1.0: Answer directly and completely addresses the question.
- 0.7-0.9: Answer addresses the question but includes minor tangential information.
- 0.4-0.6: Answer partially addresses the question, missing key aspects or going off-topic.
- 0.1-0.3: Answer barely addresses the question, mostly irrelevant.
- 0.0: Answer is completely off-topic — unrelated to the question.

Return ONLY a single float number. Do not include any other text."""


async def _extract_qa_pairs(limit: int = 100) -> list[dict]:
    """从 session_memory 抽取用户→助手问答对

    查询所有身份隔离的会话记忆，按时间序排列，提取连续的
    user→assistant 消息对。去重（相同问题只取首次）。

    Args:
        limit: 最多抽取的对数（最终采样从此池中取）

    Returns:
        [{"question": str, "answer": str, "identity": str}, ...]
    """
    pairs: list[dict] = []
    seen_questions: set[str] = set()

    try:
        async with async_session_factory() as session:
            # 查所有 session 记忆的 source 前缀
            rows = await session.execute(
                select(Document.source)
                .where(Document.source.like("memory:%:session:"))
                .distinct()
            )
            sources = [r[0] for r in rows.all()]
    except Exception as e:
        logger.warning("查询会话源失败: %s", e)
        return pairs

    logger.info("发现 %d 个会话源（身份）", len(sources))
    if not sources:
        logger.warning("无会话记忆数据，无法抽取问答对")
        return pairs

    for source in sources:
        try:
            async with async_session_factory() as session:
                docs = (
                    await session.execute(
                        select(Document)
                        .where(Document.source == source)
                        .order_by(Document.id.asc())
                    )
                ).scalars().all()
        except Exception as e:
            logger.warning("读取 source=%s 失败: %s", source, e)
            continue

        msgs: list[dict] = []
        for doc in docs:
            content = (doc.content or "").strip()
            if not content:
                continue
            role = "user"
            if doc.title and doc.title.startswith("session:"):
                role = doc.title[len("session:"):].strip()
            if role not in ("user", "assistant"):
                continue
            msgs.append({"role": role, "content": content})

        # 提取连续 user→assistant 对
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
                continue
            q = msgs[i]["content"].strip()
            if q in seen_questions:
                continue
            seen_questions.add(q)
            pairs.append({
                "question": q,
                "answer": msgs[i + 1]["content"].strip(),
                "identity": source,
            })
        if len(pairs) >= limit:
            break

    logger.info("共抽取 %d 个问答对", len(pairs))
    return pairs


async def _load_dataset_pairs(sample_size: int) -> list[dict]:
    """降级数据源：从 dataset.json 加载问题，全链路检索+生成答案

    当 session_memory 无数据时使用。对每条 dataset 问题执行
    _retrieve → generate_answer，产出问答对用于 faithfulness 评估。

    Args:
        sample_size: 最多使用多少条问题

    Returns:
        [{"question": str, "answer": str, "source": "dataset"}, ...]
    """
    from agent.reflector import reflector

    dataset_path = EVAL_DIR / "dataset.json"
    if not dataset_path.exists():
        logger.warning("dataset.json 不存在，无降级数据")
        return []

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = raw[:sample_size]
    logger.info("降级使用 dataset.json: %d 题", len(items))

    pairs: list[dict] = []
    for i, item in enumerate(items):
        question = item["question"]
        logger.info("[%d/%d] 生成答案: %s", i + 1, len(items), question[:60])
        try:
            docs = await rag_engine._retrieve(question)
            answer = await reflector.generate_answer(question, docs)
        except Exception as e:
            logger.warning("生成答案失败: %s", e)
            answer = "（生成失败）"
        pairs.append({"question": question, "answer": answer, "source": "dataset"})

    return pairs


async def _judge_faithfulness(
    judge: DeepSeekClient,
    question: str,
    answer: str,
    contexts: list[str],
) -> float:
    """LLM-as-Judge 评 faithfulness（答案对检索上下文的忠实度）"""
    ctx_text = "\n\n---\n\n".join(
        f"[Doc {i+1}] {c[:800]}" for i, c in enumerate(contexts[:5])
    )
    prompt = FAITHFULNESS_PROMPT.format(
        question=question, answer=answer[:2000], context=ctx_text,
    )
    try:
        raw = await judge.chat([{"role": "user", "content": prompt}])
        return float(raw.strip())
    except (ValueError, Exception) as e:
        logger.warning("faithfulness 评分失败: %s — raw=%s", e, str(raw)[:80] if 'raw' in dir() else '')
        return 0.0


async def _judge_relevancy(
    judge: DeepSeekClient,
    question: str,
    answer: str,
) -> float:
    """LLM-as-Judge 评 relevancy（答案对问题的相关性）"""
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer[:2000])
    try:
        raw = await judge.chat([{"role": "user", "content": prompt}])
        return float(raw.strip())
    except (ValueError, Exception) as e:
        logger.warning("relevancy 评分失败: %s", e)
        return 0.0


async def run_faithfulness_eval(sample_size: int = 50, force_dataset: bool = False) -> dict:
    """执行 faithfulness 评估

    Args:
        sample_size: 从候选池随机采样多少条
        force_dataset: 强制使用 dataset.json（跳过 session_memory）

    Returns:
        评估结果 dict（summary + per_question + metadata）
    """
    data_source = "session_memory"
    all_pairs = await _extract_qa_pairs(limit=max(sample_size * 3, 100))
    if not all_pairs:
        if force_dataset:
            logger.info("session_memory 无数据，降级使用 dataset.json")
            all_pairs = await _load_dataset_pairs(sample_size)
            data_source = "dataset"
        else:
            print(
                "\n⚠️  session_memory 中无聊天数据，无法抽取问答对。\n"
                "   原因：还没有通过 Chat 页面产生过对话记录（module-034 会话持久化）。\n"
                "   解决：加 --dataset 可从 dataset.json 加载问题并实时生成答案评估：\n"
                "     python -m eval.golden.faithfulness --sample 5 --dataset\n"
            )
            return {"error": "无会话记忆数据（加 --dataset 降级）", "summary": {}, "per_question": []}

    if len(all_pairs) > sample_size:
        all_pairs = random.sample(all_pairs, sample_size)
    logger.info("采样 %d 条问答对进行评估 (数据源: %s)", len(all_pairs), data_source)

    if not settings.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY 未配置，评估可能失败")
    judge = DeepSeekClient(temperature=0)

    results: list[dict] = []
    faith_scores: list[float] = []
    relevancy_scores: list[float] = []
    ctx_count: list[int] = []

    for i, pair in enumerate(all_pairs):
        q = pair["question"]
        a = pair["answer"]
        logger.info("[%d/%d] %s", i + 1, len(all_pairs), q[:60])

        # 1. 检索上下文（为 Q 重新检索）
        contexts: list[str] = []
        try:
            docs = await rag_engine._retrieve(q)
            contexts = [d.get("content", "") for d in docs]
        except Exception as e:
            logger.warning("检索失败，上下文为空: %s", e)
        ctx_count.append(len(contexts))

        # 2. LLM 评分
        faith_coro = _judge_faithfulness(judge, q, a, contexts)
        relevancy_coro = _judge_relevancy(judge, q, a)
        faithful = await faith_coro
        rel = await relevancy_coro

        faith_scores.append(faithful)
        relevancy_scores.append(rel)
        results.append({
            "question": q[:200],
            "answer": a[:300],
            "faithfulness": round(faithful, 4),
            "relevancy": round(rel, 4),
            "context_count": len(contexts),
        })

    n = len(results)
    avg_f = round(sum(faith_scores) / n, 4) if n else 0.0
    avg_r = round(sum(relevancy_scores) / n, 4) if n else 0.0

    summary = {
        "sample_size": n,
        "faithfulness_avg": avg_f,
        "relevancy_avg": avg_r,
        "avg_context_count": round(sum(ctx_count) / n, 1) if n else 0,
    }
    metadata = {
        "eval_type": "faithfulness",
        "data_source": data_source,
        "judge_llm": f"{settings.deepseek_model} (temperature=0)",
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "per_question": results,
    }


def print_faithfulness_report(report: dict) -> None:
    """打印 faithfulness 评估报告到控制台"""
    if report.get("error") and not report.get("per_question"):
        return  # 报错信息已在 run_faithfulness_eval 中输出，跳过空表
    summary = report.get("summary", {})
    results = report.get("per_question", [])
    n = len(results)

    print("\n" + "=" * 60)
    print("Faithfulness & Relevancy Evaluation")
    print("=" * 60)
    ds = report.get('metadata', {}).get('data_source', 'session_memory')
    print(f"Sample: {n} Q&A pairs (source: {ds})")
    print(f"Judge:  {report.get('metadata', {}).get('judge_llm', 'N/A')}")
    print("-" * 60)
    print(f"Faithfulness  avg: {summary.get('faithfulness_avg', 0):.4f}")
    print(f"Relevancy     avg: {summary.get('relevancy_avg', 0):.4f}")
    print(f"Avg contexts/QA: {summary.get('avg_context_count', 0):.1f}")
    print("-" * 60)

    # 分档统计
    bins = {"excellent (≥0.8)": 0, "good (0.6-0.8)": 0, "fair (0.4-0.6)": 0, "poor (<0.4)": 0}
    for r in results:
        f = r.get("faithfulness", 0)
        if f >= 0.8: bins["excellent (≥0.8)"] += 1
        elif f >= 0.6: bins["good (0.6-0.8)"] += 1
        elif f >= 0.4: bins["fair (0.4-0.6)"] += 1
        else: bins["poor (<0.4)"] += 1
    print("Faithfulness distribution:")
    for label, count in bins.items():
        pct = count / n * 100 if n else 0
        print(f"  {label:<22} {count:>3} ({pct:>5.1f}%)")

    # 随机样例（5 条）
    if results:
        sample = random.sample(results, min(5, n))
        print("-" * 60)
        print("Random samples (5):")
        for r in sample:
            print(f"  Q: {r['question'][:55]}...")
            print(f"  faithfulness={r['faithfulness']:.2f}  relevancy={r['relevancy']:.2f}  ctx={r['context_count']}")
            print()
    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat 对话 Faithfulness/Relevancy 评估（LLM-as-Judge）",
    )
    parser.add_argument("--sample", type=int, default=50, help="采样问答对数（默认 50）")
    parser.add_argument("--dataset", action="store_true", dest="force_dataset",
                        help="强制使用 dataset.json 问题（生成新答案），跳过 session_memory")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    args = parser.parse_args()

    report = await run_faithfulness_eval(
        sample_size=args.sample,
        force_dataset=args.force_dataset,
    )
    print_faithfulness_report(report)

    if not args.no_save and "error" not in report:
        commit = get_git_commit()
        config_snapshot = await load_rag_config()
        saved_id = await save_eval_run(
            eval_type="faithfulness",
            git_commit=commit,
            config_snapshot=config_snapshot,
            scores=report["summary"],
            per_question=report["per_question"],
        )
        if saved_id:
            logger.info("Faithfulness 评估已记录到 eval_runs (id=%s)", saved_id)

    # 输出 JSON 到文件
    output_path = EVAL_DIR / "faithfulness.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("报告已保存到: %s", output_path)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
