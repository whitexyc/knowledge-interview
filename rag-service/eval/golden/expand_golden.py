"""
Golden 检索集扩题 — 自动生成候选题目 + golden_docs（module-038）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

从知识库文档自动生成检索评估题目。

文档结构说明:
    知识库采用父块-子块层级：父块存完整文档标题+内容摘要（parent_id=NULL），
    子块存各小节标题+详细内容（parent_id 指向父块）。检索返回的是子块标题，
    golden 评估时标题匹配容忍 "文档名 > 小节名" 的层级前缀（取最左段比对）。
    因此 golden_docs 用文档名（父块标题）即可覆盖所有子块检索结果。

用法（ai_service 目录下）:
    python -m eval.golden.expand_golden --dry-run              # 仅统计，不调用 LLM
    python -m eval.golden.expand_golden --doc-ids 1,3,5        # 只处理指定 ID 的文档
    python -m eval.golden.expand_golden --doc-titles G1        # 只处理标题包含 "G1" 的文档
    python -m eval.golden.expand_golden --doc-titles G1 --dry-run  # 先看匹配了哪些
    python -m eval.golden.expand_golden --limit 10             # 限制处理 10 篇
    python -m eval.golden.expand_golden                        # 处理全部文档

⚠️ 自动生成的题目需人工审核：问题质量、golden_docs 准确性、
   类别分配等最终由人工确认后合并到 golden.json。
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import select

from src.database import async_session_factory
from llm.client import DeepSeekClient
from rag.models import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("expand_golden")

EVAL_DIR = Path(__file__).resolve().parent

QUESTION_GEN_PROMPT = """\
You are generating evaluation questions for a retrieval-augmented generation (RAG) system.

DOCUMENT TITLE: {title}

DOCUMENT SECTIONS (child blocks with detailed content):
{children}

PARENT SUMMARY:
{content}

TASK: Generate 2-4 natural-language questions (in Chinese) that a user might ask, where the CORRECT answer can be found in this document. These questions will be used for retrieval evaluation — the system must retrieve THIS document to answer correctly.

Use the CHILD BLOCK content to generate specific, answerable questions. Each child block represents a section of the document with detailed technical content.

REQUIREMENTS:
- Questions should be specific enough that this document is the best source
- Vary question types: factual ("what is..."), comparative ("how does X differ from Y"), procedural ("how to..."), conceptual ("why...")
- Questions should sound like real user queries, not test questions
- Output ONLY a JSON array of strings, nothing else.

Example format:
["G1垃圾收集器的Region分区机制是什么？", "Mixed GC和Young GC的区别是什么？"]"""

CATEGORY_PROMPT = """\
Classify the following document into ONE category from the list below.
Return ONLY the category slug, no other text.

DOCUMENT TITLE: {title}

CATEGORIES:
- java_gc: Java garbage collection topics
- java_concurrency: Java concurrency, threads, locks
- ai_llm: AI, LLM, machine learning topics
- kafka: Kafka, message queue topics
- resume: Personal resume, work experience, skills
- comprehensive: Cross-cutting or general tech topics

Category:"""


async def _fetch_parent_docs(
    limit: int = 0,
    doc_ids: list[int] | None = None,
    title_filter: str = "",
) -> list[dict]:
    """获取知识库父块文档（排除记忆类 source）

    Args:
        limit: 最大文档数（0 表示无限制，仅在无 doc_ids 时生效）
        doc_ids: 指定文档 ID 列表（None 表示全部）
        title_filter: 标题模糊匹配（空串表示不过滤）

    Returns:
        [{"id": int, "title": str, "content": str, "source": str}, ...]
    """
    docs: list[dict] = []
    try:
        async with async_session_factory() as session:
            query = (
                select(Document)
                .where(
                    Document.parent_id.is_(None),
                    Document.title != "",
                    ~Document.source.like("memory:%"),
                )
                .order_by(Document.id)
            )
            if doc_ids:
                query = query.where(Document.id.in_(doc_ids))
            elif limit > 0:
                query = query.limit(limit)
            rows = (await session.execute(query)).scalars().all()
            for doc in rows:
                if title_filter and title_filter.lower() not in doc.title.lower():
                    continue
                docs.append({
                    "id": doc.id,
                    "title": doc.title,
                    "content": (doc.content or "")[:2000],
                    "source": doc.source or "",
                })
    except Exception as e:
        logger.error("查询父块文档失败: %s", e)
        raise
    return docs


async def _fetch_children(parent_id: int) -> list[dict]:
    """获取某父块的所有子块（含标题和详细内容）"""
    try:
        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(Document)
                    .where(Document.parent_id == parent_id)
                    .order_by(Document.id)
                )
            ).scalars().all()
            return [
                {
                    "title": d.title or "(无标题)",
                    "content": (d.content or "")[:600],
                }
                for d in rows
            ]
    except Exception:
        return []


async def _generate_questions(title: str, parent_content: str, children: list[dict]) -> list[str]:
    """用 LLM 为单篇文档生成候选评估问题（含子块内容）

    Args:
        title: 文档标题（父块）
        parent_content: 父块内容摘要
        children: 子块列表 [{"title": str, "content": str}, ...]

    Returns:
        候选问题列表；失败返回空列表
    """
    children_text = ""
    if children:
        parts = []
        for c in children[:10]:  # 最多 10 个子块，避免 prompt 溢出
            parts.append(f"  [{c['title']}] {c['content'][:400]}")
        children_text = "\n".join(parts)
    else:
        children_text = "（无子块，仅父块摘要）"

    prompt = QUESTION_GEN_PROMPT.format(
        title=title,
        children=children_text,
        content=parent_content[:1200],
    )
    try:
        client = DeepSeekClient(temperature=0.7)
        raw = await client.chat([{"role": "user", "content": prompt}])
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        questions = json.loads(cleaned)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str) and len(q) > 5]
    except Exception as e:
        logger.warning("问题生成失败 [%s]: %s", title[:40], e)
    return []


async def _classify_category(title: str) -> str:
    """用 LLM 为文档分类，失败返回 'comprehensive'"""
    prompt = CATEGORY_PROMPT.format(title=title)
    try:
        client = DeepSeekClient(temperature=0)
        raw = await client.chat([{"role": "user", "content": prompt}])
        cat = raw.strip().lower()
        valid = {"java_gc", "java_concurrency", "ai_llm", "kafka", "resume", "comprehensive"}
        return cat if cat in valid else "comprehensive"
    except Exception:
        return "comprehensive"


async def expand_golden(
    limit: int = 0,
    doc_ids: list[int] | None = None,
    title_filter: str = "",
    dry_run: bool = False,
) -> list[dict]:
    """主流程：读文档（含子块）→ 生成题目 → 输出 golden 格式

    Args:
        limit: 最多处理多少篇文档（0=全部，doc_ids 优先）
        doc_ids: 指定文档 ID 列表
        title_filter: 标题模糊匹配
        dry_run: 仅统计不调用 LLM

    Returns:
        golden 格式题目列表
    """
    docs = await _fetch_parent_docs(
        limit=limit if not doc_ids else 0,
        doc_ids=doc_ids,
        title_filter=title_filter,
    )
    logger.info("匹配 %d 篇父块文档", len(docs))

    if dry_run:
        for d in docs:
            logger.info("  id=%-4d [%s] %s", d["id"], d["source"][:15], d["title"][:70])
            children = await _fetch_children(d["id"])
            if children:
                logger.info("         └─ %d 个子块: %s", len(children),
                            ", ".join(c["title"][:30] for c in children[:3]))
            logger.info("")
        return []

    entries: list[dict] = []
    for i, doc in enumerate(docs):
        logger.info("[%d/%d] 生成问题: %s", i + 1, len(docs), doc["title"][:60])
        children = await _fetch_children(doc["id"])
        if children:
            logger.info("  %d 个子块", len(children))
        questions = await _generate_questions(doc["title"], doc["content"], children)
        if not questions:
            logger.warning("  无问题生成，跳过")
            continue

        category = await _classify_category(doc["title"])
        for q in questions:
            entries.append({
                "question": q,
                "golden_docs": [doc["title"]],
                "category": category,
            })
        logger.info("  生成 %d 题 (类别: %s)", len(questions), category)

    return entries


def print_summary(entries: list[dict]) -> None:
    """打印生成统计"""
    cats: dict[str, int] = {}
    for e in entries:
        cats[e["category"]] = cats.get(e["category"], 0) + 1

    print("\n" + "=" * 60)
    print("Golden Expansion Summary")
    print("=" * 60)
    print(f"Total questions generated: {len(entries)}")
    print("-" * 60)
    for cat, count in sorted(cats.items()):
        print(f"  {cat:<20} {count:>4}")
    print("=" * 60)
    print("⚠️  自动生成题目需人工审核后合并到 golden.json")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Golden 检索集自动扩题 — 从知识库父块+子块生成候选评估问题",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="最多处理文档数（0=全部，--doc-ids 指定时忽略）")
    parser.add_argument("--doc-ids", type=str, default="",
                        help="逗号分隔的文档 ID 列表（如 1,3,5）")
    parser.add_argument("--doc-titles", type=str, default="",
                        help="标题模糊匹配（如 G1、Kafka），大小写不敏感")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计文档+子块数量，不调用 LLM")
    args = parser.parse_args()

    doc_ids = None
    if args.doc_ids:
        try:
            doc_ids = [int(x.strip()) for x in args.doc_ids.split(",") if x.strip()]
        except ValueError:
            logger.error("--doc-ids 格式错误，应为逗号分隔整数（如 1,3,5）")
            sys.exit(1)

    entries = await expand_golden(
        limit=args.limit,
        doc_ids=doc_ids,
        title_filter=args.doc_titles,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    print_summary(entries)

    if entries:
        output_path = EVAL_DIR / "golden_expanded.json"
        output_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("已保存 %d 题到: %s", len(entries), output_path)
    else:
        logger.warning("无题目生成")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
