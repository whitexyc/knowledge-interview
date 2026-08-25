"""
知识库重建脚本 (module-031)
============================
背景：
  库里 45 篇文档全部是旧版导入代码写入的"整篇 1 父 + 1 子"大块（平均 2.1 万字符），
  当前父子两级分块从未真正执行过，导致嵌入 8192 token 截断（检索质量崩塌）+
  rerank 对超长块推理（200-641s）。本脚本用当前源文件重新分块 + 嵌入。

流程（每篇 .md）：
  1. 删除旧记录（title = stem 或 title LIKE 'stem > %'）
  2. chunker.chunk（Option C：## + ### 标题层级 + 父块 4000 上限）
  3. 插入父块（无向量）→ flush 获取 id
  4. 批量嵌入子块（本地 bge-m3）→ 插入子块（含 search_tokens / content_hash / parent_id）
  5. 收尾：清空检索缓存；可选重建知识图谱（清空 + 逐篇 LLM 提取）

幂等性：按 title 先删后建，可重复执行；已导入文档不重复。

用法：
  python reindex_knowledge_base.py             # 全量重建（含图谱）
  python reindex_knowledge_base.py --no-graph  # 跳过图谱重建
  python reindex_knowledge_base.py --dry-run   # 只统计不写库
"""
import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from sqlalchemy import delete, text

from src.database import async_session_factory
from rag.models import Document
from rag.retrieval.chunker import chunker
from rag.retrieval.embeddings import embedding_service
from rag.retrieval.text_tokenizer import tokenize
from src.cache import cache
from rag.graph.graph_store import graph_store, GRAPH_NAME
from rag.graph.graph_extractor import graph_extractor

logger = logging.getLogger("reindex_knowledge_base")

# 源目录：(绝对路径, source 前缀)
DOCS_DIRS = [
    (r"D:\white\Documents\obsidian\backend-push", "backend-push"),
    (r"D:\white\Documents\obsidian\llm-push", "llm-push"),
]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def collect_files() -> list[tuple[str, str, str, str]]:
    """收集源文件 → [(path, dirname, fname, title)]，处理标题冲突"""
    files: list[tuple[str, str, str, str]] = []
    used_titles: set[str] = set()
    for dirpath, prefix in DOCS_DIRS:
        if not os.path.isdir(dirpath):
            logger.warning("目录不存在，跳过: %s", dirpath)
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".md"):
                continue
            title = fname[:-3]
            if title in used_titles:
                title = f"{title}-{prefix}"
                logger.info("标题冲突，重命名: %s → %s", fname, title)
            used_titles.add(title)
            files.append((os.path.join(dirpath, fname), prefix, fname, title))
    return files


async def _delete_old_rows(session, title: str):
    """删除该文档旧记录（父块 title=stem，子块/小节 title='stem > ...'）"""
    await session.execute(delete(Document).where(
        (Document.title == title) | (Document.title.like(f"{title} > %"))
    ))


async def import_file(path: str, prefix: str, fname: str, title: str):
    """单篇：删旧 → 分块 → 嵌入 → 入库。返回 (parents, children, first_parent_id, content)"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        logger.info("跳过空文件: %s", fname)
        return None
    source = f"{prefix}:{fname}"

    async with async_session_factory() as session:
        await _delete_old_rows(session, title)

        cr = chunker.chunk(content, source=source)
        parents, children = cr["parents"], cr["children"]
        # 镜像 add_document 兜底：无父块（极短内容）→ 1 父 + 1 子
        if not parents:
            parents = [{"title": title, "content": content}]
            children = [{"title": title, "content": content, "parent_index": 0}]

        # 1. 父块（无向量，供子块引用）
        parent_objs = []
        for p in parents:
            parent_title = f"{title} > {p['title']}" if p.get("title") and p["title"] != title else title
            doc = Document(
                title=parent_title,
                content=p["content"],
                source=source,
                embedding=None,
                parent_id=None,
                content_hash=hashlib.sha256(p["content"].encode("utf-8")).hexdigest(),
            )
            session.add(doc)
            parent_objs.append(doc)
        await session.flush()

        # 2. 子块向量化
        child_texts = [c["content"] for c in children]
        embeddings = await embedding_service.embed_documents(child_texts)

        # 3. 子块（含向量 + parent_id + search_tokens）
        for i, (child, emb) in enumerate(zip(children, embeddings)):
            parent_idx = child.get("parent_index", 0)
            if parent_idx >= len(parent_objs):
                parent_idx = 0
            parent = parent_objs[parent_idx]
            child_title = f"{title} > {child.get('title', '')}" if child.get("title") else title
            session.add(Document(
                title=child_title,
                content=child["content"],
                source=source,
                embedding=emb,
                parent_id=parent.id,
                content_hash=hashlib.sha256(child["content"].encode("utf-8")).hexdigest(),
                search_tokens=tokenize(child["content"]),
            ))

        await session.commit()
        return len(parent_objs), len(children), parent_objs[0].id, content


async def clear_graph():
    """清空知识图谱（重建前置）"""
    async with async_session_factory() as session:
        await session.execute(text("LOAD 'age'"))
        await session.execute(text('SET search_path = ag_catalog, "$user", public'))
        await session.execute(text(
            f"SELECT * FROM cypher('{GRAPH_NAME}', $$ MATCH (n) DETACH DELETE n $$) AS (n agtype)"
        ))
        await session.commit()
    logger.info("知识图谱已清空")


async def rebuild_graph(docs: list[tuple[str, str, int]]):
    """逐文档 LLM 提取实体/关系并写入图（失败不阻断）。docs = [(title, content, first_parent_id)]"""
    await clear_graph()
    stats = {"ok": 0, "failed": 0, "entities": 0, "relations": 0}
    for title, content, parent_id in docs:
        try:
            extraction = await graph_extractor.extract_from_document(content)
            entities = extraction.get("entities", [])
            relations = extraction.get("relations", [])
            for ent in entities:
                name = ent.get("name", "").strip()
                if name:
                    await graph_store.upsert_entity(name, ent.get("type", "concept"), int(parent_id))
            for rel in relations:
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                if src and tgt:
                    await graph_store.upsert_relation(src, tgt)
            stats["ok"] += 1
            stats["entities"] += len(entities)
            stats["relations"] += len(relations)
            logger.info("[图完成] %s | entities=%d relations=%d", title[:30], len(entities), len(relations))
        except Exception as e:
            stats["failed"] += 1
            logger.warning("[图失败] %s: %s", title[:30], e)
    logger.info("=== 图谱重建: ok=%d failed=%d entities=%d relations=%d ===",
                stats["ok"], stats["failed"], stats["entities"], stats["relations"])


async def cleanup_orphans(imported_titles: list[str]):
    """清理未导入的残留记录（如 test_dedup、无源文件的旧文档）"""
    async with async_session_factory() as session:
        rows = await session.execute(text(
            "SELECT DISTINCT split_part(title, ' > ', 1) AS t FROM documents"
        ))
        # 注意用 r[0] 索引取值：SQLAlchemy Row 的具名属性 `t` 会与 Row 内部属性
        # 冲突（2.0.19 起弃用并返回 Row 而非值），导致 asyncpg 绑定参数报错
        existing = {r[0] for r in rows}
        orphans = existing - set(imported_titles)
        for t in sorted(orphans):
            await session.execute(delete(Document).where(
                (Document.title == t) | (Document.title.like(f"{t} > %"))
            ))
            logger.warning("清理残留文档: %s", t)
        await session.commit()
        if not orphans:
            logger.info("无残留记录")


async def load_docs_from_db() -> list[tuple[str, str, int]]:
    """从重建后的库加载图谱重建所需数据 [(doc_title, reconstructed_content, first_parent_id)]

    重建后整篇内容不再单行存储（已切成父块），按顶层 title 用 string_agg
    拼接各父块内容近似还原全文（供 LLM 实体提取）。首次导入后崩溃恢复时使用。
    """
    async with async_session_factory() as session:
        rows = await session.execute(text("""
            SELECT split_part(title, ' > ', 1) AS doc,
                   MIN(id) AS first_parent_id,
                   string_agg(content, E'\n') AS content
            FROM documents WHERE parent_id IS NULL
            GROUP BY split_part(title, ' > ', 1)
            ORDER BY MIN(id)
        """))
        return [(r[0], r[2], r[1]) for r in rows]


async def main(dry_run: bool, no_graph: bool, skip_import: bool = False):
    files = collect_files()
    if not files:
        logger.error("未找到任何源文件")
        sys.exit(1)
    logger.info("源文件总数: %d", len(files))

    # ---- dry-run：只统计预计规模 ----
    total_chars = 0
    est_parents = 0
    est_children = 0
    for path, prefix, fname, title in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        total_chars += len(content)
        cr = chunker.chunk(content, source=fname)
        est_parents += len(cr["parents"]) or 1
        est_children += len(cr["children"]) or 1
    logger.info("[dry-run] 预计: 父块=%d, 子块=%d, 总字符=%d",
                est_parents, est_children, total_chars)
    if dry_run:
        return

    # ---- 正式重建 ----
    t0 = time.time()
    total_p = total_c = 0
    graph_docs: list[tuple[str, str, int]] = []
    if skip_import:
        # 文档已导入（崩溃恢复/补图谱场景）：直接从库加载图谱数据
        logger.info("--skip-import：跳过文档导入，直接从库加载图谱数据")
        graph_docs = await load_docs_from_db()
        logger.info("从库加载图谱数据 %d 篇", len(graph_docs))
    else:
        for idx, (path, prefix, fname, title) in enumerate(files, 1):
            try:
                res = await import_file(path, prefix, fname, title)
                if res is None:
                    continue
                parents_n, children_n, first_id, content = res
                total_p += parents_n
                total_c += children_n
                graph_docs.append((title, content, first_id))
                logger.info("[%d/%d] %s | parents=%d children=%d (%.0fs)",
                            idx, len(files), title[:40], parents_n, children_n, time.time() - t0)
            except Exception as e:
                logger.error("[%d/%d] %s 失败: %s", idx, len(files), title[:40], e)

        logger.info("=== 文档重建: parents=%d children=%d 用时 %.0fs ===",
                    total_p, total_c, time.time() - t0)

    # ---- 清缓存 + 清理残留 ----
    await cache.delete_by_prefix("rag:retrieve:")
    logger.info("检索缓存已失效")
    await cleanup_orphans([t for _, _, _, t in files])

    # ---- 图谱重建 ----
    if no_graph:
        logger.info("--no-graph，跳过图谱重建（可后跑 backfill_graph.py）")
    else:
        await rebuild_graph(graph_docs)

    logger.info("=== 重建完成 ===")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="知识库重建（module-031）")
    parser.add_argument("--dry-run", action="store_true", help="只统计预计规模，不写库")
    parser.add_argument("--no-graph", action="store_true", help="跳过知识图谱重建")
    parser.add_argument("--skip-import", action="store_true",
                        help="跳过文档导入（崩溃恢复/补图谱：直接从库加载图谱数据）")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, no_graph=args.no_graph, skip_import=args.skip_import))
