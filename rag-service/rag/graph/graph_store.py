"""
Apache AGE 知识图谱存储 — Graph RAG 图操作层

在整个 RAG 链路中的位置：
  文档入库 → [GraphExtractor] 提取实体/关系 → [GraphStore] 写入 AGE
  用户查询 → [GraphExtractor] 提取查询实体 → [GraphStore.search_related] 图遍历 → 文档

依赖：
  - PostgreSQL 已安装 Apache AGE 扩展（本地 PG 自带）
  - 图名 knowledge_graph
  - 节点 == Entity(name, type, doc_ids)
  - 边 == RELATED_TO（统一关系类型）

设计决策：
  1. 为什么用 MERGE 而非 CREATE？
     MERGE 是幂等的，重复执行不会重复创建节点/边。
     适合 add_document 多次调用和初始化场景。

  2. 为什么 doc_ids 存为 JSON 数组字符串？
     AGE 不支持直接的数组类型存储。JSON 字符串可被
     SQL 解析和追加，且与 Python list 互相转换方便。

  3. 为什么 search_related 用两步查询（Cypher → SQL）？
     AGE Cypher 可以找到相关实体和它们的 doc_ids，但
     返回完整文档内容需要 JOIN PostgreSQL 的 documents 表。
     两步查询是 AGE + PG 混合模型的标准做法。

  4. 为什么不用 bindparams？
     asyncpg 在 $$...$$ dollar-quoting 内的 `:param` 不会被识别为参数绑定，
     因此使用 Python f-string + 字符转义（Cypher 级别）注入参数值。
     $$...$$ 本身已经提供 PostgreSQL 级别的 SQL 注入防护——f-string 只影响 Cypher
     表达式内部的字符串字面量，不涉及外层 SQL AST。
"""
import asyncio
import json
import logging
from typing import Optional

from sqlalchemy import text, select

from src.database import async_session_factory
from rag.models import Document

logger = logging.getLogger(__name__)

GRAPH_NAME = "knowledge_graph"


def _escape(val: str) -> str:
    """转义 Cypher 字符串字面量中的特殊字符

    替换规则：
      \\  → \\\\   （反斜杠）
      ' → \\'    （单引号）

    注意：不能转义 `}`——_escape 的输出总是插入 Cypher 字符串字面量 `'...'`
    内部，`\}` 在 openCypher（AGE 1.6）中是非法转义序列，会导致含 `}` 的
    实体/关系写入失败（实测 InvalidEscapeSequenceError，实体名如 `#{}`、`${}`）。
    属性字典的 `{...}` 花括号来自查询模板本身，无需转义值内的 `}`。
    """
    return val.replace("\\", "\\\\").replace("'", "\\'")


class GraphStore:
    """Apache AGE 知识图谱存储

    职责：
    1. ensure_graph() — 幂等创建图
    2. upsert_entity() — 创建/更新实体节点，追加 doc_ids
    3. upsert_relation() — 创建 RELATED_TO 边
    4. search_related() — 从实体出发图遍历，返回关联文档

    所有操作均 try/except 包裹，失败时静默降级（日志 warning）。
    """

    async def ensure_graph(self) -> bool:
        """确保 AGE 图和扩展已就绪（幂等）

        创建流程：
          1. CREATE EXTENSION age（首次）
          2. LOAD 'age'（每次会话）
          3. 检查图是否已存在 → 不存在才 create_graph

        注意：
          - 不再吞异常：create_graph 失败必须记录，避免"图没建出来但系统
            假装成功"的静默降级（之前导致 Graph RAG 长期无数据）。
          - 只忽略一种情况：图已存在（查询 ag_graph 表确认）。

        Returns:
            True 如果就绪，False 如果创建失败
        """
        try:
            async with async_session_factory() as session:
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS age"))
                await session.execute(text("LOAD 'age'"))
                await session.execute(text(
                    "SET search_path = ag_catalog, \"$user\", public"
                ))

                # 检查图是否已存在（避免 create_graph 抛"已存在"异常）
                exists = await session.execute(text(
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = :name LIMIT 1"
                ), {"name": GRAPH_NAME})
                if exists.scalar_one_or_none() is None:
                    await session.execute(text(
                        f"SELECT create_graph('{_escape(GRAPH_NAME)}')"
                    ))
                    logger.info("AGE 图已创建: %s", GRAPH_NAME)
                else:
                    logger.info("AGE 图已存在: %s", GRAPH_NAME)

                await session.commit()
                return True
        except Exception as e:
            logger.warning("AGE 图初始化失败: %s", e)
            return False

    async def upsert_entity(self, name: str, entity_type: str, doc_id: int) -> bool:
        """创建或更新实体节点，追加关联文档 ID

        实现说明（重要）：
          Apache AGE 1.6.0 的 openCypher 方言不支持 MERGE 的
          ON CREATE SET / ON MATCH SET 子句（语法错误），所以拆成两步：
            1. 先 MATCH 检查节点是否存在
            2. 不存在 → CREATE（doc_ids 初始为数组 ['doc_id']）
               已存在 → WHERE NOT doc_id IN e.doc_ids + SET 数组追加
          doc_ids 存为 agtype 数组，与 search_related 的 json.loads 解析兼容。

        Args:
            name: 实体名称
            entity_type: 实体类型（如 "concept", "technology"）
            doc_id: 关联的文档 ID

        Returns:
            True 如果成功，False 如果失败
        """
        try:
            safe_name = _escape(name)
            safe_type = _escape(entity_type)
            doc_id_str = str(doc_id)

            async with async_session_factory() as session:
                await session.execute(text("LOAD 'age'"))
                await session.execute(text(
                    "SET search_path = ag_catalog, \"$user\", public"
                ))

                # Step 1: 检查实体是否已存在
                exists = await session.execute(text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $$
                        MATCH (e:Entity {{name: '{safe_name}', type: '{safe_type}'}})
                        RETURN e.name
                    $$) AS (name agtype)
                """))
                if exists.fetchone() is None:
                    # Step 2a: 不存在 → 创建，doc_ids 初始为单元素数组
                    query = text(f"""
                        SELECT * FROM cypher('{GRAPH_NAME}', $$
                            CREATE (e:Entity {{name: '{safe_name}', type: '{safe_type}',
                                              doc_ids: ['{doc_id_str}']}})
                            RETURN e.name
                        $$) AS (name agtype)
                    """)
                    await session.execute(query)
                else:
                    # Step 2b: 已存在 → 若 doc_id 不在数组内则追加
                    query = text(f"""
                        SELECT * FROM cypher('{GRAPH_NAME}', $$
                            MATCH (e:Entity {{name: '{safe_name}', type: '{safe_type}'}})
                            WHERE NOT '{doc_id_str}' IN e.doc_ids
                            SET e.doc_ids = e.doc_ids || ['{doc_id_str}']
                            RETURN e.name
                        $$) AS (name agtype)
                    """)
                    await session.execute(query)

                await session.commit()
                return True
        except Exception as e:
            logger.warning("实体写入失败 [%s]: %s", name[:30], e)
            return False

    async def upsert_relation(self, source: str, target: str) -> bool:
        """创建源实体到目标实体的 RELATED_TO 边（幂等）

        Args:
            source: 源实体名称
            target: 目标实体名称

        Returns:
            True 如果成功，False 如果失败
        """
        try:
            safe_src = _escape(source)
            safe_tgt = _escape(target)

            async with async_session_factory() as session:
                await session.execute(text("LOAD 'age'"))
                await session.execute(text(
                    "SET search_path = ag_catalog, \"$user\", public"
                ))
                query = text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $$
                        MATCH (a:Entity {{name: '{safe_src}'}})
                        MATCH (b:Entity {{name: '{safe_tgt}'}})
                        MERGE (a)-[r:RELATED_TO]->(b)
                        RETURN r
                    $$) AS (r agtype)
                """)
                await session.execute(query)
                await session.commit()
                return True
        except Exception as e:
            logger.warning("关系写入失败 [%s → %s]: %s", source[:20], target[:20], e)
            return False

    async def search_related(self, entities: list[str], top_k: int = 10) -> list[dict]:
        """从查询实体出发图遍历，返回关联文档（含真实 graph_score）

        相关度 = 每篇文档被「查询实体 + 一跳邻居」引用的次数（命中实体数），
        命中越多越相关。流程：Cypher 统计命中数（_count_doc_hits）→ Python
        min-max 归一化到 [0,1]（全同分保底 0.6）→ 按真实命中数降序取 top_k。

        Args:
            entities: 查询中提取的实体名称列表
            top_k: 返回的最大文档数

        Returns:
            文档列表（与 vector_retrieval 兼容格式），失败返回空列表
        """
        if not entities:
            return []

        try:
            hit_map = await self._count_doc_hits(entities, top_k)
            if not hit_map:
                return []

            async with async_session_factory() as session:
                result = await session.execute(
                    select(Document)
                    .where(Document.id.in_(list(hit_map.keys())))
                    .where(Document.parent_id.is_(None))
                )
                docs = result.scalars().all()

            # 命中实体数 → graph_score（min-max 归一化，全同分保底 0.6）
            scores = self._normalize_graph_scores([hit_map.get(d.id, 0) for d in docs])

            # 排序用真实命中数降序（归一化只改分数值）
            ranked = sorted(zip(docs, scores),
                            key=lambda x: hit_map.get(x[0].id, 0), reverse=True)

            output = [
                {"id": d.id, "title": d.title, "content": d.content,
                 "source": d.source, "hybrid_score": s, "parent_id": None}
                for d, s in ranked[:top_k]
            ]

            logger.info("图搜索完成: entities=%d, docs=%d", len(entities), len(output))
            return output
        except Exception as e:
            logger.warning("图搜索失败，降级返回空: %s", e)
            return []

    async def _count_doc_hits(self, entities: list[str], top_k: int) -> dict[int, int]:
        """Cypher 统计每篇文档被查询实体及一跳邻居引用的次数

        命中实体数 = 多少实体（查询实体 e ∪ 一跳邻居 related）的 doc_ids
        数组包含该文档。UNWIND [e] + [related] 逐实体展开 doc_ids，
        count(DISTINCT ename) 去重计数。

        AGE 方言注意：ORDER BY 用 count(...) 表达式而非别名（别名排序报
        "could not find rte for hits"）；新查询含 e 与 related 的 doc_ids。

        Args:
            entities: 查询中提取的实体名称列表
            top_k: 返回的最大文档数（Cypher 取 2 倍候选）

        Returns:
            {doc_id: 命中实体数} 映射，无命中返回空 dict
        """
        async with async_session_factory() as session:
            await session.execute(text("LOAD 'age'"))
            await session.execute(text('SET search_path = ag_catalog, "$user", public'))
            # 构建 Cypher 兼容的列表字符串 ["a","b","c"]
            safe_entities = ",".join(f"'{_escape(e)}'" for e in entities)
            entity_str = f"[{safe_entities}]"

            query = text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $$
                    MATCH (e:Entity)
                    WHERE e.name IN {entity_str}
                    OPTIONAL MATCH (e)-[r:RELATED_TO]->(related:Entity)
                    WITH e, related
                    UNWIND [e] + CASE WHEN related IS NULL THEN [] ELSE [related] END AS ent
                    WITH ent.name AS ename, ent.doc_ids AS ids
                    UNWIND ids AS doc_id
                    RETURN doc_id, count(DISTINCT ename) AS hits
                    ORDER BY count(DISTINCT ename) DESC
                    LIMIT {top_k * 2}
                $$) AS (doc_id agtype, hits agtype)
            """)
            rows = (await session.execute(query)).fetchall()

        # 解析 (doc_id, hits)：agtype 序列化为 JSON 字符串，json.loads 还原
        hit_map: dict[int, int] = {}
        for row in rows:
            try:
                did = json.loads(str(row[0])) if row[0] is not None else None
                if isinstance(did, (int, str)) and str(did).isdigit():
                    hit_map[int(did)] = int(json.loads(str(row[1]) or "0"))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return hit_map

    @staticmethod
    def _normalize_graph_scores(counts: list[int]) -> list[float]:
        """Min-Max 归一化命中实体数到 [0, 1]

        与 retriever._normalize 同范式，但保底值不同：
          全同分/单结果时 retriever._normalize 返回 1.0，
          本方法返回 0.6 —— 与历史硬编码 0.6 一致，避免图结果
          在无区分度时给融合通道一个突兀的高分。

        Args:
            counts: 每篇文档的命中实体数列表

        Returns:
            归一化分数列表，长度与 counts 相同
        """
        if not counts:
            return []
        min_c = min(counts)
        max_c = max(counts)
        score_range = max_c - min_c
        if score_range < 1e-9:
            # 所有文档命中数相同（单结果/全同分）→ 保底 0.6
            return [0.6] * len(counts)
        return [(c - min_c) / score_range for c in counts]


# 全局单例
graph_store = GraphStore()
