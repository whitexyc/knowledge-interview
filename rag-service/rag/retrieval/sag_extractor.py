"""SAG 实体/事件抽取器 — 文档入库时 LLM 抽取（module-081）

复用 graph_extractor 的 LLM 抽取范式（_ENTITY_PROMPT / _parse_json /
LLMFactory.get_client），SAG 版 entity_type 扩展到 11 类。

在整个 SAG 链路中的位置：
  文档入库 → [sag_extractor.extract_entities_events] → 入库 sag_entities/sag_events
  用户查询 → [sag_retriever.retrieve] → SQL join 检索相关文档

设计决策：
  1. entity_type 11 类独立定义，不强行复用 graph_extractor 的 8 类（解耦）
  2. 单次 LLM 调用同时抽取 entities + events（省一次 LLM roundtrip）
  3. 失败/超时返回空 —— fail-open，不阻断入库（对齐 document_ingest 纪律）
"""
import json
import logging

from llm.client import LLMFactory

logger = logging.getLogger(__name__)

# SAG 实体类型（11 类，plan.md §5.1 拍板）
ENTITY_TYPES = [
    "concept", "technology", "algorithm", "framework", "tool",
    "person", "company", "language", "event", "metric", "method",
]

_SAG_PROMPT = """你是知识图谱构建专家。从以下文档中抽取实体和事件。

实体类型限以下 11 类：{entity_types}
每个实体提取 name 和 type。
每个事件提取 text（事件描述）和 entity_names（关联的实体名称列表）。

文档:
{document}

返回 JSON 格式（只返回 JSON，不要其他文字）：
{{"entities": [{{"name": "实体名", "type": "类型"}}, ...], "events": [{{"text": "事件描述", "entity_names": ["实体1", ...]}}, ...]}}

JSON:"""


def _parse_json(raw: str) -> dict:
    """解析 LLM 输出的 JSON，多级回退（复用 graph_extractor 范式）"""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        pass
    return {}


async def extract_entities_events(document_text: str) -> dict:
    """从文档文本中抽取实体和事件

    Args:
        document_text: 文档全文（截断到 2000 字符）

    Returns:
        {"entities": [{"name": str, "type": str}, ...],
         "events": [{"text": str, "entity_names": [str, ...]}, ...]}
        失败时返回空值默认结构
    """
    doc_text = document_text[:2000] if len(document_text) > 2000 else document_text
    if not doc_text.strip():
        return {"entities": [], "events": []}
    try:
        client = LLMFactory.get_client("fallback", temperature=0.1)
        prompt = _SAG_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            document=doc_text,
        )
        raw = await client.generate(prompt)
        data = _parse_json(raw)
        entities = data.get("entities", [])
        events = data.get("events", [])
        # 过滤非法实体类型
        entities = [e for e in entities if e.get("name") and e.get("type") in ENTITY_TYPES]
        logger.info("SAG 抽取完成: entities=%d, events=%d", len(entities), len(events))
        return {"entities": entities, "events": events}
    except Exception as e:
        logger.warning("SAG 抽取失败，返回空: %s", e)
        return {"entities": [], "events": []}


async def ingest_sag_data(doc_id: int, document_text: str) -> None:
    """入库 hook：抽取实体/事件并写入 SAG 三表（fire-and-forget，fail-open）

    仅当 retrieval_mode in ("sag", "hybrid_sag") 时由 document_ingest 调用。
    抽取失败只记录日志，绝不阻断入库。

    Args:
        doc_id: 入库文档 ID
        document_text: 文档全文
    """
    from src.database import async_session_factory
    from sqlalchemy import text as sql_text

    try:
        result = await extract_entities_events(document_text)
        if not result["entities"] and not result["events"]:
            return

        async with async_session_factory() as session:
            # 写入实体（追加模式：同名同类型合并 source_doc_ids）
            entity_id_map = {}
            for ent in result["entities"]:
                name = ent["name"]
                etype = ent["type"]
                # INSERT ... ON CONFLICT 合并 source_doc_ids
                stmt = sql_text("""
                    INSERT INTO sag_entities (name, entity_type, source_doc_ids)
                    VALUES (:name, :etype, :doc_ids::jsonb)
                    ON CONFLICT (name, entity_type) DO UPDATE
                    SET source_doc_ids = (
                        SELECT jsonb_agg(DISTINCT x)
                        FROM jsonb_array_elements(
                            sag_entities.source_doc_ids || EXCLUDED.source_doc_ids
                        ) AS t(x)
                    )
                    RETURNING id
                """)
                row = await session.execute(stmt, {
                    "name": name, "etype": etype,
                    "doc_ids": json.dumps([doc_id]),
                })
                eid = row.scalar()
                if eid:
                    entity_id_map[name] = eid

            # 写入事件
            for evt in result["events"]:
                event_text = evt.get("text", "")
                if not event_text:
                    continue
                entity_ids = []
                for ename in evt.get("entity_names", []):
                    if ename in entity_id_map:
                        entity_ids.append(entity_id_map[ename])
                stmt = sql_text("""
                    INSERT INTO sag_events (event_text, entity_ids, source_doc_id)
                    VALUES (:text, :eids::jsonb, :doc_id)
                """)
                await session.execute(stmt, {
                    "text": event_text,
                    "eids": json.dumps(entity_ids),
                    "doc_id": doc_id,
                })

            # 写入关系（从 events 的 entity_names 推导：相邻实体间隐含关系）
            for evt in result["events"]:
                names = evt.get("entity_names", [])
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        src_id = entity_id_map.get(names[i])
                        tgt_id = entity_id_map.get(names[j])
                        if src_id and tgt_id:
                            stmt = sql_text("""
                                INSERT INTO sag_relations
                                    (source_entity_id, target_entity_id, relation_type, source_doc_id)
                                VALUES (:src, :tgt, :rtype, :doc_id)
                            """)
                            await session.execute(stmt, {
                                "src": src_id, "tgt": tgt_id,
                                "rtype": "co_mention",
                                "doc_id": doc_id,
                            })

            await session.commit()
        logger.info("SAG 数据入库完成: doc_id=%d, entities=%d, events=%d",
                     doc_id, len(result["entities"]), len(result["events"]))
    except Exception as e:
        logger.warning("SAG 数据入库失败（fail-open，不阻断文档入库）: %s", e)
