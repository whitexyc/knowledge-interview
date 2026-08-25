"""
知识图谱实体提取器 — Graph RAG LLM 层
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置：
  文档入库 → [GraphExtractor.extract_from_document] 提取实体/关系
  用户查询 → [GraphExtractor.extract_from_query] 提取查询中的实体

依赖：
  - LLMFactory.get_client() 用于调用 LLM
  - 所有提取操作均 try/except，失败返回空

设计决策：
  1. 为什么 extract_from_document 分两次 LLM 调用？
     一次提取实体、一次提取关系。分开处理的原因：
     - 实体类型信息在文档标题和关键词中更明确
     - 关系提取需要理解段落间的语义连接
     - 分步处理降低单次 prompt 复杂度，提高提取质量

  2. 为什么 extract_from_query 只提取实体名称？
     查询通常很短（<50 字），只需要提取实体名称用于图检索入口。
     不需要提取关系。

  3. 为什么 _parse_json 有 n 回退策略？
     LLM 输出可能夹杂 markdown 包裹或被截断。
     逐级回退确保总能得到有效数据或空值（绝不抛异常）。
"""
import json
import logging

from llm.client import LLMFactory

logger = logging.getLogger(__name__)

# 实体提取 prompt：从文档中抽取出技术实体
_ENTITY_PROMPT = """你是一个知识图谱构建专家。阅读以下文档片段，提取其中的技术实体。
实体类型包括但不限于：concept, technology, algorithm, framework, tool, person, company, language。
每个实体提取其 name 和 type。

文档:
{document}

返回 JSON 格式（只返回 JSON，不要其他文字）：
{{"entities": [{{"name": "实体名", "type": "类型"}}, ...]}}

JSON:"""

# 关系提取 prompt：基于已提取的实体和文档内容，找出实体间的关系
_RELATION_PROMPT = """你是一个知识图谱构建专家。基于以下文档和已提取的实体，找出实体间的关系。

实体列表:
{entities}

文档摘要:
{document}

返回 JSON 格式（只返回 JSON，不要其他文字）：
{{"relations": [{{"source": "源实体", "target": "目标实体"}}, ...]}}

JSON:"""

# 查询实体提取 prompt：从用户问题中提取实体名称
_QUERY_ENTITY_PROMPT = """你是一个搜索意图分析专家。从用户问题中提取技术实体名称。
只返回逗号分隔的实体名称（不要 JSON，不要解释）。

用户问题: {query}

实体名称:"""


class GraphExtractor:
    """基于 LLM 的知识图谱实体和关系提取器

    职责：
    1. extract_from_document() — 从文档提取 {entities, relations}
    2. extract_from_query() — 从查询提取 [entity_names]

    所有方法均 try/except 包裹，LLM 失败或解析失败时返回空值。
    """

    async def extract_from_document(self, content: str) -> dict:
        """从文档内容提取实体和关系

        分两步 LLM 调用：
        1. 提取实体（name + type）
        2. 基于实体列表提取关系（source → target）

        Args:
            content: 文档全文内容（会被截断到 2000 字符）

        Returns:
            {"entities": [{"name": str, "type": str}, ...],
             "relations": [{"source": str, "target": str}, ...]}
            失败时返回空值的默认结构
        """
        # 截断超长文档，避免超出 token 限制
        doc_text = content[:2000] if len(content) > 2000 else content
        if not doc_text.strip():
            return {"entities": [], "relations": []}

        entities = []
        relations = []

        try:
            client = LLMFactory.get_client()

            # Step 1: 提取实体
            entity_prompt = _ENTITY_PROMPT.format(document=doc_text)
            entity_raw = await client.generate(entity_prompt)
            entity_data = self._parse_json(entity_raw)
            entities = entity_data.get("entities", [])

            if not entities:
                return {"entities": [], "relations": []}

            # Step 2: 基于实体提取关系
            entity_names = [e.get("name", "") for e in entities if e.get("name")]
            entity_list_str = json.dumps(entity_names, ensure_ascii=False)

            relation_prompt = _RELATION_PROMPT.format(
                entities=entity_list_str,
                document=doc_text[:1000],  # 关系提取用更短的文档摘要
            )
            relation_raw = await client.generate(relation_prompt)
            relation_data = self._parse_json(relation_raw)
            relations = relation_data.get("relations", [])

            logger.info("实体提取完成: entities=%d, relations=%d",
                        len(entities), len(relations))
        except Exception as e:
            logger.warning("实体提取失败，返回空: %s", e)

        return {
            "entities": entities,
            "relations": relations,
        }

    async def extract_from_query(self, query: str) -> list[str]:
        """从用户查询中提取实体名称

        用于 graph_store.search_related() 的入口参数。

        Args:
            query: 用户查询文本

        Returns:
            实体名称列表，失败时返回空列表
        """
        if not query or not query.strip():
            return []

        try:
            client = LLMFactory.get_client()
            prompt = _QUERY_ENTITY_PROMPT.format(query=query)
            raw = await client.generate(prompt)
            # 解析逗号分隔的实体名称
            names = []
            for part in raw.split(","):
                name = part.strip().strip("'\"").strip()
                if name and len(name) > 1:  # 过滤单字符
                    names.append(name)
            logger.info("查询实体提取: query=%s, entities=%d", query[:40], len(names))
            return names[:10]  # 限制数量，避免过度匹配
        except Exception as e:
            logger.warning("查询实体提取失败: %s", e)
            return []

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """解析 LLM 输出的 JSON，多级回退策略

        回退顺序：
        1. 直接 json.loads
        2. 提取 {...} 包裹的 JSON 再解析
        3. 返回空 dict

        Args:
            raw: LLM 原始输出文本

        Returns:
            解析后的 dict，失败时返回 {}
        """
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


# 全局单例
graph_extractor = GraphExtractor()
