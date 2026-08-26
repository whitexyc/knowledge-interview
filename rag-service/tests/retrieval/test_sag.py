"""SAG 检索模式单测（module-081）

覆盖：DDL 幂等 / 抽取 mock 合法+非法+空 / hook 开关三态 /
检索实体匹配+一跳+空 / 开关切换零回归
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestSAGDDLIdempotent:
    """DDL 幂等：二次运行不报错"""

    @pytest.mark.asyncio
    async def test_sag_ddl_idempotent(self):
        """CREATE TABLE IF NOT EXISTS 幂等（模拟二次运行）"""
        from src.database import ensure_sag_tables
        with patch("src.database.async_session_factory") as mock_factory:
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            await ensure_sag_tables()
            await ensure_sag_tables()
            assert mock_session.execute.call_count > 0


class TestSAGExtractEntitiesEvents:
    """SAG 抽取 mock：合法/非法/空 JSON"""

    @pytest.mark.asyncio
    async def test_extract_entities_events_success(self):
        from rag.retrieval.sag_extractor import extract_entities_events
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=json.dumps({
            "entities": [{"name": "G1 GC", "type": "technology"}],
            "events": [{"text": "G1 GC introduced", "entity_names": ["G1 GC"]}],
        }))
        with patch("rag.retrieval.sag_extractor.LLMFactory") as mock_factory:
            mock_factory.get_client.return_value = mock_client
            result = await extract_entities_events("G1 GC is a garbage collector")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "G1 GC"
        assert len(result["events"]) == 1

    @pytest.mark.asyncio
    async def test_extract_entities_events_invalid_json(self):
        from rag.retrieval.sag_extractor import extract_entities_events
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="not json at all")
        with patch("rag.retrieval.sag_extractor.LLMFactory") as mock_factory:
            mock_factory.get_client.return_value = mock_client
            result = await extract_entities_events("some text")
        assert result["entities"] == []
        assert result["events"] == []

    @pytest.mark.asyncio
    async def test_extract_entities_events_exception(self):
        from rag.retrieval.sag_extractor import extract_entities_events
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=Exception("LLM error"))
        with patch("rag.retrieval.sag_extractor.LLMFactory") as mock_factory:
            mock_factory.get_client.return_value = mock_client
            result = await extract_entities_events("some text")
        assert result["entities"] == []
        assert result["events"] == []

    @pytest.mark.asyncio
    async def test_extract_empty_text(self):
        from rag.retrieval.sag_extractor import extract_entities_events
        result = await extract_entities_events("")
        assert result["entities"] == []
        assert result["events"] == []

    @pytest.mark.asyncio
    async def test_extract_filters_invalid_entity_type(self):
        from rag.retrieval.sag_extractor import extract_entities_events
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=json.dumps({
            "entities": [
                {"name": "Valid", "type": "concept"},
                {"name": "Invalid", "type": "unknown_type"},
            ],
            "events": [],
        }))
        with patch("rag.retrieval.sag_extractor.LLMFactory") as mock_factory:
            mock_factory.get_client.return_value = mock_client
            result = await extract_entities_events("text")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Valid"


class TestSAGIngestHook:
    """SAG 入库 hook：开关开/关/抽取失败 fail-open"""

    @pytest.mark.asyncio
    async def test_ingest_hook_enabled(self):
        """开关开（sag）→ asyncio.create_task 被调用（SAG hook 触发）"""
        from rag.retrieval.document_ingest import ingest_document
        with patch("rag.retrieval.document_ingest.settings") as mock_settings, \
             patch("rag.retrieval.document_ingest.document_parser") as mock_parser, \
             patch("rag.retrieval.document_ingest.image_pipeline") as mock_pipeline, \
             patch("rag.retrieval.document_ingest.document_cleaner") as mock_cleaner, \
             patch("rag.retrieval.document_ingest.document_dedup") as mock_dedup, \
             patch("rag.retrieval.document_ingest.save_original") as mock_save:
            from src.config import settings as real_settings
            for attr in dir(real_settings):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real_settings, attr))
                    except Exception:
                        pass
            mock_settings.retrieval_mode = "sag"
            mock_settings.doc_dedup_semantic_enabled = False
            mock_parser.parse_document.return_value = MagicMock(text="content", format="md", page_count=1)
            mock_pipeline.process_pdf_images.return_value = "content"
            mock_cleaner.clean.return_value = "content"
            mock_cleaner.normalize.return_value = "content"
            mock_dedup.exact_hash.return_value = "abc123"
            mock_save.return_value = "/tmp/test"

            with patch("rag.retrieval.document_ingest._find_exact_duplicate", new_callable=AsyncMock, return_value=None), \
                 patch("rag.engine.rag_engine.add_document", new_callable=AsyncMock, return_value={"id": 42, "title": "test", "chunks": 1}), \
                 patch("asyncio.create_task") as mock_task:
                result = await ingest_document(b"content", "test.md")
            assert result["id"] == 42
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_hook_disabled(self):
        """开关关（hybrid）→ asyncio.create_task 不被调用"""
        from rag.retrieval.document_ingest import ingest_document
        with patch("rag.retrieval.document_ingest.settings") as mock_settings, \
             patch("rag.retrieval.document_ingest.document_parser") as mock_parser, \
             patch("rag.retrieval.document_ingest.image_pipeline") as mock_pipeline, \
             patch("rag.retrieval.document_ingest.document_cleaner") as mock_cleaner, \
             patch("rag.retrieval.document_ingest.document_dedup") as mock_dedup, \
             patch("rag.retrieval.document_ingest.save_original") as mock_save:
            from src.config import settings as real_settings
            for attr in dir(real_settings):
                if not attr.startswith("_"):
                    try:
                        setattr(mock_settings, attr, getattr(real_settings, attr))
                    except Exception:
                        pass
            mock_settings.retrieval_mode = "hybrid"
            mock_settings.doc_dedup_semantic_enabled = False
            mock_parser.parse_document.return_value = MagicMock(text="content", format="md", page_count=1)
            mock_pipeline.process_pdf_images.return_value = "content"
            mock_cleaner.clean.return_value = "content"
            mock_cleaner.normalize.return_value = "content"
            mock_dedup.exact_hash.return_value = "abc123"
            mock_save.return_value = "/tmp/test"

            with patch("rag.retrieval.document_ingest._find_exact_duplicate", new_callable=AsyncMock, return_value=None), \
                 patch("rag.engine.rag_engine.add_document", new_callable=AsyncMock, return_value={"id": 42, "title": "test", "chunks": 1}), \
                 patch("asyncio.create_task") as mock_task:
                result = await ingest_document(b"content", "test.md")
            mock_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_hook_fail_open(self):
        """抽取失败不阻断入库（fail-open）"""
        from rag.retrieval.sag_extractor import ingest_sag_data
        with patch("rag.retrieval.sag_extractor.extract_entities_events",
                   new_callable=AsyncMock, side_effect=Exception("LLM down")):
            await ingest_sag_data(99, "some document text")


class TestSAGRetrieve:
    """SAG 检索：实体匹配/一跳关系/空结果/无匹配"""

    @pytest.mark.asyncio
    async def test_sag_retrieve_entity_match(self):
        from rag.retrieval.sag_retriever import retrieve
        mock_row = (1, "Test Title", "Test Content", "test_source", {})
        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            mock_ge.extract_from_query = AsyncMock(return_value=["G1 GC"])
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: [([1],)]),
                MagicMock(fetchall=lambda: [mock_row]),
            ])
            docs = await retrieve("What is G1 GC?", top_k=5)
        assert len(docs) >= 1
        assert docs[0]["id"] == 1
        assert docs[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_sag_retrieve_relation_hop(self):
        from rag.retrieval.sag_retriever import retrieve
        mock_doc_row = (2, "Related Doc", "Related Content", "source2", {})
        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge, \
             patch("rag.retrieval.sag_retriever.async_session_factory") as mock_factory:
            mock_ge.extract_from_query = AsyncMock(return_value=["Kafka"])
            mock_session = AsyncMock()
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = mock_ctx
            mock_session.execute = AsyncMock(side_effect=[
                MagicMock(fetchall=lambda: []),
                MagicMock(fetchall=lambda: [(2,)]),
                MagicMock(fetchall=lambda: [mock_doc_row]),
            ])
            docs = await retrieve("Kafka topics", top_k=5)
        assert len(docs) >= 1
        assert docs[0]["score"] == 0.8

    @pytest.mark.asyncio
    async def test_sag_retrieve_empty_query(self):
        from rag.retrieval.sag_retriever import retrieve
        docs = await retrieve("", top_k=5)
        assert docs == []

    @pytest.mark.asyncio
    async def test_sag_retrieve_no_match(self):
        from rag.retrieval.sag_retriever import retrieve
        with patch("rag.retrieval.sag_retriever.graph_extractor") as mock_ge:
            mock_ge.extract_from_query = AsyncMock(return_value=[])
            docs = await retrieve("random query with no entities", top_k=5)
        assert docs == []


class TestSAGModeSwitch:
    """开关切换：hybrid 零回归（conftest 钉住）"""

    def test_default_retrieval_mode_is_hybrid(self):
        from src.config import settings
        assert settings.retrieval_mode == "hybrid"

    def test_entity_types_list(self):
        from rag.retrieval.sag_extractor import ENTITY_TYPES
        assert len(ENTITY_TYPES) == 11
        assert "concept" in ENTITY_TYPES
        assert "method" in ENTITY_TYPES
