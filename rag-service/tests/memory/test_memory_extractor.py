"""Module-033 长期记忆自动写入单元测试

覆盖（验收 §1.1-1.5 + §2 接口 + §3 质量）：
- extract_facts：结构化 JSON 提取 / importance 过滤 / 空 content 丢弃 /
  失败降级 / 超时降级 / 空 answer 不调 LLM / markdown 包裹 JSON 解析
- _find_duplicate：cosine>0.95 命中 / 低相似 None / 嵌入失败 None / DB 失败 None
- save(dedup=True)：重复 → 更新旧父块（追加合并，条数不涨）/ 无重复 → 正常新增 /
  去重失败降级为正常新增
- recall 动态 K：均值>0.85→5 / 0.75-0.85→3 / <0.75→1 / 空候选返回空
- format_memory_line + engine._recall_memory：'[长期记忆 - 日期]：内容' 格式化注入
- _persist_memory：提取→逐条 save(dedup=True) / 空 answer 跳过 / 失败降级
- engine.chat：knowledge 路径触发后台写入 / casual_chat / realtime 跳过
- main.schedule_stream_persist + chat_stream 端点：knowledge 触发 / casual / realtime 跳过

实现说明：同步用例内 asyncio.run 执行，不依赖 pytest-asyncio
（与套件其余用例同款模式）；mock 打桩 LLM / 检索 / DB，不依赖真实依赖。
"""
import asyncio
from contextlib import nullcontext
from unittest import mock

import httpx

import main
from rag.memory import memory_service, format_memory_line
from rag.memory_extractor import extract_facts
from rag.engine import rag_engine
from rag.schemas import ChatRequest
from main import schedule_stream_persist


# ─── 测试桩工具（与 test_memory.py / test_stream_memory.py 同款） ───


class _FakeSession:
    """假 AsyncSession：记录 add 的对象 + 可配置 execute 结果"""

    def __init__(self, scalar=None, scalars=None):
        self.added: list = []
        self._scalar = scalar
        self._scalars = scalars or []
        self.rolled_back = False

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for i, obj in enumerate(self.added):
            if getattr(obj, "parent_id", None) is None:
                obj.id = i + 1

    async def commit(self):
        pass

    async def rollback(self):
        self.rolled_back = True

    async def execute(self, stmt):
        result = mock.MagicMock()
        result.scalar.return_value = self._scalar
        result.scalars.return_value = mock.MagicMock(
            all=mock.MagicMock(return_value=self._scalars),
        )
        return result


class _FakeSessionGet(_FakeSession):
    """假 AsyncSession：额外支持 session.get（去重更新父块路径用）"""

    def __init__(self, get_return):
        super().__init__(scalar=0)
        self._get_return = get_return

    async def get(self, model, pk):
        return self._get_return


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器"""
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


def _chunk_single(content):
    """短记忆分块桩：单个父块 + 单个子块"""
    return {
        "parents": [{"title": "记忆", "content": content}],
        "children": [{"title": "记忆", "content": content, "parent_index": 0}],
    }


def _parse_sse(body: bytes) -> list[dict]:
    """把 SSE 响应体解析成事件列表 [{event, data}, ...]"""
    events = []
    for block in body.decode("utf-8").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        evt = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                evt["event"] = line[len("event: "):]
            elif line.startswith("data: "):
                evt["data"] = line[len("data: "):]
        if evt:
            events.append(evt)
    return events


def _doc(doc_id: int = 1) -> dict:
    return {
        "id": doc_id, "title": "测试文档", "content": "这是一段测试内容。",
        "source": "test", "hybrid_score": 0.95,
    }


class _GenCapture:
    """捕获 generate_answer_stream 调用参数，并用假 token 流式产出"""

    def __init__(self, tokens=None):
        self.calls: list[dict] = []
        self.tokens = tokens or ["答", "案"]

    def make_gen(self):
        async def fake_generate_answer_stream(query, documents, history=None, memory=""):
            self.calls.append({
                "query": query, "documents": documents,
                "history": history, "memory": memory,
            })
            for tok in self.tokens:
                yield tok
        return fake_generate_answer_stream


class _FakeLLM:
    """casual_chat / 降级路径用的假 LLM 客户端"""

    def __init__(self):
        async def generate_stream(prompt):
            yield "你好"
        self.generate_stream = generate_stream


# ─── 记忆提取器 extract_facts ───


class TestExtractFacts:
    """extract_facts：LLM 提取 / 过滤 / 降级 / JSON 结构"""

    @staticmethod
    def _mock_client(raw):
        fake = mock.MagicMock()
        fake.generate = mock.AsyncMock(return_value=raw)
        return fake

    def test_extracts_and_filters_facts(self):
        raw = ('{"facts": [{"content": "用户偏好简短回答", "importance": 0.9},'
               '{"content": "一次性临时问题", "importance": 0.3}]}')

        async def run():
            fake = self._mock_client(raw)
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("回答风格", "可以，我会简短回答", [])
            # module-062：返回结构加 type 字段（缺失 → 默认 fact，增量字段）
            assert facts == [{"content": "用户偏好简短回答", "importance": 0.9, "type": "fact"}]
        asyncio.run(run())

    def test_empty_content_dropped(self):
        raw = ('{"facts": [{"content": "  ", "importance": 0.9},'
               '{"content": "", "importance": 0.8}]}')

        async def run():
            fake = self._mock_client(raw)
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            assert facts == []
        asyncio.run(run())

    def test_importance_at_threshold_kept(self):
        # importance=0.6（阈值）→ 保留（>= 阈值）；0.59 → 丢弃
        raw = ('{"facts": [{"content": "边界事实", "importance": 0.6},'
               '{"content": "低重要", "importance": 0.59}]}')

        async def run():
            fake = self._mock_client(raw)
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            # module-062：缺失 type → 默认 fact（增量字段）
            assert facts == [{"content": "边界事实", "importance": 0.6, "type": "fact"}]
        asyncio.run(run())

    def test_non_numeric_importance_dropped(self):
        raw = '{"facts": [{"content": "无分数", "importance": "high"}]}'

        async def run():
            fake = self._mock_client(raw)
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            assert facts == []
        asyncio.run(run())

    def test_llm_failure_returns_empty(self):
        async def run():
            fake = mock.MagicMock()
            fake.generate = mock.AsyncMock(side_effect=RuntimeError("llm down"))
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            assert facts == []
        asyncio.run(run())

    def test_llm_timeout_returns_empty(self):
        async def run():
            fake = mock.MagicMock()
            fake.generate = mock.AsyncMock(side_effect=asyncio.TimeoutError())
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            assert facts == []
        asyncio.run(run())

    def test_empty_answer_no_llm_call(self):
        async def run():
            with mock.patch("rag.memory_extractor.LLMFactory.get_client") as gc:
                assert await extract_facts("q", "  ", []) == []
                gc.assert_not_called()
        asyncio.run(run())

    def test_json_with_markdown_fence_parsed(self):
        raw = '```json\n{"facts": [{"content": "用户偏好", "importance": 0.95}]}\n```'

        async def run():
            fake = self._mock_client(raw)
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            # module-062：缺失 type → 默认 fact（增量字段）
            assert facts == [{"content": "用户偏好", "importance": 0.95, "type": "fact"}]
        asyncio.run(run())

    def test_parse_failure_returns_empty(self):
        raw = "抱歉，我无法理解你的问题。"

        async def run():
            fake = self._mock_client(raw)
            with mock.patch("rag.memory_extractor.LLMFactory.get_client", return_value=fake):
                facts = await extract_facts("q", "a", [])
            assert facts == []
        asyncio.run(run())


# ─── 语义去重 ───


class TestFindDuplicate:
    """_find_duplicate：cosine 相似度 > 阈值命中 / 降级 None"""

    def test_high_cosine_returns_duplicate(self):
        existing = mock.MagicMock(id=7, parent_id=5, embedding=[0.99, 0.0, 0.0])

        async def run():
            with mock.patch("rag.memory.embedding_service") as emb:
                emb.embed_text = mock.AsyncMock(return_value=[1.0, 0.0, 0.0])
                with mock.patch("rag.memory.async_session_factory",
                                _fake_factory(_FakeSession(scalars=[existing]))):
                    dup = await memory_service._find_duplicate("用户偏好", "42")
            assert dup is not None
            assert dup.id == 7  # 0.99 > 0.85 → 命中重复
        asyncio.run(run())

    def test_low_cosine_returns_none(self):
        existing = mock.MagicMock(id=7, parent_id=5, embedding=[0.4, 0.0, 0.0])

        async def run():
            with mock.patch("rag.memory.embedding_service") as emb:
                emb.embed_text = mock.AsyncMock(return_value=[1.0, 0.0, 0.0])
                with mock.patch("rag.memory.async_session_factory",
                                _fake_factory(_FakeSession(scalars=[existing]))):
                    dup = await memory_service._find_duplicate("另一个偏好", "42")
            assert dup is None  # 0.4 ≤ 0.85 → 不同事实
        asyncio.run(run())

    def test_embedding_failure_returns_none(self):
        async def run():
            with mock.patch("rag.memory.embedding_service") as emb:
                emb.embed_text = mock.AsyncMock(side_effect=RuntimeError("embed down"))
                with mock.patch("rag.memory.async_session_factory",
                                _fake_factory(_FakeSession())):
                    dup = await memory_service._find_duplicate("偏好", "42")
            assert dup is None  # 去重向量化失败降级
        asyncio.run(run())

    def test_db_failure_returns_none(self):
        async def run():
            sess = mock.MagicMock()
            sess.__aenter__ = mock.AsyncMock(side_effect=RuntimeError("db down"))
            sess.__aexit__ = mock.AsyncMock(return_value=False)
            with mock.patch("rag.memory.embedding_service") as emb:
                emb.embed_text = mock.AsyncMock(return_value=[1.0, 0.0, 0.0])
                with mock.patch("rag.memory.async_session_factory",
                                mock.MagicMock(return_value=sess)):
                    dup = await memory_service._find_duplicate("偏好", "42")
            assert dup is None  # 去重检索失败降级
        asyncio.run(run())


class TestSaveDedup:
    """save(dedup=True)：重复更新 / 无重复新增 / 失败降级"""

    def test_duplicate_merges_into_parent_no_new_rows(self):
        dup = mock.MagicMock(id=10, parent_id=5)
        parent = mock.MagicMock(id=5, title="记忆-2026-08-01-01", content="旧内容")

        async def run():
            fs = _FakeSessionGet(parent)
            with mock.patch.object(memory_service, "_find_duplicate",
                                   new=mock.AsyncMock(return_value=dup)):
                with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                    result = await memory_service.save("同义新措辞", "42", dedup=True)
            assert result["status"] == "updated"
            assert result["id"] == 5
            assert fs.added == []  # 不新增行（库内条数不涨）
            assert parent.content == "旧内容\n同义新措辞"  # 追加合并到既有父块
        asyncio.run(run())

    def test_no_duplicate_normal_new_insert(self):
        async def run():
            fs = _FakeSession(scalar=0)
            with mock.patch.object(memory_service, "_find_duplicate",
                                   new=mock.AsyncMock(return_value=None)):
                with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                    with mock.patch("rag.memory.chunker") as chunker_mock:
                        with mock.patch("rag.memory.embedding_service") as emb_mock:
                            chunker_mock.chunk.return_value = _chunk_single("不同事实")
                            emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                            result = await memory_service.save("不同事实", "42", dedup=True)
            assert result["status"] == "saved"
            assert len(fs.added) == 2  # 父块 + 子块正常新增
        asyncio.run(run())

    def test_dedup_failure_degrades_to_new_insert(self):
        async def run():
            fs = _FakeSession(scalar=0)
            with mock.patch.object(memory_service, "_find_duplicate",
                                   new=mock.AsyncMock(side_effect=RuntimeError("db down"))):
                with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                    with mock.patch("rag.memory.chunker") as chunker_mock:
                        with mock.patch("rag.memory.embedding_service") as emb_mock:
                            chunker_mock.chunk.return_value = _chunk_single("新事实")
                            emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                            result = await memory_service.save("新事实", "42", dedup=True)
            assert result["status"] == "saved"  # 去重异常降级为正常新增
        asyncio.run(run())

    def test_dedup_disabled_skips_find(self):
        async def run():
            fs = _FakeSession(scalar=0)
            with mock.patch.object(memory_service, "_find_duplicate",
                                   new=mock.AsyncMock()) as find:
                with mock.patch("rag.memory.async_session_factory", _fake_factory(fs)):
                    with mock.patch("rag.memory.chunker") as chunker_mock:
                        with mock.patch("rag.memory.embedding_service") as emb_mock:
                            chunker_mock.chunk.return_value = _chunk_single("事实")
                            emb_mock.embed_documents = mock.AsyncMock(return_value=[[0.1]])
                            await memory_service.save("事实", "42", dedup=False)
            find.assert_not_called()
        asyncio.run(run())


# ─── 动态 K 召回 ───


class TestRecallDynamicK:
    """recall 动态 K：按候选平均绝对余弦调整召回条数（module-035 口径，宁缺毋滥）"""

    @staticmethod
    def _retrieve_docs(cosines):
        # module-035：候选 embedding 由 mock 提供 → 绝对余弦（id/parent_id 一一对应）
        return [{"id": i, "parent_id": i, "hybrid_score": 0.5}
                for i in range(len(cosines))]

    def _recall(self, cosines, expanded):
        async def run():
            with mock.patch("rag.memory.hybrid_retriever") as ret:
                ret.retrieve = mock.AsyncMock(return_value=self._retrieve_docs(cosines))
                with mock.patch("rag.memory.embedding_service") as emb:
                    emb.embed_text = mock.AsyncMock(return_value=[1.0, 0.0])
                    with mock.patch.object(memory_service, "_child_embeddings",
                                           new=mock.AsyncMock(return_value={
                                               i: [c, 0.0] for i, c in enumerate(cosines)})):
                        with mock.patch.object(memory_service, "_expand_to_parents",
                                               new=mock.AsyncMock(return_value=expanded)):
                            return await memory_service.recall("q", "42")
        return asyncio.run(run())

    def test_high_quality_recalls_five(self):
        expanded = [{"content": f"c{i}", "score": 0.9, "title": "t"} for i in range(5)]
        memories = self._recall([0.9] * 5, expanded)
        assert len(memories) == 5  # 绝对余弦均值 0.9 > 0.85 → K=5

    def test_mid_quality_recalls_three(self):
        expanded = [{"content": f"c{i}", "score": 0.8, "title": "t"} for i in range(5)]
        memories = self._recall([0.8] * 5, expanded)
        assert len(memories) == 3  # 绝对余弦均值 0.8 ∈ [0.75,0.85] → K=3

    def test_low_quality_recalls_one(self):
        expanded = [{"content": f"c{i}", "score": 0.7, "title": "t"} for i in range(5)]
        memories = self._recall([0.7] * 5, expanded)
        assert len(memories) == 1  # 绝对余弦均值 0.7 < 0.75 → K=1（宁缺毋滥）

    def test_empty_candidates_returns_empty(self):
        async def run():
            with mock.patch("rag.memory.hybrid_retriever") as ret:
                ret.retrieve = mock.AsyncMock(return_value=[])
                with mock.patch.object(memory_service, "_expand_to_parents",
                                       new=mock.AsyncMock()) as expand:
                    assert await memory_service.recall("q", "42") == []
                    expand.assert_not_called()
        asyncio.run(run())

    def test_dynamic_k_thresholds(self):
        assert memory_service._dynamic_k(0.86) == 5
        assert memory_service._dynamic_k(0.85) == 3
        assert memory_service._dynamic_k(0.75) == 3
        assert memory_service._dynamic_k(0.74) == 1


# ─── 格式化注入 ───


class TestFormatting:
    """'[长期记忆 - 日期]：内容' 格式化注入（无日期省略）"""

    def test_format_line_with_date(self):
        line = format_memory_line({"content": "偏好", "created_at": "2026-08-05"})
        assert line == "[长期记忆 - 2026-08-05]：偏好"

    def test_format_line_without_date(self):
        line = format_memory_line({"content": "偏好"})
        assert line == "[长期记忆]：偏好"

    def test_recall_memory_formats_with_date(self):
        async def run():
            with mock.patch("rag.engine.memory_service.recall",
                            new=mock.AsyncMock(return_value=[
                                {"content": "偏好简短回答", "created_at": "2026-08-05"},
                                {"content": "旧事实", "created_at": None},
                            ])):
                with mock.patch("rag.engine.memory_service.recall_short",
                                new=mock.AsyncMock(return_value=[])):
                    text = await rag_engine._recall_memory("回答风格", "42", top_k=3)
            assert text == ("历史记忆:\n[长期记忆 - 2026-08-05]：偏好简短回答\n"
                            "[长期记忆]：旧事实")
        asyncio.run(run())


# ─── 异步自动写入 _persist_memory ───


class TestPersistMemory:
    """_persist_memory：提取→逐条 save(dedup=True)；失败降级"""

    def test_extracts_and_saves_each_fact(self):
        facts = [{"content": "事实A", "importance": 0.9},
                 {"content": "事实B", "importance": 0.8}]

        async def run():
            with mock.patch("rag.engine.extract_facts", new=mock.AsyncMock(return_value=facts)):
                with mock.patch("rag.engine.memory_service.save", new=mock.AsyncMock()) as save:
                    with mock.patch("rag.engine.memory_service.save_short", new=mock.AsyncMock()) as save_short:
                        await rag_engine._persist_memory("问题", "答案", "42", [])
            assert save.call_count == 2
            args0, kwargs0 = save.call_args_list[0]
            assert args0[0] == "事实A"
            assert args0[1] == "42"
            assert kwargs0.get("dedup") is True  # 逐条语义去重
            args1, _ = save.call_args_list[1]
            assert args1[0] == "事实B"
            # module-034：同一批事实沉淀短期记忆（TTL 7 天），同样逐条去重
            assert save_short.call_count == 2
            assert save_short.call_args_list[0][0][0] == "事实A"
        asyncio.run(run())

    def test_empty_answer_skips_extract(self):
        async def run():
            with mock.patch("rag.engine.extract_facts", new=mock.AsyncMock()) as extract:
                with mock.patch("rag.engine.memory_service.save", new=mock.AsyncMock()) as save:
                    await rag_engine._persist_memory("问题", "  ", "42", [])
            extract.assert_not_called()
            save.assert_not_called()
        asyncio.run(run())

    def test_extract_failure_no_save(self):
        async def run():
            with mock.patch("rag.engine.extract_facts",
                            new=mock.AsyncMock(side_effect=RuntimeError("llm down"))):
                with mock.patch("rag.engine.memory_service.save", new=mock.AsyncMock()) as save:
                    await rag_engine._persist_memory("问题", "答案", "42", [])
            save.assert_not_called()
        asyncio.run(run())

    def test_single_save_failure_degrades(self):
        async def run():
            with mock.patch("rag.engine.extract_facts",
                            new=mock.AsyncMock(return_value=[{"content": "A", "importance": 0.9}])):
                with mock.patch("rag.engine.memory_service.save",
                                new=mock.AsyncMock(side_effect=RuntimeError("db down"))):
                    with mock.patch("rag.engine.memory_service.save_short", new=mock.AsyncMock()):
                        # 单条 save 失败仅日志降级，不抛错（短期 save_short 仍继续）
                        await rag_engine._persist_memory("问题", "答案", "42", [])
        asyncio.run(run())


# ─── engine.chat 触发 ───


class TestChatPersistTrigger:
    """engine.chat：knowledge 路径异步触发；casual_chat / realtime 跳过"""

    def test_knowledge_triggers_background_persist(self):
        persist_calls = []

        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(return_value={"intent": "knowledge"})):
                with mock.patch("rag.engine.memory_service.recall", new=mock.AsyncMock(return_value=[])):
                    with mock.patch("rag.engine.memory_service.recall_short", new=mock.AsyncMock(return_value=[])):
                        with mock.patch("rag.engine.hybrid_retriever.retrieve",
                                        new=mock.AsyncMock(return_value=[])):
                            with mock.patch("rag.engine.reranker.rerank",
                                            new=mock.AsyncMock(side_effect=lambda q, docs, top_k=5: docs)):
                                with mock.patch("rag.engine.reflector.check_sufficiency",
                                                new=mock.AsyncMock(return_value={"sufficient": True})):
                                    with mock.patch("rag.engine.LLMFactory.get_client") as gc:
                                        fake = mock.MagicMock()
                                        fake.generate = mock.AsyncMock(return_value="这是答案")
                                        gc.return_value = fake
                                        with mock.patch.object(
                                            rag_engine, "_persist_memory",
                                            new=mock.AsyncMock(
                                                side_effect=lambda *a, **k: persist_calls.append(a)),
                                        ):
                                            with mock.patch.object(rag_engine, "_schedule_session_persist",
                                                                   new=mock.MagicMock()):
                                                result = await rag_engine.chat(
                                                    ChatRequest(query="问题",
                                                                history=[{"role": "user", "content": "hi"}]),
                                                    identity="42")
                                                await asyncio.sleep(0)  # 让后台任务跑完
            assert result.answer == "这是答案"
            assert len(persist_calls) == 1
            args = persist_calls[0]
            assert args[0] == "问题"
            assert args[1] == "这是答案"
            assert args[2] == "42"
            assert args[3] == [{"role": "user", "content": "hi"}]
        asyncio.run(run())

    def test_casual_chat_skips_persist(self):
        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(return_value={"intent": "casual_chat"})):
                with mock.patch("rag.engine.memory_service.recall", new=mock.AsyncMock(return_value=[])):
                    with mock.patch("rag.engine.LLMFactory.get_client") as gc:
                        fake = mock.MagicMock()
                        fake.chat = mock.AsyncMock(return_value="好的")
                        gc.return_value = fake
                        with mock.patch.object(rag_engine, "_persist_memory",
                                               new=mock.AsyncMock()) as persist:
                            result = await rag_engine.chat(ChatRequest(query="你好"), identity="42")
                            await asyncio.sleep(0)
            assert result.message == "casual_chat"
            persist.assert_not_called()
        asyncio.run(run())

    def test_realtime_skips_persist(self):
        async def run():
            with mock.patch("rag.engine.router_agent.classify",
                            new=mock.AsyncMock(return_value={"intent": "realtime"})):
                with mock.patch.object(rag_engine, "_persist_memory",
                                       new=mock.AsyncMock()) as persist:
                    result = await rag_engine.chat(ChatRequest(query="现在几点"), identity="42")
                    await asyncio.sleep(0)
            assert "开发中" in result.answer
            persist.assert_not_called()
        asyncio.run(run())


# ─── main.schedule_stream_persist（纯函数） ───


class TestScheduleStreamPersist:
    """schedule_stream_persist：intent=knowledge 且 answer 非空才触发"""

    @staticmethod
    def _capture(intent, answer, identity="42"):
        calls = []

        async def run():
            with mock.patch.object(rag_engine, "_persist_memory",
                                   new=mock.AsyncMock(
                                       side_effect=lambda *a, **k: calls.append(a))):
                schedule_stream_persist(intent, "问题", answer, identity, [])
                await asyncio.sleep(0)
        asyncio.run(run())
        return calls

    def test_knowledge_with_answer_triggers(self):
        calls = self._capture("knowledge", "答案")
        assert len(calls) == 1
        assert calls[0][0] == "问题"
        assert calls[0][1] == "答案"
        assert calls[0][2] == "42"

    def test_casual_skips(self):
        assert self._capture("casual_chat", "答案") == []

    def test_realtime_skips(self):
        assert self._capture("realtime", "答案") == []

    def test_empty_answer_skips(self):
        assert self._capture("knowledge", "") == []
        assert self._capture("knowledge", "   ") == []


# ─── chat_stream 端点集成 ───


def _hit_stream_persist(classify_intent="knowledge", xff="9.9.9.9"):
    """打桩全链路后访问 /ai/rag/chat/stream，收集后台 _persist_memory 调用

    返回 (sse_events, persist_calls, gen_capture)。
    """
    gen = _GenCapture()
    persist_calls = []

    async def run():
        llm_ctx = (mock.patch("llm.client.LLMFactory.get_client", return_value=_FakeLLM())
                   if classify_intent == "casual_chat" else nullcontext())
        with llm_ctx:
            with mock.patch("agent.router.router_agent.classify",
                            new=mock.AsyncMock(return_value={"intent": classify_intent})):
                with mock.patch("rag.engine.rag_engine._retrieve",
                                new=mock.AsyncMock(return_value=[_doc()])):
                    with mock.patch("rag.engine.rag_engine._rerank",
                                    new=mock.AsyncMock(side_effect=lambda q, docs: docs)):
                        with mock.patch("rag.engine.rag_engine._recall_memory",
                                        new=mock.AsyncMock(return_value="")):
                            with mock.patch("agent.reflector.reflector.check_sufficiency",
                                            new=mock.AsyncMock(
                                                return_value={"sufficient": True, "reason": ""})):
                                with mock.patch("agent.reflector.reflector.generate_answer_stream",
                                                new=gen.make_gen()):
                                    with mock.patch("rag.engine.rag_engine._resolve_session_history",
                                                    new=mock.AsyncMock(
                                                        side_effect=lambda identity, h: h)):
                                        with mock.patch("rag.engine.rag_engine._schedule_session_persist",
                                                        new=mock.MagicMock()):
                                            with mock.patch.object(
                                                rag_engine, "_persist_memory",
                                                new=mock.AsyncMock(
                                                    side_effect=lambda *a, **k: persist_calls.append(a)),
                                            ):
                                                transport = httpx.ASGITransport(
                                                    app=main.app, raise_app_exceptions=True)
                                                async with httpx.AsyncClient(
                                                        transport=transport,
                                                        base_url="http://test") as client:
                                                    resp = await client.post(
                                                        "/ai/rag/chat/stream",
                                                        headers={"X-Forwarded-For": xff},
                                                        json={"query": "回答风格",
                                                              "history": [{"role": "user",
                                                                           "content": "之前聊过"}]},
                                                    )
                                                    events = _parse_sse(resp.content)
                                                await asyncio.sleep(0)  # 让后台任务跑完
        return events

    events = asyncio.run(run())
    return events, persist_calls, gen


class TestChatStreamPersist:
    """chat_stream：knowledge 生成结束触发后台写入；casual / realtime 跳过"""

    def test_knowledge_triggers_persist_with_accumulated_answer(self):
        events, persist_calls, gen = _hit_stream_persist("knowledge")
        assert events and events[-1]["event"] == "done"
        assert len(persist_calls) == 1
        args = persist_calls[0]
        assert args[0] == "回答风格"
        assert args[1] == "".join(gen.tokens)  # 完整答案 = 各 token 拼接
        assert args[2] == "9.9.9.9"  # identity 从 request.state 透传
        assert args[3] == [{"role": "user", "content": "之前聊过"}]

    def test_casual_chat_skips_persist(self):
        events, persist_calls, _ = _hit_stream_persist("casual_chat")
        assert persist_calls == []  # 闲聊不提取

    def test_realtime_skips_persist(self):
        events, persist_calls, _ = _hit_stream_persist("realtime")
        assert persist_calls == []  # 实时路径不提取
