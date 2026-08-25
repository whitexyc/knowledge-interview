"""Module-034 会话记忆持久化 + Module-046 WP2 会话摘要单元测试

覆盖（验收 §4.1 test_session_memory.py：会话保存/恢复/隔离/TTL）：
- save_session_messages：写入 source='memory:<identity>:session:'（按身份隔离）、
  空消息/空 content 跳过、content_hash 去重幂等、超上限滚动删除最旧
- get_session_messages：恢复最近会话（时间升序、limit 截断）、按身份隔离、
  无记录返回空列表
- 身份规范化：通配符 identity 降级 'unknown'（复用 memory._normalize_identity）

module-046 WP2（验收 §2 会话摘要 + 降级 §4）：
- 摘要维护：超限滚动删除前最旧消息段 LLM 压缩成摘要（source='memory:<id>:
  session_summary:'，title='session_summary'，无向量）；增量更新（新摘要 =
  摘要(旧摘要 + 新对话段)，MemGPT 递归公式）；仅顺序读最新一条
- 摘要 LLM 失败 → 跳过摘要（fail-open），滚动删除照常执行，不抛异常
- 分层注入：engine._resolve_session_history = 早期摘要段 + 最近 N 条原样；
  无摘要 → 与旧行为逐字节一致（零回归）

实现说明：mock async_session_factory 打桩 AsyncSession（按语句类型/编译后
SQL 路由结果），不依赖真实数据库；同步用例内 asyncio.run 执行（与套件同款
模式）；摘要 LLM 用 mock LLMFactory.get_client 打桩。
"""
import asyncio
import hashlib
from unittest import mock

from rag.engine import rag_engine
from rag.models import Document
from rag.session_memory import (
    session_memory_service, _session_source, _session_summary_source,
)
from src.config import settings


class _FakeSession:
    """假 AsyncSession：按语句类型路由 execute 结果 + 记录 add / delete"""

    def __init__(self, existing_hashes=None, session_count=0, oldest_ids=(), docs=None):
        self.added: list = []
        self.existing_hashes = list(existing_hashes or [])
        self.session_count = session_count
        self.oldest_ids = list(oldest_ids)
        self.docs = list(docs or [])
        self.deleted_stmts = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def execute(self, stmt):
        sql = str(stmt).lower()
        result = mock.MagicMock()
        if "count(" in sql:
            result.scalar.return_value = self.session_count
        elif "delete" in sql:
            self.deleted_stmts.append(stmt)
        elif "order by" in sql:
            # get_session_messages（全列）与 _trim 的 id 查询都带 ORDER BY；
            # 全列查询用 scalars().all()（docs），id 查询用 all()（oldest_ids）
            result.all.return_value = [(i,) for i in self.oldest_ids]
            result.scalars.return_value = mock.MagicMock(
                all=mock.MagicMock(return_value=self.docs),
            )
        elif "content_hash" in sql:
            # save 的去重幂等查询：select(Document.content_hash)，无 ORDER BY
            result.all.return_value = [(h,) for h in self.existing_hashes]
        else:
            result.all.return_value = [(i,) for i in self.oldest_ids]
        return result


def _fake_factory(session):
    """把 async_session_factory 打桩成返回 session 的异步上下文管理器"""
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=session)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return mock.MagicMock(return_value=cm)


class TestSessionSource:
    """_session_source：source='memory:<identity>:session:'"""

    def test_session_source_format(self):
        assert _session_source("42") == "memory:42:session:"
        assert _session_source("1.1.1.1") == "memory:1.1.1.1:session:"
        # 与长期/短期 source 互不混淆
        assert _session_source("42") != "memory:42:"
        assert _session_source("42") != "memory:42:short:"


class TestSaveSession:
    """save_session_messages：写入 / 幂等 / 上限滚动"""

    def test_save_writes_session_source_with_roles(self):
        async def run():
            fs = _FakeSession(existing_hashes=[])
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                n = await session_memory_service.save_session_messages("42", [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ])
            assert n == 2
            # 父/子无 embedding 平铺一条消息，source='memory:42:session:'（与长期/短期区分）
            assert {getattr(d, "source", None) for d in fs.added} == {"memory:42:session:"}
            titles = sorted(d.title for d in fs.added)
            assert titles == ["session:assistant", "session:user"]
            assert {d.content for d in fs.added} == {"你好", "你好！"}
        asyncio.run(run())

    def test_save_empty_messages_returns_zero(self):
        async def run():
            with mock.patch("rag.session_memory.async_session_factory") as fac:
                assert await session_memory_service.save_session_messages("42", []) == 0
                fac.assert_not_called()  # 空消息不碰 DB
        asyncio.run(run())

    def test_save_skips_empty_content_and_duplicate_hash(self):
        digest = hashlib.sha256("你好".encode("utf-8")).hexdigest()

        async def run():
            fs = _FakeSession(existing_hashes=[digest])
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                n = await session_memory_service.save_session_messages("42", [
                    {"role": "user", "content": "你好"},   # content_hash 重复 → 跳过
                    {"role": "user", "content": "   "},    # 空 content → 跳过
                    {"role": "user", "content": "新问题"},  # 新增
                ])
            assert n == 1
            assert [d.content for d in fs.added] == ["新问题"]
        asyncio.run(run())

    def test_save_trims_oldest_when_over_cap(self):
        """每 identity 会话超上限滚动删除最旧（防止 documents 表膨胀）"""
        cap = settings.memory_session_max_messages

        async def run():
            fs = _FakeSession(existing_hashes=[], session_count=cap + 2, oldest_ids=[1, 2])
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                n = await session_memory_service.save_session_messages("42", [
                    {"role": "user", "content": "新问题"},
                ])
            assert n == 1
            # 超限 2 条 → 发起一次 DELETE（删除最旧 id 1、2）
            assert len(fs.deleted_stmts) == 1
            assert "id IN" in str(fs.deleted_stmts[0])
        asyncio.run(run())

    def test_save_wildcard_identity_normalized(self):
        """通配符 identity 降级 'unknown'（复用 memory._normalize_identity）"""
        async def run():
            fs = _FakeSession(existing_hashes=[])
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                await session_memory_service.save_session_messages("%", [
                    {"role": "user", "content": "x"},
                ])
            assert {getattr(d, "source", None) for d in fs.added} == {"memory:unknown:session:"}
        asyncio.run(run())


class TestGetSession:
    """get_session_messages：恢复 / limit / 隔离 / 空"""

    def test_get_returns_ordered_recent(self):
        docs = [
            Document(title="session:user", content="第一轮"),
            Document(title="session:assistant", content="答复一"),
            Document(title="session:user", content="第二轮"),
        ]

        async def run():
            fs = _FakeSession(docs=docs)
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                msgs = await session_memory_service.get_session_messages("42", limit=10)
            assert msgs == [
                {"role": "user", "content": "第一轮"},
                {"role": "assistant", "content": "答复一"},
                {"role": "user", "content": "第二轮"},
            ]
        asyncio.run(run())

    def test_get_respects_limit(self):
        docs = [Document(title="session:user", content=f"消息{i}") for i in range(5)]

        async def run():
            fs = _FakeSession(docs=docs)
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                msgs = await session_memory_service.get_session_messages("42", limit=2)
            assert [m["content"] for m in msgs] == ["消息3", "消息4"]  # 最近 2 条
        asyncio.run(run())

    def test_get_isolated_by_identity(self):
        """恢复只查本身份会话：SQL 按 source='memory:<identity>:session:' 过滤"""
        captured = {}

        class _CaptureSession(_FakeSession):
            async def execute(self, stmt):
                captured["stmt"] = stmt
                return await super().execute(stmt)

        async def run():
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(_CaptureSession(docs=[]))):
                await session_memory_service.get_session_messages("42")

        asyncio.run(run())
        sql = str(captured["stmt"].compile(compile_kwargs={"literal_binds": True}))
        assert "memory:42:session:" in sql
        assert "memory:43:session:" not in sql  # 不含他人身份

    def test_get_empty_returns_empty(self):
        async def run():
            fs = _FakeSession(docs=[])
            with mock.patch("rag.session_memory.async_session_factory", _fake_factory(fs)):
                assert await session_memory_service.get_session_messages("42") == []
        asyncio.run(run())

    def test_get_failure_returns_empty(self):
        """恢复失败 → 返回空列表（调用方降级用当前请求 history，零回归）"""
        async def run():
            sess = mock.MagicMock()
            sess.__aenter__ = mock.AsyncMock(side_effect=RuntimeError("db down"))
            sess.__aexit__ = mock.AsyncMock(return_value=False)
            with mock.patch("rag.session_memory.async_session_factory",
                            mock.MagicMock(return_value=sess)):
                assert await session_memory_service.get_session_messages("42") == []
        asyncio.run(run())


class _SummaryFakeSession:
    """module-046 WP2 摘要流假 AsyncSession（按编译后 SQL literal binds 路由）

    覆盖 save_session_messages 摘要路径的查询：
      - count → session_count（trim 超限判定）
      - delete → 记录 deleted_stmts（含摘要行替换删除 + trim 滚动删除）
      - source 含 'session_summary:' → 旧摘要读取（select content, order by id
        desc limit 1）→ first() 返回 latest_summary 元组
      - content_hash → 去重幂等查询（返回空）
      - order by（source=session）→ scalars().all() 返回 segment_docs（待摘要段）
        + all() 返回 [(id,)]（trim 待删除 id 列表）
    """

    def __init__(self, session_count=0, segment_docs=(), latest_summary=None,
                 oldest_ids=()):
        self.added: list = []
        self.deleted_stmts: list = []
        self.deleted_sql: list = []  # 编译后（literal binds）的删除语句文本
        self.session_count = session_count
        self.segment_docs = list(segment_docs)
        self.latest_summary = latest_summary  # tuple (content,) 或 None
        self.oldest_ids = list(oldest_ids)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def execute(self, stmt):
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
        result = mock.MagicMock()
        if "count(" in sql:
            result.scalar.return_value = self.session_count
        elif "delete" in sql:
            self.deleted_stmts.append(stmt)
            self.deleted_sql.append(sql)
        elif "session_summary" in sql:
            # 旧摘要读取（select content, order by id desc limit 1）→ first()
            result.first.return_value = self.latest_summary
        elif "order by" in sql:
            # 待摘要段查询（select(Document) 全列含 content_hash 列，须先于
            # content_hash 分支判断；scalars().all() 返回 segment_docs）
            result.scalars.return_value = mock.MagicMock(
                all=mock.MagicMock(return_value=self.segment_docs),
            )
            result.all.return_value = [(i,) for i in self.oldest_ids]
        elif "content_hash" in sql:
            result.all.return_value = []
        else:
            result.all.return_value = []
        return result


class TestSessionSummary:
    """module-046 WP2：滚动删除前 LLM 摘要 + 增量更新 + fail-open"""

    @staticmethod
    def _fake_llm(text="摘要文本", prompts=None):
        fake = mock.MagicMock()
        if prompts is not None:
            async def _gen(prompt):
                prompts.append(prompt)
                return text
            fake.generate = _gen
        else:
            fake.generate = mock.AsyncMock(return_value=text)
        return fake

    @staticmethod
    def _segment_docs():
        return [
            Document(title="session:user", content="早期问题"),
            Document(title="session:assistant", content="早期回答"),
        ]

    def test_summary_written_before_trim_when_over_cap(self):
        """超限滚动删除前：最旧消息段压缩成摘要行（documents 表，无向量）"""
        cap = settings.memory_session_max_messages

        async def run():
            fs = _SummaryFakeSession(
                session_count=cap + 2,
                segment_docs=self._segment_docs(),
                latest_summary=None,
                oldest_ids=[1, 2],
            )
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                with mock.patch("rag.session_memory.LLMFactory.get_client",
                                return_value=self._fake_llm("压缩后的早期摘要")):
                    n = await session_memory_service.save_session_messages("42", [
                        {"role": "user", "content": "新问题"},
                    ])
            assert n == 1
            # 摘要行写入 documents：source='memory:42:session_summary:'，无向量
            summaries = [d for d in fs.added
                         if getattr(d, "source", None) == "memory:42:session_summary:"]
            assert len(summaries) == 1
            assert summaries[0].title == "session_summary"
            assert summaries[0].content == "压缩后的早期摘要"
            assert summaries[0].embedding is None
            # 滚动删除照常执行（摘要行替换删除 + trim 删除）
            assert any("session_summary" in s for s in fs.deleted_sql)
            assert any("id in" in s for s in fs.deleted_sql)
        asyncio.run(run())

    def test_summary_incremental_includes_old_summary(self):
        """增量更新：新摘要 = 摘要(旧摘要 + 新对话段)（MemGPT 递归公式）"""
        cap = settings.memory_session_max_messages
        prompts = []

        async def run():
            fs = _SummaryFakeSession(
                session_count=cap + 2,
                segment_docs=self._segment_docs(),
                latest_summary=("已有早期摘要",),  # 旧摘要已存在
                oldest_ids=[1, 2],
            )
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                with mock.patch("rag.session_memory.LLMFactory.get_client",
                                return_value=self._fake_llm(prompts=prompts)):
                    await session_memory_service.save_session_messages("42", [
                        {"role": "user", "content": "新问题"},
                    ])
            assert len(prompts) == 1
            # prompt 同时含旧摘要与新对话段（增量合并而非覆盖）
            assert "已有早期摘要" in prompts[0]
            assert "早期问题" in prompts[0]
            assert "早期回答" in prompts[0]
        asyncio.run(run())

    def test_summary_llm_failure_fail_open_trim_still_runs(self):
        """摘要 LLM 失败 → 跳过摘要（fail-open），滚动删除照常，不抛异常"""
        cap = settings.memory_session_max_messages

        async def run():
            fs = _SummaryFakeSession(
                session_count=cap + 2,
                segment_docs=self._segment_docs(),
                latest_summary=None,
                oldest_ids=[1, 2],
            )
            fake = mock.MagicMock()
            fake.generate = mock.AsyncMock(side_effect=RuntimeError("llm down"))
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                with mock.patch("rag.session_memory.LLMFactory.get_client",
                                return_value=fake):
                    n = await session_memory_service.save_session_messages("42", [
                        {"role": "user", "content": "新问题"},
                    ])
            assert n == 1  # 保存不受影响
            # 未写摘要行（fail-open），滚动删除仍执行
            assert not [d for d in fs.added
                        if getattr(d, "source", None) == "memory:42:session_summary:"]
            assert any("id in" in s for s in fs.deleted_sql)
        asyncio.run(run())

    def test_summary_empty_llm_output_skips(self):
        """LLM 返回空文本 → 视为失败，跳过摘要写入（fail-open）"""
        cap = settings.memory_session_max_messages

        async def run():
            fs = _SummaryFakeSession(
                session_count=cap + 2,
                segment_docs=self._segment_docs(),
                oldest_ids=[1],
            )
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                with mock.patch("rag.session_memory.LLMFactory.get_client",
                                return_value=self._fake_llm("  ")):
                    await session_memory_service.save_session_messages("42", [
                        {"role": "user", "content": "新问题"},
                    ])
            assert not [d for d in fs.added
                        if getattr(d, "source", None) == "memory:42:session_summary:"]
        asyncio.run(run())

    def test_no_summary_when_within_cap(self):
        """未超限 → 不触发摘要 LLM（零回归）"""
        cap = settings.memory_session_max_messages

        async def run():
            fs = _SummaryFakeSession(session_count=cap)
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                with mock.patch("rag.session_memory.LLMFactory.get_client") as gc:
                    await session_memory_service.save_session_messages("42", [
                        {"role": "user", "content": "新问题"},
                    ])
            gc.assert_not_called()
            assert not [d for d in fs.added
                        if getattr(d, "source", None) == "memory:42:session_summary:"]
        asyncio.run(run())

    def test_get_session_summary_returns_latest(self):
        """读取摘要：仅顺序读最新一条（source='memory:42:session_summary:'）"""
        async def run():
            fs = _SummaryFakeSession(latest_summary=("最新摘要",))
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                text = await session_memory_service.get_session_summary("42")
            assert text == "最新摘要"
        asyncio.run(run())

    def test_get_session_summary_empty_and_failure(self):
        """无摘要 → 空串；读取失败 → 空串（调用方跳过摘要段，零回归）"""
        async def run():
            fs = _SummaryFakeSession(latest_summary=None)
            with mock.patch("rag.session_memory.async_session_factory",
                            _fake_factory(fs)):
                assert await session_memory_service.get_session_summary("42") == ""
            sess = mock.MagicMock()
            sess.__aenter__ = mock.AsyncMock(side_effect=RuntimeError("db down"))
            sess.__aexit__ = mock.AsyncMock(return_value=False)
            with mock.patch("rag.session_memory.async_session_factory",
                            mock.MagicMock(return_value=sess)):
                assert await session_memory_service.get_session_summary("42") == ""
        asyncio.run(run())

    def test_summary_source_format_isolated(self):
        """摘要 source 与会话 source 前缀隔离（尾冒号分隔，不互扰）"""
        assert _session_summary_source("42") == "memory:42:session_summary:"
        assert _session_summary_source("42") != "memory:42:session:"
        assert _session_summary_source("42") != "memory:42:"


class TestLayeredInjection:
    """module-046 WP2：engine 组装 history = 早期摘要段 + 最近 N 条原样（零回归）"""

    @staticmethod
    def _recent(n=20):
        return [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"消息{i}"} for i in range(n)]

    @staticmethod
    def _patch_service(messages, summary, summary_raises=False):
        svc = mock.MagicMock()
        svc.get_session_messages = mock.AsyncMock(return_value=messages)
        if summary_raises:
            svc.get_session_summary = mock.AsyncMock(
                side_effect=RuntimeError("summary down"))
        else:
            svc.get_session_summary = mock.AsyncMock(return_value=summary)
        return mock.patch("rag.engine.session_memory_service", svc)

    def test_summary_segment_prepended(self):
        """有摘要 → history = [早期摘要段] + 最近 20 条原样"""
        recent = self._recent(20)
        with self._patch_service(recent, "早期会话摘要"):
            history = asyncio.run(
                rag_engine._resolve_session_history("42", [])
            )
        assert history[0] == {"role": "assistant",
                              "content": "[早期会话摘要]\n早期会话摘要"}
        assert history[1:] == recent  # 最近 20 条原样

    def test_no_summary_byte_identical(self):
        """无摘要 → 返回持久化会话原样（≤20 条与旧行为逐字节一致）"""
        recent = self._recent(10)
        with self._patch_service(recent, ""):
            history = asyncio.run(
                rag_engine._resolve_session_history("42", [])
            )
        assert history == recent  # 逐字节一致（零回归）

    def test_summary_failure_skips_segment(self):
        """摘要读取失败 → 跳过摘要段，持久化会话原样返回（fail-open）"""
        recent = self._recent(20)
        with self._patch_service(recent, None, summary_raises=True):
            history = asyncio.run(
                rag_engine._resolve_session_history("42", [])
            )
        assert history == recent

    def test_empty_persisted_falls_back_request_history(self):
        """无持久化会话 → 回退当前请求 history（摘要不注入，零回归）"""
        request_history = [{"role": "user", "content": "本次请求"}]
        with self._patch_service([], None):
            history = asyncio.run(
                rag_engine._resolve_session_history("42", request_history)
            )
        assert history == request_history

    def test_empty_identity_returns_request_history(self):
        """身份为空 → 直接回退当前请求 history（不查 DB）"""
        request_history = [{"role": "user", "content": "x"}]
        history = asyncio.run(
            rag_engine._resolve_session_history("", request_history)
        )
        assert history == request_history
