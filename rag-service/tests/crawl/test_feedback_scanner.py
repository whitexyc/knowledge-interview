"""
反向闭环扫描器单元测试（module-080）：低分题 → 待学笔记 → 优先级队列

覆盖验收矩阵：拉取成功/超时/HTTP 错误/JSON 异常（fail-open 空跑）、
主题提取（截断/空白折叠/缺省回退）、笔记模板、入队去重/失败降级、
扫描编排（过滤高分、单条失败不中断、空列表空跑）、调度器开关。
全 mock：httpx / memory_service.save / async_session_factory，不触网不触库。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.crawl.feedback_scanner import (
    build_learning_note,
    enqueue_priority,
    extract_topic,
    fetch_low_score_questions,
    scan_and_generate,
    setup_feedback_scheduler,
)

SAMPLE = {
    "sessionId": "sess-abc",
    "questionNumber": "1",
    "questionContent": "请解释 Redis 持久化机制 RDB 与 AOF 的区别是什么",
    "score": 40,
    "totalScore": 100,
    "feedback": "RDB 快照原理不清楚",
    "endTime": "1720000000000",
}


def _ok_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _mock_http_client(resp):
    instance = AsyncMock()
    instance.get = AsyncMock(return_value=resp)
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    return instance


def _db_factory(dup_scalar=None, execute_side_effect=None):
    result = MagicMock()
    result.scalar.return_value = dup_scalar
    session = AsyncMock()
    if execute_side_effect:
        session.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        session.execute = AsyncMock(return_value=result)
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


# ─── 拉取（验收 1.3：fail-open） ───


class TestFetchLowScoreQuestions:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        """Java 返回正常 → 条目列表 + 请求参数/内部 token 头正确"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_java_base_url", "http://java:8002")
        monkeypatch.setattr(settings, "feedback_low_score_threshold", 60)
        monkeypatch.setattr(settings, "feedback_internal_token", "tok-123")
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   return_value=_mock_http_client(_ok_response({"code": "0", "data": [SAMPLE]}))) as m:
            items = await fetch_low_score_questions()
        assert items == [SAMPLE]
        called_url = m.return_value.get.await_args.args[0]
        called_params = m.return_value.get.await_args.kwargs["params"]
        called_headers = m.return_value.get.await_args.kwargs["headers"]
        assert called_url == "http://java:8002/api/xunzhi/v1/interview/weak-points"
        assert called_params == {"threshold": 60, "days": 7, "limit": 50}
        assert called_headers == {"X-Internal-Token": "tok-123"}

    @pytest.mark.asyncio
    async def test_no_token_header_when_empty(self, monkeypatch):
        """内部 token 为空 → 不带头（Java 联调阶段 fail-open 语义）"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_internal_token", "")
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   return_value=_mock_http_client(_ok_response({"data": []}))):
            items = await fetch_low_score_questions()
        assert items == []

    @pytest.mark.asyncio
    async def test_http_error_fail_open(self):
        """非 200 → 空列表 + 不抛异常"""
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   side_effect=Exception("connection refused")):
            items = await fetch_low_score_questions()
        assert items == []

    @pytest.mark.asyncio
    async def test_timeout_fail_open(self):
        """超时 → 空列表"""
        resp = MagicMock()
        resp.raise_for_status = MagicMock(side_effect=__import__("httpx").TimeoutException("t"))
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   return_value=_mock_http_client(resp)):
            items = await fetch_low_score_questions()
        assert items == []

    @pytest.mark.asyncio
    async def test_malformed_json_fail_open(self):
        """JSON 解析失败 → 空列表"""
        resp = _ok_response({})
        resp.json.side_effect = ValueError("bad json")
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   return_value=_mock_http_client(resp)):
            items = await fetch_low_score_questions()
        assert items == []

    @pytest.mark.asyncio
    async def test_malformed_payload_fail_open(self):
        """data 非列表 / 非 dict 项被剔除 → 空/仅 dict 项"""
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   return_value=_mock_http_client(_ok_response({"data": "oops"}))):
            assert await fetch_low_score_questions() == []
        with patch("rag.crawl.feedback_scanner.httpx.AsyncClient",
                   return_value=_mock_http_client(_ok_response({"data": [SAMPLE, "bad", None]}))):
            items = await fetch_low_score_questions()
        assert items == [SAMPLE]


# ─── 主题提取 / 笔记模板 ───


class TestTopicAndNote:
    def test_extract_topic_truncates_and_collapses(self):
        topic = extract_topic({"questionContent": "  请解释 Redis\n持久化  机制 RDB 与 AOF 的区别是什么"})
        assert len(topic) <= 30
        assert "\n" not in topic
        assert topic.startswith("请解释 Redis 持久化")

    def test_extract_topic_fallback(self):
        assert extract_topic({"questionContent": ""}) == "未知主题"
        assert extract_topic({}) == "未知主题"

    def test_build_learning_note_contains_fields(self):
        note = build_learning_note(SAMPLE, "Redis 持久化")
        assert "【待学笔记】Redis 持久化" in note
        assert "面试问题: 请解释 Redis 持久化机制" in note
        assert "本题得分: 40/100" in note
        assert "面试反馈: RDB 快照原理不清楚" in note
        assert "来源会话: sess-abc" in note


# ─── 优先级入队（验收 1.1/1.2：去重防堆积、失败 fail-open） ───


class TestEnqueuePriority:
    @pytest.mark.asyncio
    async def test_inserts_when_no_pending_dup(self):
        with patch("src.database.async_session_factory", _db_factory(dup_scalar=None)) as factory:
            ok = await enqueue_priority(SAMPLE, "Redis 持久化", "note")
        assert ok is True
        session = factory.return_value.__aenter__.return_value
        insert_sql = str(session.execute.call_args_list[-1][0][0])
        assert "INSERT INTO crawl_priority" in insert_sql
        params = session.execute.call_args_list[-1][0][1]
        assert params["t"] == "Redis 持久化"
        assert params["sc"] == 40

    @pytest.mark.asyncio
    async def test_skips_when_pending_dup(self):
        """同 topic pending 已存在 → 不重复入队"""
        with patch("src.database.async_session_factory", _db_factory(dup_scalar=1)) as factory:
            ok = await enqueue_priority(SAMPLE, "Redis 持久化", "note")
        assert ok is False
        session = factory.return_value.__aenter__.return_value
        sqls = [str(c[0][0]) for c in session.execute.call_args_list]
        assert not any("INSERT" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_db_error_fail_open(self):
        with patch("src.database.async_session_factory",
                   MagicMock(side_effect=Exception("db down"))):
            assert await enqueue_priority(SAMPLE, "Redis 持久化", "note") is False


# ─── 扫描编排（验收 1.1 核心链路 / 1.2 空跑 / 1.3 异常） ───


class TestScanAndGenerate:
    @pytest.mark.asyncio
    async def test_full_chain(self, monkeypatch):
        """N 条低分题 → {scanned:N, noted:N, enqueued:N, errors:0}"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_learning_identity", "learning")
        monkeypatch.setattr(settings, "feedback_low_score_threshold", 60)
        with patch("rag.crawl.feedback_scanner.fetch_low_score_questions",
                   new_callable=AsyncMock, return_value=[SAMPLE, dict(SAMPLE, sessionId="s2",
                                                                     questionContent="讲一下 MySQL 索引原理", score=55)]), \
             patch("rag.crawl.feedback_scanner.memory_service.save",
                   new_callable=AsyncMock, return_value={"id": 1, "status": "saved"}) as mock_save, \
             patch("src.database.async_session_factory", _db_factory(dup_scalar=None)):
            summary = await scan_and_generate()
        assert summary == {"scanned": 2, "noted": 2, "enqueued": 2, "errors": 0}
        assert mock_save.call_count == 2
        note = mock_save.call_args_list[0][0][0]
        assert "【待学笔记】" in note

    @pytest.mark.asyncio
    async def test_high_score_filtered(self, monkeypatch):
        """score >= threshold 的题不写笔记不入队（验收 1.1 低分题过滤）"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_low_score_threshold", 60)
        items = [SAMPLE, dict(SAMPLE, questionContent="高分题", score=80)]
        with patch("rag.crawl.feedback_scanner.fetch_low_score_questions",
                   new_callable=AsyncMock, return_value=items), \
             patch("rag.crawl.feedback_scanner.memory_service.save",
                   new_callable=AsyncMock) as mock_save, \
             patch("rag.crawl.feedback_scanner.enqueue_priority",
                   new_callable=AsyncMock, return_value=True):
            summary = await scan_and_generate()
        assert summary == {"scanned": 2, "noted": 1, "enqueued": 1, "errors": 0}
        assert mock_save.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_fail_open(self):
        """空列表 → 空跑零汇总不报错（验收 1.2）"""
        with patch("rag.crawl.feedback_scanner.fetch_low_score_questions",
                   new_callable=AsyncMock, return_value=[]):
            summary = await scan_and_generate()
        assert summary == {"scanned": 0, "noted": 0, "enqueued": 0, "errors": 0}

    @pytest.mark.asyncio
    async def test_single_save_failure_continues(self, monkeypatch):
        """单条笔记写入失败 → errors+1，其余条继续（验收 1.3）"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_low_score_threshold", 60)
        items = [SAMPLE, dict(SAMPLE, sessionId="s2", questionContent="第二题")]

        async def flaky_save(*args, **kwargs):
            if "sess-abc" in str(args):
                raise RuntimeError("记忆保存失败")
            return {"id": 2, "status": "saved"}

        with patch("rag.crawl.feedback_scanner.fetch_low_score_questions",
                   new_callable=AsyncMock, return_value=items), \
             patch("rag.crawl.feedback_scanner.memory_service.save",
                   side_effect=flaky_save), \
             patch("src.database.async_session_factory", _db_factory(dup_scalar=None)):
            summary = await scan_and_generate()
        assert summary == {"scanned": 2, "noted": 1, "enqueued": 1, "errors": 1}

    @pytest.mark.asyncio
    async def test_dedup_enqueued_skip(self):
        """入队去重命中（同 topic pending 已存在）→ noted 计但 enqueued 不计"""
        items = [SAMPLE]
        with patch("rag.crawl.feedback_scanner.fetch_low_score_questions",
                   new_callable=AsyncMock, return_value=items), \
             patch("rag.crawl.feedback_scanner.memory_service.save",
                   new_callable=AsyncMock, return_value={"status": "updated"}), \
             patch("src.database.async_session_factory", _db_factory(dup_scalar=1)):
            summary = await scan_and_generate()
        assert summary == {"scanned": 1, "noted": 1, "enqueued": 0, "errors": 0}


# ─── 调度器生命周期（验收 1.3：开关关不启动） ───


class TestFeedbackScheduler:
    def test_disable_when_disabled(self, monkeypatch):
        """feedback_reverse_enabled=false → 不创建调度器（验收 1.3）"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_reverse_enabled", False)
        with patch("apscheduler.schedulers.asyncio.AsyncIOScheduler") as m:
            setup_feedback_scheduler(True)
        m.assert_not_called()

    def test_enable_registers_job(self, monkeypatch):
        """开启 → 注册 feedback_reverse_loop 定时任务；关闭 → shutdown"""
        from src.config import settings
        monkeypatch.setattr(settings, "feedback_reverse_enabled", True)
        monkeypatch.setattr(settings, "feedback_scan_interval_minutes", 1440)
        with patch("apscheduler.schedulers.asyncio.AsyncIOScheduler") as m:
            setup_feedback_scheduler(True)
            setup_feedback_scheduler(False)
        assert m.return_value.add_job.call_count == 1
        job_kwargs = m.return_value.add_job.call_args
        assert job_kwargs.kwargs["id"] == "feedback_reverse_loop"
        assert job_kwargs.kwargs["replace_existing"] is True
        assert m.return_value.shutdown.call_count == 1

    def test_shutdown_noop_when_none(self):
        setup_feedback_scheduler(False)  # 未启动过 → 不抛异常
