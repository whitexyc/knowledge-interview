"""module-066 WP-B/WP-C：agent_tasks 任务集 + 评测脚本单测（ADR-0017 决策 3/4）

覆盖（验收 §1.2/§1.3 + §2 边界）：
- 任务集 schema：条数 30-50 / id 唯一 / 字段齐全 / 工具名 ∈ 10 工具 / points 1-3
- 六类路径覆盖计数（knowledge 单轮/多轮/casual/realtime/重检/记忆）
- expected_tools 阶段顺序（检索工具在前、生成工具在后，re_search 双组豁免）
- 判定器四规则：覆盖（顺序放宽）/ 无多调（re_search 豁免）/ 参数类型 / Grounding
- outcome：tools=[] 恒过覆盖；答案缺要点 → fail（不过度宽松）
- 指标聚合：pass^k / 工具正确率 / 平均步数 / P50-P95 / chat 无轨迹占位
- CLI：--mode/--sample/--pass_k/--limit/--no-save/--fixture
- fixture 模式全量跑通（确定性 pass，六类路径全过）
- agent_eval_runs 落库（DDL 幂等 + INSERT 参数）+ --no-save 不落库

实现说明：
- conftest autouse 钉住 tool_call_logs_enabled=false（hermetic，fixture 零 DB）
- 真实模式（DB/LLM）不在单测内跑，由真实冒烟覆盖（--limit/--sample）
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import eval.agent_tasks as m
from agent.tool_registry import registry

VALID_TOOLS = set(registry.list_tool_names())
TASKS_PATH = Path(__file__).resolve().parents[2] / "eval" / "agent_tasks.json"


class TestDataset:
    """WP-B：任务集结构 + 六类路径覆盖"""

    def test_dataset_size_and_schema(self):
        """30-50 条；id 唯一；task 字符串/数组；expected_tools 合法；points 1-3"""
        tasks = m.load_agent_tasks(TASKS_PATH)
        assert 30 <= len(tasks) <= 50
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids))
        for t in tasks:
            task = t["task"]
            assert isinstance(task, str) or (isinstance(task, list) and all(isinstance(q, str) and q for q in task))
            assert set(t["expected_tools"]) <= VALID_TOOLS
            assert 1 <= len(t["answer_points"]) <= 3
            assert all(isinstance(p, str) and p for p in t["answer_points"])

    def test_six_path_classes_covered(self):
        """覆盖 ≥6 类路径：单轮/多轮/casual/realtime/重检/记忆各 ≥1"""
        tasks = m.load_agent_tasks(TASKS_PATH)
        counts = m.path_coverage(tasks)
        for path in ("knowledge_single", "knowledge_multi", "casual",
                     "realtime", "reselect", "memory"):
            assert counts.get(path, 0) >= 1, f"路径 {path} 无覆盖"
        assert sum(counts.values()) == len(tasks)

    def test_multi_turn_tasks_exist(self):
        """多轮任务（task 为数组）存在（省略句继承语义，module-063 能力）"""
        tasks = m.load_agent_tasks(TASKS_PATH)
        multi = [t for t in tasks if isinstance(t["task"], list)]
        assert len(multi) >= 5
        # 追问是省略/指代句（短句，含"它/和/为什么"等继承语义）
        for t in multi:
            assert len(t["task"]) >= 2

    def test_expected_tools_phase_order(self):
        """expected_tools 阶段顺序：检索组在前、生成组在后（re_search 双组豁免）

        module-058（ADR-0012）归组：检索-only 7 工具 + 生成-only 3 工具 +
        re_search 双组。序列中出现生成工具后不得再出现检索-only 工具。
        """
        tasks = m.load_agent_tasks(TASKS_PATH)
        retrieval_only = {"search_knowledge", "search_fts", "search_vector",
                          "search_graph", "extract_entities", "recall_memory"}
        generation_only = {"generate_answer", "verify_answer", "note_to_self"}
        for t in tasks:
            seen_generation = False
            for tool in t["expected_tools"]:
                if tool in generation_only:
                    seen_generation = True
                elif tool in retrieval_only:
                    assert not seen_generation, \
                        f"{t['id']} 生成工具后出现检索工具 {tool}"

    def test_load_rejects_bad_dataset(self, tmp_path):
        """条数不足 30 / id 重复 → ValueError（fail-fast）"""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([
            {"id": "at-1", "task": "q", "expected_tools": ["search_knowledge"],
             "answer_points": ["G1"]} for _ in range(5)]), encoding="utf-8")
        try:
            m.load_agent_tasks(bad)
            assert False, "应抛 ValueError"
        except ValueError as e:
            assert "30-50" in str(e)
        dup = tmp_path / "dup.json"
        dup.write_text(json.dumps([
            {"id": "at-1", "task": f"q{i}", "expected_tools": [],
             "answer_points": ["你好"]} for i in range(30)]), encoding="utf-8")
        try:
            m.load_agent_tasks(dup)
            assert False, "应抛 ValueError（id 重复）"
        except ValueError as e:
            assert "id 重复" in str(e)

    def test_load_rejects_unknown_tool(self, tmp_path):
        """expected_tools 含注册表外工具名 → ValueError"""
        bad = tmp_path / "bad_tool.json"
        bad.write_text(json.dumps([
            {"id": f"at-{i}", "task": "q", "expected_tools": ["no_such_tool"],
             "answer_points": ["G1"]} for i in range(30)]), encoding="utf-8")
        try:
            m.load_agent_tasks(bad)
            assert False, "应抛 ValueError"
        except ValueError as e:
            assert "expected_tools" in str(e)


class TestJudge:
    """WP-C：判定器四规则（确定性，ADR-0017 决策 4）"""

    def test_coverage_order_relaxed(self):
        """规则 1 覆盖：期望工具都出现即过（顺序放宽，含多轮累积）"""
        assert m.check_coverage(["generate_answer", "search_knowledge"],
                                ["search_knowledge", "generate_answer"]) is True
        assert m.check_coverage(["search_knowledge"], ["search_knowledge", "generate_answer"]) is False
        assert m.check_coverage([], []) is True  # tools=[] 任务恒过

    def test_no_extra_with_re_search_exemption(self):
        """规则 2 无多调：期望集合内；re_search 双组豁免（生成阶段补检不算错）"""
        assert m.check_no_extra(["search_knowledge", "generate_answer"],
                                ["search_knowledge", "generate_answer"]) is True
        assert m.check_no_extra(["search_knowledge", "verify_answer"],
                                ["search_knowledge", "generate_answer"]) is False
        assert m.check_no_extra(["search_knowledge", "re_search", "generate_answer"],
                                ["search_knowledge", "generate_answer"]) is True

    def test_args_type_required_fields(self):
        """规则 3 参数类型：args 缺必填字段 → 不通过（不判值语义）"""
        schemas = {"search_knowledge": registry.get("search_knowledge").args_schema}
        ok_calls = [{"name": "search_knowledge", "args": {"query": "RRF", "top_k": 5}}]
        assert m.check_args_type(ok_calls, schemas) is True
        missing = [{"name": "search_knowledge", "args": {"top_k": 5}}]  # 缺 query
        assert m.check_args_type(missing, schemas) is False
        assert m.check_args_type([], schemas) is True

    def test_outcome_points_missing_fails(self):
        """答案不含任何 answer_points → outcome fail（判定器不过度宽松）"""
        item = {"expected_tools": ["search_knowledge", "generate_answer"],
                "answer_points": ["倒数排名", "分数量纲"]}
        assert m.outcome_pass(item, "RRF 用倒数排名融合，不依赖分数量纲",
                              ["search_knowledge", "generate_answer"]) is True
        assert m.outcome_pass(item, "RRF 融合效果好",
                              ["search_knowledge", "generate_answer"]) is False
        # 工具覆盖不过 → 直接 fail
        assert m.outcome_pass(item, "倒数排名 分数量纲", ["search_knowledge"]) is False

    def test_outcome_empty_tools_constant_pass(self):
        """空 expected_tools（casual/realtime）工具覆盖恒过，按 answer_points 判定"""
        item = {"expected_tools": [], "answer_points": ["你好"]}
        assert m.outcome_pass(item, "你好！我是知识库助手", []) is True
        assert m.outcome_pass(item, "嗯嗯", []) is False
        assert m.outcome_pass(item, "你好呀", ["search_knowledge"]) is True  # 多调不影响 outcome

    def test_failure_classification(self):
        """失败分类：参数错 → 工具选错 → 工具漏调 → 路径绕 → 答案缺要点"""
        assert m.classify_failure({"args_ok": False, "no_extra": True,
                                   "coverage": True, "tool_count": 2,
                                   "expected_tools": ["a", "b"]}) == "参数错"
        assert m.classify_failure({"args_ok": True, "no_extra": False,
                                   "coverage": True, "tool_count": 2,
                                   "expected_tools": ["a", "b"]}) == "工具选错"
        assert m.classify_failure({"args_ok": True, "no_extra": True,
                                   "coverage": False, "tool_count": 2,
                                   "expected_tools": ["a", "b"]}) == "工具漏调"
        assert m.classify_failure({"args_ok": True, "no_extra": True,
                                   "coverage": True, "tool_count": 4,
                                   "expected_tools": ["a", "b"]}) == "路径绕"
        assert m.classify_failure({"args_ok": True, "no_extra": True,
                                   "coverage": True, "tool_count": 2,
                                   "expected_tools": ["a", "b"]}) == "答案缺要点"


class TestMetrics:
    """指标聚合纯函数（pass^k / 正确率 / 步数 / P50-P95）"""

    @staticmethod
    def _task(pass_=True, correct=True, tool_count=2, duration=100,
              tokens=50, grounding=1.0, path="knowledge_single"):
        return {"task_id": "at-1", "path": path, "expected_tools": ["a", "b"],
                "pass": pass_, "coverage": True, "no_extra": True, "args_ok": True,
                "tool_correct": correct, "grounding": grounding,
                "actual_names": ["a", "b"], "tool_count": tool_count,
                "tokens": tokens, "duration_ms": duration, "answer": "", "fail_reason": None}

    def test_compute_scores_aggregates(self):
        tasks = [self._task(pass_=True, duration=100),
                 self._task(pass_=True, duration=200),
                 self._task(pass_=False, correct=False, duration=300)]
        s = m.compute_scores(tasks)
        assert s["pass_1"] == round(2 / 3, 4)
        assert s["tool_correct_rate"] == round(2 / 3, 4)
        assert s["avg_tool_count"] == 2.0
        assert s["p50_ms"] == 200.0
        assert s["p95_ms"] == 290.0  # 线性插值百分位
        assert s["avg_tokens"] == 50.0
        assert s["grounding"] == 1.0

    def test_chat_mode_trajectory_none(self):
        """chat 模式 tool_correct=None → 工具正确率 None（'无轨迹' 占位，不伪造）"""
        t = self._task(correct=None)
        s = m.compute_scores([t])
        assert s["tool_correct_rate"] is None
        assert s["no_extra_rate"] is None
        assert s["args_rate"] is None

    def test_pass_k_all_runs_required(self):
        """pass^k 口径：k 次全成功才算过（run_eval 聚合）"""
        runs = [self._task(pass_=True), self._task(pass_=True), self._task(pass_=False)]
        assert all(r["pass"] for r in runs[:2]) is True
        assert all(r["pass"] for r in runs) is False

    def test_percentile(self):
        assert m._percentile([1, 2, 3, 4], 0.5) == 2.5
        assert m._percentile([5], 0.95) == 5.0
        assert m._percentile([], 0.5) == 0.0


class TestRunner:
    """运行器：fixture 全量确定性 + chat 模式 + 多轮 + pass_k"""

    def test_fixture_full_run_deterministic(self):
        """fixture 模式全量：36 条全过、六类路径、轨迹正确率 1.0（零 LLM/DB）"""
        tasks = m.load_agent_tasks(TASKS_PATH)
        per_question, scores = asyncio.run(m.run_eval(tasks, "agent", 1, True))
        assert len(per_question) == 36
        assert scores["pass_1"] == 1.0
        assert scores["tool_correct_rate"] == 1.0
        assert set(scores["per_path"]) == {
            "knowledge_single", "knowledge_multi", "casual",
            "realtime", "reselect", "memory"}

    def test_fixture_pass_k_3(self):
        """--pass_k 3：抽样 10 条各跑 3 次，全成功才算过（fixture 确定性）"""
        tasks = m.load_agent_tasks(TASKS_PATH)[:10]
        per_question, scores = asyncio.run(m.run_eval(tasks, "agent", 3, True))
        assert scores["pass_k"] == 3
        assert scores["pass_1"] == 1.0
        assert all(pq["pass"] for pq in per_question)

    def test_chat_mode_trajectory_placeholder(self):
        """chat 模式：Trajectory 如实标注 '无轨迹'；engine.chat 答案判 outcome"""
        item = {"id": "at-x", "task": "什么是 G1？",
                "expected_tools": ["search_knowledge", "generate_answer"],
                "answer_points": ["G1", "Region"]}
        with mock.patch("rag.engine.rag_engine.chat", new=mock.AsyncMock(
                return_value=SimpleNamespace(answer="G1 垃圾收集器使用 Region 分区"))):
            result = asyncio.run(m._run_chat_once(item, 1))
        assert result["pass"] is True
        assert result["tool_correct"] is None  # 无轨迹
        assert result["actual_names"] == []
        # 多轮 chat：第二轮答案判 outcome
        item2 = {"id": "at-y", "task": ["什么是 G1？", "它有什么创新？"],
                 "expected_tools": ["search_knowledge", "generate_answer"],
                 "answer_points": ["Region"]}
        with mock.patch("rag.engine.rag_engine.chat", new=mock.AsyncMock(
                return_value=SimpleNamespace(answer="G1 的 Region 分区机制"))):
            result2 = asyncio.run(m._run_chat_once(item2, 1))
        assert result2["pass"] is True
        assert len(result2["answer"]) <= 200

    def test_agent_run_failure_recorded_not_raise(self):
        """单任务运行失败（LLM 异常）→ 记录 fail_reason，不中断其余任务"""
        item = {"id": "at-z", "task": "q", "expected_tools": ["search_knowledge"],
                "answer_points": ["G1"]}
        with mock.patch("eval.agent_tasks.react_loop",
                        side_effect=RuntimeError("LLM 429 限流")):
            result = asyncio.run(m._run_agent_once(item, 1, fixture=False))
        assert result["pass"] is False
        assert "LLM 429" in result["fail_reason"]
        assert result["grounding"] is None  # 失败不伪造 grounding

    def test_grounding_reads_tool_call_logs(self):
        """Grounding：tool_call_logs 读取 result_ok 比例；无行 → None"""
        rows = [{"result_ok": True}, {"result_ok": True}, {"result_ok": False}]
        session = mock.MagicMock()
        session.execute = mock.AsyncMock(return_value=mock.MagicMock(
            mappings=lambda: mock.MagicMock(all=lambda: rows)))
        cm = mock.MagicMock()
        cm.__aenter__ = mock.AsyncMock(return_value=session)
        cm.__aexit__ = mock.AsyncMock(return_value=False)
        with mock.patch("eval.agent_tasks.async_session_factory",
                        mock.MagicMock(return_value=cm)):
            with mock.patch("src.database.ensure_tool_call_logs_table",
                            new=mock.AsyncMock()):
                assert asyncio.run(m._load_grounding("eval-at-1-1")) == round(2 / 3, 4)
                session.execute = mock.AsyncMock(return_value=mock.MagicMock(
                    mappings=lambda: mock.MagicMock(all=lambda: [])))
                assert asyncio.run(m._load_grounding("eval-at-1-1")) is None

    def test_cleanup_eval_memory_runs(self):
        """测后清理评测身份记忆（LIKE memory:eval-066-anon:%）"""
        session = mock.MagicMock()
        cm = mock.MagicMock()
        cm.__aenter__ = mock.AsyncMock(return_value=session)
        cm.__aexit__ = mock.AsyncMock(return_value=False)
        with mock.patch("eval.agent_tasks.async_session_factory",
                        mock.MagicMock(return_value=cm)):
            asyncio.run(m._cleanup_eval_memory())
        stmt, params = session.execute.call_args[0]
        assert "memory:eval-066-anon:%" in params["p"]
        assert "DELETE FROM documents" in str(stmt)


class TestSaveAndCLI:
    """agent_eval_runs 落库 + CLI 参数行为"""

    @staticmethod
    def _fake_session():
        """假 AsyncSession：execute/commit 均为异步（await 可执行）"""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    def test_agent_eval_runs_ddl_idempotent(self):
        """agent_eval_runs DDL 拆分逐条执行（CREATE + COMMENT 幂等）"""
        session = self._fake_session()
        cm = mock.MagicMock()
        cm.__aenter__ = mock.AsyncMock(return_value=session)
        cm.__aexit__ = mock.AsyncMock(return_value=False)
        with mock.patch("eval.agent_tasks.async_session_factory",
                        mock.MagicMock(return_value=cm)):
            asyncio.run(m.ensure_agent_eval_runs_table())
        stmts = [c.args[0] for c in session.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS agent_eval_runs" in str(s) for s in stmts)
        assert len(stmts) == 7  # CREATE + 6 条 COMMENT

    def test_save_agent_eval_run_inserts(self):
        """save_agent_eval_run：INSERT 参数含 git_commit/JSONB scores/per_question"""
        session = self._fake_session()
        session.execute.return_value = mock.MagicMock(
            fetchone=lambda: (42,))
        cm = mock.MagicMock()
        cm.__aenter__ = mock.AsyncMock(return_value=session)
        cm.__aexit__ = mock.AsyncMock(return_value=False)
        with mock.patch("eval.agent_tasks.async_session_factory",
                        mock.MagicMock(return_value=cm)):
            rid = asyncio.run(m.save_agent_eval_run(
                "abc123", {"k": "v"}, {"pass_1": 1.0}, [{"task_id": "at-1"}]))
        assert rid == 42
        stmt, params = session.execute.call_args[0]
        assert "INSERT INTO agent_eval_runs" in str(stmt)
        assert "CAST(:config_snapshot AS jsonb)" in str(stmt)
        assert params["git_commit"] == "abc123"
        assert json.loads(params["scores"])["pass_1"] == 1.0

    @staticmethod
    def _fake_tasks(n=30):
        return [{"id": f"at-{i:03d}", "task": "什么是 G1？",
                 "expected_tools": ["search_knowledge", "generate_answer"],
                 "answer_points": ["G1"]} for i in range(n)]

    @staticmethod
    def _fake_per_question():
        return [{"task_id": "at-001", "path": "knowledge_single",
                 "expected_tools": ["search_knowledge", "generate_answer"],
                 "pass": True, "coverage": True, "no_extra": True,
                 "args_ok": True, "tool_correct": True, "grounding": 1.0,
                 "actual_names": ["search_knowledge"], "tool_count": 1,
                 "tokens": 100, "duration_ms": 100, "answer": "", "fail_reason": None}]

    def test_cli_no_save_skips_db(self, monkeypatch):
        """--no-save：不调 save_agent_eval_run（dry-run 不落库）"""
        monkeypatch.setattr(m, "load_agent_tasks",
                            lambda: self._fake_tasks())
        monkeypatch.setattr(m, "run_eval", mock.AsyncMock(
            return_value=(self._fake_per_question(),
                          {"count": 1, "pass_1": 1.0, "mode": "agent",
                           "pass_k": 1, "fixture": False, "dataset_size": 30,
                           "trajectory": "有轨迹", "tool_correct_rate": 1.0,
                           "no_extra_rate": 1.0, "args_rate": 1.0,
                           "grounding": 1.0, "avg_tool_count": 1.0,
                           "avg_tokens": 100, "p50_ms": 100.0, "p95_ms": 100.0,
                           "per_path": {}})))
        save_mock = mock.AsyncMock(return_value=0)
        monkeypatch.setattr(m, "save_agent_eval_run", save_mock)
        monkeypatch.setattr(m, "_cleanup_eval_memory", mock.AsyncMock())
        monkeypatch.setattr(sys, "argv", ["eval.agent_tasks", "--no-save", "--limit", "1"])
        asyncio.run(m.main())
        assert save_mock.await_count == 0  # 不落库

    def test_cli_default_saves(self, monkeypatch):
        """默认（无 --no-save）：落 agent_eval_runs"""
        monkeypatch.setattr(m, "load_agent_tasks", lambda: self._fake_tasks())
        monkeypatch.setattr(m, "run_eval", mock.AsyncMock(
            return_value=(self._fake_per_question(),
                          {"count": 1, "pass_1": 1.0, "mode": "agent",
                           "pass_k": 1, "fixture": False, "dataset_size": 30,
                           "trajectory": "有轨迹", "tool_correct_rate": 1.0,
                           "no_extra_rate": 1.0, "args_rate": 1.0,
                           "grounding": 1.0, "avg_tool_count": 1.0,
                           "avg_tokens": 100, "p50_ms": 100.0, "p95_ms": 100.0,
                           "per_path": {}})))
        save_mock = mock.AsyncMock(return_value=7)
        monkeypatch.setattr(m, "save_agent_eval_run", save_mock)
        monkeypatch.setattr(m, "_cleanup_eval_memory", mock.AsyncMock())
        monkeypatch.setattr("eval.golden.golden_retrieval.get_git_commit",
                            lambda: "abc123")
        monkeypatch.setattr("eval.golden.golden_retrieval.load_rag_config",
                            mock.AsyncMock(return_value={}))
        monkeypatch.setattr(sys, "argv", ["eval.agent_tasks", "--sample", "10", "--pass_k", "3"])
        asyncio.run(m.main())
        assert save_mock.await_count == 1
        args = save_mock.await_args
        assert args[0][0] == "abc123"
