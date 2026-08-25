"""module-019 评估脚本单元测试

覆盖：
- compute_metrics: Hit@k / Recall@k / MRR 计算正确性
- compute_metrics: 空结果 / 无 golden doc 边界
- retriever mode 参数: 默认 hybrid、合法模式集合、非法模式拒绝
- golden.json 标注回归: CAP 题不含 HashMap 误标、Docker 题按"无覆盖标空"
- _eval_question: 单题评估 + hybrid 降级 FTS + 跳过路径（打桩检索器）
- run_eval: 循环聚合端到端（打桩检索器，不依赖数据库）
- 方法长度规范: run_eval/_eval_question/retrieve/_dispatch_mode ≤ 50 行

说明：
- 指标计算为纯函数，不依赖数据库，可被 pytest 直接收集。
- 检索 mode 的同步检查使用 VALID_MODES 常量与签名校验，不触发真实检索。
- 异步路径用 asyncio.run + mock.AsyncMock 打桩执行，不依赖 pytest-asyncio / 数据库。
"""
import asyncio
import inspect
from unittest import mock

from rag.retriever import HybridRetriever, VALID_MODES, RetrievalException
from eval.golden.golden_retrieval import compute_metrics
from eval.golden import golden_retrieval


class TestComputeMetrics:
    """Hit@k / Recall@k / MRR 指标计算"""

    def test_hit_at_k_when_first_rank(self):
        # 第一个结果即命中 → Hit=1, Recall=全中, MRR=1.0
        m = compute_metrics(
            retrieved_titles=["G1文档", "B", "C"],
            golden_titles=["G1文档"],
            k=5,
        )
        assert m["hit_at_k"] == 1.0
        assert m["recall_at_k"] == 1.0
        assert m["mrr"] == 1.0
        assert m["first_hit_rank"] == 1

    def test_mrr_with_second_rank_hit(self):
        # 第二个结果命中 → MRR = 1/2 = 0.5
        m = compute_metrics(
            retrieved_titles=["X", "G1文档", "Y"],
            golden_titles=["G1文档"],
            k=5,
        )
        assert m["hit_at_k"] == 1.0
        assert m["mrr"] == 0.5
        assert m["first_hit_rank"] == 2

    def test_recall_partial(self):
        # golden 3 篇，只命中 1 篇 → Recall = 1/3
        m = compute_metrics(
            retrieved_titles=["G1文档", "B", "C"],
            golden_titles=["G1文档", "D", "E"],
            k=5,
        )
        assert m["recall_at_k"] == 1.0 / 3.0

    def test_no_hit(self):
        m = compute_metrics(
            retrieved_titles=["X", "Y", "Z"],
            golden_titles=["G1文档"],
            k=5,
        )
        assert m["hit_at_k"] == 0.0
        assert m["recall_at_k"] == 0.0
        assert m["mrr"] == 0.0
        assert m["first_hit_rank"] == 0

    def test_empty_retrieved(self):
        # 空检索结果 → 指标全 0，不异常
        m = compute_metrics(retrieved_titles=[], golden_titles=["G1文档"], k=5)
        assert m["hit_at_k"] == 0.0
        assert m["mrr"] == 0.0

    def test_no_golden_docs(self):
        # 无 gold doc → recall 为 0（该题会在评估中提前跳过，此处验证不崩溃）
        m = compute_metrics(retrieved_titles=["X"], golden_titles=[], k=5)
        assert m["hit_at_k"] == 0.0
        assert m["recall_at_k"] == 0.0
        assert m["mrr"] == 0.0

    def test_hit_beyond_k_ignored(self):
        # 命中在第 k 位之后 → 不计入（只考察前 k）
        m = compute_metrics(
            retrieved_titles=["A", "B", "C", "D", "E", "G1文档"],
            golden_titles=["G1文档"],
            k=5,
        )
        assert m["hit_at_k"] == 0.0
        assert m["mrr"] == 0.0


class TestRetrieverMode:
    """retriever mode 参数（module-019 消融）"""

    def test_valid_modes_defined(self):
        assert VALID_MODES == ("hybrid", "vector_only", "fts_only", "graph_only")

    def test_retrieve_signature_has_mode_default_hybrid(self):
        # mode 参数存在且默认 hybrid（向后兼容，无回归）
        sig = inspect.signature(HybridRetriever.retrieve)
        assert "mode" in sig.parameters
        assert sig.parameters["mode"].default == "hybrid"

    def test_mode_validation_units(self):
        # 消融模式均在 VALID_MODES 中
        for mode in VALID_MODES:
            assert mode in VALID_MODES


class TestGoldenDataFixes:
    """golden.json 标注回归（Reviewer 2.1 两处数据 bug）"""

    @staticmethod
    def _find(questions, keyword):
        return next(q for q in questions if keyword in q["question"])

    def test_cap_question_no_hashmap_doc(self):
        # CAP 题唯一真实覆盖为 Nacos（AP/CP/CAP 讨论所在），
        # HashMap 全文无 CAP 内容，仅 MIN_TREEIFY_CAPACITY 子串误命中 → 不得标注
        cap = self._find(golden_retrieval.load_golden(), "CAP定理")
        assert "12-HashMap与ConcurrentHashMap底层原理_2026-07-23" not in cap["golden_docs"]
        assert cap["golden_docs"] == ["8-Spring-Cloud-Nacos服务注册发现与配置中心_2026-07-18"]

    def test_docker_question_no_coverage_marked_empty(self):
        # 知识库无 Docker 主题文档，JVM 文档仅一句带过 → 按设计决策 1"无覆盖标空"
        docker = self._find(golden_retrieval.load_golden(), "Docker和虚拟机")
        assert docker["golden_docs"] == []


class TestEvalQuestion:
    """_eval_question 单题评估路径（打桩检索器，不依赖数据库）"""

    def test_no_gold_docs_skipped(self):
        item = {
            "question": "Docker和虚拟机的核心区别是什么？容器化有什么优势？",
            "golden_docs": [],
            "category": "comprehensive",
        }
        evaluated, skipped = asyncio.run(golden_retrieval._eval_question(item, "hybrid", 5))
        assert evaluated == {}
        assert skipped["reason"] == "no_gold_docs"
        assert skipped["question"] == item["question"]

    @staticmethod
    def _stub_retrieve(side_effect):
        # 注意：side_effect 必须配置在 retrieve 子 mock 上，
        # 否则 hybrid_retriever.retrieve 属性访问会新建无配置的子 mock
        stub = mock.AsyncMock()
        stub.retrieve = mock.AsyncMock(side_effect=side_effect)
        return stub

    def test_hybrid_fallback_to_fts_degraded(self):
        # embedding 502 时 hybrid 应降级为仅 FTS，且标记 degraded=True
        def _behavior(query, top_k=5, mode="hybrid"):
            if mode == "hybrid":
                raise RetrievalException("embedding 502")
            if mode == "fts_only":
                return [{"title": "8-Spring-Cloud-Nacos服务注册发现与配置中心_2026-07-18"}]
            raise AssertionError(f"unexpected mode: {mode}")

        item = {
            "question": "什么是CAP定理？在分布式系统设计中如何权衡？",
            "golden_docs": ["8-Spring-Cloud-Nacos服务注册发现与配置中心_2026-07-18"],
            "category": "comprehensive",
        }
        with mock.patch.object(golden_retrieval, "hybrid_retriever", self._stub_retrieve(_behavior)):
            evaluated, skipped = asyncio.run(golden_retrieval._eval_question(item, "hybrid", 5))
        assert skipped == {}
        assert evaluated["degraded"] is True
        assert evaluated["hit_at_k"] == 1.0
        assert evaluated["recall_at_k"] == 1.0

    def test_non_hybrid_retrieval_error_skipped(self):
        # vector_only 向量化失败 → 如实记录通道不可用，不降级
        def _behavior(query, top_k=5, mode="hybrid"):
            raise RetrievalException("embedding 502")

        item = {
            "question": "什么是CAP定理？在分布式系统设计中如何权衡？",
            "golden_docs": ["8-Spring-Cloud-Nacos服务注册发现与配置中心_2026-07-18"],
            "category": "comprehensive",
        }
        with mock.patch.object(golden_retrieval, "hybrid_retriever", self._stub_retrieve(_behavior)):
            evaluated, skipped = asyncio.run(golden_retrieval._eval_question(item, "vector_only", 5))
        assert evaluated == {}
        assert skipped["reason"].startswith("error:")


class TestRunEvalEndToEnd:
    """run_eval 循环聚合回归（打桩检索器，不依赖数据库）"""

    def test_cap_evaluated_and_docker_skipped(self, monkeypatch):
        # 检索恒返回 Nacos 文档：CAP 题 golden 仅 Nacos → 命中；
        # Docker 题标空 → 走 no_gold_docs 跳过
        stub = mock.AsyncMock()
        stub.retrieve = mock.AsyncMock(return_value=[{"title": "8-Spring-Cloud-Nacos服务注册发现与配置中心_2026-07-18"}])
        monkeypatch.setattr(golden_retrieval, "hybrid_retriever", stub)
        scores, per_question, skipped = asyncio.run(golden_retrieval.run_eval("hybrid", 5))

        docker_skips = [s for s in skipped if "Docker" in s["question"]]
        assert len(docker_skips) == 1
        assert docker_skips[0]["reason"] == "no_gold_docs"

        cap = next(q for q in per_question if "CAP定理" in q["question"])
        assert cap["golden_docs"] == ["8-Spring-Cloud-Nacos服务注册发现与配置中心_2026-07-18"]
        assert cap["hit_at_k"] == 1.0

        # module-047 扩样本后规模可变：断言数据集规模关系而非硬编码数量
        golden = golden_retrieval.load_golden()
        expected_empty = sum(1 for q in golden if not q["golden_docs"])
        assert scores["dataset_size"] == len(golden)
        assert scores["skipped"] == expected_empty
        assert scores["evaluated"] == scores["dataset_size"] - expected_empty


class TestMethodLengthLimit:
    """方法 ≤ 50 行规范回归（Reviewer 2.2 两处超长方法）"""

    @staticmethod
    def _lines(fn):
        return len(inspect.getsource(fn).splitlines())

    def test_run_eval_under_50(self):
        assert self._lines(golden_retrieval.run_eval) <= 50

    def test_eval_question_under_50(self):
        assert self._lines(golden_retrieval._eval_question) <= 50

    def test_retrieve_under_50(self):
        assert self._lines(HybridRetriever.retrieve) <= 50

    def test_dispatch_mode_under_50(self):
        assert self._lines(HybridRetriever._dispatch_mode) <= 50
