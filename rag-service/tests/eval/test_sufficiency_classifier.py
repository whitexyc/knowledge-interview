"""module-045 WP4: 充分性分类器训练脚本测试（mock 特征）

覆盖 acceptance-criteria §4/§7：
- fit/predict_proba 校准概率（mock 特征，充分/不充分线性可分）/ 落盘 / 重载
- 模型缺失返回 False / 未加载 predict 抛 RuntimeError
- 样本不足 10 条 → 明确报错退出（SystemExit(1)）
- 数据源契约：SUFFICIENCY_DATASET 100 条（充分 50 + 不充分 50）
- 特征文本拼接（问题 + 检索文档）

同步 def + 函数内 asyncio.run 执行，不依赖 pytest-asyncio
（与套件其余用例同款模式）。不依赖真实 bge-m3 / sklearn 外的任何服务。
"""
import asyncio
import os

import pytest

from eval.train import train_sufficiency_classifier as tsc
from eval.train.train_sufficiency_classifier import (
    SufficiencyClassifier, build_feature_text, load_training_samples,
)

# (question, documents, label)；特征文本由 build_feature_text 生成
_RAW_SAMPLES: list[tuple[str, list[str], str]] = [
    # sufficient：文档含 G1/Kafka/AQS 核心术语
    ("什么是G1垃圾收集器？", ["G1（Garbage First）垃圾收集器是 JDK 9 之后的默认垃圾收集器，核心设计是 Region 分区机制。"], "sufficient"),
    ("Kafka 的 ISR 机制是什么？", ["Kafka 可靠性核心是 ISR 机制，Leader 负责读写，Follower 拉取同步。"], "sufficient"),
    ("AQS 的工作原理？", ["AQS 是 Java 并发包基石，volatile int state 字段 + CLH 变体 FIFO 等待队列。"], "sufficient"),
    ("G1 和 ZGC 的区别？", ["ZGC 着色指针与读屏障，G1 Region 分区 + RSet 回收。"], "sufficient"),
    ("Kafka 分区怎么路由？", ["Kafka 分区路由：key.hashCode() % 分区数保证相同 key 进同一分区。"], "sufficient"),
    ("AQS 的 Condition 怎么用？", ["ReentrantLock 通过 newCondition() 创建条件队列，await/signal 实现等待唤醒。"], "sufficient"),
    ("Kafka 零拷贝怎么实现？", ["Kafka 高吞吐依赖顺序写、页缓存与零拷贝。"], "sufficient"),
    # insufficient：文档与问题无关（无核心术语）
    ("什么是G1垃圾收集器？", ["Redis 持久化：RDB 定时快照，AOF 追加写命令日志。"], "insufficient"),
    ("Kafka 的 ISR 机制是什么？", ["MySQL B+ 树索引：非叶子只存键，叶子有序串联。"], "insufficient"),
    ("AQS 的工作原理？", ["Docker 镜像分层：OverlayFS 堆叠，可写层 Copy-on-Write。"], "insufficient"),
    ("什么是G1垃圾收集器？", ["JWT 三段式：Header.Payload.Signature，验签通过即信任。"], "insufficient"),
    ("Kafka 分区怎么路由？", ["Spring Bean 生命周期：实例化、属性填充、BeanPostProcessor。"], "insufficient"),
]


def _samples() -> list[tuple[str, str]]:
    return [
        (build_feature_text(q, [{"content": d} for d in docs]), label)
        for q, docs, label in _RAW_SAMPLES
    ]


class _FakeEmbedding:
    """mock 特征：充分/不充分按样本地图给确定性向量（线性可分）

    embed_text 对未知文本返回 [0.5, 0.5]（等概率特征，不干扰方向断言）。
    """

    def __init__(self, vec_map: dict[str, list[float]]):
        self._map = vec_map

    async def embed_text(self, text: str) -> list:
        return self._map.get(text, [0.5, 0.5])

    async def embed_documents(self, texts: list) -> list:
        return [await self.embed_text(t) for t in texts]


def _make_clf(tmp_path, samples=None):
    samples = samples if samples is not None else _samples()
    vec_map = {
        text: ([1.0, 0.0] if label == "sufficient" else [0.0, 1.0])
        for text, label in samples
    }
    model_path = os.path.join(str(tmp_path), "sufficiency_clf.joblib")
    clf = SufficiencyClassifier(model_path=model_path,
                                embedding_service=_FakeEmbedding(vec_map))
    return clf, samples, model_path


class TestBuildFeatureText:
    """特征文本拼接（问题 + 检索文档）"""

    def test_concatenates_question_and_docs(self):
        text = build_feature_text("什么是GC", [
            {"content": "G1 详解"}, {"content": "调优参数"},
        ])
        assert "问题：什么是GC" in text
        assert "G1 详解" in text and "调优参数" in text

    def test_empty_documents_ok(self):
        text = build_feature_text("什么是GC", [])
        assert text == "问题：什么是GC\n文档："


class TestLoadTrainingSamples:
    """数据源契约：SUFFICIENCY_DATASET 100 条（充分 50 / 不充分 50）"""

    def test_dataset_contract(self):
        samples = load_training_samples()
        assert len(samples) == 100
        labels = {lbl for _, lbl in samples}
        assert labels == {"sufficient", "insufficient"}
        counts = {lbl: sum(1 for _, l in samples if l == lbl) for lbl in labels}
        assert counts == {"sufficient": 50, "insufficient": 50}


class TestSufficiencyClassifierFit:
    """AC §4/§7: fit/predict_proba（mock 特征）/ 落盘 / 重载"""

    def test_fit_and_predict_proba(self, tmp_path):
        clf, samples, model_path = _make_clf(tmp_path)
        metrics = asyncio.run(clf.fit(samples))
        assert metrics["n_samples"] == 12
        assert set(metrics["classes"]) == {"sufficient", "insufficient"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "insufficient" in metrics["per_class"]  # P/R/F1 含 insufficient
        assert os.path.isfile(model_path)  # 模型落盘

        # 重载后 predict_proba：两类键齐全、和≈1、充分样本占优
        clf2, _, _ = _make_clf(tmp_path)
        assert asyncio.run(clf2.load()) is True
        q, docs, _ = _RAW_SAMPLES[0]
        probs = asyncio.run(clf2.predict_proba(q, [{"content": docs[0]}]))
        assert set(probs) == {"sufficient", "insufficient"}
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert probs["sufficient"] > probs["insufficient"]

    def test_predict_proba_insufficient_direction(self, tmp_path):
        clf, samples, _ = _make_clf(tmp_path)
        asyncio.run(clf.fit(samples, save=False))
        q, docs, _ = _RAW_SAMPLES[-1]  # 不充分样本
        probs = asyncio.run(clf.predict_proba(q, [{"content": docs[0]}]))
        assert probs["insufficient"] > probs["sufficient"]

    def test_no_save_skips_file(self, tmp_path):
        clf, samples, model_path = _make_clf(tmp_path)
        asyncio.run(clf.fit(samples, save=False))
        assert not os.path.isfile(model_path)

    def test_load_missing_model_returns_false(self, tmp_path):
        clf, _, _ = _make_clf(tmp_path)
        assert asyncio.run(clf.load()) is False

    def test_predict_without_model_raises(self, tmp_path):
        clf, _, _ = _make_clf(tmp_path)
        with pytest.raises(RuntimeError):
            asyncio.run(clf.predict_proba("什么是GC"))


class TestTrainScript:
    """训练入口：样本不足明确报错退出（AC §4）"""

    def test_insufficient_samples_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tsc, "load_training_samples",
                            lambda: [("q", "sufficient")] * 3)
        with pytest.raises(SystemExit) as exc:
            asyncio.run(tsc.train("x.joblib", save=False))
        assert exc.value.code == 1

    def test_main_has_cli_args(self):
        """--model-path / --no-save 参数可配（与 train_intent_classifier 对齐）"""
        import inspect
        sig = inspect.signature(tsc.train)
        assert "model_path" in sig.parameters and "save" in sig.parameters
