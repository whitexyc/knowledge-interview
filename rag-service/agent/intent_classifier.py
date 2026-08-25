"""
L4 意图分类器 — bge-m3 冻结特征 + 逻辑回归头（ADR-0003）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个意图链路中的位置（module-043 / ADR-0003 L4）：
  LLM 分类（现实现） ──L4 上线后──> 本分类器（决策主体可换，router 可注入）

为什么用 bge-m3 + 逻辑回归替代 LLM 分类：
  1. intent 是简单分类任务：bge-m3 已本地部署（1024 维冻结特征），
     逻辑回归头 ~1025 参数（1024 权重 + 1 偏置，多类时每类一份），
     CPU 毫秒级推理，比 LLM API 调用便宜几个数量级
  2. 输出经训练校准的**真概率**（sigmoid/softmax），从根上解决
     LLM 自报 confidence 不可信的问题（ADR-0003 修订版依据之一）
  3. 可解释：权重可分析（训练源口径见下方数据约束）

数据约束（已知边界）：
  - 训练源（module-056 训练/评测分离口径）：人造标注集
    eval/intent_train_dataset.json + golden.json knowledge 天然样本；
    golden_intent 100 条评测集只作评测不进训练（防泄漏）
  - 真实飞轮数据（前端 👍/👎）未积累，人造集为方向性验证；飞轮接口预留：
    fit() 接受 (query, label) 样本列表，数据到位后并入样本增量重训即可，
    无需改推理路径
  - 训练/加载失败一律由调用方（router）回退 LLM 分类，零影响

依赖：sklearn / joblib（惰性导入，仅 fit/load 时用到；主链路不硬依赖）。
"""
import logging
import os
from typing import Optional

from rag.retrieval.embeddings import embedding_service as default_embedding_service

logger = logging.getLogger(__name__)

# 模型落盘路径（对齐 ai_service/models/ 本地模型存放约定；训练产物不进仓库）
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "intent_clf.joblib",
)

# 三类意图契约（与 router 白名单一致；训练集缺类时补 0 保契约）
_INTENT_LABELS = ("knowledge", "casual_chat", "realtime")


class IntentClassifier:
    """bge-m3 冻结特征 + sklearn 逻辑回归头的意图分类器

    - 特征：复用 rag.retrieval.embeddings.embedding_service（bge-m3 冻结，1024 维）
    - 分类头：LogisticRegression（class_weight="balanced" 抗类别不平衡——
      golden 集天然 knowledge 多，不补权会学成"永远猜 knowledge"）
    - fit(): 标注样本训练 → 落盘（joblib）
    - predict_proba(): 返回 {intent: 校准概率}（与调用方契约）
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH,
                 embedding_service: Optional[object] = None):
        self.model_path = model_path
        self._embedding_service = embedding_service or default_embedding_service
        self._model = None

    async def load(self) -> bool:
        """加载落盘模型；缺失/损坏 → False（调用方回退 LLM 分类，零影响）

        Returns:
            True 加载成功；False 模型缺失/损坏
        """
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            return True
        except Exception as e:
            logger.warning("L4 分类器模型加载失败（回退 LLM 分类）: %s", e)
            self._model = None
            return False

    async def fit(self, samples: list[tuple[str, str]], save: bool = True) -> dict:
        """用标注样本训练逻辑回归头（可选落盘）

        Args:
            samples: [(query, intent_label), ...]，来源人造标注集
                intent_train_dataset.json + golden.json knowledge 天然样本
                （module-056 训练/评测分离口径：golden_intent 评测集不进
                训练）；飞轮 👍/👎 数据回流后并入重训
            save: 是否落盘 joblib 模型（--no-save 评估场景可跳过）

        Returns:
            {"classes": [...], "n_samples": int, "accuracy": float,
             "report": str} 评估指标

        Raises:
            样本/嵌入失败时抛异常，由调用方（训练脚本）处理
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        queries = [q for q, _ in samples]
        labels = [lbl for _, lbl in samples]
        X = await self._embedding_service.embed_documents(queries)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X_tr, y_tr)
        self._model = model

        if save:
            import joblib
            joblib.dump(model, self.model_path)
            logger.info("L4 分类器训练完成并落盘: %s, samples=%d",
                        self.model_path, len(samples))
        else:
            logger.info("L4 分类器训练完成（未落盘，--no-save）: samples=%d",
                        len(samples))

        pred = model.predict(X_te)
        # module-056: 补充混淆矩阵（训练脚本对比输出用，additive 兼容旧调用方）
        cm = confusion_matrix(y_te, pred, labels=list(model.classes_)).tolist()
        return {
            "classes": list(model.classes_),
            "n_samples": len(samples),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "report": classification_report(y_te, pred, zero_division=0),
            "confusion_matrix": {"classes": list(model.classes_), "matrix": cm},
        }

    async def predict_proba(self, query: str,
                            prev_user_query: Optional[str] = None) -> dict[str, float]:
        """返回 {intent: 校准概率}（三类键齐全，和≈1）

        module-063（WP-A）：prev_user_query 提供时拼接最近一轮 user query 向量
        （list 拼接 2048 维，训练时同构——参考 memory_conflict_clf 两条嵌入先例）。
        **注意**：当前落盘模型 intent_clf.joblib 为单 query 1024 维训练，传入
        prev 会触发 sklearn 特征维度不匹配抛 ValueError → 调用方（router）捕获
        回退 LLM 分类（fail-open 零回归）；待多轮标注数据重训（config
        intent_classifier_multi_turn 置 true）后生效。（2026-08-16 架构评估：
        多轮拼接已降级不做——能力已被 LLM 路径覆盖 + WP-B 规则层兜底，
        性价比不足，见 METRICS 待办区 #8；prev 拼接代码保留不删，开关恒 false）

        模型未加载时抛 RuntimeError——调用方（router）捕获后回退 LLM 分类，
        不阻断主链路。

        Args:
            query: 用户问题
            prev_user_query: 最近一轮 user query（多轮场景；None = 单轮）
                （None → 单 query 1024 维，与存量模型契约一致零回归）

        Returns:
            {"knowledge": float, "casual_chat": float, "realtime": float}
            每类概率（round 到 4 位）；训练集缺类补 0.0
        """
        if self._model is None:
            raise RuntimeError("L4 分类器模型未加载")
        if prev_user_query is not None and str(prev_user_query).strip():
            # 多轮拼接：当前 query 向量 + 最近一轮 user query 向量（2048 维）
            vec = await self._embedding_service.embed_text(query.strip())
            prev_vec = await self._embedding_service.embed_text(prev_user_query.strip())
            feature = list(vec) + list(prev_vec)
        else:
            vec = await self._embedding_service.embed_text(query.strip())
            feature = vec
        proba = self._model.predict_proba([feature])[0]
        probs = {cls: round(float(p), 4) for cls, p in zip(self._model.classes_, proba)}
        for label in _INTENT_LABELS:
            probs.setdefault(label, 0.0)
        return probs
