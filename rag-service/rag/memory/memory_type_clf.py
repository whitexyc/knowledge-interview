"""
记忆类型分类器 — bge-m3 冻结特征 + 逻辑回归头（module-062 WP1 方案 A）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在类型判断链路中的位置（module-062 / ADR-0007 P2）：
  extract_facts（LLM 判型，方案 B） ──对比──> 本分类器（方案 A）
  谁达标（Accuracy≥0.8）谁上，`memory_type_mode`（clf/llm/none）定生产注入。

为什么用 bge-m3 + 逻辑回归替代 LLM 判型（与 module-056 intent 分类器同款哲学）：
  1. 类型是三分类简单任务：bge-m3 已本地部署（1024 维冻结特征），逻辑回归头
     ~1025 参数（每类一份），CPU 毫秒级推理，比 LLM API 调用便宜几个数量级
  2. 训练/评测分离：人造标注集（eval/memory_type_dataset 训练集）训练，评测集
     与训练集字符串零重叠防泄漏
  3. 推理失败/模型缺失 → 调用方回退 llm_type/默认 fact（fail-open，零影响）

数据约束（已知边界，写入 changelog）：
  - 训练源为人造标注集（build_memory_type_dataset.py 120 条），非真实用户数据，
    方向性验证；真实分布以飞轮数据积累后重训为准
  - 记忆内容短句（一句话偏好/事实/事件），与 intent 问题查询分布不同，单独训练
  - 依赖 sklearn / joblib（惰性导入，仅 fit/load 时用到；主链路不硬依赖）
"""
import logging
import os
from typing import Optional

from rag.retrieval.embeddings import embedding_service as default_embedding_service
from rag.memory.memory_extractor import MEMORY_TYPES

logger = logging.getLogger(__name__)

# 模型落盘路径（对齐 ai_service/models/ 本地模型存放约定；训练产物不进仓库）
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "memory_type_clf.joblib",
)


class MemoryTypeClassifier:
    """bge-m3 冻结特征 + sklearn 逻辑回归头的记忆类型分类器

    - 特征：复用 rag.retrieval.embeddings.embedding_service（bge-m3 冻结，1024 维）
      对记忆内容编码（短句偏好/事实/事件）
    - 分类头：LogisticRegression（class_weight="balanced" 抗类别不平衡）
    - fit(): 标注样本训练 → 落盘（joblib）
    - classify()/predict_proba(): 单条记忆内容判型（preference/fact/event）
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH,
                 embedding_service: Optional[object] = None):
        self.model_path = model_path
        self._embedding_service = embedding_service or default_embedding_service
        self._model = None

    async def load(self) -> bool:
        """加载落盘模型；缺失/损坏 → False（调用方回退 LLM 判型/默认 fact，零影响）

        Returns:
            True 加载成功；False 模型缺失/损坏
        """
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            return True
        except Exception as e:
            logger.warning("记忆类型分类器模型加载失败（回退 llm/默认 fact）: %s", e)
            self._model = None
            return False

    async def fit(self, samples: list[tuple[str, str]], save: bool = True) -> dict:
        """用标注样本训练逻辑回归头（可选落盘）

        Args:
            samples: [(content, type_label), ...]，type_label ∈ MEMORY_TYPES，
                来源人造标注集（build_memory_type_dataset.py，与评测集零重叠）
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

        texts = [t for t, _ in samples]
        labels = [lbl for _, lbl in samples]
        X = await self._embedding_service.embed_documents(texts)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X_tr, y_tr)
        self._model = model

        if save:
            import joblib
            joblib.dump(model, self.model_path)
            logger.info("记忆类型分类器训练完成并落盘: %s, samples=%d",
                        self.model_path, len(samples))
        else:
            logger.info("记忆类型分类器训练完成（未落盘，--no-save）: samples=%d",
                        len(samples))

        pred = model.predict(X_te)
        cm = confusion_matrix(y_te, pred, labels=list(model.classes_)).tolist()
        return {
            "classes": list(model.classes_),
            "n_samples": len(samples),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "report": classification_report(y_te, pred, zero_division=0),
            "confusion_matrix": {"classes": list(model.classes_), "matrix": cm},
        }

    async def predict_proba(self, content: str) -> dict[str, float]:
        """返回 {type: 校准概率}（三类键齐全，和≈1）

        模型未加载时抛 RuntimeError——调用方捕获后回退 llm_type/默认 fact。

        Args:
            content: 记忆内容

        Returns:
            {"preference": float, "fact": float, "event": float}（round 到 4 位）
        """
        if self._model is None:
            raise RuntimeError("记忆类型分类器模型未加载")
        vec = await self._embedding_service.embed_text(content)
        proba = self._model.predict_proba([vec])[0]
        probs = {cls: round(float(p), 4) for cls, p in zip(self._model.classes_, proba)}
        for label in MEMORY_TYPES:
            probs.setdefault(label, 0.0)
        return probs

    async def classify(self, content: str) -> str:
        """判单条记忆内容的类型（最高概率类）

        Args:
            content: 记忆内容

        Returns:
            "preference" / "fact" / "event"

        Raises:
            RuntimeError: 模型未加载
        """
        probs = await self.predict_proba(content)
        return max(probs, key=probs.get)


# 全局单例 — 整个应用共享一个 MemoryTypeClassifier 实例（无状态，延迟加载）
memory_type_clf = MemoryTypeClassifier()


async def resolve_memory_type(content: str, llm_type: Optional[str] = None) -> str:
    """按 memory_type_mode 决策单条记忆的类型（生产注入，module-062 WP1）

    - clf：分类模型判型（加载/推理失败 → 回退 llm_type → 默认 fact，fail-open）
    - llm：直接采用 extract_facts 输出的 llm_type（缺失/非法 → fact）
    - none：不判型，全部默认 fact（类型化衰减零生效 = 零回归回退，不预设成功）

    Args:
        content: 记忆内容（clf 判型输入）
        llm_type: extract_facts 输出的 type（可空）

    Returns:
        "preference" / "fact" / "event"
    """
    from src.config import settings

    mode = settings.memory_type_mode
    llm_type = str(llm_type or "").strip().lower()
    if mode == "llm":
        return llm_type if llm_type in MEMORY_TYPES else "fact"
    if mode == "clf":
        try:
            # 先 load（幂等，首次加载落盘模型并缓存；缺失/损坏返回 False → 回退）
            if not await memory_type_clf.load():
                return llm_type if llm_type in MEMORY_TYPES else "fact"
            return await memory_type_clf.classify(content)
        except Exception as e:
            logger.warning("记忆类型分类器推理失败，回退 llm_type/默认 fact: %s", e)
            return llm_type if llm_type in MEMORY_TYPES else "fact"
    return "fact"  # none / 其他
