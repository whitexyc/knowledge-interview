"""
记忆矛盾检测分类器 — bge-m3 嵌入新旧两条记忆 + 逻辑回归二分类（module-062 WP4）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在矛盾消解链路中的位置（module-061 P1 + module-062 WP4）：
  `_merge_duplicate` 语义去重命中后判新事实 vs 旧记忆是否矛盾（contradiction →
  旧父块标 SUPERSEDED + 新内容按正常新增）。判定器可切换：
    clf（本分类器，bge-m3+LR 二分类，自建 100+ 案例训练）
    nli（module-061 mDeBERTa NLI 三分类）
  同评测集（eval/memory_conflict_dataset.py 30 条）对比 Accuracy/P/R/F1，
  **contradiction Precision ≥ 0.8 者启用**（用户决策：宁可漏检也不错标），
  `PW_MEMORY_CONFLICT_JUDGE` 选型，达标后 `PW_MEMORY_CONFLICT=true`。

为什么用"新旧两条分别嵌入"而非拼接文本整体嵌入（与 sufficiency 拼接文本不同）：
  矛盾 = 两条记忆之间的关系（改口/迁移/过时），分别嵌入后再在特征层拼接/做差，
  让线性头学到"两条嵌入的差异方向"（如 {喜欢咖啡} vs {讨厌咖啡} 在嵌入空间
  的位移）——比整句拼接对"关系判别"更直接。

数据约束（已知边界，写入 changelog）：
  - 训练源为自建人造矛盾案例（build_memory_conflict_train.py，100+ 条，
    改口/迁移/过时/升级冲突/正例中性），与评测集字符串零重叠防泄漏；
    人工构造非真实用户改口数据，方向性验证
  - 二分类（contradiction/non_conflict），与 mDeBERTa 三分类口径不同——
    对比时三分类映射为二值（contradiction vs 其余）
  - 依赖 sklearn / joblib（惰性导入，仅 fit/load 时用到；主链路不硬依赖）
"""
import logging
import os
from typing import Optional

import numpy as np

from rag.retrieval.embeddings import embedding_service as default_embedding_service

logger = logging.getLogger(__name__)

# 模型落盘路径（对齐 ai_service/models/ 本地模型存放约定；训练产物不进仓库）
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "memory_conflict_clf.joblib",
)

# 二分类标签契约：contradiction（矛盾）/ non_conflict（非矛盾）
CONFLICT_CLASSES = ("contradiction", "non_conflict")


class MemoryConflictClassifier:
    """bge-m3 冻结特征 + sklearn 逻辑回归的记忆矛盾分类器

    - 特征：新旧两条记忆分别 bge-m3 嵌入（1024 维）→ 拼接 + 差值 + 绝对差值
      （约 4096 维），编码"两条记忆的关系"
    - 分类头：LogisticRegression（class_weight="balanced" 抗类别不平衡）
    - fit(): 标注样本训练 → 落盘（joblib）
    - predict(): (premise 旧记忆, hypothesis 新事实) → "contradiction" / "non_conflict"
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH,
                 embedding_service: Optional[object] = None):
        self.model_path = model_path
        self._embedding_service = embedding_service or default_embedding_service
        self._model = None

    async def load(self) -> bool:
        """加载落盘模型；缺失/损坏 → False（调用方回退 nli/旧行为，零影响）

        Returns:
            True 加载成功；False 模型缺失/损坏
        """
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            return True
        except Exception as e:
            logger.warning("记忆矛盾分类器模型加载失败（回退 nli/旧行为）: %s", e)
            self._model = None
            return False

    @staticmethod
    def _feature(a: list[float], b: list[float]) -> list[float]:
        """新旧嵌入拼接 + 差值 + 绝对差值（关系判别特征）

        Args:
            a: 旧记忆（premise）嵌入
            b: 新事实（hypothesis）嵌入

        Returns:
            拼接特征向量（[a, b, a-b, |a-b|]）
        """
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        diff = a_arr - b_arr
        return np.concatenate([a_arr, b_arr, diff, np.abs(diff)]).tolist()

    async def fit(self, samples: list[tuple[str, str, str]], save: bool = True) -> dict:
        """用标注样本训练逻辑回归头（可选落盘）

        Args:
            samples: [(premise, hypothesis, label), ...]，
                label ∈ CONFLICT_CLASSES；来源 build_memory_conflict_train.py
            save: 是否落盘 joblib 模型（--no-save 评估场景可跳过）

        Returns:
            {"classes": [...], "n_samples": int, "accuracy": float,
             "report": str, "per_class": {cls: {precision/recall/f1}}}
            评估指标（per_class 供报告重点展示 contradiction Precision）

        Raises:
            样本/嵌入失败时抛异常，由调用方（训练脚本）处理
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, classification_report, precision_recall_fscore_support,
        )
        from sklearn.model_selection import train_test_split

        premises = [p for p, _, _ in samples]
        hypotheses = [h for _, h, _ in samples]
        labels = [lbl for _, _, lbl in samples]
        emb_a = await self._embedding_service.embed_documents(premises)
        emb_b = await self._embedding_service.embed_documents(hypotheses)
        X = [self._feature(a, b) for a, b in zip(emb_a, emb_b)]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X_tr, y_tr)
        self._model = model

        if save:
            import joblib
            joblib.dump(model, self.model_path)
            logger.info("记忆矛盾分类器训练完成并落盘: %s, samples=%d",
                        self.model_path, len(samples))
        else:
            logger.info("记忆矛盾分类器训练完成（未落盘，--no-save）: samples=%d",
                        len(samples))

        pred = model.predict(X_te)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, pred, average=None, zero_division=0)
        per_class = {
            cls: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                  "f1": round(float(f), 4)}
            for cls, p, r, f in zip(model.classes_, prec, rec, f1)
        }
        return {
            "classes": list(model.classes_),
            "n_samples": len(samples),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "report": classification_report(y_te, pred, zero_division=0),
            "per_class": per_class,
        }

    async def predict_proba(self, premise: str, hypothesis: str) -> dict[str, float]:
        """返回 {label: 校准概率}（二类键齐全，和≈1）

        模型未加载时抛 RuntimeError——调用方捕获后走降级路径。

        Args:
            premise: 旧记忆内容
            hypothesis: 新事实内容

        Returns:
            {"contradiction": float, "non_conflict": float}（round 到 4 位）
        """
        if self._model is None:
            raise RuntimeError("记忆矛盾分类器模型未加载")
        a = await self._embedding_service.embed_text(premise)
        b = await self._embedding_service.embed_text(hypothesis)
        vec = self._feature(a, b)
        proba = self._model.predict_proba([vec])[0]
        probs = {cls: round(float(p), 4) for cls, p in zip(self._model.classes_, proba)}
        for label in CONFLICT_CLASSES:
            probs.setdefault(label, 0.0)
        return probs

    async def predict(self, premise: str, hypothesis: str) -> Optional[str]:
        """判旧记忆与新事实是否矛盾（最高概率类）

        任何异常（模型缺失/加载失败/推理失败/嵌入失败）→ None（降级信号，
        由上层回退 nli/旧行为，不抛异常）。

        Args:
            premise: 旧记忆内容
            hypothesis: 新事实内容

        Returns:
            "contradiction" / "non_conflict"；不可用 → None
        """
        if not premise or not premise.strip() or not hypothesis or not hypothesis.strip():
            return None
        try:
            probs = await self.predict_proba(premise, hypothesis)
            return max(probs, key=probs.get)
        except Exception as e:
            logger.warning("记忆矛盾分类器推理失败（降级 None）: %s", e)
            return None


# 全局单例 — 整个应用共享一个 MemoryConflictClassifier 实例（无状态，延迟加载）
memory_conflict_clf = MemoryConflictClassifier()
