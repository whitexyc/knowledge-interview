"""
充分性分类器训练脚本 — SUFFICIENCY_DATASET 训练 + 模型落盘（module-045 WP4）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.train.train_sufficiency_classifier            # 默认训练并落盘
    python -m eval.train.train_sufficiency_classifier --no-save  # 只训练评估，不落盘
    python -m eval.train.train_sufficiency_classifier --model-path models/sufficiency_clf.joblib

训练数据: eval.golden.golden_sufficiency.SUFFICIENCY_DATASET（100 条：充分 50 /
不充分 50，2026-08-09 自造扩充至 100）。每条含 question / documents /
sufficient（bool）——问题借 golden 集真实题目，文档为代表性内容。

特征设计: 充分性由"问题能否被检索文档回答"决定，单靠问题无法判别——
特征 = bge-m3 冻结 embedding 对"问题 + 检索文档内容"拼接文本编码
（复用 rag.retrieval.embeddings.embedding_service，1024 维；文档内容按序拼接）。

分类头: LogisticRegression（max_iter=500, class_weight="balanced" 抗类别
不平衡，与 intent 分类器同款结构；训练/推理接口对齐 IntentClassifier）。

输出: Accuracy + P/R/F1（重点看 insufficient Recall——漏判"不充分"会把
基于无关文档的硬答放行，最致命，报告里单独大字标出）。

已知边界:
  - 训练样本即 golden 评测集（golden 集即训练集，与意图分类器同哲学）
  - 依赖 sklearn / joblib（本地已安装；requirements.txt 已声明，module-045 WP5）；
    模型落盘 ai_service/models/sufficiency_clf.joblib（对齐本地模型存放约定，
    训练产物不进仓库）
  - 样本不足 10 条明确报错退出（sys.exit(1)，与 train_intent_classifier 一致）
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from rag.retrieval.embeddings import embedding_service as default_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_sufficiency_classifier")

# 模型落盘路径（对齐 ai_service/models/ 本地模型约定）
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "models" / "sufficiency_clf.joblib"
)

# 充分性标签契约（与 golden_sufficiency.SUFFICIENCY_CLASSES 一致）
SUFFICIENCY_CLASSES = ("sufficient", "insufficient")


def build_feature_text(question: str, documents: list[dict]) -> str:
    """特征文本：问题 + 检索文档拼接（充分性由问题能否被文档回答决定）

    Args:
        question: 用户问题
        documents: 检索文档列表（取 content 拼接）

    Returns:
        "问题：...\n文档：..." 拼接文本
    """
    docs_text = " ".join((d.get("content") or "") for d in documents)
    return f"问题：{question}\n文档：{docs_text}"


def load_training_samples() -> list[tuple[str, str]]:
    """从 SUFFICIENCY_DATASET 组装训练样本（特征文本 → 充分性标签）

    Returns:
        [(feature_text, "sufficient"|"insufficient"), ...]

    Raises:
        ValueError: 数据集校验失败（load_sufficiency_dataset 抛出）
    """
    from eval.golden.golden_sufficiency import load_sufficiency_dataset

    dataset = load_sufficiency_dataset()
    samples = [
        (build_feature_text(item["question"], item["documents"]),
         "sufficient" if item["sufficient"] else "insufficient")
        for item in dataset
    ]
    counts = {
        lbl: sum(1 for _, l in samples if l == lbl) for lbl in SUFFICIENCY_CLASSES
    }
    logger.info("训练样本组装完成: %d 条, 类别分布: %s", len(samples), counts)
    return samples


class SufficiencyClassifier:
    """bge-m3 冻结特征 + sklearn 逻辑回归头的充分性分类器

    - 特征：复用 rag.retrieval.embeddings.embedding_service（bge-m3 冻结，1024 维）
      对"问题 + 检索文档"拼接文本编码
    - 分类头：LogisticRegression（class_weight="balanced" 抗类别不平衡）
    - fit(): 标注样本训练 → 落盘（joblib）
    - predict_proba(): 返回 {"sufficient": 校准概率, "insufficient": 校准概率}
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 embedding_service: Optional[object] = None):
        self.model_path = model_path
        self._embedding_service = embedding_service or default_embedding_service
        self._model = None

    async def load(self) -> bool:
        """加载落盘模型；缺失/损坏 → False（调用方自行处理，零影响）

        Returns:
            True 加载成功；False 模型缺失/损坏
        """
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            return True
        except Exception as e:
            logger.warning("充分性分类器模型加载失败: %s", e)
            self._model = None
            return False

    async def fit(self, samples: list[tuple[str, str]], save: bool = True) -> dict:
        """用标注样本训练逻辑回归头（可选落盘）

        Args:
            samples: [(feature_text, "sufficient"|"insufficient"), ...]，
                特征文本由 build_feature_text 生成（问题 + 检索文档）
            save: 是否落盘 joblib 模型（--no-save 评估场景可跳过）

        Returns:
            {"classes": [...], "n_samples": int, "accuracy": float,
             "report": str, "per_class": {cls: {precision/recall/f1}}}
            评估指标（per_class 供报告重点展示 insufficient Recall）

        Raises:
            样本/嵌入失败时抛异常，由调用方（训练脚本）处理
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, classification_report, precision_recall_fscore_support,
        )
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
            logger.info("充分性分类器训练完成并落盘: %s, samples=%d",
                        self.model_path, len(samples))
        else:
            logger.info("充分性分类器训练完成（未落盘，--no-save）: samples=%d",
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

    async def predict_proba(self, question: str,
                            documents: Optional[list[dict]] = None) -> dict[str, float]:
        """返回 {"sufficient": 校准概率, "insufficient": 校准概率}

        模型未加载时抛 RuntimeError——调用方捕获后走降级路径。

        Args:
            question: 用户问题
            documents: 检索文档列表（可选，缺省按无文档编码）

        Returns:
            {"sufficient": float, "insufficient": float}（round 到 4 位）
        """
        if self._model is None:
            raise RuntimeError("充分性分类器模型未加载")
        text = build_feature_text(question, documents or [])
        vec = await self._embedding_service.embed_text(text)
        proba = self._model.predict_proba([vec])[0]
        return {cls: round(float(p), 4) for cls, p in zip(self._model.classes_, proba)}


async def train(model_path: str, save: bool) -> None:
    """训练并评估充分性分类器（样本不足时明确报错退出）"""
    samples = load_training_samples()
    if len(samples) < 10:
        logger.error("训练样本不足（%d 条），无法训练——请先补充标注数据"
                     "（SUFFICIENCY_DATASET 需 ≥ 10 条）", len(samples))
        sys.exit(1)

    clf = SufficiencyClassifier(model_path=model_path)
    metrics = await clf.fit(samples, save=save)

    print("\n" + "=" * 60)
    print("Sufficiency Classifier Training (bge-m3 + LogisticRegression)")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']} | Classes: {metrics['classes']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("-" * 60)
    print("Classification report (test split):")
    print(metrics["report"])
    print("-" * 60)
    ins = metrics["per_class"].get("insufficient")
    if ins:
        print(f"★ insufficient Recall: {ins['recall']:.4f}"
              "（漏判'不充分'→ 基于无关文档硬答，最致命）")
    print("=" * 60)
    if save:
        print(f"Model saved to: {model_path}")
    else:
        print("--no-save：未落盘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="充分性分类器训练（SUFFICIENCY_DATASET，bge-m3 + 逻辑回归）")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"模型落盘路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--no-save", action="store_true", help="只训练评估，不落盘")
    args = parser.parse_args()
    asyncio.run(train(args.model_path, save=not args.no_save))


if __name__ == "__main__":
    main()
