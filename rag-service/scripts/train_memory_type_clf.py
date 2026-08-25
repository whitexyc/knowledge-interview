"""
记忆类型分类器训练脚本 — 人造标注集训练 + 模型落盘（module-062 WP1 方案 A）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python scripts/train_memory_type_clf.py            # 默认训练并落盘
    python scripts/train_memory_type_clf.py --no-save  # 只训练评估，不落盘

训练数据: eval/memory_type_train_dataset.json（build_memory_type_dataset.py 人造
    120 条：preference 40 / fact 40 / event 40，与评测集零重叠防泄漏）。

特征设计: 记忆类型由内容语义决定（偏好/事实/事件），特征 = bge-m3 冻结 embedding
    对记忆内容短句编码（复用 rag.retrieval.embeddings.embedding_service，1024 维）。

分类头: LogisticRegression（max_iter=500, class_weight="balanced" 抗类别不平衡，
    与 intent 分类器同款结构；训练/推理接口对齐 MemoryTypeClassifier）。

输出: Accuracy + P/R/F1（重点看整体 Accuracy——达标线 ≥0.8，谁达标谁上）。

已知边界（写入 changelog）:
  - 人造数据非真实用户记忆分布（方向性验证）；真实分布以飞轮积累后重训为准
  - 依赖 sklearn / joblib（本地已安装）；模型落盘 ai_service/models/
    memory_type_clf.joblib（训练产物不进仓库）
  - 样本不足 10 条明确报错退出（sys.exit(1)，与 train_intent_classifier 一致）
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_memory_type_clf")

# 本文件所在目录（scripts/），训练集在 eval/
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
DATASET_PATH = EVAL_DIR / "datasets" / "memory_type_train_dataset.json"
# 模型落盘路径（与 memory_type_clf 默认一致）
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "models" / "memory_type_clf.joblib"
)


def load_training_samples(path: Path = DATASET_PATH) -> list[tuple[str, str]]:
    """从人造标注集加载样本 [(content, type), ...]

    JSON 结构 [{"content", "type"}, ...]；缺文件 → 警告返回空（不阻断）。
    """
    if not path.exists():
        logger.warning("memory_type_train_dataset.json 不存在，跳过: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        (item["content"], item["type"])
        for item in data
        if item.get("content") and item.get("type")
    ]
    labels = sorted({t for _, t in samples})
    logger.info("训练样本加载 %d 条, 类别分布: %s",
                len(samples), {lbl: sum(1 for _, t in samples if t == lbl) for lbl in labels})
    return samples


async def train(model_path: str, save: bool) -> None:
    """训练并评估记忆类型分类器（样本不足时明确报错退出）"""
    from rag.memory.memory_type_clf import MemoryTypeClassifier

    samples = load_training_samples()
    if len(samples) < 10:
        logger.error("训练样本不足（%d 条），无法训练——请先运行 "
                     "python -m eval.datasets.build_memory_type_dataset 补充标注数据", len(samples))
        sys.exit(1)

    clf = MemoryTypeClassifier(model_path=model_path)
    metrics = await clf.fit(samples, save=save)

    print("\n" + "=" * 60)
    print("Memory Type Classifier Training (bge-m3 + LogisticRegression)")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']} | Classes: {metrics['classes']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}（达标线 ≥0.8，谁达标谁上）")
    print("-" * 60)
    print("Classification report (test split):")
    print(metrics["report"])
    print("-" * 60)
    print("Confusion matrix (row=label, col=predicted):")
    cm = metrics.get("confusion_matrix")
    if cm:
        classes = cm["classes"]
        print(f"{'':<12}" + "".join(f"{c[:10]:>12}" for c in classes))
        for i, label in enumerate(classes):
            print(f"{label:<12}" + "".join(f"{v:>12}" for v in cm["matrix"][i]))
    print("=" * 60)
    if save:
        print(f"Model saved to: {model_path}")
        print("上线：由 eval/memory_type_dataset.py 对比 clf vs LLM 达标后，"
              "设 PW_MEMORY_TYPE_MODE=clf/llm 注入生产；不达标保持 none（回退）")
    else:
        print("--no-save：未落盘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆类型分类器训练（人造集，bge-m3 + 逻辑回归）")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"模型落盘路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--no-save", action="store_true", help="只训练评估，不落盘")
    args = parser.parse_args()
    asyncio.run(train(args.model_path, save=not args.no_save))


if __name__ == "__main__":
    main()
