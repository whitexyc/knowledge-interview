"""
记忆矛盾分类器训练脚本 — 人造 100+ 案例训练 + 模型落盘（module-062 WP4）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python scripts/train_memory_conflict_clf.py            # 默认训练并落盘
    python scripts/train_memory_conflict_clf.py --no-save  # 只训练评估，不落盘

训练数据: eval/memory_conflict_train_dataset.json（build_memory_conflict_train.py
    人造 100+ 条：contradiction（改口/迁移/过时/升级冲突/其它互斥）+ non_conflict
    （entailment/neutral），与评测集零重叠防泄漏）。

特征设计: 矛盾 = 新旧两条记忆的关系（改口/迁移/过时），分别 bge-m3 嵌入后特征层
    拼接 + 差值 + 绝对差值（约 4096 维），让线性头学到"两条嵌入的差异方向"。

分类头: LogisticRegression（max_iter=500, class_weight="balanced" 抗类别不平衡，
    与 intent/sufficiency 分类器同款结构；训练/推理接口对齐 MemoryConflictClassifier）。

输出: Accuracy + 每类 P/R/F1（重点看 contradiction Precision——达标线 ≥0.8，
    宁可漏检也不错标，用户决策）。

启用现状（module-062 WP4）:
  达标判定 = eval/memory_conflict_dataset.py 同评测集对比 clf vs mDeBERTa NLI，
  **Precision ≥ 0.8 者启用**（clf 达标用 clf，mDeBERTa 达标用 nli，双达标取
  Precision 高者）→ 达标后 PW_MEMORY_CONFLICT=true（config）+ PW_MEMORY_CONFLICT_JUDGE
  选型；不达标保持关如实标注（Recall 后续提升入 backlog）。

已知边界（写入 changelog）:
  - 人造数据非真实用户改口分布（方向性验证）
  - 依赖 sklearn / joblib；模型落盘 ai_service/models/memory_conflict_clf.joblib
    （训练产物不进仓库）
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
logger = logging.getLogger("train_memory_conflict_clf")

# 本文件所在目录（scripts/），训练集在 eval/
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
DATASET_PATH = EVAL_DIR / "datasets" / "memory_conflict_train_dataset.json"
# 模型落盘路径（与 memory_conflict_clf 默认一致）
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "models" / "memory_conflict_clf.joblib"
)


def load_training_samples(path: Path = DATASET_PATH) -> list[tuple[str, str, str]]:
    """从人造标注集加载样本 [(premise, hypothesis, label), ...]

    JSON 结构 [{"premise", "hypothesis", "label"}...]；缺文件 → 警告返回空（不阻断）。
    """
    if not path.exists():
        logger.warning("memory_conflict_train_dataset.json 不存在，跳过: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        (item["premise"], item["hypothesis"], item["label"])
        for item in data
        if item.get("premise") and item.get("hypothesis") and item.get("label")
    ]
    labels = sorted({lbl for _, _, lbl in samples})
    logger.info("训练样本加载 %d 条, 类别分布: %s",
                len(samples), {lbl: sum(1 for _, _, l in samples if l == lbl) for lbl in labels})
    return samples


async def train(model_path: str, save: bool) -> None:
    """训练并评估记忆矛盾分类器（样本不足时明确报错退出）"""
    from rag.memory.memory_conflict_clf import MemoryConflictClassifier

    samples = load_training_samples()
    if len(samples) < 10:
        logger.error("训练样本不足（%d 条），无法训练——请先运行 "
                     "python -m eval.datasets.build_memory_conflict_train 补充标注数据", len(samples))
        sys.exit(1)

    clf = MemoryConflictClassifier(model_path=model_path)
    metrics = await clf.fit(samples, save=save)

    print("\n" + "=" * 60)
    print("Memory Conflict Classifier Training (bge-m3 + LogisticRegression)")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']} | Classes: {metrics['classes']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("-" * 60)
    print("Classification report (test split):")
    print(metrics["report"])
    print("-" * 60)
    cont = metrics["per_class"].get("contradiction")
    if cont:
        print(f"★ contradiction Precision: {cont['precision']:.4f}"
              "（达标线 ≥0.8，宁可漏检也不错标——误判会误标正常记忆过期）")
    print("=" * 60)
    if save:
        print(f"Model saved to: {model_path}")
        print("启用：由 eval/memory_conflict_dataset.py 对比 clf vs mDeBERTa 达标后，"
              "设 PW_MEMORY_CONFLICT=true + PW_MEMORY_CONFLICT_JUDGE=clf；"
              "不达标保持 PW_MEMORY_CONFLICT=false")
    else:
        print("--no-save：未落盘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="记忆矛盾分类器训练（人造 100+ 案例，bge-m3 + 逻辑回归）")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"模型落盘路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--no-save", action="store_true", help="只训练评估，不落盘")
    args = parser.parse_args()
    asyncio.run(train(args.model_path, save=not args.no_save))


if __name__ == "__main__":
    main()
