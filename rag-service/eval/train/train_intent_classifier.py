"""
L4 意图分类器训练脚本 — 人造标注集 + golden 训练 + 模型落盘（ADR-0003 L4）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.train.train_intent_classifier          # 默认训练并落盘
    python -m eval.train.train_intent_classifier --no-save  # 只训练评估，不落盘

训练数据（优先级从高到低）:
  1. eval/intent_train_dataset.json（module-056 人造标注集：337 条，三类
     平衡 + 边界易混 + 专有术语 + 口语化；构造脚本 eval/build_intent_dataset.py）
  2. eval/golden.json（knowledge 天然标注，112 题）
  （module-056 起 golden_intent 评测集与内置样本不再进入训练——训练/评测
    分离防泄漏：golden_intent 100 条是独立外部队列，只用于评测不用于训练）

已知边界（写入验收）:
  - 真实飞轮数据（前端 👍/👎）未积累：先以人造集训练；飞轮接口已预留——
    样本回流后并入 load_training_samples 的返回列表重训即可，
    intent_classifier.fit() 的接口无需变更
  - 人造数据非真实用户分布（方向性验证）；真实分布以 golden_intent 评测为准
  - 依赖 sklearn / joblib（本地已安装）；模型落盘 ai_service/models/
    intent_clf.joblib（对齐本地模型存放约定，训练产物不进仓库）

L4 启用现状（module-056 达标启用）:
  模型已落盘 + PW_INTENT_CLASSIFIER_ENABLED 默认 true（router 惰性加载，
  加载/推理失败自动回退 LLM 分类）；本脚本重训出新模型覆盖落盘即生效，
  无需改配置；回退开关 PW_INTENT_CLASSIFIER_ENABLED=false 保持 LLM 路径。
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_intent_classifier")

# 本文件所在目录（eval/train/）；数据源分别位于 eval/golden/ 与 eval/datasets/
EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "golden.json"
DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "intent_train_dataset.json"
# 模型落盘路径（与 intent_classifier 默认一致，ai_service/models/）
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "models" / "intent_clf.joblib"
)


def load_intent_train_dataset(path: Path = DATASET_PATH) -> list[tuple[str, str]]:
    """从人造标注集加载样本（module-056，训练集主源，优先级最高）

    JSON 结构 [{"query", "intent", "note"?}, ...]；缺文件 → 警告返回空
    （回退 golden.json，不阻断）。
    """
    if not path.exists():
        logger.warning("intent_train_dataset.json 不存在，跳过: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        (item["query"], item["intent"])
        for item in data
        if item.get("query") and item.get("intent")
    ]
    logger.info("intent_train_dataset.json 加载 %d 条", len(samples))
    return samples


def load_golden_knowledge(path: Path = GOLDEN_PATH) -> list[tuple[str, str]]:
    """从 golden.json 加载 knowledge 样本（天然标注，每题一条）"""
    if not path.exists():
        logger.warning("golden.json 不存在，跳过 knowledge 样本: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        (item["question"], "knowledge")
        for item in data if item.get("question")
    ]
    logger.info("golden.json 加载 knowledge 样本 %d 条", len(samples))
    return samples


def load_training_samples() -> list[tuple[str, str]]:
    """组装训练样本：人造标注集优先 → golden.json knowledge

    去重（按 query），保留首见标注（人造集 > golden.json）。

    训练/评测分离（module-056）：golden_intent 评测集 100 条不进入训练，
    独立外部队列（防泄漏）；评测集内 knowledge 题源自 golden.json，故
    golden.json 进训练属计划内重叠，casual/realtime 评测样本零混入。
    """
    samples: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _append(items: list[tuple[str, str]]) -> None:
        for query, label in items:
            q = query.strip()
            if q and q not in seen:
                samples.append((q, label))
                seen.add(q)

    _append(load_intent_train_dataset())
    _append(load_golden_knowledge())

    labels = sorted({lbl for _, lbl in samples})
    logger.info("训练样本组装完成: %d 条, 类别分布: %s",
                len(samples), {lbl: sum(1 for _, l in samples if l == lbl) for lbl in labels})
    return samples


async def train(model_path: str, save: bool) -> None:
    """训练并评估 L4 分类器（样本不足时明确报错退出）"""
    from agent.intent_classifier import IntentClassifier

    samples = load_training_samples()
    if len(samples) < 10:
        logger.error("训练样本不足（%d 条），无法训练——请先补充标注数据", len(samples))
        sys.exit(1)

    clf = IntentClassifier(model_path=model_path)
    metrics = await clf.fit(samples, save=save)

    print("\n" + "=" * 60)
    print("Intent Classifier Training (bge-m3 + LogisticRegression)")
    print("=" * 60)
    print(f"Samples: {metrics['n_samples']} | Classes: {metrics['classes']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
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
        print("上线：默认已启用（PW_INTENT_CLASSIFIER_ENABLED 默认 true，router 惰性加载）；"
              "回退可用 PW_INTENT_CLASSIFIER_ENABLED=false 保持 LLM 路径")
    else:
        print("--no-save：未落盘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="L4 意图分类器训练（人造集 + golden）")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"模型落盘路径（默认 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--no-save", action="store_true", help="只训练评估，不落盘")
    args = parser.parse_args()
    asyncio.run(train(args.model_path, save=not args.no_save))


if __name__ == "__main__":
    main()
