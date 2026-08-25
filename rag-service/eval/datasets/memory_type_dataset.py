"""
记忆类型评测 — 分类模型 vs LLM 同集对比（module-062 WP1 双方案，谁达标谁上）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.datasets.memory_type_dataset                   # clf vs LLM 同集对比 + 落库
    python -m eval.datasets.memory_type_dataset --clf-only         # 只跑分类模型（不依赖 LLM）
    python -m eval.datasets.memory_type_dataset --fixture          # 关键词启发式（确定性，不依赖模型/LLM）
    python -m eval.datasets.memory_type_dataset --no-save          # 纯跑分不写 eval_runs

评测口径:
    样本 = 记忆内容短句 + 人工标注类型（preference/fact/event，各 10 条共 30，
    与训练集 build_memory_type_dataset.py 字符串零重叠防泄漏）。判定器：
      方案 A 分类模型：memory_type_clf.classify(content)（bge-m3+LR，train 后落盘）
      方案 B LLM：extract_facts 对"用户说该内容"的对话提取，取匹配事实的 type
                  （_EXTRACT_PROMPT 加 type few-shot；缺失/非法默认 fact）
    同集对比 Accuracy + 每类 P/R/F1（eval_runs eval_type='memory_type'，scores 含
    model 字段区分 clf/llm）。

达标线: 类型 Accuracy ≥ 0.8 —— **谁达标谁上**（双达标取高分者）；都不达标 →
    类型化回退（memory_type_mode='none'，type 按默认 fact，不预设成功）。

诚实边界:
    1. 记忆类型样本为人工构造（非真实用户记忆），方向性验证。
    2. LLM 判定走 extract_facts（含 importance≥0.6 过滤）——未提取出事实的样本
       记 skipped（该样本类型不可判，生产表现为不写记忆），与提取召回耦合如实标注。
    3. 分类模型 Accuracy 依赖 bge-m3 嵌入 + LR 训练（train_memory_type_clf.py
       落盘 models/memory_type_clf.joblib）；模型缺失/加载失败 → 明确报错。
"""
import argparse
import asyncio
import logging
import sys

from eval.golden.golden_memory import _dialogue_to_extract_inputs, _fact_match
from rag.memory.memory_extractor import extract_facts, MEMORY_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("memory_type_dataset")

GATE_TYPE_ACCURACY = 0.8

# 记忆类型评测集：content 短句 + 人工标注 type（与训练集字符串零重叠，防泄漏）
MEMORY_TYPE_DATASET: list[dict] = [
    # ---- preference（偏好/习惯/兴趣，10）----
    {"content": "用户偏好用英文回答", "type": "preference", "keywords": ["英文"]},
    {"content": "用户喜欢喝绿茶", "type": "preference", "keywords": ["绿茶"]},
    {"content": "用户习惯睡前刷手机", "type": "preference", "keywords": ["睡前"]},
    {"content": "用户偏爱高效简洁的代码", "type": "preference", "keywords": ["简洁"]},
    {"content": "用户喜欢看悬疑剧", "type": "preference", "keywords": ["悬疑剧"]},
    {"content": "用户偏好步行上班", "type": "preference", "keywords": ["步行"]},
    {"content": "用户习惯用双屏", "type": "preference", "keywords": ["双屏"]},
    {"content": "用户喜欢交朋友", "type": "preference", "keywords": ["交朋友"]},
    {"content": "用户偏爱米色系穿搭", "type": "preference", "keywords": ["米色"]},
    {"content": "用户习惯定时喝水", "type": "preference", "keywords": ["喝水"]},
    # ---- fact（客观事实，10）----
    {"content": "用户是产品经理", "type": "fact", "keywords": ["产品经理"]},
    {"content": "用户有四年数据分析经验", "type": "fact", "keywords": ["数据分析"]},
    {"content": "用户住在深圳", "type": "fact", "keywords": ["深圳"]},
    {"content": "用户在中型电商公司工作", "type": "fact", "keywords": ["电商"]},
    {"content": "用户的本科专业是软件工程", "type": "fact", "keywords": ["软件工程"]},
    {"content": "用户负责订单系统", "type": "fact", "keywords": ["订单系统"]},
    {"content": "用户的团队 12 人", "type": "fact", "keywords": ["12"]},
    {"content": "用户的英语达到六级", "type": "fact", "keywords": ["六级"]},
    {"content": "用户持有 PMP 证书", "type": "fact", "keywords": ["PMP"]},
    {"content": "用户的上家公司是外企", "type": "fact", "keywords": ["外企"]},
    # ---- event（带时间临时事件，10）----
    {"content": "用户这周三有牙医预约", "type": "event", "keywords": ["周三"]},
    {"content": "用户明天下午去面试", "type": "event", "keywords": ["明天"]},
    {"content": "用户这周末去露营", "type": "event", "keywords": ["周末"]},
    {"content": "用户下个月去香港出差", "type": "event", "keywords": ["下个月"]},
    {"content": "用户今晚有线上课程", "type": "event", "keywords": ["今晚"]},
    {"content": "用户这周五要交季度报告", "type": "event", "keywords": ["周五"]},
    {"content": "用户下周一有体检", "type": "event", "keywords": ["周一"]},
    {"content": "用户明天上午约了理发", "type": "event", "keywords": ["理发"]},
    {"content": "用户这周末参加同学聚会", "type": "event", "keywords": ["聚会"]},
    {"content": "用户下个月办理护照", "type": "event", "keywords": ["护照"]},
]


def load_memory_type_dataset() -> list[dict]:
    """加载并校验评测集结构

    Raises:
        ValueError: 样本 < 30、每类 < 10、缺 content/type、type 非法
    """
    data = MEMORY_TYPE_DATASET
    if len(data) < 30:
        raise ValueError(f"记忆类型评测集过小：需 ≥ 30 条，当前 {len(data)}")
    for item in data:
        if not str(item.get("content") or "").strip():
            raise ValueError(f"样本缺 content: {item}")
        if item.get("type") not in MEMORY_TYPES:
            raise ValueError(f"type 须为 {MEMORY_TYPES}: {item.get('content', '')[:30]}")
    for cls in MEMORY_TYPES:
        if sum(1 for i in data if i["type"] == cls) < 10:
            raise ValueError(f"类型 {cls} 样本需 ≥ 10 条")
    return data


# ──────────────────────────────────────────────────────────────
# 指标（纯函数，可单测）
# ──────────────────────────────────────────────────────────────

def type_metrics(human: list[str], pred: list[str]) -> dict:
    """类型 Accuracy + 每类 P/R/F1（macro 口径）

    Args:
        human: 人工类型标注列表
        pred: 判定器预测类型列表（长度一致）

    Returns:
        {"accuracy", "classes", "per_class": {cls: {precision/recall/f1}}}
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
    )
    accuracy = round(float(accuracy_score(human, pred)), 4)
    classes = sorted(set(MEMORY_TYPES))
    prec, rec, f1, _ = precision_recall_fscore_support(
        human, pred, labels=classes, average=None, zero_division=0)
    per_class = {
        cls: {"precision": round(float(p), 4), "recall": round(float(r), 4),
              "f1": round(float(f), 4)}
        for cls, p, r, f in zip(classes, prec, rec, f1)
    }
    return {"accuracy": accuracy, "classes": classes, "per_class": per_class}


def gate_passed(scores: dict) -> bool:
    """达标线：类型 Accuracy ≥ 0.8"""
    return scores.get("accuracy", 0.0) >= GATE_TYPE_ACCURACY


# ──────────────────────────────────────────────────────────────
# 判定器
# ──────────────────────────────────────────────────────────────

def fixture_judge(item: dict) -> str:
    """fixture 关键词启发式（确定性，不依赖模型/LLM，仅演示评测管线）

    内容含时间词（明天/下周/这周/今晚/下月/下个/本周/今天/周X）→ event；
    含喜好词（喜欢/偏好/偏爱/习惯）→ preference；否则 → fact。不代表真实判别能力。
    """
    content = item["content"]
    import re
    if re.search(r"明天|下周|这周|今晚|下个月|下个|本周|今天|周[一二三四五六日天]", content):
        return "event"
    if any(w in content for w in ("喜欢", "偏好", "偏爱", "习惯")):
        return "preference"
    return "fact"


def clf_judge_factory():
    """分类模型判定器（依赖 models/memory_type_clf.joblib，bge-m3 嵌入）"""
    from rag.memory.memory_type_clf import memory_type_clf

    async def _judge(item: dict) -> str:
        loaded = await memory_type_clf.load()
        if not loaded:
            raise RuntimeError("memory_type_clf 模型缺失/加载失败（先跑 train_memory_type_clf.py）")
        return await memory_type_clf.classify(item["content"])
    return _judge


def llm_judge_factory():
    """LLM 判定器：extract_facts 对"用户说该内容"的对话提取，取匹配事实的 type

    生产路径即 extract_facts（_EXTRACT_PROMPT 带 type few-shot）。无法映射对话 /
    未提取出事实 → 返回 None（该样本记 skipped，类型不可判）。
    """
    async def _judge(item: dict) -> str | None:
        content = item["content"]
        dialogue = f"用户: {content}\n助手: 好的，我记住了。"
        inputs = _dialogue_to_extract_inputs(dialogue)
        if inputs is None:
            return None
        query, answer, history = inputs
        facts = await extract_facts(query, answer, history)
        for f in facts:
            if _fact_match(str(f.get("content") or ""), content):
                return str(f.get("type") or "fact")
        if facts:
            return str(facts[0].get("type") or "fact")
        return None  # 未提取出事实 → 类型不可判（记 skipped）
    return _judge


# ──────────────────────────────────────────────────────────────
# 运行
# ──────────────────────────────────────────────────────────────

async def run_eval(judge=None, dataset=None) -> tuple:
    """执行一次记忆类型评估

    Args:
        judge: 判定器 async callable (item) -> str|None；缺省 None（由调用方指定）
        dataset: 评测样本列表；默认 load_memory_type_dataset()

    Returns:
        (scores, per_question, skipped)
    """
    items = dataset if dataset is not None else load_memory_type_dataset()
    if judge is None:
        raise ValueError("必须指定 judge（clf/llm/fixture）")
    human = [i["type"] for i in items]
    pred: list[str] = []
    per_question: list[dict] = []
    skipped: list[dict] = []

    for i, item in enumerate(items):
        try:
            p = await judge(item)
            if p is None or p not in MEMORY_TYPES:
                raise ValueError(f"判定器返回非法/空 type: {p}")
        except Exception as e:
            logger.error("[%d/%d] 类型判定失败: %s — %s",
                         i + 1, len(items), item["content"][:30], e)
            skipped.append({"content": item["content"][:40], "reason": f"error: {e}"})
            pred.append("fact")  # 失败样本按默认 fact 计数（不污染 accuracy 统计）
            per_question.append({
                "content": item["content"][:40],
                "label": item["type"], "predicted": "fact", "skipped": True,
            })
            continue
        pred.append(p)
        per_question.append({
            "content": item["content"][:40],
            "label": item["type"], "predicted": p, "skipped": False,
        })

    scores = type_metrics(human, pred)
    scores["dataset_size"] = len(items)
    scores["evaluated"] = len([q for q in per_question if not q["skipped"]])
    scores["skipped"] = len(skipped)
    return scores, per_question, skipped


async def record_eval_run(scores: dict, per_question: list[dict], model: str) -> tuple[str, int]:
    """版本化落库 eval_runs（eval_type='memory_type'，scores 含 model 字段）"""
    from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run
    commit = get_git_commit()
    config_snapshot = await load_rag_config()
    saved_id = await save_eval_run(
        eval_type="memory_type",
        git_commit=commit,
        config_snapshot=config_snapshot,
        scores={**scores, "model": model,
                "gate_accuracy": GATE_TYPE_ACCURACY,
                "gate_passed": gate_passed(scores)},
        per_question=per_question,
    )
    return commit, saved_id


def print_report(scores: dict, per_question: list[dict], skipped: list[dict],
                 saved_id: int, commit: str, model: str) -> None:
    print("\n" + "=" * 60)
    print(f"Memory Type Eval  [model={model}]")
    print("=" * 60)
    print(f"Dataset: {scores['dataset_size']} | Evaluated: {scores['evaluated']} "
          f"| Skipped: {scores['skipped']}")
    print(f"Accuracy: {scores['accuracy']:.4f}")
    for cls, m in scores["per_class"].items():
        print(f"  {cls:<12} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    passed = gate_passed(scores)
    print("-" * 60)
    print(f"达标线: Accuracy≥{GATE_TYPE_ACCURACY}  "
          f"→ {'✅ 达标' if passed else '❌ 未达标（类型化回退，不预设成功）'}")
    if per_question:
        print("Per-Item (first 12):")
        for q in per_question[:12]:
            tag = "SKIP" if q["skipped"] else ("ok " if q["label"] == q["predicted"] else "MIS")
            print(f"  [{tag}] 人工={q['label']:<11} 预测={q['predicted']:<11} {q['content'][:26]}")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  [{s['reason'][:30]}] {s['content'][:44]}")
    print("=" * 60)
    if saved_id:
        print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to eval_runs")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="记忆类型评测：分类模型 vs LLM 同集对比（Accuracy/P/R/F1 + 达标判定）")
    parser.add_argument("--clf-only", action="store_true",
                        help="只跑分类模型（不依赖 LLM）")
    parser.add_argument("--fixture", action="store_true",
                        help="fixture 模式：关键词启发式（确定性，不依赖模型/LLM）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    args = parser.parse_args()

    load_memory_type_dataset()
    results: list[tuple[str, dict, list[dict], list[dict]]] = []
    models_to_run = []
    if args.fixture:
        models_to_run = ["fixture"]
    elif args.clf_only:
        models_to_run = ["clf"]
    else:
        models_to_run = ["clf", "llm"]

    for model in models_to_run:
        if model == "clf":
            judge = clf_judge_factory()
        elif model == "llm":
            judge = llm_judge_factory()
        else:
            async def _fixture(item):
                return fixture_judge(item)
            judge = _fixture
        scores, per_question, skipped = await run_eval(judge=judge)
        results.append((model, scores, per_question, skipped))
        saved_id = 0
        commit = ""
        if not args.no_save:
            try:
                commit, saved_id = await record_eval_run(scores, per_question, model)
            except Exception as e:
                logger.warning("eval_runs 落库失败（不中断）: %s", e)
        print_report(scores, per_question, skipped, saved_id, commit, model)

    if len(results) == 2:
        print("\n" + "=" * 60)
        print("双方案对比（谁达标谁上）")
        print("=" * 60)
        for model, scores, _, _ in results:
            print(f"  {model:<5} Accuracy={scores['accuracy']:.4f} "
                  f"{'✅ 达标' if gate_passed(scores) else '❌ 未达标'}")
        print("-" * 60)
        print(f"达标线: Accuracy≥{GATE_TYPE_ACCURACY}；双达标取高分者；都不达标 → "
              "memory_type_mode='none'（类型化回退，不预设成功）")
        print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
