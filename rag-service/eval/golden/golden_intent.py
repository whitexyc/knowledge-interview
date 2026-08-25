"""
Golden Intent 评测脚本 — 意图分类混淆矩阵 + 版本化回归（module-043 L1）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.golden.golden_intent                    # 默认跑全量 + 落库
    python -m eval.golden.golden_intent --no-save          # 纯跑分，不写 eval_runs

指标定义:
    Accuracy      全部样本判对比例
    Precision     该类预测中判对比例（predict 正确率）
    Recall        该类真实样本中被抓回比例（漏检率 = 1 - recall）
    F1            精确率与召回率的调和平均
    Confusion Matrix  行=真实意图，列=预测意图；重点看 knowledge 行（漏检分布）

数据集:
    内嵌 INTENT_DATASET：knowledge / casual_chat / realtime 三类 +
    边界易混样本（"你们网站有什么功能"看似闲聊实为知识库——LLM 易误判）。

版本化回归:
    每次运行记录 eval_runs 表（eval_type='intent'，git_commit + rag_config 快照 +
    scores/per_question），对齐 eval/golden_retrieval.py 的落库模式。
    改 router prompt 后跑分对比，量化 intent 误判率变化。
    注意（module-056 起）：intent_classifier_enabled 默认 true，默认运行测量
    **生产 router 管线**（L4 分类器优先、失败回退 LLM+L2）；纯 LLM 测量需
    PW_INTENT_CLASSIFIER_ENABLED=false。--compare-classifier 模式的 LLM 侧
    已显式钉住 L4 关闭（见 run_compare_classifier docstring，防自污染）。

降级策略:
    - 单条分类失败 → 跳过并记录错误，其余继续
    - 数据库不可用 → 分数记录失败打印警告，评估仍完成
"""
import argparse
import asyncio
import logging
import sys

from agent.router import router_agent
from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("golden_intent")

INTENT_CLASSES = ("knowledge", "casual_chat", "realtime")

# intent 评测集：knowledge / casual_chat / realtime + 边界易混样本
# 共 100 条（2026-08-09 自造扩充：50/30/20）。
# 边界样本是 LLM 常见误判区：含"你们/网站/功能"等闲聊外壳但实为知识库问答
INTENT_DATASET: list[dict] = [
    # ---- knowledge（知识库问答，50 题，含 10 边界易混）----
    {"query": "什么是G1垃圾收集器？它的核心创新是什么？", "intent": "knowledge"},
    {"query": "CMS和G1的主要区别是什么？", "intent": "knowledge"},
    {"query": "CAP定理在分布式系统设计中如何权衡？", "intent": "knowledge"},
    {"query": "Nacos 作为注册中心和配置中心的核心原理是什么？", "intent": "knowledge"},
    {"query": "HashMap 和 ConcurrentHashMap 的底层实现有什么区别？", "intent": "knowledge"},
    {"query": "Spring Cloud 微服务架构中怎么做熔断降级？", "intent": "knowledge"},
    {"query": "JWT 和 Session 的区别是什么？各自的适用场景？", "intent": "knowledge"},
    {"query": "Redis 的持久化机制 RDB 和 AOF 有什么区别？", "intent": "knowledge"},
    {"query": "MySQL 的索引为什么选择 B+ 树而不是红黑树？", "intent": "knowledge"},
    {"query": "你们网站有什么功能？", "intent": "knowledge", "note": "边界易混：看似闲聊实为知识库"},
    {"query": "你能做什么？这个系统能帮我解决什么问题？", "intent": "knowledge", "note": "边界易混：系统能力问答"},
    {"query": "这个知识库都收录了哪些主题的内容？", "intent": "knowledge", "note": "边界易混：关于知识库自身"},
    {"query": "我该从哪篇文档开始学这个框架？", "intent": "knowledge", "note": "边界易混：导学型问题"},
    {"query": "你们最近新增了哪些文档？", "intent": "knowledge", "note": "边界易混：含时间外壳实为知识库"},
    # （以下为 2026-08-09 自造扩充，目标 100 条：knowledge 50 / casual_chat 30 / realtime 20）
    {"query": "JVM 的内存区域是怎么划分的？堆和栈有什么区别？", "intent": "knowledge"},
    {"query": "JVM 类加载机制是什么？双亲委派模型怎么工作的？", "intent": "knowledge"},
    {"query": "Full GC 频繁怎么排查和解决？", "intent": "knowledge"},
    {"query": "MySQL 事务隔离级别有哪些？默认是哪个？", "intent": "knowledge"},
    {"query": "MySQL 索引失效的场景有哪些？", "intent": "knowledge"},
    {"query": "Redis 缓存穿透、击穿、雪崩有什么区别？怎么解决？", "intent": "knowledge"},
    {"query": "Redis 哨兵和集群模式有什么区别？", "intent": "knowledge"},
    {"query": "Spring 事务失效的场景有哪些？", "intent": "knowledge"},
    {"query": "Spring Bean 的生命周期是怎样的？", "intent": "knowledge"},
    {"query": "Netty 的 Reactor 线程模型是怎么工作的？", "intent": "knowledge"},
    {"query": "TCP 三次握手为什么是三次而不是两次？", "intent": "knowledge"},
    {"query": "HTTP 和 HTTPS 的区别是什么？", "intent": "knowledge"},
    {"query": "分布式事务有哪些解决方案？两阶段提交是什么？", "intent": "knowledge"},
    {"query": "分布式锁怎么实现？Redis 和 Zookeeper 怎么选？", "intent": "knowledge"},
    {"query": "接口幂等性怎么设计？", "intent": "knowledge"},
    {"query": "消息队列怎么选型？RabbitMQ 和 Kafka 的适用场景？", "intent": "knowledge"},
    {"query": "微服务之间怎么做服务发现？", "intent": "knowledge"},
    {"query": "Docker 和虚拟机有什么区别？", "intent": "knowledge"},
    {"query": "Kubernetes 的核心组件有哪些？", "intent": "knowledge"},
    {"query": "线程池参数怎么设置？核心线程数和最大线程数怎么配？", "intent": "knowledge"},
    {"query": "ThreadLocal 的原理是什么？为什么会有内存泄漏问题？", "intent": "knowledge"},
    {"query": "CAS 和 synchronized 分别适合什么场景？", "intent": "knowledge"},
    {"query": "ConcurrentHashMap 为什么是线程安全的？", "intent": "knowledge"},
    {"query": "CMS 和 G1 的停顿表现有什么区别？", "intent": "knowledge"},
    {"query": "JWT 令牌过期怎么处理？刷新机制是什么？", "intent": "knowledge"},
    {"query": "Nacos 配置中心动态刷新的原理是什么？", "intent": "knowledge"},
    {"query": "Sentinel 限流熔断的原理是什么？", "intent": "knowledge"},
    {"query": "数据库索引为什么用 B+ 树而不是红黑树？", "intent": "knowledge"},
    {"query": "数据库连接池参数怎么配置合理？", "intent": "knowledge"},
    {"query": "HashMap 1.7 和 1.8 的实现有什么区别？", "intent": "knowledge"},
    {"query": "你知道 Java 并发编程里有哪些锁吗？", "intent": "knowledge", "note": "边界易混：闲聊外壳实为知识库问答"},
    {"query": "帮我推荐一下学习 JVM 的顺序", "intent": "knowledge", "note": "边界易混：导学型问题"},
    {"query": "这个系统能不能帮我查技术问题？", "intent": "knowledge", "note": "边界易混：系统能力问答"},
    {"query": "数据库相关的文档都有哪些？", "intent": "knowledge", "note": "边界易混：关于知识库自身"},
    {"query": "怎么才能学好分布式？", "intent": "knowledge", "note": "边界易混：导学型问题"},
    {"query": "你的知识库里有没有讲线程池的？", "intent": "knowledge", "note": "边界易混：关于知识库自身"},
    # ---- casual_chat（闲聊寒暄，30 题）----
    {"query": "你好呀", "intent": "casual_chat"},
    {"query": "在吗？", "intent": "casual_chat"},
    {"query": "早上好，今天心情不错", "intent": "casual_chat"},
    {"query": "哈哈，讲个笑话听听", "intent": "casual_chat"},
    {"query": "谢谢你，帮大忙了", "intent": "casual_chat"},
    {"query": "再见，下次再聊", "intent": "casual_chat"},
    {"query": "介绍一下你自己吧", "intent": "casual_chat"},
    {"query": "随便聊聊，今天工作好累啊", "intent": "casual_chat"},
    {"query": "你叫什么名字？", "intent": "casual_chat"},
    {"query": "嗨", "intent": "casual_chat"},
    {"query": "吃了没", "intent": "casual_chat"},
    {"query": "周末过得怎么样", "intent": "casual_chat"},
    {"query": "你现在忙吗", "intent": "casual_chat"},
    {"query": "最近在忙什么呀", "intent": "casual_chat"},
    {"query": "哈哈哈哈哈", "intent": "casual_chat"},
    {"query": "真棒，给你点个赞", "intent": "casual_chat"},
    {"query": "你也太厉害了吧", "intent": "casual_chat"},
    {"query": "辛苦了", "intent": "casual_chat"},
    {"query": "没关系，下次注意就好", "intent": "casual_chat"},
    {"query": "无聊死了，陪我聊聊天", "intent": "casual_chat"},
    {"query": "今天心情不太好", "intent": "casual_chat"},
    {"query": "猜猜我今天干嘛了", "intent": "casual_chat"},
    {"query": "我们换个话题吧", "intent": "casual_chat"},
    {"query": "拜拜", "intent": "casual_chat"},
    {"query": "晚点再聊", "intent": "casual_chat"},
    {"query": "谢谢你的帮助", "intent": "casual_chat"},
    {"query": "好的好的", "intent": "casual_chat"},
    {"query": "没问题", "intent": "casual_chat"},
    {"query": "是的没错", "intent": "casual_chat"},
    {"query": "嗯嗯，明白了", "intent": "casual_chat"},
    # ---- realtime（实时数据，20 题）----
    {"query": "现在几点了？", "intent": "realtime"},
    {"query": "今天天气怎么样？", "intent": "realtime"},
    {"query": "现在是北京时间几点？", "intent": "realtime"},
    {"query": "明天会下雨吗？", "intent": "realtime"},
    {"query": "今天是星期几？", "intent": "realtime"},
    {"query": "现在几度？", "intent": "realtime"},
    {"query": "这周末天气适合出去玩吗？", "intent": "realtime"},
    {"query": "现在北京时间几点了", "intent": "realtime"},
    {"query": "现在是上午还是下午", "intent": "realtime"},
    {"query": "今天几号", "intent": "realtime"},
    {"query": "今年是几年", "intent": "realtime"},
    {"query": "距离周末还有几天", "intent": "realtime"},
    {"query": "明天天气怎么样", "intent": "realtime"},
    {"query": "这周气温怎么样", "intent": "realtime"},
    {"query": "现在外面多少度", "intent": "realtime"},
    {"query": "今天有什么新闻", "intent": "realtime"},
    {"query": "现在股市行情怎么样", "intent": "realtime"},
    {"query": "今天人民币汇率是多少", "intent": "realtime"},
    {"query": "最近有什么热门电影", "intent": "realtime"},
    {"query": "现在流行什么", "intent": "realtime"},
]


def load_intent_dataset() -> list[dict]:
    """加载 intent 评测集，校验结构

    Returns:
        样本列表，每项含 query / intent（可含 note 标注）

    Raises:
        ValueError: 样本 < 10、query 为空或 intent 非法、三类不齐全
    """
    data = INTENT_DATASET
    if len(data) < 10:
        raise ValueError(f"intent 评测集过小：需 ≥ 10 条，当前 {len(data)}")
    for item in data:
        if not item.get("query", "").strip():
            raise ValueError(f"intent 评测集存在空 query: {item}")
        if item.get("intent") not in INTENT_CLASSES:
            raise ValueError(f"intent 非法（须 ∈ {INTENT_CLASSES}）: {item.get('query', '')[:30]}")
    missing = set(INTENT_CLASSES) - {item["intent"] for item in data}
    if missing:
        raise ValueError(f"intent 评测集缺少类别: {missing}")
    return data


def compute_confusion_matrix(labels: list[str], predictions: list[str]) -> dict:
    """计算混淆矩阵 + per-class 精确率/召回率/F1 + 整体准确率

    Args:
        labels: 每样本真实意图
        predictions: 每样本预测意图

    Returns:
        {"classes": [...], "matrix": {label: {pred: count}},
         "per_class": {cls: {"precision", "recall", "f1", "support"}},
         "accuracy": float}
        空输入 → accuracy 0.0，matrix/per_class 空 dict。
    """
    classes = sorted(set(labels) | set(predictions))
    matrix = {label: {pred: 0 for pred in classes} for label in classes}
    for label, pred in zip(labels, predictions):
        matrix[label][pred] += 1

    per_class = {}
    for cls in classes:
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in classes if other != cls)
        fn = sum(matrix[cls][pred] for pred in classes if pred != cls)
        denom_p = tp + fp
        denom_f = 2 * tp + fp + fn
        per_class[cls] = {
            "precision": round(tp / denom_p, 4) if denom_p else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            "f1": round(2 * tp / denom_f, 4) if denom_f else 0.0,
            "support": tp + fn,
        }

    n = len(labels)
    accuracy = round(
        sum(1 for l, p in zip(labels, predictions) if l == p) / n, 4
    ) if n else 0.0
    return {"classes": classes, "matrix": matrix, "per_class": per_class, "accuracy": accuracy}


async def classify_intent(query: str) -> str:
    """调用 router_agent 分类，返回 intent 标签

    失败降级由 router 内部兜底（LLM 失败保守返回 knowledge），此处不重复处理。
    注意（module-056 起）：测量的是生产 router 管线——intent_classifier_enabled
    默认 true 时优先走 L4 分类器路径；纯 LLM 测量需 PW_INTENT_CLASSIFIER_ENABLED=false。
    """
    result = await router_agent.classify(query)
    return result.get("intent", "knowledge")


async def run_eval(classifier=None, dataset=None) -> tuple[dict, list[dict], list[dict]]:
    """执行一次 intent 评估

    Args:
        classifier: 分类协程 (query) -> intent 标签；默认走 router_agent
        dataset: 评测样本列表；默认 load_intent_dataset()

    Returns:
        (scores, per_question, skipped)
        - scores: accuracy + 混淆矩阵 + per-class 指标 + 统计
        - per_question: 每题明细（label/predicted/correct）
        - skipped: 分类失败的样本记录
    """
    items = dataset if dataset is not None else load_intent_dataset()
    classify = classifier if classifier is not None else classify_intent
    per_question: list[dict] = []
    skipped: list[dict] = []

    for i, item in enumerate(items):
        query = item["query"]
        try:
            # module-072（WP-C）：短路路由测量——query_rewrite_enabled 开启时
            # 引擎在 classify 前先走短路（分诊命中 FTS 术语且非规则词 →
            # knowledge，engine.chat 同款确定性信号，零 LLM）。短路样本在
            # per_question 打 reason 标记（与 engine.chat 字符串逐字一致），
            # 供确定性后处理过滤算判对率（预期 100%）。
            shortcut = False
            if settings.query_rewrite_enabled:
                from rag.retrieval.query_rewrite import triage
                shortcut = (await triage(query) == "precise"
                            and not router_agent._rule_hits(query))
            predicted = "knowledge" if shortcut else await classify(query)
        except Exception as e:
            logger.error("[%d/%d] 分类失败: %s — %s", i + 1, len(items), query[:40], e)
            skipped.append({"query": query, "label": item["intent"], "reason": f"error: {e}"})
            continue
        per_question.append({
            "query": query,
            "label": item["intent"],
            "predicted": predicted,
            "correct": predicted == item["intent"],
            **({"reason": "分诊命中 FTS 术语，短路 knowledge"} if shortcut else {}),
        })

    conf = compute_confusion_matrix(
        [q["label"] for q in per_question],
        [q["predicted"] for q in per_question],
    )
    # 短路样本统计（module-072 WP-C）：precise AND NOT rule_hits → knowledge
    # 是纯确定性规则，判对率预期 100%（短路样本 = 非闲聊/实时特征的术语句）
    shortcut_samples = [q for q in per_question
                        if "分诊命中 FTS 术语" in q.get("reason", "")]
    shortcut_fired = len(shortcut_samples)
    scores = {
        "dataset_size": len(items),
        "evaluated": len(per_question),
        "skipped": len(skipped),
        "accuracy": conf["accuracy"],
        "confusion_matrix": conf["matrix"],
        "per_class": conf["per_class"],
        "classes": conf["classes"],
        "shortcut_fired": shortcut_fired,
        "shortcut_correct": sum(1 for q in shortcut_samples if q["correct"]),
        "shortcut_accuracy": round(
            sum(1 for q in shortcut_samples if q["correct"]) / shortcut_fired, 4
        ) if shortcut_fired else None,
    }
    return scores, per_question, skipped


async def record_eval_run(scores: dict, per_question: list[dict],
                          eval_type: str = "intent") -> tuple[str, int]:
    """版本化落库：git_commit + rag_config 快照 + eval_type 可指定

    Args:
        scores: 整体指标 dict
        per_question: 每题明细 list
        eval_type: 评估类型（module-056 起分类器侧用 'intent_classifier'）

    Returns:
        (commit, saved_id)；落库失败 saved_id=0（save_eval_run 内部已捕获并警告）
    """
    commit = get_git_commit()
    config_snapshot = await load_rag_config()
    # module-056 Review 修复：补运行时开关快照（rag_config 表无此键），
    # 使 eval_runs 可回溯本次评估的 L4 启用态——--compare-classifier 的
    # LLM 侧钉住 false 时此处如实记录 false
    config_snapshot["intent_classifier_enabled"] = str(settings.intent_classifier_enabled)
    # module-072（WP-C）：补两开关运行时快照（rag_config 表无此键），使
    # eval_runs 可回溯本次评估的 query_rewrite_enabled / contextual_rewrite_enabled
    # 启用态（短路路由 off/on 四跑可区分）
    config_snapshot["query_rewrite_enabled"] = str(settings.query_rewrite_enabled)
    config_snapshot["contextual_rewrite_enabled"] = str(settings.contextual_rewrite_enabled)
    saved_id = await save_eval_run(
        eval_type=eval_type,
        git_commit=commit,
        config_snapshot=config_snapshot,
        scores=scores,
        per_question=per_question,
    )
    return commit, saved_id


class _ClassifierAdapter:
    """包装 L4 分类器为 run_eval 可用的 (query) -> intent 协程

    决策口径与 router.classify 的 L4 路径一致：max(probs, key=probs.get)
    （"谁高选谁"）。注意：LLM 侧走 router_agent（含 L2 确定性信号确认），
    分类器侧为裸分类器——对比的是「生产 LLM+L2 管线 vs 纯分类器」。
    """

    def __init__(self, clf):
        self.clf = clf

    async def classify(self, query: str) -> str:
        probs = await self.clf.predict_proba(query)
        return max(probs, key=probs.get)


def print_comparison(llm_scores: dict, llm_per_q: list[dict],
                     clf_scores: dict, clf_per_q: list[dict]) -> None:
    """打印 LLM vs 分类器对比表（同 100 条）：Accuracy + 每类 P/R/F1 + 混淆矩阵"""
    print("\n" + "=" * 64)
    print("LLM vs L4 Classifier (same dataset)")
    print("=" * 64)
    print(f"Dataset: {llm_scores['dataset_size']} queries | "
          f"LLM Accuracy: {llm_scores['accuracy']:.4f} | "
          f"Classifier Accuracy: {clf_scores['accuracy']:.4f}")
    print("-" * 64)
    print(f"{'':<12}{'LLM P':>9}{'LLM R':>9}{'LLM F1':>9}"
          f"{'|  Clf P':>10}{'Clf R':>9}{'Clf F1':>9}")
    for cls in llm_scores["classes"]:
        lpc = llm_scores["per_class"][cls]
        cpc = clf_scores["per_class"].get(cls, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        print(f"{cls:<12}{lpc['precision']:>9.4f}{lpc['recall']:>9.4f}{lpc['f1']:>9.4f}"
              f"{'|':>3}{cpc['precision']:>9.4f}{cpc['recall']:>9.4f}{cpc['f1']:>9.4f}")
    print("-" * 64)
    print("knowledge Recall（漏检率 = 1 - recall，最高风险指标）: "
          f"LLM {llm_scores['per_class']['knowledge']['recall']:.4f} / "
          f"Classifier {clf_scores['per_class']['knowledge']['recall']:.4f}")
    print("-" * 64)
    for name, scores in (("LLM", llm_scores), ("Classifier", clf_scores)):
        print(f"\n{name} Confusion Matrix (row=label, col=predicted):")
        classes = scores["classes"]
        print(f"{'':<12}" + "".join(f"{c[:10]:>12}" for c in classes))
        for label in classes:
            row = scores["confusion_matrix"][label]
            print(f"{label:<12}" + "".join(f"{row[pred]:>12}" for pred in classes))
    mis_llm = [q for q in llm_per_q if not q["correct"]]
    mis_clf = [q for q in clf_per_q if not q["correct"]]
    if mis_llm or mis_clf:
        print("-" * 64)
        if mis_llm:
            print(f"LLM misclassified ({len(mis_llm)}):")
            for q in mis_llm:
                print(f"  {q['label']:<12} -> {q['predicted']:<12} | {q['query'][:40]}")
        if mis_clf:
            print(f"Classifier misclassified ({len(mis_clf)}):")
            for q in mis_clf:
                print(f"  {q['label']:<12} -> {q['predicted']:<12} | {q['query'][:40]}")
    print("=" * 64)
    print()


async def run_compare_classifier(no_save: bool) -> None:
    """module-056 WP-3：golden_intent 真实模式 LLM vs 分类器同 100 条对比

    - LLM 侧：router_agent.classify（生产管线，含 L2 确定性信号确认）
    - 分类器侧：IntentClassifier（bge-m3 + 逻辑回归，max 概率决策）
    - 落库：LLM 侧 eval_type='intent'；分类器侧 eval_type='intent_classifier'
    - LLM 不可用（全部降级为保守 knowledge，闲聊/实时 0 命中）→ 记声明，
      仅分类器单侧落库（如实声明）
    - 分类器模型缺失/加载失败 → 打印跳过，LLM 单侧完成

    防自污染（module-056 Review 修复）：module-056 起 intent_classifier_enabled
    默认 true——直接跑会令「LLM 侧」静默走 router 的 L4 分类器路径，「LLM vs
    分类器」退化为「分类器 vs 分类器」（双 1.0000 恒成立、对比失去意义）。
    故 LLM 侧运行前显式钉住 settings.intent_classifier_enabled=False，并重置
    router_agent 已缓存的分类器状态（_classifier_tried/_intent_classifier，
    防同一进程内早前按启用态加载过模型）；分类器侧使用独立 IntentClassifier
    实例，不受钉住影响；finally 恢复原开关值与缓存。
    """
    from agent.intent_classifier import IntentClassifier

    dataset = load_intent_dataset()

    # ── 钉住 LLM 侧：禁 L4 分类器 + 重置 router_agent 缓存（防自污染）──
    orig_enabled = settings.intent_classifier_enabled
    orig_clf = router_agent._intent_classifier
    orig_tried = router_agent._classifier_tried
    settings.intent_classifier_enabled = False
    router_agent._intent_classifier = None
    router_agent._classifier_tried = False
    try:
        print("== LLM 侧（router_agent，含 L2 确定性信号确认；L4 已钉住禁用）==")
        llm_scores, llm_per_q, llm_skipped = await run_eval(dataset=dataset)
        print_report(llm_scores, llm_per_q, llm_skipped, 0, "")

        clf = IntentClassifier()
        if not await clf.load():
            print("\n[skip] L4 分类器模型不可用（load 失败），LLM 单侧完成。")
            if not no_save:
                commit, saved_id = await record_eval_run(llm_scores, llm_per_q)
                print(f"LLM 结果已保存 eval_runs (id={saved_id}, commit={commit[:8]})")
            return

        print("\n== 分类器侧（bge-m3 + LogisticRegression，裸分类器）==")
        adapter = _ClassifierAdapter(clf)
        clf_scores, clf_per_q, clf_skipped = await run_eval(
            classifier=adapter.classify, dataset=dataset)
        print_report(clf_scores, clf_per_q, clf_skipped, 0, "")

        llm_non_kb_hits = sum(
            1 for q in llm_per_q if q["label"] in ("casual_chat", "realtime") and q["correct"])
        if llm_non_kb_hits == 0:
            print("\n[skip-decl] LLM 侧闲聊/实时 0 命中——LLM 不可用或链路降级为"
                  "全量保守 knowledge，LLM 侧结论不可信，记 skipped，仅分类器单侧落库。")
            if not no_save:
                commit, saved_id = await record_eval_run(
                    clf_scores, clf_per_q, eval_type="intent_classifier")
                print(f"分类器结果已保存 eval_runs (id={saved_id}, commit={commit[:8]})")
            return

        print_comparison(llm_scores, llm_per_q, clf_scores, clf_per_q)
        if not no_save:
            commit, llm_id = await record_eval_run(llm_scores, llm_per_q)
            _, clf_id = await record_eval_run(
                clf_scores, clf_per_q, eval_type="intent_classifier")
            print(f"Saved to eval_runs: LLM id={llm_id} (eval_type='intent') | "
                  f"Classifier id={clf_id} (eval_type='intent_classifier'), "
                  f"commit={commit[:8]}")
    finally:
        settings.intent_classifier_enabled = orig_enabled
        router_agent._intent_classifier = orig_clf
        router_agent._classifier_tried = orig_tried


def print_report(scores: dict, per_question: list[dict], skipped: list[dict], saved_id: int, commit: str) -> None:
    """打印评估报告到控制台：混淆矩阵 + per-class 指标 + 误判明细"""
    classes = scores["classes"]
    print("\n" + "=" * 60)
    print("Golden Intent Eval")
    print("=" * 60)
    print(f"Dataset: {scores['dataset_size']} queries | Evaluated: {scores['evaluated']} | Skipped: {scores['skipped']}")
    print("-" * 60)
    print(f"Accuracy: {scores['accuracy']:.4f}")
    if scores.get("shortcut_fired"):
        print(f"短路路由（module-072 WP-C，确定性零 LLM）: 触发 "
              f"{scores['shortcut_fired']} 条，判对 "
              f"{scores['shortcut_correct']}/{scores['shortcut_fired']} "
              f"（判对率 {scores['shortcut_accuracy']:.4f}）")
    print("-" * 60)
    print("Confusion Matrix (row=label, col=predicted):")
    print(f"{'':<12}" + "".join(f"{c[:10]:>12}" for c in classes))
    for label in classes:
        row = scores["confusion_matrix"][label]
        print(f"{label:<12}" + "".join(f"{row[pred]:>12}" for pred in classes))
    print("-" * 60)
    print("Per-Class Precision/Recall/F1:")
    for cls in classes:
        pc = scores["per_class"][cls]
        print(f"  {cls:<12} precision={pc['precision']:.4f} recall={pc['recall']:.4f} "
              f"f1={pc['f1']:.4f} support={pc['support']}")
    mis = [q for q in per_question if not q["correct"]]
    if mis:
        print("-" * 60)
        print(f"Misclassified ({len(mis)}):")
        for q in mis:
            print(f"  {q['label']:<12} -> {q['predicted']:<12} | {q['query'][:40]}")
    if skipped:
        print("-" * 60)
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  [{s['reason'][:30]}] {s['query'][:50]}")
    print("=" * 60)
    if saved_id:
        print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to eval_runs")
    print()


async def main() -> None:
    """评测脚本入口"""
    parser = argparse.ArgumentParser(description="Golden intent 评测：混淆矩阵 + 版本化回归")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    parser.add_argument("--compare-classifier", action="store_true",
                        help="LLM vs L4 分类器同 100 条对比（module-056 WP-3）")
    args = parser.parse_args()

    if args.compare_classifier:
        await run_compare_classifier(no_save=args.no_save)
        return

    load_intent_dataset()
    scores, per_question, skipped = await run_eval()

    saved_id = 0
    commit = ""
    if not args.no_save:
        commit, saved_id = await record_eval_run(scores, per_question)
    print_report(scores, per_question, skipped, saved_id, commit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
