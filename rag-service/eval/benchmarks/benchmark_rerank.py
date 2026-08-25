"""
Rerank 截断基准测试 — 250/500/1000/2000 字符截断的分数与耗时对比（ADR-0004 TODO 验证）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.benchmark_rerank                       # 默认 500 字符、2 pair
    python -m eval.benchmarks.benchmark_rerank --max-chars 250       # 250 字符截断
    python -m eval.benchmarks.benchmark_rerank --pairs 6             # 2 pair 之外再加 6 pair
    python -m eval.benchmarks.benchmark_rerank --max-chars 1000 --pairs 6

输出:
    - 每对 (query, doc) 的 rerank 分数
    - 总耗时 / 平均每对耗时（先预热 1 对，排除模型加载与首次推理缓存）

数据:
    文档用知识库代表性内容构造（借 golden 集真实主题：G1 GC / Kafka / AQS），
    每篇核心内容 1500 字左右 + 填充段补到 3200+ 字符（超过 2000 截断上限，
    使截断真正生效），模拟知识库中"无 ## 标题整篇入库"的超长父块——
    即 ADR-0004 决策 2 背景里"父块可达数万字符"的代表性子块。

决策规则（对齐 ADR-0004 TODO）:
    250 分数 ≥ 0.98（相对 500 无损）且耗时显著下降 → 建议采纳 250；
    否则保持 500。结果如实记录，两种可能都接受。
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark_rerank")

# 本地模型目录（与 rag/reranker.py 同源：ai_service/models/bge-reranker-v2-m3）
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "bge-reranker-v2-m3"

# 基准 query：借 golden 集真实问题（eval/golden.json）
QUERIES = {
    "g1": "什么是G1垃圾收集器？它的核心创新是什么？",
    "kafka": "Kafka的ISR机制是如何保证消息可靠性的？",
    "aqs": "AQS (AbstractQueuedSynchronizer) 的工作原理是什么？ReentrantLock如何基于AQS实现？",
}

# 主题核心内容（真实技术要点浓缩，代表知识库文档"标题+开头段落"信号区）
_DOC_CORES = {
    "g1": {
        "doc1": """## G1垃圾收集器的Region分区机制与MixedGC全流程

G1（Garbage First）垃圾收集器是 JDK 9 之后的默认垃圾收集器，核心设计是把堆划分为大小相等的 Region 区域（默认 2048 个，每个 Region 1MB~32MB），不再像 CMS 那样物理连续划分年轻代和老年代。

Region 分区机制：G1 将整个堆分成多个 Region，每个 Region 都可以独立扮演 Eden、Survivor 或 Old 的角色，角色可以动态切换。这样收集器在回收时只处理部分 Region，而不是扫描整个堆，实现"只回收部分区域"的增量回收目标。G1 通过 Remembered Set（RSet）记录每个 Region 中被其他 Region 对象引用的入口，回收一个 Region 时只需扫描其 RSet 而非全堆。

MixedGC 全流程：G1 的回收分为 YoungGC 和 MixedGC 两种。YoungGC 只回收年轻代 Region，速度快；当老年代占用达到 IHOP（Initiating Heap Occupancy Percent，默认 45%）时触发并发标记，标记完成后进入 MixedGC——同时回收年轻代和部分高收益的老年代 Region。

G1 的核心创新：1）Region 分区 + RSet 使回收粒度从"整堆"降到"区域级"，停顿时间可预测（通过 -XX:MaxGCPauseMillis 设定目标，默认 200ms）；2）回收价值优先（Garbage First）——优先回收垃圾最多、回收收益最大的 Region；3）并发标记 + 写屏障（SATB 快照）实现与用户线程并发执行；4）无碎片——Region 间的对象移动通过"复制"完成，避免 CMS 的标记-清除碎片问题。""",
        "doc2": """## G1垃圾收集器的调优参数与踩坑记录

本篇记录 G1 在实际项目中的调优经验，包括常用参数与常见问题排查。

常用参数：-XX:+UseG1GC 启用 G1；-XX:MaxGCPauseMillis=200 设定停顿目标；-XX:G1HeapRegionSize=8m 指定 Region 大小；-XX:InitiatingHeapOccupancyPercent=45 控制并发标记触发阈值；-XX:G1NewSizePercent 与 -XX:G1MaxNewSizePercent 控制年轻代占比范围。

调优经验：1）大对象（Humongous Object）直接分配在连续的大 Region 中，超过 Region 大小一半的对象不参与正常的复制回收，频繁分配大对象会导致提前触发并发标记，必要时增大 Region 大小；2）RSet 占用内存约 5%-10% 的堆空间，Region 越小 RSet 越精细但内存开销越大；3）MixedGC 的候选 Region 集合（CSet）选择会动态调整，可通过 -XX:G1MixedGCLiveThresholdPercent 控制存活对象占比过高的 Region 不进入 CSet。

踩坑记录：曾遇到"to-space exhausted"（晋升空间不足）错误，原因是 Survivor 区复制失败，解决方式是增大 -XX:SurvivorRatio 或减少晋升对象；也曾遇到 Full GC 频繁，排查发现是动态 IHOP 计算偏低导致并发标记过早触发，调高 IHOP 阈值后缓解。""",
    },
    "kafka": {
        "doc1": """## Kafka消息可靠性与高吞吐设计

Kafka 的可靠性核心是 ISR（In-Sync Replicas）机制。每个 Partition 有多个副本，其中 Leader 负责读写，Follower 从 Leader 拉取消息。ISR 是"与 Leader 保持同步"的副本集合——Follower 通过拉取落后于 Leader 的消息，超过 replica.lag.time.max.ms（默认 30s）未同步即被踢出 ISR。

生产者可靠性：acks 参数控制确认级别——acks=0 发完即走不确认，可能丢消息；acks=1 Leader 写入本地日志即确认，Leader 宕机可能丢；acks=all（-1）要求 ISR 中所有副本都写入才确认，配合 min.insync.replicas=2 可保证至少一个副本确认后才返回。

消费端可靠性：消费者处理完消息后手动提交 offset，避免消息丢失；开启 enable.auto.commit=false 后由业务控制提交时机，处理成功再提交，防止"先提交后处理失败导致丢消息"。

Kafka 高吞吐的三个底层机制：1）顺序写——消息追加写入 Partition 日志文件末尾，磁盘顺序写速度接近内存；2）页缓存（Page Cache）——读写都经过 OS 页缓存，避免用户态/内核态拷贝，消费时直接从缓存读；3）零拷贝（Zero-Copy）——使用 sendfile 系统调用，数据从页缓存直接发送到网卡，跳过用户态拷贝。""",
        "doc2": """## Kafka分区机制、消费者组与Rebalance

Partition 机制：每个 Topic 分为多个 Partition，消息按 Key 哈希取模路由到具体 Partition，同一 Partition 内消息有序（FIFO），不同 Partition 之间无序。Partition 是 Kafka 并行度的基础——生产端可并行写入多个 Partition，消费端每个 Partition 同时只被组内一个消费者消费，分区数决定了消费并行度的上限。

Consumer Group 与 Rebalance：同一组的消费者共同消费一个 Topic，每个 Partition 只会被组内一个消费者消费。当消费者加入/退出或分区数变化时触发 Rebalance（再平衡），重新分配 Partition。旧版 Rebalance 采用"停止世界"策略（全部消费者停止消费重新分配），会引发一段时间的消费停滞；新版（KIP-429 增量协调）实现了增量 Rebalance，只迁移受影响的分区，减少停顿。

分区分配策略：RangeAssignor 按分区号范围分配（可能产生倾斜），RoundRobinAssignor 轮询分配（更均匀），StickyAssignor 在 Rebalance 时尽量保持已有分配不变（减少分区迁移）。实际生产中一般推荐 StickyAssignor。""",
    },
    "aqs": {
        "doc1": """## AQS抽象队列同步器与ReentrantLock实现原理

AQS（AbstractQueuedSynchronizer）是 Java 并发包的基石，ReentrantLock、Semaphore、CountDownLatch、ReentrantReadWriteLock 等同步器都基于它实现。AQS 的核心是一个 volatile int state 字段 + 一个 CLH 变体的 FIFO 等待队列。

state 字段：表示同步状态，子类通过 getState/setState/compareAndSetState 操作它。ReentrantLock 中 state 表示"锁被重入的次数"——0 表示无线程持有，n 表示被某线程重入 n 次；Semaphore 中 state 表示剩余许可数。

CLH 等待队列：获取同步状态失败的线程被封装为 Node 节点挂入队尾，通过自旋 + park 阻塞，前驱节点释放后唤醒后继。队列用 CAS 保证入队原子性。

ReentrantLock 获取锁流程（公平/非公平）：非公平锁在 tryAcquire 时先直接 CAS 抢锁（不管队列是否有人排队），抢失败才进队列——吞吐优先；公平锁则严格按 FIFO，队列中有前驱就排队——公平优先。释放锁：state 减到 0 时唤醒队首后继节点（signal）。

独占与共享：AQS 支持独占模式（acquire/release，如 ReentrantLock）与共享模式（acquireShared/releaseShared，如 Semaphore/CountDownLatch），内部通过 tryAcquire/tryRelease（独占）和 tryAcquireShared/tryReleaseShared（共享）模板方法由子类实现。""",
        "doc2": """## ReentrantLock实战：Condition、可中断锁与性能对比

本篇记录 ReentrantLock 在生产代码中的使用经验与与 synchronized 的对比。

Condition：ReentrantLock 通过 newCondition() 创建多个条件队列，await() 释放锁并等待、signal() 唤醒等待线程。典型场景是生产者-消费者模型——ArrayBlockingQueue 内部就用两个 Condition（notEmpty/notFull）实现。Condition 相比 synchronized 的 wait/notify 优势是支持多个独立条件队列，避免"唤醒错线程"问题。

可中断与超时：lockInterruptibly() 允许线程在等待锁时响应中断；tryLock(timeout, unit) 支持限时抢锁，超过时间返回 false——避免无限期等待，这是 synchronized 不具备的能力。

公平性测试：公平锁在竞争激烈时吞吐约为非公平锁的 60%-70%，但线程等待时间更均匀，无"饿死"风险。实际业务中默认用非公平锁（吞吐优先），对等待公平性敏感的场景（如任务调度）才用公平锁。

AQS 与 synchronized 底层对比：synchronized 基于监视器锁（Monitor），JDK 6 之后有偏向锁→轻量级锁→重量级锁的升级路径，由 JVM 管理；AQS 是纯 Java 实现 + CAS + park/unpark，可控性更强。两者在无竞争时性能接近，有竞争时 AQS 更可预测。""",
    },
}

# 填充段：模拟超长父块"中后段"（信号稀疏的正文/案例部分），补到超过 2000 字符
_FILLER = """
在后续的工程实践中，我们对上述机制进行了多轮验证与复盘。团队内部的技术评审会上，主讲人从原理层到实现层逐层拆解，与会者针对边界场景提出了若干追问，例如极端并发下的退化行为、资源受限环境中的水位线设置，以及监控指标与告警阈值的联动方案。这些问题最终沉淀为一份内部 FAQ，并被纳入下一版本的代码走查清单。值得注意的是，在压测环境中观察到延迟毛刺，经排查确认与操作系统层面的调度策略相关，而非应用逻辑本身的问题。为此我们引入了更细粒度的观测点，将关键路径上的耗时分布记录到日志与指标系统中，供后续容量规划参考。此外，文档末尾补充了与主流开源实现的对照表格，并列出社区中讨论度较高的十个相关议题，作为延伸阅读材料，帮助读者建立更完整的知识图谱。
"""


def load_dataset() -> dict[str, list[str]]:
    """构造基准数据集：query → 文档内容列表（每篇 3200+ 字符）

    Returns:
        {"g1": [doc1, doc2], "kafka": [...], "aqs": [...]}
    """
    dataset = {}
    for topic, docs in _DOC_CORES.items():
        contents = []
        for doc in docs.values():
            text = doc + _FILLER
            while len(text) < 3200:
                text += _FILLER
            contents.append(text)
        dataset[topic] = contents
    return dataset


def build_pairs(max_chars: int, pair_count: int) -> tuple[list[tuple[str, str]], list[str]]:
    """按截断值构造 (query, doc[:max_chars]) 对

    Args:
        max_chars: 截断字符数
        pair_count: 期望 pair 数（2 或 6）

    Returns:
        (pairs, labels)：labels 与 pairs 一一对应，形如 "g1/doc1"
    """
    dataset = load_dataset()
    pairs: list[tuple[str, str]] = []
    labels: list[str] = []
    for topic in ("g1", "kafka", "aqs"):
        query = QUERIES[topic]
        docs = dataset[topic]
        for i, content in enumerate(docs):
            if len(pairs) >= pair_count:
                return pairs, labels
            pairs.append((query, content[:max_chars]))
            labels.append(f"{topic}/doc{i + 1}")
    return pairs, labels


def run_benchmark(max_chars: int, pairs_2: bool, pairs_6: bool,
                  model: Optional[CrossEncoder] = None,
                  warmup: bool = True) -> None:
    """执行截断基准：2 pair 与 6 pair 各计时打分

    Args:
        max_chars: 截断字符数
        pairs_2: 是否跑 2 pair
        pairs_6: 是否跑 6 pair
        model: 已加载模型（sweep 模式复用，避免重复加载 2.17GB）
        warmup: 是否预热 1 对（sweep 模式下后续档位跳过）
    """
    if model is None:
        model = CrossEncoder(str(MODEL_DIR))
        logger.info("模型加载完成: %s", MODEL_DIR.name)

    if warmup:
        # 预热 1 对（排除模型首次推理的缓存/线程池初始化影响）
        warmup_pairs = [("预热query", "预热文档内容" * 40)]
        model.predict(warmup_pairs)

    print("\n" + "=" * 64)
    print(f"Rerank Truncation Benchmark — max_chars={max_chars}")
    print("=" * 64)

    for pair_count in (2, 6):
        if pair_count == 2 and not pairs_2:
            continue
        if pair_count == 6 and not pairs_6:
            continue
        pairs, labels = build_pairs(max_chars, pair_count)
        chars = [len(doc) for _, doc in pairs]

        start = time.perf_counter()
        scores = model.predict(pairs)
        elapsed = time.perf_counter() - start

        print(f"\n-- {pair_count} pair (截断后实际 {chars[0]} 字符/pair) --")
        for label, doc, score in zip(labels, pairs, scores):
            print(f"  {label:<12} score={float(score):.6f}  len={len(doc[1])}")
        print(f"  耗时: {elapsed:.3f}s | 平均 {elapsed / pair_count:.3f}s/pair")


def main() -> None:
    """基准测试入口"""
    parser = argparse.ArgumentParser(description="Rerank 截断基准：分数与耗时（ADR-0004 TODO 验证）")
    parser.add_argument("--max-chars", type=int, default=500,
                        choices=[50, 75, 100, 150, 200, 250, 500, 1000, 2000],
                        help="截断字符数（默认 500）")
    parser.add_argument("--pairs", type=int, default=2,
                        choices=[2, 6], help="6 pair 规模（默认仅 2 pair）")
    parser.add_argument("--sweep", action="store_true",
                        help="拐点扫描：一次加载模型跑 2000/1000/500/250/200/150/100/75/50 全档（6 pair）")
    args = parser.parse_args()

    if args.sweep:
        logger.info("sweep 模式：一次加载模型，全档位 6 pair")
        model = CrossEncoder(str(MODEL_DIR))
        model.predict([("预热query", "预热文档内容" * 40)])
        for mc in (2000, 1000, 500, 250, 200, 150, 100, 75, 50):
            run_benchmark(mc, pairs_2=False, pairs_6=True, model=model, warmup=False)
        return

    run_benchmark(args.max_chars, pairs_2=True, pairs_6=(args.pairs == 6))


if __name__ == "__main__":
    main()
