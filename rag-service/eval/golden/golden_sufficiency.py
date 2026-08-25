"""
Golden Sufficiency 评测脚本 — 充分性判断 Accuracy/P/R/F1 + 混淆矩阵 + 版本化回归（module-044 层 0）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.golden.golden_sufficiency                 # 真实模式：reflector.check_sufficiency 判每条样本 + 落库
    python -m eval.golden.golden_sufficiency --fixture       # fixture 模式：启发式判断器，不依赖 LLM/DB（管线演示）
    python -m eval.golden.golden_sufficiency --no-save       # 纯跑分，不写 eval_runs

指标定义（对齐 ADR-0005 层 0）:
    Accuracy      全部样本判对比例
    Precision     该类预测中判对比例
    Recall        该类真实样本中被抓回比例
    F1            精确率与召回率的调和平均
    Confusion Matrix  行=真实充分性，列=预测充分性
    **重点看 insufficient 类的 Recall**：漏判"不充分"（把不充分判成充分）会导致
    基于无关文档硬答，最致命——报告里单独大字标出。

数据集:
    内嵌 SUFFICIENCY_DATASET：问题借 golden 集真实题目（eval/golden.json），
    注入代表性文档（相关文档/不相关文档），人工标注充分/不充分两类，共 100 条
    （充分 50 + 不充分 50；2026-08-09 自造扩充至 100）；每条 2 篇文档——兼容
    层 1 数量闸门（文档数 < 2 → 直接判不充分，零 LLM），确保真实模式测到
    LLM 判断而非被数量闸门短路。fixture 模式不依赖 DB 检索与 LLM。

版本化回归:
    每次运行记录 eval_runs 表（eval_type='sufficiency'，git_commit + rag_config
    快照 + scores/per_question），对齐 eval/golden_retrieval.py 的落库模式。
    改 check_sufficiency（层 1-3）后跑分对比，量化充分性判断误判率变化。

降级策略:
    - 单条判断失败 → 跳过并记录错误，其余继续
    - 数据库不可用 → 分数记录失败打印警告，评估仍完成
    - reflector.check_sufficiency 内部失败默认充分（现有降级哲学，不误杀）
"""
import argparse
import asyncio
import logging
import sys

from agent.reflector import reflector
from eval.golden.golden_retrieval import get_git_commit, load_rag_config, save_eval_run
from eval.golden.golden_intent import compute_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("golden_sufficiency")

SUFFICIENCY_CLASSES = ("sufficient", "insufficient")

# 充分性标注集：问题借 golden 集真实题目，文档为代表性内容（相关/不相关），
# 人工标注充分性。keywords 为问题核心术语，供 fixture 启发式判断器使用。
SUFFICIENCY_DATASET: list[dict] = [
    # ---- 充分（相关文档能回答问题，6 条）----
    {
        "question": "什么是G1垃圾收集器？它的核心创新是什么？",
        "documents": [{
            "title": "1-G1垃圾收集器的Region分区机制与MixedGC全流程_2026-07-11",
            "content": "G1（Garbage First）垃圾收集器是 JDK 9 之后的默认垃圾收集器。"
                       "核心设计是把堆划分为大小相等的 Region 区域，每个 Region 可独立扮演 Eden、"
                       "Survivor 或 Old 角色，实现增量回收。G1 的核心创新：1）Region 分区 + "
                       "Remembered Set 使回收粒度降到区域级，停顿时间可预测；2）回收价值优先，"
                       "优先回收垃圾最多的 Region；3）并发标记 + SATB 写屏障与用户线程并发执行；"
                       "4）复制式回收避免 CMS 的碎片问题。MixedGC 在并发标记完成后同时回收"
                       "年轻代与高收益的老年代 Region。",
        }, {
            "title": "1-G1垃圾收集器的Region分区机制与MixedGC全流程_2026-07-11 > 板块3 > 调优参数",
            "content": "G1 常用调优参数：-XX:MaxGCPauseMillis=200 设定停顿目标；"
                       "-XX:G1HeapRegionSize=8m 指定 Region 大小；"
                       "-XX:InitiatingHeapOccupancyPercent=45 控制并发标记触发阈值。"
                       "调优经验：大对象（Humongous）直接分配在连续 Region，频繁分配会提前触发"
                       "并发标记；RSet 占用约 5%-10% 堆空间，Region 越小 RSet 越精细但开销越大。",
        }],
        "sufficient": True,
        "keywords": ["G1", "Region", "MixedGC"],
        "category": "java_gc",
    },
    {
        "question": "Kafka的ISR机制是如何保证消息可靠性的？",
        "documents": [{
            "title": "5-Kafka消息可靠性与高吞吐设计_2026-07-15",
            "content": "Kafka 可靠性核心是 ISR（In-Sync Replicas）机制：每个 Partition 有多个副本，"
                       "Leader 负责读写，Follower 拉取同步，超过 replica.lag.time.max.ms 未同步即被踢出 ISR。"
                       "生产者端 acks=all 配合 min.insync.replicas=2 保证 ISR 中至少一个副本确认；"
                       "消费端手动提交 offset 防丢消息。高吞吐依赖顺序写、页缓存与零拷贝。",
        }, {
            "title": "5-Kafka消息可靠性与高吞吐设计_2026-07-15 > 板块2 > 生产端配置",
            "content": "生产端可靠性配置：acks=0 发完即走可能丢消息；acks=1 Leader 写入即确认，"
                       "Leader 宕机可能丢；acks=all 要求 ISR 所有副本写入，配合 min.insync.replicas=2 "
                       "与 retries 重试参数，实现不丢消息。幂等性（enable.idempotence=true）配合"
                       "事务机制进一步防止重复写入。",
        }],
        "sufficient": True,
        "keywords": ["ISR", "副本", "acks"],
        "category": "kafka",
    },
    {
        "question": "AQS (AbstractQueuedSynchronizer) 的工作原理是什么？ReentrantLock如何基于AQS实现？",
        "documents": [{
            "title": "15-AQS抽象队列同步器与ReentrantLock实现原理_2026-07-26",
            "content": "AQS 是 Java 并发包基石，核心是 volatile int state 字段 + CLH 变体 FIFO 等待队列。"
                       "state 表示同步状态，ReentrantLock 中表示重入次数；获取失败的线程封装为 Node "
                       "入队，通过自旋 + park 阻塞，前驱释放后唤醒后继。ReentrantLock 非公平锁先 CAS "
                       "抢锁再排队，公平锁严格 FIFO；释放时 state 减到 0 唤醒队首。AQS 支持独占"
                       "（acquire/release）与共享（acquireShared/releaseShared）两种模式。",
        }, {
            "title": "15-AQS抽象队列同步器与ReentrantLock实现原理_2026-07-26 > 板块2 > Condition与实战",
            "content": "ReentrantLock 通过 newCondition() 创建多个条件队列，await() 释放锁并等待、"
                       "signal() 唤醒等待线程，典型场景是 ArrayBlockingQueue 的 notEmpty/notFull "
                       "两个 Condition。lockInterruptibly() 支持响应中断，tryLock(timeout) 支持限时"
                       "抢锁——这是 synchronized 不具备的能力。",
        }],
        "sufficient": True,
        "keywords": ["AQS", "ReentrantLock", "state"],
        "category": "java_concurrency",
    },
    {
        "question": "volatile关键字的作用和实现原理是什么？",
        "documents": [{
            "title": "16-volatile与Java内存模型JMM_2026-07-27",
            "content": "volatile 是 Java 提供的轻量级同步机制，两大作用：1）可见性——写 volatile 变量"
                       "会插入 StoreStore/StoreLoad 内存屏障，强制刷新到主内存，读时从主内存读取，"
                       "避免线程工作内存缓存导致的值过期；2）有序性——禁止指令重排序，防止单例"
                       "双重检查锁中对象未初始化完成即被发布。volatile 不保证原子性，复合操作"
                       "（如 i++）仍需 synchronized 或原子类。",
        }, {
            "title": "16-volatile与Java内存模型JMM_2026-07-27 > 板块2 > 双检锁示例",
            "content": "双重检查锁（Double-Checked Locking）单例：外层判空避免无谓加锁，内层"
                       "synchronized 保证只实例化一次，instance 字段声明为 volatile 防止指令重排"
                       "导致返回未完成初始化的对象。这是 volatile 有序性语义的经典应用场景。",
        }],
        "sufficient": True,
        "keywords": ["volatile", "可见性", "内存屏障"],
        "category": "java_concurrency",
    },
    {
        "question": "Redis的持久化方式RDB和AOF有什么区别？如何选择？",
        "documents": [{
            "title": "10-Redis持久化机制_2026-07-20",
            "content": "Redis 两种持久化：RDB 定时生成全量快照（fork 子进程写临时文件），恢复快但可能"
                       "丢最后一次快照后的数据；AOF 追加写命令日志，默认 everysec 每秒 fsync，"
                       "最多丢 1 秒数据，文件会不断增大需 AOF 重写压缩。选择：能接受少量丢数据、"
                       "看重恢复速度用 RDB；数据安全要求高用 AOF；生产一般 RDB + AOF 组合。",
        }, {
            "title": "10-Redis持久化机制_2026-07-20 > 板块2 > 混合持久化",
            "content": "Redis 4.0 引入混合持久化：AOF 重写时把历史数据以 RDB 格式写入 AOF 文件头部，"
                       "后续增量用命令追加——兼顾 RDB 的恢复速度与 AOF 的数据安全，是生产环境的"
                       "推荐配置（aof-use-rdb-preamble yes）。",
        }],
        "sufficient": True,
        "keywords": ["RDB", "AOF", "持久化"],
        "category": "comprehensive",
    },
    {
        "question": "synchronized的底层实现原理是什么？锁升级过程是怎样的？",
        "documents": [{
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23",
            "content": "synchronized 底层基于对象头 Mark Word + Monitor 监视器锁。锁升级路径："
                       "无锁 → 偏向锁（单线程重入，CAS 记录线程 ID）→ 轻量级锁（竞争时自旋 CAS "
                       "抢锁）→ 重量级锁（阻塞挂起，依赖操作系统互斥量）。JDK 6 之后的优化使"
                       "无竞争场景开销极低。锁升级是单向不可逆的，避免频繁竞争可减少升级到重量级锁。",
        }, {
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23 > 板块3 > 锁对比",
            "content": "synchronized 与 ReentrantLock 对比：synchronized 由 JVM 管理锁升级，写法简单"
                       "自动释放；ReentrantLock 支持可中断、限时、公平性与多 Condition，灵活但需"
                       "手动释放。无竞争时两者性能接近，有竞争时 AQS 表现更可预测。",
        }],
        "sufficient": True,
        "keywords": ["synchronized", "Monitor", "锁升级"],
        "category": "java_concurrency",
    },
    # ---- 充分扩充（2026-08-09 自造，目标 100 条：充分 50 / 不充分 50）----
    {
        "question": "ZGC的特点和适用场景是什么？",
        "documents": [{
            "title": "2-ZGC超低停顿垃圾收集器原理_2026-07-12",
            "content": "ZGC 是追求超低停顿的垃圾收集器，目标停顿时间不随堆大小增长（10ms 级别）。"
                       "核心机制：着色指针（Colored Pointers）把 GC 状态编码进指针高位，"
                       "读屏障（Load Barrier）在对象访问时修正指针；回收阶段可与用户线程并发执行，"
                       "堆分区可动态扩展。适用场景：大堆（几十 GB 以上）、低延迟敏感、"
                       "需要可预测停顿的服务。JDK 15+ 生产可用。",
        }, {
            "title": "2-ZGC超低停顿垃圾收集器原理_2026-07-12 > 板块3 > 与G1对比",
            "content": "ZGC 与 G1 对比：G1 通过 Region 分区 + RSet 把停顿降到可预测但随堆增大，"
                       "ZGC 通过着色指针 + 读屏障让停顿基本恒定；G1 并发标记阶段仍需 Stop-The-World，"
                       "ZGC 大部分阶段并发。代价是 ZGC 内存占用略高（指针染色位）且 CPU 开销更大，"
                       "小堆场景优势不明显。",
        }],
        "sufficient": True,
        "keywords": ["ZGC", "着色指针", "读屏障"],
        "category": "java_gc",
    },
    {
        "question": "CMS垃圾收集器的原理和缺陷是什么？",
        "documents": [{
            "title": "3-CMS垃圾收集器原理与缺陷分析_2026-07-13",
            "content": "CMS（Concurrent Mark Sweep）以最短回收停顿为目标，四个阶段：初始标记（STW）、"
                       "并发标记、重新标记（STW）、并发清除。全程与用户线程并发执行，停顿低。"
                       "主要缺陷：1）并发模式失败（Concurrent Mode Failure）退化为 Serial Old 全停顿；"
                       "2）标记-清除产生碎片，大对象分配失败触发 Full GC；3）占用 CPU 资源；"
                       "4）浮动垃圾无法在本次清除。JDK 9 起被 G1 取代，JDK 14 移除。",
        }, {
            "title": "3-CMS垃圾收集器原理与缺陷分析_2026-07-13 > 板块2 > 参数调优",
            "content": "CMS 常用参数：-XX:+UseConcMarkSweepGC 启用；-XX:CMSInitiatingOccupancyFraction=70 "
                       "控制并发标记触发阈值（预留空间防 Concurrent Mode Failure）；"
                       "-XX:+UseCMSCompactAtFullCollection 在 Full GC 时压缩碎片。"
                       "生产上 CMS 触发频繁是常见故障点，通常调整触发比例或迁移 G1。",
        }],
        "sufficient": True,
        "keywords": ["CMS", "并发标记", "碎片"],
        "category": "java_gc",
    },
    {
        "question": "ThreadPoolExecutor的核心参数和工作流程是什么？",
        "documents": [{
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16",
            "content": "ThreadPoolExecutor 六大核心参数：corePoolSize（核心线程数，常驻）、"
                       "maximumPoolSize（最大线程数）、workQueue（任务队列）、keepAliveTime（"
                       "非核心线程空闲存活）、threadFactory（线程工厂）、handler（拒绝策略）。"
                       "工作流程：任务提交 → 核心线程未满直接执行 → 队列未满入队 → "
                       "线程数未到上限扩容执行 → 队列满且线程满走拒绝策略。",
        }, {
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16 > 板块2 > 拒绝策略",
            "content": "四种拒绝策略：AbortPolicy 直接抛 RejectedExecutionException（默认）；"
                       "CallerRunsPolicy 由提交线程自己执行（降速背压）；DiscardPolicy 静默丢弃；"
                       "DiscardOldestPolicy 丢弃最旧任务。生产常用 CallerRunsPolicy 做背压，"
                       "或自定义策略记录告警。",
        }],
        "sufficient": True,
        "keywords": ["ThreadPoolExecutor", "核心线程", "拒绝策略"],
        "category": "java_concurrency",
    },
    {
        "question": "HashMap的底层实现原理是什么？",
        "documents": [{
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23",
            "content": "HashMap 底层是数组 + 链表 + 红黑树：key 经 hashCode 扰动后定位数组槽位，"
                       "冲突时链式挂载；链表长度超 8 且数组容量 ≥64 时树化为红黑树（查找 O(log n)）。"
                       "默认容量 16、负载因子 0.75，超阈值扩容为 2 倍并 rehash。非线程安全，"
                       "并发 put 可能死循环（1.7 头插法）或丢数据。",
        }, {
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23 > 板块3 > 1.7与1.8对比",
            "content": "HashMap 1.7 与 1.8 区别：1.7 数组+链表、头插法（并发扩容可能成环死循环）、"
                       "无树化；1.8 数组+链表+红黑树、尾插法（并发丢数据但不死循环）、"
                       "树化/退化阈值 8/6。1.8 的扩容迁移按高低位拆分，效率更高。",
        }],
        "sufficient": True,
        "keywords": ["HashMap", "红黑树", "扩容"],
        "category": "java_collection",
    },
    {
        "question": "ConcurrentHashMap 1.8 是如何保证线程安全的？",
        "documents": [{
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23 > 板块4 > 并发实现",
            "content": "ConcurrentHashMap 1.8 线程安全策略：1）Node 数组初始化用 CAS（sizeCtl 控制）；"
                       "2）put 时槽位为空用 CAS 直接写入，非空则对头节点 synchronized 加锁（锁粒度"
                       "细化为单个桶）；3）扩容用 ForwardingNode 标记 + 多线程协助迁移（transfer）；"
                       "4）size 用 CounterCell 分散计数避免全局竞争。抛弃了 1.7 的 Segment 分段锁，"
                       "并发度更高。",
        }, {
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23 > 板块4 > 使用建议",
            "content": "使用建议：读多写少场景可用读写锁或 CopyOnWriteArrayList；高并发写场景"
                       "用 ConcurrentHashMap；需要复合操作（先查后写）时用 computeIfAbsent 等"
                       "原子方法而非先 get 再 put。注意 size() 是弱一致的（遍历期间可能有并发修改）。",
        }],
        "sufficient": True,
        "keywords": ["ConcurrentHashMap", "CAS", "synchronized"],
        "category": "java_collection",
    },
    {
        "question": "MySQL 的 B+ 树索引为什么能加速查询？",
        "documents": [{
            "title": "8-MySQL索引原理与B+树_2026-07-18",
            "content": "B+ 树索引结构：非叶子节点只存键值（扇出大，3 层可容纳千万级数据），"
                       "叶子节点存数据并按序串联（范围查询只需顺序遍历）。相比红黑树：树高更矮"
                       "（磁盘 IO 次数少）、范围查询高效；相比哈希索引：支持排序与范围。"
                       "InnoDB 主键索引叶子存整行（聚簇），二级索引叶子存主键值，回表取数据。",
        }, {
            "title": "8-MySQL索引原理与B+树_2026-07-18 > 板块2 > 索引优化",
            "content": "索引优化要点：最左前缀原则（联合索引按顺序匹配）；覆盖索引避免回表；"
                       "区分度高的列放前面；避免索引列上做函数运算（导致失效）；"
                       "like '%xx' 前缀模糊无法用索引。EXPLAIN 的 type 列：const > ref > range > "
                       "index > ALL，目标是避免 ALL 全表扫描。",
        }],
        "sufficient": True,
        "keywords": ["B+树", "聚簇索引", "回表"],
        "category": "mysql",
    },
    {
        "question": "MySQL 的事务隔离级别有哪些？各自解决什么问题？",
        "documents": [{
            "title": "9-MySQL事务与隔离级别_2026-07-19",
            "content": "MySQL 四种隔离级别：读未提交（脏读）、读已提交（不可重复读）、"
                       "可重复读（默认，幻读）、串行化（全隔离但性能差）。InnoDB 可重复读下"
                       "通过 MVCC（多版本并发控制）快照读解决不可重复读，通过间隙锁/临键锁"
                       "解决幻读。事务隔离与并发性能成反比，生产默认可重复读。",
        }, {
            "title": "9-MySQL事务与隔离级别_2026-07-19 > 板块2 > MVCC实现",
            "content": "MVCC 实现：每行隐藏 trx_id（最近修改事务）与回滚指针，undo log 串联版本链；"
                       "Read View 记录活跃事务集合，快照读按可见性规则取版本。当前读（for update）"
                       "走最新版本并加锁。MVCC 让读不加锁、读写不互斥，是 InnoDB 高并发的基础。",
        }],
        "sufficient": True,
        "keywords": ["隔离级别", "MVCC", "幻读"],
        "category": "mysql",
    },
    {
        "question": "慢查询优化的一般思路是什么？",
        "documents": [{
            "title": "11-MySQL慢查询优化实践_2026-07-22",
            "content": "慢查询优化流程：1）开启慢查询日志（long_query_time=1s）定位问题 SQL；"
                       "2）EXPLAIN 分析执行计划，关注 type（是否全表扫描）、rows（扫描行数）、"
                       "extra（Using filesort/Using temporary 是危险信号）；3）加索引/改索引"
                       "（覆盖索引、联合索引调整列序）；4）改写 SQL（避免 select *、函数包裹列、"
                       "大范围 in）；5）数据量过大考虑分表分库或归档。",
        }, {
            "title": "11-MySQL慢查询优化实践_2026-07-22 > 板块3 > 实战案例",
            "content": "实战案例：一条订单查询 3 秒，EXPLAIN 显示 type=ALL 扫描 500 万行。"
                       "根因：status 列无索引且查询条件 status IN (1,2,3) 加时间范围。"
                       "优化：建联合索引 (status, create_time) 覆盖查询条件，耗时降至 30ms。"
                       "教训：先看执行计划再谈优化，避免盲目加索引。",
        }],
        "sufficient": True,
        "keywords": ["慢查询", "EXPLAIN", "索引"],
        "category": "mysql",
    },
    {
        "question": "Redis缓存穿透、击穿、雪崩有什么区别？分别怎么解决？",
        "documents": [{
            "title": "13-Redis缓存三大问题与解决方案_2026-07-24",
            "content": "三大问题区分：穿透——查不存在的数据（缓存和 DB 都没有），恶意攻击打垮 DB；"
                       "击穿——热点 key 过期瞬间大量请求打到 DB；雪崩——大量 key 同时过期或"
                       "Redis 宕机，DB 瞬间承压。穿透解决：布隆过滤器 + 空值缓存；击穿解决："
                       "互斥锁/逻辑过期；雪崩解决：过期时间加随机值、多级缓存、集群高可用。",
        }, {
            "title": "13-Redis缓存三大问题与解决方案_2026-07-24 > 板块2 > 缓存一致性",
            "content": "缓存一致性方案：Cache Aside（先更新 DB 再删缓存，延迟双删防旧值回写）；"
                       "读多写少可用 TTL 容忍短暂不一致；严格一致用 Canal 订阅 binlog 异步删缓存"
                       "或分布式锁。删缓存失败会引入脏数据，用消息队列重试保证最终一致。",
        }],
        "sufficient": True,
        "keywords": ["穿透", "击穿", "雪崩"],
        "category": "redis",
    },
    {
        "question": "Redis哨兵模式是怎么实现高可用的？",
        "documents": [{
            "title": "14-Redis高可用架构：主从+哨兵_2026-07-25",
            "content": "哨兵（Sentinel）实现高可用：独立进程监控主从节点，通过心跳（PING）判断"
                       "存活；主节点客观下线（多数哨兵同意）后触发故障转移——从从节点中选举"
                       "新主（复制偏移量最大优先）、修改配置、通知客户端重连。哨兵集群自身"
                       "至少 3 个（奇数）防脑裂，客户端通过哨兵获取最新主节点地址。",
        }, {
            "title": "14-Redis高可用架构：主从+哨兵_2026-07-25 > 板块3 > 与集群对比",
            "content": "哨兵 vs 集群：哨兵只解决高可用（主从切换），数据仍单点写入；"
                       "集群（Cluster）通过 16384 个槽位分片，支持多主多从横向扩展，"
                       "每个主节点负责一段槽位，读写压力分散。数据量大、写并发高的场景用集群，"
                       "数据量小要高可用用哨兵。",
        }],
        "sufficient": True,
        "keywords": ["哨兵", "故障转移", "高可用"],
        "category": "redis",
    },
    {
        "question": "Spring事务的传播行为有哪些？",
        "documents": [{
            "title": "17-Spring事务管理与传播行为_2026-07-28",
            "content": "Spring 事务传播行为：REQUIRED（默认，有则加入无则新建）、REQUIRES_NEW"
                       "（挂起当前新建独立事务）、NESTED（嵌套事务，Savepoint 回滚）、"
                       "SUPPORTS（有则参与无则无事务）、NOT_SUPPORTED（无事务执行）、"
                       "MANDATORY（必须有事务否则抛异常）、NEVER（必须无事务）。"
                       "事务失效常见场景：同类调用（自调用不走代理）、非 public 方法、"
                       "异常被 catch、rollbackFor 未配置。",
        }, {
            "title": "17-Spring事务管理与传播行为_2026-07-28 > 板块2 > 事务失效排查",
            "content": "事务失效排查清单：1）方法是否 public（Spring AOP 默认只代理 public）；"
                       "2）是否同类自调用（this.method() 不走代理，需注入自身或拆分 Bean）；"
                       "3）异常是否被吞（catch 后不抛出，事务感知不到）；4）rollbackFor 是否"
                       "包含该异常类型（默认只回滚 RuntimeException）；5）数据库引擎是否 InnoDB。",
        }],
        "sufficient": True,
        "keywords": ["传播行为", "REQUIRED", "事务失效"],
        "category": "spring",
    },
    {
        "question": "Spring Bean 的生命周期是怎样的？",
        "documents": [{
            "title": "18-SpringBean生命周期与扩展点_2026-07-29",
            "content": "Spring Bean 生命周期：实例化（构造器）→ 属性填充（依赖注入）→ Aware 回调"
                       "（BeanNameAware/BeanFactoryAware/ApplicationContextAware）→ "
                       "BeanPostProcessor.postProcessBeforeInitialization → @PostConstruct/"
                       "InitializingBean.afterPropertiesSet/init-method → "
                       "BeanPostProcessor.postProcessAfterInitialization → 使用中 → "
                       "@PreDestroy/DisposableBean.destroy。核心扩展点是 BeanPostProcessor。",
        }, {
            "title": "18-SpringBean生命周期与扩展点_2026-07-29 > 板块2 > 常用扩展点",
            "content": "常用扩展点：BeanPostProcessor（全局拦截所有 Bean 初始化前后，AOP 代理就挂在这里）；"
                       "BeanFactoryPostProcessor（BeanDefinition 注册后修改定义，如占位符替换）；"
                       "ApplicationListener（监听容器事件如 ContextRefreshedEvent）。"
                       "掌握生命周期顺序是排查初始化顺序问题的关键。",
        }],
        "sufficient": True,
        "keywords": ["Bean生命周期", "BeanPostProcessor", "初始化"],
        "category": "spring",
    },
    {
        "question": "Spring AOP 的实现原理是什么？",
        "documents": [{
            "title": "19-SpringAOP原理与动态代理_2026-07-30",
            "content": "Spring AOP 基于动态代理：有接口用 JDK 动态代理（Proxy + InvocationHandler），"
                       "无接口用 CGLIB（字节码生成子类）。切面织入发生在 BeanPostProcessor"
                       "阶段，代理对象替换原 Bean。@Transactional/@Async 都是 AOP 应用。"
                       "通知类型：Before/AfterReturning/AfterThrowing/After/Around（环绕最强）。",
        }, {
            "title": "19-SpringAOP原理与动态代理_2026-07-30 > 板块2 > 代理失效场景",
            "content": "代理失效场景：1）同类内部调用（this 调用不走代理）；2）final 类/方法"
                       "（CGLIB 无法继承）；3）静态方法（不走实例代理）；4）代理对象未被注入"
                       "（自己 new 的实例）。排查时先确认对象是否代理类（打印 class 名），"
                       "再检查调用路径。",
        }],
        "sufficient": True,
        "keywords": ["AOP", "动态代理", "CGLIB"],
        "category": "spring",
    },
    {
        "question": "Netty 粘包拆包问题是怎么解决的？",
        "documents": [{
            "title": "20-Netty编解码与粘包拆包_2026-07-31",
            "content": "TCP 是流式协议，无消息边界，多包粘在一起（粘包）或一包拆成多段（拆包）。"
                       "Netty 解决方式：1）定长解码器 FixedLengthFrameDecoder；2）分隔符解码器"
                       "LineBasedFrameDecoder/DelimiterBasedFrameDecoder；3）长度域解码器"
                       "LengthFieldBasedFrameDecoder（最常用，头 4 字节存长度）；"
                       "4）自定义 ByteToMessageDecoder。服务端收到的是完整消息后交给业务 handler。",
        }, {
            "title": "20-Netty编解码与粘包拆包_2026-07-31 > 板块2 > 自定义协议设计",
            "content": "自定义协议设计要点：魔数（校验帧合法性）+ 版本号 + 消息类型 + 长度域 + "
                       "业务数据 + 可选校验和。解码器用 LengthFieldBasedFrameDecoder 按长度域"
                       "切帧，配合解码后的 POJO（MessageToMessageDecoder）分层处理。"
                       "注意长度域最大帧长设置，防止恶意大包拖垮内存。",
        }],
        "sufficient": True,
        "keywords": ["粘包", "拆包", "LengthFieldBasedFrameDecoder"],
        "category": "netty",
    },
    {
        "question": "HTTPS 的握手过程是怎样的？",
        "documents": [{
            "title": "21-HTTPS与TLS握手原理_2026-08-01",
            "content": "TLS 1.2 握手流程：1）ClientHello（客户端随机数 + 支持的加密套件）；"
                       "2）ServerHello + 证书（服务端随机数 + 证书链）；3）客户端验证证书"
                       "（CA 信任链 + 域名 + 有效期）；4）客户端生成预主密钥，用服务端公钥加密"
                       "发送（RSA 场景），双方各自计算会话密钥；5）ChangeCipherSpec 切换加密，"
                       "Finished 校验。TLS 1.3 精简为 1 次往返（1-RTT），0-RTT 可带早期数据。",
        }, {
            "title": "21-HTTPS与TLS握手原理_2026-08-01 > 板块3 > 性能优化",
            "content": "HTTPS 性能优化：1）会话复用（Session ID/Session Ticket，减少握手次数）；"
                       "2）TLS 1.3 缩短握手往返；3）OCSP Stapling 缓存证书吊销状态；"
                       "4）CDN 边缘节点终结 TLS；5）硬件加速卡卸载加解密。"
                       "握手成本主要在 RSA 非对称运算，ECC 证书可显著降低计算量。",
        }],
        "sufficient": True,
        "keywords": ["HTTPS", "TLS", "握手"],
        "category": "network",
    },
    {
        "question": "Reactor 线程模型在 Netty 中是怎么实现的？",
        "documents": [{
            "title": "7-Netty高性能IO与Reactor线程模型_2026-07-17",
            "content": "Reactor 模型核心是事件分发：Boss 线程（主 Reactor）accept 新连接，"
                       "注册到 Worker 线程组（子 Reactor）；Worker 线程处理该连接的读写事件，"
                       "通过 ChannelPipeline 逐 handler 处理。Netty 默认 NioEventLoop 数量为 "
                       "CPU 核数 * 2，一个连接生命周期内绑定固定线程，避免线程切换。"
                       "IO 事件与业务处理解耦，handler 里阻塞操作会拖垮该线程。",
        }, {
            "title": "7-Netty高性能IO与Reactor线程模型_2026-07-17 > 板块2 > 线程模型演进",
            "content": "线程模型演进：单线程 Reactor（一个线程管全部，瓶颈明显）→ 多线程 Reactor"
                       "（IO 线程 + 业务线程池分离）→ 主从 Reactor（Boss/Worker 两级，Netty 采用）。"
                       "业务处理建议丢到独立线程池（Channel 上 add 自定义 EventExecutor），"
                       "避免阻塞 IO 线程导致事件积压。",
        }],
        "sufficient": True,
        "keywords": ["Reactor", "Boss", "Worker"],
        "category": "netty",
    },
    {
        "question": "JWT 认证的流程是怎样的？和 Session 有什么区别？",
        "documents": [{
            "title": "22-JWT认证机制详解_2026-08-02",
            "content": "JWT 流程：用户登录 → 服务端签发 JWT（Header.Payload.Signature 三段，"
                       "Base64Url 编码）→ 客户端存储并在后续请求带 Authorization: Bearer <token>"
                       "→ 服务端验签（HMAC/非对称）通过即信任载荷。无状态：服务端不存会话，"
                       "天然适合分布式。签名防篡改，但载荷明文可读，不放敏感信息。",
        }, {
            "title": "22-JWT认证机制详解_2026-08-02 > 板块3 > 与Session对比",
            "content": "JWT vs Session：Session 服务端存储（Redis 共享），可主动吊销，"
                       "但分布式需共享存储；JWT 无状态免查询，但签发后无法撤销（黑名单方案"
                       "又回到有状态）、载荷有体积开销。安全实践：JWT 过期时间短 + refresh token"
                       "轮换；HTTPS 传输防中间人；算法固定 HS256/RS256 防算法混淆攻击。",
        }],
        "sufficient": True,
        "keywords": ["JWT", "签名", "无状态"],
        "category": "auth",
    },
    {
        "question": "分布式场景下 Session 共享怎么解决？",
        "documents": [{
            "title": "23-分布式Session共享方案_2026-08-03",
            "content": "分布式 Session 共享方案：1）Session 存 Redis（最常用，spring-session-data-redis），"
                       "多实例读取同一份数据；2）粘性会话（Sticky Session，负载均衡按 Session "
                       "路由固定实例，扩容/宕机有风险）；3）客户端存储（JWT/加密 Cookie）；"
                       "4）Session 同步广播（只适合小集群）。生产首选 Redis 方案，"
                       "注意序列化性能与过期策略。",
        }, {
            "title": "23-分布式Session共享方案_2026-08-03 > 板块2 > 实践建议",
            "content": "实践建议：Spring Session + Redis 替换容器 Session，配置 SessionRepositoryFilter "
                       "后无感迁移；Session 里只放必要数据（减少 Redis 内存与序列化开销）；"
                       "设置合理过期时间；集群 Session 一致性要求不高时可用最终一致（Redis 主从）。"
                       "更现代的替代是无状态 JWT + 短过期。",
        }],
        "sufficient": True,
        "keywords": ["Session共享", "Redis", "粘性会话"],
        "category": "distributed",
    },
    {
        "question": "两阶段提交（2PC）的流程和缺点是什么？",
        "documents": [{
            "title": "24-分布式事务：2PC与3PC详解_2026-08-04",
            "content": "两阶段提交流程：阶段一（准备）——协调者向所有参与者发 prepare，参与者"
                       "执行事务到可提交状态（写 undo/redo 日志）并返回 yes/no；阶段二（提交/中止）"
                       "——全部 yes 则广播 commit，任一 no 或超时则广播 rollback。"
                       "缺点：同步阻塞（参与者持有资源等待协调者）、协调者单点（宕机则悬挂）、"
                       "脑裂风险（部分提交部分回滚）。",
        }, {
            "title": "24-分布式事务：2PC与3PC详解_2026-08-04 > 板块2 > 3PC改进",
            "content": "3PC 在 2PC 基础上改进：增加阶段 0 CanCommit（预询问）、引入超时机制，"
                       "参与者超时后自主提交而非无限等待，缓解同步阻塞；但网络分区下仍可能"
                       "不一致，且实现复杂。实际生产中 2PC/3PC 因协调者单点与性能问题少用，"
                       "更多采用 TCC、最终一致（MQ）、Seata AT 等柔性方案。",
        }],
        "sufficient": True,
        "keywords": ["两阶段提交", "2PC", "协调者"],
        "category": "distributed",
    },
    {
        "question": "Seata AT 模式是怎么实现分布式事务的？",
        "documents": [{
            "title": "25-Seata分布式事务方案_2026-08-05",
            "content": "Seata AT 模式核心是三个角色：TC（事务协调器，独立服务）、TM（事务管理器，"
                       "业务发起方）、RM（资源管理器，每个数据库分支）。流程：TM 开启全局事务"
                       "→ 分支事务执行（RM 写 undo_log 快照）→ TM 提交 → TC 协调各分支 → "
                       "全部成功则各 RM 删除 undo_log，失败则按 undo_log 反向补偿回滚。"
                       "对业务无侵入，适用数据库一致性场景。",
        }, {
            "title": "25-Seata分布式事务方案_2026-08-05 > 板块2 > 与TCC对比",
            "content": "AT vs TCC：AT 自动生成 undo_log 补偿，业务零侵入，但依赖数据库（回滚靠"
                       "快照，不适合纯缓存/消息等非事务资源）；TCC 三阶段（Try/Confirm/Cancel）"
                       "业务自实现补偿逻辑，适用范围广（任何资源），但开发量大。"
                       "选择：数据库为主用 AT，跨资源（DB+MQ+缓存）用 TCC 或 MQ 最终一致。",
        }],
        "sufficient": True,
        "keywords": ["Seata", "AT模式", "undo_log"],
        "category": "distributed",
    },
    {
        "question": "Redisson 分布式锁的实现原理是什么？",
        "documents": [{
            "title": "26-Redisson分布式锁原理_2026-08-06",
            "content": "Redisson 分布式锁：基于 Redis SET NX EX 加锁（key=锁名，value=线程标识+"
                       "看门狗续期），释放用 Lua 脚本校验持有者后 DEL（防误删他人锁）。"
                       "看门狗机制：默认 30s 过期，持有线程存活则每 10s 续期，防死锁又防"
                       "业务未完成锁先过期。可重入：同一线程多次 lock 计数。"
                       "支持红锁（RedLock）多节点防单点故障，但工程上少用（性能与一致性权衡）。",
        }, {
            "title": "26-Redisson分布式锁原理_2026-08-06 > 板块2 > 注意事项",
            "content": "使用注意：1）锁粒度要小（按业务键加锁而非全局锁）；2）锁内操作要快"
                       "（长时间持锁会阻塞其他请求）；3）主从切换时锁可能丢失（Redis 主宕机"
                       "未同步从节点，新主无锁记录）——敏感场景用 RedLock 或 Zookeeper 锁；"
                       "4）tryLock(waitTime) 设置合理等待时间，避免无限等待。",
        }],
        "sufficient": True,
        "keywords": ["Redisson", "看门狗", "分布式锁"],
        "category": "distributed",
    },
    {
        "question": "接口幂等性怎么设计？",
        "documents": [{
            "title": "27-接口幂等性设计实践_2026-08-07",
            "content": "幂等设计核心：让同一请求重复执行结果一致。方案：1）唯一键约束——"
                       "下单/支付类业务用业务单号做数据库唯一索引，重复插入直接冲突忽略；"
                       "2）幂等表/去重表——请求方带幂等键（UUID），服务端记录已处理键；"
                       "3）状态机——订单状态流转加条件更新（update ... where status=待支付）；"
                       "4）Redis SETNX 分布式锁防并发重复提交。",
        }, {
            "title": "27-接口幂等性设计实践_2026-08-07 > 板块2 > 常见坑",
            "content": "常见坑：1）重试导致重复扣款——支付回调必须幂等（按支付单号查重）；"
                       "2）并发重复请求——先查后写非原子，需唯一索引或分布式锁兜底；"
                       "3）MQ 消费重复——消费端按消息 ID 去重，配合 at-least-once 语义；"
                       "4）幂等键过期时间——过期后重复请求可能再执行，键有效期应覆盖重试窗口。",
        }],
        "sufficient": True,
        "keywords": ["幂等", "唯一键", "去重"],
        "category": "distributed",
    },
    {
        "question": "消息队列是怎么实现削峰填谷的？",
        "documents": [{
            "title": "28-消息队列削峰与异步解耦_2026-08-08",
            "content": "削峰原理：生产端突发流量先写入 MQ（瞬间吞吐极高），消费端按自身能力"
                       "匀速拉取处理，把峰值压力摊平到低谷时段。同时实现异步解耦：下单 → "
                       "发消息 → 库存/积分/通知各自消费，互不阻塞。选型：秒杀等超高吞吐用 "
                       "Kafka（顺序写、页缓存）；强事务/路由复杂用 RocketMQ/RabbitMQ。",
        }, {
            "title": "28-消息队列削峰与异步解耦_2026-08-08 > 板块2 > 削峰实践",
            "content": "秒杀削峰实践：请求先入队（限流 + 排队）→ 异步下单 → 前端轮询结果；"
                       "关键点：1）队列容量要有上限（拒绝策略兜底）；2）消费并发与 DB 能力匹配"
                       "（过度消费照样打垮 DB）；3）消息不丢（生产者 confirm、消费者手动 ack）；"
                       "4）重复消费幂等。削峰本质是流量整形，不是提升系统总容量。",
        }],
        "sufficient": True,
        "keywords": ["削峰", "异步", "消息队列"],
        "category": "mq",
    },
    {
        "question": "Kafka 消息是怎么分区和路由的？",
        "documents": [{
            "title": "29-Kafka分区与路由机制_2026-08-09",
            "content": "Kafka 分区路由：producer 发送消息时——指定 partition 则直接使用；"
                       "未指定但 key 存在则 key.hashCode() % 分区数（保证相同 key 进同一分区，"
                       "消息有序）；key 为空则轮询/黏性分区（批量优化）。分区是并行度与有序性的"
                       "平衡：分区数越多并行度越高，但单分区内才严格有序，且分区过多增加"
                       "文件句柄与重平衡开销。",
        }, {
            "title": "29-Kafka分区与路由机制_2026-08-09 > 板块2 > 分区数选择",
            "content": "分区数选择经验：分区数 = max(生产吞吐需求, 消费并行度) 且不超过"
                       "broker 数 * 2 的整数倍；同 key 消息严格有序需要单分区（或 key 粒度分区）；"
                       "重平衡成本与分区数正相关，分区数过大时消费者增减会频繁触发重平衡"
                       "（KIP-429 增量协调缓解）。上线后扩分区可行但消息顺序会被打乱。",
        }],
        "sufficient": True,
        "keywords": ["分区", "路由", "key"],
        "category": "kafka",
    },
    {
        "question": "Kafka 消费者组重平衡是怎么触发的？",
        "documents": [{
            "title": "30-Kafka消费者组与重平衡_2026-08-10",
            "content": "重平衡（Rebalance）触发条件：1）消费者加入/退出（含崩溃，心跳超时被踢）；"
                       "2）订阅主题或分区数变化；3）消费者组订阅了新主题。流程：组内消费者"
                       "选举 Leader（第一个加入的）→ Leader 分配分区 → 同步给所有成员。"
                       "旧版 Stop-The-World 式全量重平衡（EAGER）导致消费停滞，新版 KIP-429 "
                       "增量式（COOPERATIVE）只迁移受影响分区。",
        }, {
            "title": "30-Kafka消费者组与重平衡_2026-08-10 > 板块2 > 重平衡风暴",
            "content": "重平衡风暴：某消费者 GC 停顿超时被踢 → 触发重平衡 → 其他消费者暂停消费 → "
                       "积压加剧 → 更多超时被踢 → 连锁重平衡。缓解：调大 max.poll.interval.ms "
                       "与 session.timeout.ms、处理耗时长用异步或调小 max.poll.records、"
                       "保持组成员稳定（避免频繁启停）、静态成员（static membership）防止"
                       "瞬间重平衡。",
        }],
        "sufficient": True,
        "keywords": ["重平衡", "消费者组", "KIP-429"],
        "category": "kafka",
    },
    {
        "question": "微服务的服务发现是怎么实现的？",
        "documents": [{
            "title": "31-微服务注册中心与服务发现_2026-08-11",
            "content": "服务发现三类角色：服务提供者（启动时向注册中心注册，心跳续约）、"
                       "服务消费者（拉取服务列表并本地缓存，按负载策略选择实例）、注册中心"
                       "（维护实例信息，摘除失活实例）。实现方式：客户端发现（消费者直连实例，"
                       "如 Nacos/Consul/Eureka）与服务端发现（通过网关/负载均衡器转发）。"
                       "心跳超时与保护模式（Eureka 自我保护）是常见故障点。",
        }, {
            "title": "31-微服务注册中心与服务发现_2026-08-11 > 板块2 > 注册中心对比",
            "content": "Nacos vs Eureka vs Consul：Nacos 支持 AP/CP 切换、配置中心二合一、"
                       "临时/永久实例；Eureka 纯 AP（自我保护防误删），已进入维护期；"
                       "Consul 强一致（Raft）+ 自带 KV/健康检查，但需额外运维。"
                       "国内主流 Nacos（阿里生态），K8s 环境也可用内置 Service 发现替代。",
        }],
        "sufficient": True,
        "keywords": ["服务发现", "注册中心", "Nacos"],
        "category": "microservice",
    },
    {
        "question": "Nacos 配置中心动态刷新的原理是什么？",
        "documents": [{
            "title": "32-Nacos配置中心原理_2026-08-12",
            "content": "Nacos 动态刷新原理：客户端启动时向服务端拉取配置并注册监听（长轮询）；"
                       "配置变更时服务端推送变更通知（UDP）或客户端长轮询探测到数据变化后"
                       "重新拉取；本地拿到新配置后刷新上下文（@RefreshScope 重建 Bean / "
                       "@NacosValue 回调）。长轮询是核心：请求挂起等待，有变化立即返回，"
                       "避免频繁短轮询。",
        }, {
            "title": "32-Nacos配置中心原理_2026-08-12 > 板块2 > 使用建议",
            "content": "使用建议：敏感配置（数据库密码）用 Nacos 加密插件；配置变更灰度发布"
                       "（先小流量验证）；客户端本地缓存兜底（配置中心不可用时用上次配置启动）；"
                       "命名空间隔离环境（dev/test/prod）；配置变更要可回滚、可审计。"
                       "@RefreshScope 的 Bean 重建有短暂抖动，避免在热点路径 Bean 上滥用。",
        }],
        "sufficient": True,
        "keywords": ["Nacos", "动态刷新", "长轮询"],
        "category": "microservice",
    },
    {
        "question": "服务熔断和降级的原理是什么？",
        "documents": [{
            "title": "33-熔断降级与Sentinel原理_2026-08-13",
            "content": "熔断（Circuit Breaker）：统计调用失败率，超过阈值（如 50%）熔断器打开，"
                       "后续请求直接快速失败（Fallback），不再打到下游，给下游恢复时间；"
                       "熔断半开状态放少量探测流量，成功则关闭。降级：主动牺牲非核心功能"
                       "（如缓存兜底、返回默认值）保核心链路。Sentinel 提供并发线程数/"
                       "QPS 限流 + 熔断降级 + 热点防护。",
        }, {
            "title": "33-熔断降级与Sentinel原理_2026-08-13 > 板块2 > 实践要点",
            "content": "实践要点：1）熔断阈值按下游健康度设定，超时时间要比下游 RT 合理；"
                       "2）Fallback 必须快（不能 Fallback 里再调慢服务）；3）隔离手段："
                       "线程池隔离（舱壁模式）或信号量隔离；4）监控熔断状态与降级次数，"
                       "防止熔断长期打开无人发现；5）配合限流防止流量直接打垮自身。",
        }],
        "sufficient": True,
        "keywords": ["熔断", "降级", "Sentinel"],
        "category": "microservice",
    },
    {
        "question": "令牌桶和漏桶限流算法有什么区别？",
        "documents": [{
            "title": "34-限流算法：令牌桶与漏桶_2026-08-14",
            "content": "漏桶：请求先进桶排队，桶按固定速率出水——输出速率恒定，天然削峰，"
                       "但无法应对突发流量（恒定速率限制）。令牌桶：以固定速率生成令牌存桶"
                       "（桶容量 = 最大突发），请求需拿令牌——允许一定突发（桶里积攒的令牌），"
                       "平滑限流。令牌桶更常用（Guava RateLimiter、Sentinel QPS 模式）。"
                       "还有滑动窗口计数（固定窗口的改良）与并发数限流。",
        }, {
            "title": "34-限流算法：令牌桶与漏桶_2026-08-14 > 板块2 > 分布式限流",
            "content": "分布式限流：单机限流在集群下各自独立，总量不可控；方案：1）Redis "
                       "Lua 脚本原子计数（固定窗口/滑动窗口，如 INCR + EXPIRE）；"
                       "2）Redis 令牌桶（Lua 实现，Redis 7 有原生限流模块）；3）网关层限流"
                       "（Sentinel 接入网关、Kong/APISIX 插件）；4）配额中心（配额按服务下发）。"
                       "注意 Redis 单点与时钟一致性，敏感场景用本地 + 集群双层限流。",
        }],
        "sufficient": True,
        "keywords": ["令牌桶", "漏桶", "限流"],
        "category": "microservice",
    },
    {
        "question": "Docker 镜像的分层机制是怎么工作的？",
        "documents": [{
            "title": "35-Docker镜像分层与构建优化_2026-08-15",
            "content": "Docker 镜像分层：每个 RUN/COPY 指令生成一层只读层，多层通过联合文件系统"
                       "（OverlayFS）堆叠成统一视图；容器层（可写）在最上。分层好处：复用——"
                       "基础镜像层被多镜像共享（Registry 只传增量）、构建缓存——层不变直接复用。"
                       "运行时可写层写入产生 Copy-on-Write（修改底层文件先复制），性能略降。",
        }, {
            "title": "35-Docker镜像分层与构建优化_2026-08-15 > 板块2 > 构建优化",
            "content": "构建优化实践：1）指令顺序——变化少的放前面（依赖安装先于代码拷贝），"
                       "最大化缓存命中；2）多阶段构建——编译环境构建产物，运行镜像只留运行环境，"
                       "大幅减小体积；3）合并 RUN（每层都是新镜像层）；4）.dockerignore 排除"
                       "无关文件；5）镜像瘦身——alpine 基础镜像、清理包缓存。",
        }],
        "sufficient": True,
        "keywords": ["镜像分层", "OverlayFS", "多阶段构建"],
        "category": "devops",
    },
    {
        "question": "Kubernetes 的 Pod 是怎么被调度到节点的？",
        "documents": [{
            "title": "36-Kubernetes调度器原理_2026-08-16",
            "content": "Kube-scheduler 调度流程分两阶段：过滤（Predicates）——排除不满足条件的"
                       "节点（资源不足、污点不容忍、亲和性不匹配）；打分（Priorities）——"
                       "按资源余量、Pod 分布等加权评分，选最高分节点绑定。调度结果写入 "
                       "Pod 的 nodeName，kubelet 看到后拉镜像创建容器。可自定义调度器"
                       "与调度框架（Scheduling Framework）扩展点。",
        }, {
            "title": "36-Kubernetes调度器原理_2026-08-16 > 板块2 > 调度策略",
            "content": "常用调度策略：nodeSelector（硬性标签选择）、nodeAffinity（亲和性，支持"
                       "软硬偏好）、podAffinity/antiAffinity（Pod 间聚散）、污点 Taints + 容忍 "
                       "Tolerations（节点专属/隔离）、资源 requests/limits（调度与驱逐依据）、"
                       "拓扑分布约束（多可用区打散）。调度失败常见原因：资源不足、污点未容忍、"
                       "亲和性无法满足。",
        }],
        "sufficient": True,
        "keywords": ["Pod调度", "亲和性", "污点"],
        "category": "devops",
    },
    {
        "question": "双亲委派模型是什么？为什么要这么设计？",
        "documents": [{
            "title": "37-JVM类加载机制与双亲委派_2026-08-17",
            "content": "双亲委派：类加载请求先交给父加载器（应用加载器 → 扩展加载器 → 引导加载器），"
                       "父加载不了才由子加载。保证：1）核心类一致性——java.lang.String 永远由"
                       "引导加载器加载，避免重复加载与安全风险（防止自定义同名类冒充核心类）；"
                       "2）避免重复加载。JDK 9 模块化后扩展加载器改为平台加载器，"
                       "类路径机制变为模块化。",
        }, {
            "title": "37-JVM类加载机制与双亲委派_2026-08-17 > 板块2 > 打破场景",
            "content": "打破双亲委派的场景：1）SPI（Service Provider Interface）——JDBC 驱动"
                       "由引导加载器加载接口、应用加载器加载实现，线程上下文类加载器解决；"
                       "2）Tomcat 等 Web 容器——每个应用独立类加载器（WebAppClassLoader）"
                       "隔离不同应用的依赖版本，先自己加载再委派父；3）OSGi 模块化。"
                       "热部署/插件系统都依赖类加载器隔离。",
        }],
        "sufficient": True,
        "keywords": ["双亲委派", "类加载器", "SPI"],
        "category": "jvm",
    },
    {
        "question": "JVM 类加载的完整过程是什么？",
        "documents": [{
            "title": "38-JVM类加载过程详解_2026-08-18",
            "content": "类加载五阶段：加载（读取字节码生成 Class 对象）、验证（文件格式/元数据/"
                       "字节码验证，防恶意字节码）、准备（静态变量分配内存并赋默认值，常量"
                       "直接赋值）、解析（符号引用转直接引用，类/接口/字段/方法解析）、初始化"
                       "（执行 static 块与静态变量赋值，触发条件：new/反射/访问静态成员等）。"
                       "初始化失败（static 抛异常）会导致类不可用，后续使用抛 "
                       "ExceptionInInitializerError。",
        }, {
            "title": "38-JVM类加载过程详解_2026-08-18 > 板块2 > 初始化时机",
            "content": "类初始化触发时机：1）new、getstatic、putstatic、invokestatic 指令；"
                       "2）反射（Class.forName 默认初始化）；3）初始化子类先初始化父类；"
                       "4）main 类；5）JDK 7 动态语言支持。不触发：子类引用父类静态字段"
                       "（只初始化父类）、定义数组、访问常量池常量（编译期已内联）。"
                       "理解初始化时机对排查静态资源初始化顺序问题很重要。",
        }],
        "sufficient": True,
        "keywords": ["类加载", "初始化", "验证"],
        "category": "jvm",
    },
    {
        "question": "JMM（Java内存模型）解决什么问题？",
        "documents": [{
            "title": "39-Java内存模型JMM详解_2026-08-19",
            "content": "JMM 定义线程与主内存的抽象关系：每条线程有工作内存（缓存变量副本），"
                       "线程读写变量需与主内存交互（load/store 等 8 种操作）。解决三大问题："
                       "可见性（写不刷新读不到）、原子性（复合操作被中断）、有序性（指令重排）。"
                       "规范了 happens-before 规则与内存屏障语义，编译器/CPU 在规则内自由优化。"
                       "volatile、synchronized、final 是 JMM 层的保障机制。",
        }, {
            "title": "39-Java内存模型JMM详解_2026-08-19 > 板块2 > 内存屏障",
            "content": "内存屏障（Memory Barrier）：LoadLoad/LoadStore/StoreLoad/StoreStore 四类。"
                       "volatile 写插入 StoreStore + StoreLoad（写后强制刷新主内存并禁止重排到"
                       "后续读），volatile 读插入 LoadLoad + LoadStore（读后禁止重排且取最新）。"
                       "锁操作同样插入屏障。屏障是 CPU 指令级保证，JMM 通过它定义重排序边界。",
        }],
        "sufficient": True,
        "keywords": ["JMM", "可见性", "内存屏障"],
        "category": "jvm",
    },
    {
        "question": "Happens-Before 规则有哪些？",
        "documents": [{
            "title": "40-Happens-Before规则_2026-08-20",
            "content": "Happens-Before 八条规则：程序次序（单线程内前写后读）、管程锁（解锁 happens-"
                       "before 加锁）、volatile（写 happens-before 读）、线程启动（start happens-"
                       "before 线程内动作）、线程终止（线程动作 happens-before join 返回）、"
                       "中断（interrupt happens-before 检测到中断）、对象终结（构造完成 "
                       "happens-before finalize）、传递性。规则是判断数据竞争的法律依据——"
                       "两个动作无 HB 关系且共享变量即有竞争。",
        }, {
            "title": "40-Happens-Before规则_2026-08-20 > 板块2 > 应用",
            "content": "应用示例：单例双重检查锁——instance 声明 volatile 正是利用 HB 规则"
                       "（volatile 写 happens-before 读），保证返回的实例初始化完成；"
                       "线程池中任务提交 happens-before 执行（内部有同步点）；"
                       "Thread 的 sleep/yield 不提供 HB 关系，不能用作同步手段。"
                       "并发编程先想 HB 关系再写代码。",
        }],
        "sufficient": True,
        "keywords": ["Happens-Before", "volatile", "传递性"],
        "category": "jvm",
    },
    {
        "question": "CAS 的原理和缺点是什么？",
        "documents": [{
            "title": "41-CAS与原子操作原理_2026-08-21",
            "content": "CAS（Compare And Swap）：比较内存值与期望值，相等则替换为新值，"
                       "全程 CPU 原子指令（cmpxchg，锁总线/缓存行锁）。Java 中 Unsafe/"
                       "VarHandle 提供，AtomicInteger.incrementAndGet 就是 CAS 循环。"
                       "缺点：1）ABA 问题（值被改回原值无法察觉，AtomicStampedReference 解决）；"
                       "2）自旋开销（高竞争下 CPU 空转）；3）只能保证单个变量的原子性"
                       "（复合操作用锁）。",
        }, {
            "title": "41-CAS与原子操作原理_2026-08-21 > 板块2 > 与锁对比",
            "content": "CAS vs 锁：CAS 无锁（乐观），线程不阻塞，适合竞争低的场景（短临界区）；"
                       "synchronized 悲观锁，竞争激烈时避免自旋浪费（JDK 15+ 偏向锁废弃，"
                       "轻量级锁就是 CAS 实现）。选择：单变量/计数器用原子类；多变量一致性"
                       "用锁；读多写少用读写锁。LongAdder 用分段计数降低竞争，高并发计数首选。",
        }],
        "sufficient": True,
        "keywords": ["CAS", "ABA", "原子类"],
        "category": "java_concurrency",
    },
    {
        "question": "AtomicInteger 是怎么实现线程安全的？",
        "documents": [{
            "title": "42-原子类与LongAdder实现_2026-08-22",
            "content": "AtomicInteger 原理：volatile int value + Unsafe 的 compareAndSwapInt "
                       "原子更新。incrementAndGet 循环 CAS：读当前值 → CAS 尝试 +1 → 失败重试"
                       "直到成功（高竞争下自旋次数多）。LongAdder 优化：内部维护 base + "
                       "Cell[] 分段，线程各自 CAS 自己的 Cell，求和时汇总——竞争分散到多个"
                       "热点，高并发计数性能远超 AtomicLong。",
        }, {
            "title": "42-原子类与LongAdder实现_2026-08-22 > 板块2 > 适用场景",
            "content": "选择：计数频繁更新（如 QPS 统计、点击量）用 LongAdder；需要精确值且"
                       "更新不频繁用 AtomicLong；需要 CAS 语义的复合逻辑用 AtomicReference/"
                       "AtomicStampedReference（ABA 防重）。原子数组（AtomicIntegerArray）"
                       "按索引原子更新。注意原子类只保证单操作原子，多个原子操作组合仍"
                       "需要同步。",
        }],
        "sufficient": True,
        "keywords": ["AtomicInteger", "CAS", "LongAdder"],
        "category": "java_concurrency",
    },
    {
        "question": "ThreadLocal 的实现原理是什么？为什么会内存泄漏？",
        "documents": [{
            "title": "43-ThreadLocal原理与内存泄漏_2026-08-23",
            "content": "ThreadLocal 原理：每个线程持有 ThreadLocalMap（Entry 的 key 是 ThreadLocal "
                       "弱引用），get/set 操作当前线程的 map，实现线程隔离。内存泄漏根因："
                       "Entry 的 value 是强引用，key 是弱引用——ThreadLocal 被回收后 key 变 "
                       "null，但 value 仍被线程的 map 持有（线程池线程长期存活则永不释放）。"
                       "解决：用完 remove()；Entry 在 get/set 时顺带清理 null key。",
        }, {
            "title": "43-ThreadLocal原理与内存泄漏_2026-08-23 > 板块2 > 实践建议",
            "content": "实践建议：1）ThreadLocal 声明为 private static final（防 key 丢失）；"
                       "2）finally 中 remove()（尤其线程池场景，线程复用导致脏数据串线——"
                       "用户 A 的数据被用户 B 看到）；3）避免存大对象（如整个请求体）；"
                       "4）子线程传值用 InheritableThreadLocal（线程池场景需自定义包装，"
                       "否则传递失效）。",
        }],
        "sufficient": True,
        "keywords": ["ThreadLocal", "内存泄漏", "remove"],
        "category": "java_concurrency",
    },
    {
        "question": "Java 内存泄漏怎么排查？",
        "documents": [{
            "title": "44-Java内存泄漏排查实战_2026-08-24",
            "content": "排查流程：1）监控 heap 曲线——持续上升不回落是泄漏信号（配合 GC 日志）；"
                       "2）jmap 导出堆转储（或 OOM 时 -XX:+HeapDumpOnOutOfMemoryError 自动）；"
                       "3）MAT/Eclipse Memory Analyzer 分析——Dominator Tree 看大对象，"
                       "Leak Suspects 报告给出嫌疑根；4）对象直方图对比多次快照（增量增长的对象"
                       "即泄漏对象）；5）定位持有链（GC Roots 路径），找到泄漏根因代码。",
        }, {
            "title": "44-Java内存泄漏排查实战_2026-08-24 > 板块2 > 常见泄漏",
            "content": "常见泄漏：1）静态集合只增不减（缓存无淘汰策略）；2）ThreadLocal 未 remove"
                       "（线程池场景）；3）连接/流未关闭（数据库、HTTP、文件）；4）监听器"
                       "注册未反注册；5）内部类隐式持有外部类引用（非静态内部类持有 this）；"
                       "6）String.intern 滥用。排查工具：jmap/jstat/jstack + MAT/Arthas。",
        }],
        "sufficient": True,
        "keywords": ["内存泄漏", "MAT", "堆转储"],
        "category": "jvm",
    },
    {
        "question": "JVM 的 OOM 有哪几种类型？",
        "documents": [{
            "title": "45-JVMOOM类型与应对_2026-08-25",
            "content": "OOM 类型：1）Java heap space——堆内存不足（对象泄漏/堆太小）；"
                       "2）GC overhead limit exceeded——98% 时间做 GC 但回收 <2% 堆；"
                       "3）Metaspace——元空间不足（动态生成类，如反射/字节码代理）；"
                       "4）unable to create new native thread——线程数超 OS 限制（ulimit/"
                       "pid 限制）；5）Direct buffer memory——堆外内存不足（NIO 未释放）；"
                       "6）Requested array size exceeds VM limit。",
        }, {
            "title": "45-JVMOOM类型与应对_2026-08-25 > 板块2 > 应对策略",
            "content": "应对：1）HeapDumpOnOutOfMemoryError 自动导出堆转储，OOM 前 dump 配合"
                       "（-XX:+ExitOnOutOfMemoryError 可让服务退出重启而非带病运行）；"
                       "2）区分泄漏与容量不足——泄漏修代码，容量不足调参（-Xmx 与容器内存"
                       "匹配，勿超配）；3）线程 OOM 检查线程池配置与 OS 限制；"
                       "4）容器场景注意 cgroup 限制与 -Xmx 的一致性。",
        }],
        "sufficient": True,
        "keywords": ["OOM", "堆溢出", "Metaspace"],
        "category": "jvm",
    },
    {
        "question": "Full GC 频繁怎么定位和解决？",
        "documents": [{
            "title": "46-FullGC频繁排查实践_2026-08-26",
            "content": "Full GC 排查链路：1）GC 日志（-Xlog:gc* 或 -verbose:gc）确认 Full GC 频率"
                       "与耗时；2）jstat -gcutil 看各区使用率（老年代持续高位是信号）；"
                       "3）堆转储分析大对象/泄漏（MAT 直方图）；4）常见根因：大对象直接进"
                       "老年代（-XX:PretenureSizeThreshold）、内存泄漏、缓存无上限、"
                       "线程池任务积压、类加载过多（Metaspace 溢出触发 Full GC）、"
                       "GC 参数不合理（新代过小导致晋升频繁）。",
        }, {
            "title": "46-FullGC频繁排查实践_2026-08-26 > 板块2 > 解决方案",
            "content": "解决方案：1）代码层修泄漏/降大对象（分页、流式处理）；2）参数调优——"
                       "调整新生代比例（-Xmn）、晋升阈值（-XX:MaxTenuringThreshold）、"
                       "大对象阈值；3）G1 场景调 IHOP（-XX:InitiatingHeapOccupancyPercent）；"
                       "4）缓存加容量上限与淘汰策略；5）扩容/换机器。目标是让对象尽量"
                       "在新生代回收，老年代只留长生命周期对象。",
        }],
        "sufficient": True,
        "keywords": ["FullGC", "老年代", "jstat"],
        "category": "jvm",
    },
    {
        "question": "JWT 刷新机制是怎么设计的？",
        "documents": [{
            "title": "47-JWT刷新与双Token机制_2026-08-27",
            "content": "双 Token 方案：Access Token 短有效期（如 15 分钟~2 小时）用于正常请求，"
                       "Refresh Token 长有效期（如 7~30 天，存 Redis/数据库可撤销）用于续期。"
                       "流程：Access 过期 → 客户端带 Refresh 调刷新接口 → 服务端校验 Refresh "
                       "有效性 → 签发新 Access（+ 可选轮换 Refresh，旧 Refresh 作废防重放）。"
                       "Refresh Token 泄漏风险更高，需 HTTPS + 短时窗口 + 绑定设备/指纹。",
        }, {
            "title": "47-JWT刷新与双Token机制_2026-08-27 > 板块2 > 安全细节",
            "content": "安全细节：1）Refresh Token 存 HttpOnly Cookie 防 XSS 窃取；"
                       "2）Redis 存 Refresh 白名单（版本号递增实现轮换失效）；"
                       "3）敏感操作（改密/支付）要求重新认证，不自动续期；"
                       "4）登出时服务端删除 Refresh（Access 靠短过期自然失效）；"
                       "5）Access 里少放敏感信息，验签算法锁定 RS256 防算法混淆。",
        }],
        "sufficient": True,
        "keywords": ["JWT刷新", "RefreshToken", "双Token"],
        "category": "auth",
    },
    {
        "question": "缓存一致性 Cache Aside 模式是怎么工作的？",
        "documents": [{
            "title": "48-缓存一致性CacheAside模式_2026-08-28",
            "content": "Cache Aside 流程：读——先查缓存，未命中查 DB 回填缓存；写——先更新 DB，"
                       "再删缓存（而非更新缓存，避免写写并发下旧值覆盖新值）。删缓存失败会"
                       "导致旧缓存继续命中——用延迟双删（删后延时再删一次）或 MQ 异步重试兜底。"
                       "为什么删不更新：更新缓存存在并发写时序问题，删让下次读自然重建，"
                       "简单可靠。",
        }, {
            "title": "48-缓存一致性CacheAside模式_2026-08-28 > 板块2 > 极端情况",
            "content": "极端情况：读请求先于删缓存回填了旧值（读写并发窗口）——延迟双删 "
                       "（先删 → 等回填窗口 → 再删）解决；严格一致要求高可用分布式锁"
                       "（写锁 + 删缓存）或 Canal 监听 binlog 异步删缓存。取舍：业务可容忍"
                       "秒级不一致时，Cache Aside + TTL 已足够；强一致场景直接读 DB 或"
                       "用本地事务同步双写。",
        }],
        "sufficient": True,
        "keywords": ["CacheAside", "缓存一致性", "延迟双删"],
        "category": "redis",
    },
    {
        "question": "分布式 ID 生成方案有哪些？",
        "documents": [{
            "title": "49-分布式ID生成方案_2026-08-29",
            "content": "分布式 ID 方案：1）数据库自增/号段模式（批量取号段缓存内存，Leaf-segment）；"
                       "2）Redis INCR（简单但依赖 Redis）；3）UUID（无序、长，不适合作索引主键）；"
                       "4）雪花算法（Snowflake）——时间戳 41 位 + 机器 ID 10 位 + 序列号 12 位，"
                       "64 位自增趋势有序，时钟回拨需处理（记录上次时间，回拨则等待/用备用 ID）；"
                       "5）美团 Leaf（号段 + 雪花融合）。选型看有序性、吞吐、可用性要求。",
        }, {
            "title": "49-分布式ID生成方案_2026-08-29 > 板块2 > 雪花算法细节",
            "content": "雪花算法实现细节：1）机器 ID 用 ZooKeeper/数据库分配或 IP 哈希；"
                       "2）时钟回拨处理：记录 lastTimestamp，当前时间小于它——等待回拨差值"
                       "或抛出异常（ID 不能回退，否则重复）；3）序列号溢出（同毫秒超 4096）"
                       "自旋等下一毫秒；4）定制位段：时间位减少留业务位（订单号带业务前缀）。"
                       "单机 69 年用尽时间位（41 位毫秒）。",
        }],
        "sufficient": True,
        "keywords": ["分布式ID", "雪花算法", "号段"],
        "category": "distributed",
    },
    # ---- 不充分（文档无法回答问题，50 条）----
    {
        "question": "什么是G1垃圾收集器？它的核心创新是什么？",
        "documents": [{
            "title": "5-Kafka消息可靠性与高吞吐设计_2026-07-15",
            "content": "Kafka 可靠性核心是 ISR（In-Sync Replicas）机制：每个 Partition 有多个副本，"
                       "Leader 负责读写，Follower 拉取同步，超过 replica.lag.time.max.ms 未同步即被"
                       "踢出 ISR。生产者端 acks=all 配合 min.insync.replicas=2 保证确认；消费端手动"
                       "提交 offset 防丢消息。",
        }, {
            "title": "7-Netty高性能IO与Reactor线程模型_2026-07-17",
            "content": "Netty 基于 Reactor 线程模型：Boss 线程负责 accept 连接并注册到 Worker 线程，"
                       "Worker 线程处理读写事件。零拷贝通过堆外内存与 CompositeByteBuf 实现，"
                       "避免多次内存拷贝，是高吞吐网络编程的基础设施。",
        }],
        "sufficient": False,
        "keywords": ["G1", "Region", "MixedGC"],
        "category": "java_gc",
        "note": "完全不沾边：问 G1 却检索到 Kafka/Netty 文档",
    },
    {
        "question": "ZGC的特点和适用场景是什么？",
        "documents": [{
            "title": "1-G1垃圾收集器的Region分区机制与MixedGC全流程_2026-07-11",
            "content": "G1（Garbage First）垃圾收集器是 JDK 9 之后的默认垃圾收集器。核心设计是把堆"
                       "划分为大小相等的 Region 区域，每个 Region 可独立扮演 Eden、Survivor 或 Old "
                       "角色。G1 的核心创新：Region 分区 + Remembered Set、回收价值优先、"
                       "并发标记 + SATB 写屏障。MixedGC 在并发标记完成后回收年轻代与高收益 Region。",
        }, {
            "title": "1-G1垃圾收集器的Region分区机制与MixedGC全流程_2026-07-11 > 板块3 > 调优参数",
            "content": "G1 常用调优参数：-XX:MaxGCPauseMillis=200、-XX:G1HeapRegionSize=8m、"
                       "-XX:InitiatingHeapOccupancyPercent=45。调优经验：大对象分配在连续 Region，"
                       "频繁分配会提前触发并发标记；RSet 占用约 5%-10% 堆空间。",
        }],
        "sufficient": False,
        "keywords": ["ZGC", "适用场景"],
        "category": "java_gc",
        "note": "主题错位：文档全篇讲 G1，不含 ZGC 任何内容",
    },
    {
        "question": "Kafka的零拷贝(Zero-Copy)技术是如何实现的？",
        "documents": [{
            "title": "15-AQS抽象队列同步器与ReentrantLock实现原理_2026-07-26",
            "content": "AQS 是 Java 并发包基石，核心是 volatile int state 字段 + CLH 变体 FIFO 等待"
                       "队列。state 表示同步状态；获取失败的线程封装为 Node 入队，通过自旋 + park "
                       "阻塞。ReentrantLock 非公平锁先 CAS 抢锁再排队，公平锁严格 FIFO。",
        }, {
            "title": "16-volatile与Java内存模型JMM_2026-07-27",
            "content": "volatile 是 Java 提供的轻量级同步机制：可见性——写 volatile 变量插入内存屏障"
                       "强制刷新主内存；有序性——禁止指令重排序。volatile 不保证原子性，复合操作"
                       "仍需 synchronized 或原子类。",
        }],
        "sufficient": False,
        "keywords": ["零拷贝", "Zero-Copy"],
        "category": "kafka",
        "note": "完全不沾边：问 Kafka 零拷贝却检索到 AQS/volatile 文档",
    },
    {
        "question": "什么是CAP定理？在分布式系统设计中如何权衡？",
        "documents": [{
            "title": "10-Redis持久化机制_2026-07-20",
            "content": "Redis 两种持久化：RDB 定时生成全量快照（fork 子进程写临时文件），恢复快但"
                       "可能丢最后一次快照后的数据；AOF 追加写命令日志，默认 everysec 每秒 fsync。",
        }, {
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16",
            "content": "ThreadPoolExecutor 核心参数：corePoolSize、maximumPoolSize、workQueue、"
                       "keepAliveTime、threadFactory、handler。任务提交先复用核心线程，队列满后"
                       "扩容到最大线程数，仍满则走拒绝策略。",
        }],
        "sufficient": False,
        "keywords": ["CAP"],
        "category": "comprehensive",
        "note": "完全不沾边：问 CAP 定理却检索到 Redis/线程池文档",
    },
    {
        "question": "CompletableFuture和Future有什么区别？如何使用CompletableFuture进行异步编排？",
        "documents": [{
            "title": "16-volatile与Java内存模型JMM_2026-07-27",
            "content": "volatile 是 Java 提供的轻量级同步机制，两大作用：1）可见性——写 volatile "
                       "变量插入内存屏障强制刷新主内存；2）有序性——禁止指令重排序。volatile "
                       "不保证原子性，复合操作仍需 synchronized 或原子类。",
        }, {
            "title": "16-volatile与Java内存模型JMM_2026-07-27 > 板块2 > 双检锁示例",
            "content": "双重检查锁（Double-Checked Locking）单例：外层判空避免无谓加锁，内层"
                       "synchronized 保证只实例化一次，instance 字段声明为 volatile 防止指令重排。",
        }],
        "sufficient": False,
        "keywords": ["CompletableFuture", "Future"],
        "category": "java_concurrency",
        "note": "主题错位：文档讲 volatile，不含 Future 任何内容",
    },
    {
        "question": "synchronized的底层实现原理是什么？锁升级过程是怎样的？",
        "documents": [{
            "title": "5-Kafka消息可靠性与高吞吐设计_2026-07-15",
            "content": "Kafka 可靠性核心是 ISR（In-Sync Replicas）机制：每个 Partition 有多个副本，"
                       "Leader 负责读写，Follower 拉取同步，超过 replica.lag.time.max.ms 未同步即被"
                       "踢出 ISR。生产者端 acks=all 配合 min.insync.replicas=2 保证确认。",
        }, {
            "title": "7-KVCache与注意力优化_2026-07-17",
            "content": "KV Cache 是 LLM 推理优化：自回归生成时缓存历史 token 的 K/V 向量，避免每步"
                       "重复计算，显存换算力的经典手段。配合 PagedAttention 减少显存碎片。",
        }],
        "sufficient": False,
        "keywords": ["synchronized", "锁升级"],
        "category": "java_concurrency",
        "note": "完全不沾边：问 synchronized 却检索到 Kafka/KV Cache 文档",
    },
    # ---- 不充分扩充（2026-08-09 自造，目标 100 条：充分 50 / 不充分 50）----
    {
        "question": "ZGC的着色指针和读屏障是怎么工作的？",
        "documents": [{
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16",
            "content": "ThreadPoolExecutor 六大参数：corePoolSize、maximumPoolSize、workQueue、"
                       "keepAliveTime、threadFactory、handler。任务先复用核心线程，队列满后"
                       "扩容，仍满走拒绝策略。",
        }, {
            "title": "10-Redis持久化机制_2026-07-20",
            "content": "Redis 两种持久化：RDB 定时快照（恢复快可能丢数据），AOF 追加命令日志"
                       "（默认每秒 fsync）。生产一般 RDB + AOF 组合。",
        }],
        "sufficient": False,
        "keywords": ["ZGC", "着色指针"],
        "category": "java_gc",
        "note": "完全不沾边：问 ZGC 却检索到线程池/Redis 文档",
    },
    {
        "question": "CMS 并发模式失败是怎么触发的？",
        "documents": [{
            "title": "5-Kafka消息可靠性与高吞吐设计_2026-07-15",
            "content": "Kafka ISR 机制：Leader 负责读写，Follower 拉取同步，超时未同步被踢出"
                       "ISR；acks=all 配合 min.insync.replicas=2 保证不丢消息。",
        }, {
            "title": "7-Netty高性能IO与Reactor线程模型_2026-07-17",
            "content": "Netty 基于 Reactor 线程模型：Boss 线程 accept 连接，Worker 线程处理"
                       "读写事件，零拷贝通过堆外内存实现。",
        }],
        "sufficient": False,
        "keywords": ["CMS", "并发模式失败"],
        "category": "java_gc",
        "note": "完全不沾边：问 CMS 却检索到 Kafka/Netty 文档",
    },
    {
        "question": "线程池拒绝策略的背压原理是什么？",
        "documents": [{
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23",
            "content": "HashMap 底层数组+链表+红黑树，冲突链式挂载，链表超 8 树化；默认容量 16、"
                       "负载因子 0.75。",
        }, {
            "title": "22-JWT认证机制详解_2026-08-02",
            "content": "JWT 三段式：Header.Payload.Signature，验签通过即信任；无状态免查询，"
                       "但签发后无法撤销。",
        }],
        "sufficient": False,
        "keywords": ["拒绝策略", "背压"],
        "category": "java_concurrency",
        "note": "完全不沾边：问线程池拒绝策略却检索到 HashMap/JWT 文档",
    },
    {
        "question": "HashMap 树化的阈值为什么是 8？",
        "documents": [{
            "title": "15-AQS抽象队列同步器与ReentrantLock实现原理_2026-07-26",
            "content": "AQS 核心是 volatile int state + CLH FIFO 等待队列；ReentrantLock 通过"
                       "CAS 抢锁、park 阻塞、signal 唤醒实现。",
        }, {
            "title": "16-volatile与Java内存模型JMM_2026-07-27",
            "content": "volatile 两大作用：可见性（内存屏障刷新主内存）与有序性（禁止指令重排）；"
                       "不保证原子性。",
        }],
        "sufficient": False,
        "keywords": ["HashMap", "树化"],
        "category": "java_collection",
        "note": "主题错位：文档讲 AQS/volatile，不含 HashMap 树化内容",
    },
    {
        "question": "ConcurrentHashMap 扩容时怎么保证并发安全？",
        "documents": [{
            "title": "8-MySQL索引原理与B+树_2026-07-18",
            "content": "B+ 树索引：非叶子只存键，叶子存数据有序串联；树高矮磁盘 IO 少，"
                       "支持范围查询。",
        }, {
            "title": "35-Docker镜像分层与构建优化_2026-08-15",
            "content": "镜像分层：每个指令一层只读层，OverlayFS 堆叠统一视图；层可复用共享，"
                       "可写层 CoW。",
        }],
        "sufficient": False,
        "keywords": ["ConcurrentHashMap", "扩容"],
        "category": "java_collection",
        "note": "完全不沾边：问 ConcurrentHashMap 扩容却检索到 MySQL/Docker 文档",
    },
    {
        "question": "B+ 树和红黑树在数据库场景下的取舍是什么？",
        "documents": [{
            "title": "17-Spring事务管理与传播行为_2026-07-28",
            "content": "Spring 传播行为：REQUIRED/REQUIRES_NEW/NESTED 等；事务失效常见："
                       "同类调用、非 public、异常被吞。",
        }, {
            "title": "43-ThreadLocal原理与内存泄漏_2026-08-23",
            "content": "ThreadLocal：线程持有 ThreadLocalMap，key 弱引用 value 强引用导致泄漏；"
                       "用完 remove()。",
        }],
        "sufficient": False,
        "keywords": ["B+树", "红黑树"],
        "category": "mysql",
        "note": "完全不沾边：问索引数据结构却检索到 Spring/ThreadLocal 文档",
    },
    {
        "question": "MVCC 的 Read View 是怎么生成的？",
        "documents": [{
            "title": "20-Netty编解码与粘包拆包_2026-07-31",
            "content": "TCP 流式协议无消息边界，粘包拆包用 LengthFieldBasedFrameDecoder 按"
                       "长度域切帧解决。",
        }, {
            "title": "41-CAS与原子操作原理_2026-08-21",
            "content": "CAS 比较并交换，CPU 原子指令；缺点 ABA、自旋开销、单变量原子。",
        }],
        "sufficient": False,
        "keywords": ["MVCC", "ReadView"],
        "category": "mysql",
        "note": "完全不沾边：问 MVCC 却检索到 Netty/CAS 文档",
    },
    {
        "question": "EXPLAIN 里 Using filesort 怎么优化掉？",
        "documents": [{
            "title": "47-JWT刷新与双Token机制_2026-08-27",
            "content": "双 Token：Access 短效 + Refresh 长效，刷新接口校验后签发新 Access，"
                       "Refresh 可撤销。",
        }, {
            "title": "49-分布式ID生成方案_2026-08-29",
            "content": "雪花算法：时间戳+机器 ID+序列号 64 位有序；时钟回拨需处理。",
        }],
        "sufficient": False,
        "keywords": ["filesort", "EXPLAIN"],
        "category": "mysql",
        "note": "完全不沾边：问执行计划优化却检索到 JWT/分布式 ID 文档",
    },
    {
        "question": "缓存雪崩的过期时间随机化是怎么做的？",
        "documents": [{
            "title": "37-JVM类加载机制与双亲委派_2026-08-17",
            "content": "双亲委派：类加载先交父加载器，保证核心类一致与防冒充；SPI 用线程上下文"
                       "类加载器打破。",
        }, {
            "title": "31-微服务注册中心与服务发现_2026-08-11",
            "content": "服务发现：提供者注册+心跳，消费者拉取缓存；Nacos 支持 AP/CP 切换。",
        }],
        "sufficient": False,
        "keywords": ["雪崩", "随机化"],
        "category": "redis",
        "note": "完全不沾边：问缓存雪崩却检索到类加载/服务发现文档",
    },
    {
        "question": "哨兵选举新主节点的依据是什么？",
        "documents": [{
            "title": "45-JVMOOM类型与应对_2026-08-25",
            "content": "OOM 类型：heap space、GC overhead、Metaspace、native thread、Direct "
                       "buffer；应对：dump 分析、调参、修泄漏。",
        }, {
            "title": "27-接口幂等性设计实践_2026-08-07",
            "content": "幂等：唯一索引、幂等表、状态机条件更新、SETNX 防重；重复执行结果一致。",
        }],
        "sufficient": False,
        "keywords": ["哨兵", "选举"],
        "category": "redis",
        "note": "完全不沾边：问哨兵选举却检索到 OOM/幂等文档",
    },
    {
        "question": "REQUIRES_NEW 传播行为什么时候用？",
        "documents": [{
            "title": "13-Redis缓存三大问题与解决方案_2026-07-24",
            "content": "缓存三大问题：穿透（布隆过滤+空值）、击穿（互斥锁）、雪崩（随机过期）；"
                       "一致性用 Cache Aside+延迟双删。",
        }, {
            "title": "36-Kubernetes调度器原理_2026-08-16",
            "content": "Pod 调度：过滤（资源/污点/亲和）+ 打分（加权评分）；污点 Taints+容忍"
                       "Tolerations 隔离节点。",
        }],
        "sufficient": False,
        "keywords": ["REQUIRES_NEW", "传播行为"],
        "category": "spring",
        "note": "完全不沾边：问事务传播却检索到 Redis 缓存/K8s 文档",
    },
    {
        "question": "Bean 销毁阶段 @PreDestroy 的执行顺序是什么？",
        "documents": [{
            "title": "29-Kafka分区与路由机制_2026-08-09",
            "content": "Kafka 分区路由：指定分区直接用，key 哈希取模保有序，无 key 轮询；"
                       "分区数平衡并行度与开销。",
        }, {
            "title": "39-Java内存模型JMM详解_2026-08-19",
            "content": "JMM：线程工作内存与主内存交互，解决可见性/原子性/有序性；内存屏障"
                       "四类 Load/Store。",
        }],
        "sufficient": False,
        "keywords": ["PreDestroy", "销毁"],
        "category": "spring",
        "note": "完全不沾边：问 Bean 销毁却检索到 Kafka/JMM 文档",
    },
    {
        "question": "AOP 多个切面的执行顺序怎么控制？",
        "documents": [{
            "title": "44-Java内存泄漏排查实战_2026-08-24",
            "content": "泄漏排查：监控 heap 曲线、jmap 堆转储、MAT 分析大对象与持有链、"
                       "对比快照定位增长对象。",
        }, {
            "title": "33-熔断降级与Sentinel原理_2026-08-13",
            "content": "熔断：失败率超阈值打开快速失败，半开放探测；降级主动牺牲非核心保主链路。",
        }],
        "sufficient": False,
        "keywords": ["AOP", "切面顺序"],
        "category": "spring",
        "note": "完全不沾边：问切面顺序却检索到内存泄漏/熔断文档",
    },
    {
        "question": "Netty 的零拷贝是怎么实现的？",
        "documents": [{
            "title": "9-MySQL事务与隔离级别_2026-07-19",
            "content": "MySQL 隔离级别：读未提交/读已提交/可重复读（默认）/串行化；MVCC 快照读"
                       "+ 间隙锁解决幻读。",
        }, {
            "title": "37-JVM类加载机制与双亲委派_2026-08-17",
            "content": "类加载五阶段：加载、验证、准备、解析、初始化；static 块在初始化执行。",
        }],
        "sufficient": False,
        "keywords": ["零拷贝", "Zero-Copy"],
        "category": "netty",
        "note": "完全不沾边：问 Netty 零拷贝却检索到 MySQL/类加载文档",
    },
    {
        "question": "TLS 1.3 相比 1.2 少了哪次往返？",
        "documents": [{
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16",
            "content": "ThreadPoolExecutor 六大参数：corePoolSize、maximumPoolSize、workQueue、"
                       "keepAliveTime、threadFactory、handler。",
        }, {
            "title": "14-Redis高可用架构：主从+哨兵_2026-07-25",
            "content": "哨兵监控主从心跳，主下线多数同意后故障转移；哨兵集群至少 3 个防脑裂。",
        }],
        "sufficient": False,
        "keywords": ["TLS1.3", "往返"],
        "category": "network",
        "note": "完全不沾边：问 TLS 握手却检索到线程池/Redis 文档",
    },
    {
        "question": "JWT 的签名算法混淆攻击怎么防御？",
        "documents": [{
            "title": "46-FullGC频繁排查实践_2026-08-26",
            "content": "Full GC 排查：GC 日志确认频率，jstat 看区使用率，堆转储 MAT 分析；"
                       "常见根因：大对象、泄漏、缓存无上限。",
        }, {
            "title": "35-Docker镜像分层与构建优化_2026-08-15",
            "content": "镜像分层：指令生成只读层，OverlayFS 堆叠；多阶段构建减小体积。",
        }],
        "sufficient": False,
        "keywords": ["JWT", "算法混淆"],
        "category": "auth",
        "note": "完全不沾边：问 JWT 安全却检索到 Full GC/Docker 文档",
    },
    {
        "question": "Session 过期时间设多长合适？",
        "documents": [{
            "title": "41-CAS与原子操作原理_2026-08-21",
            "content": "CAS 比较并交换原子指令；缺点：ABA、自旋开销、单变量原子；"
                       "AtomicStampedReference 解决 ABA。",
        }, {
            "title": "30-Kafka消费者组与重平衡_2026-08-10",
            "content": "重平衡触发：消费者增减、分区数变化；KIP-429 增量式减少消费停滞。",
        }],
        "sufficient": False,
        "keywords": ["Session", "过期"],
        "category": "auth",
        "note": "完全不沾边：问 Session 过期却检索到 CAS/Kafka 文档",
    },
    {
        "question": "2PC 协调者宕机后事务怎么恢复？",
        "documents": [{
            "title": "42-原子类与LongAdder实现_2026-08-22",
            "content": "AtomicInteger 循环 CAS 更新；LongAdder 分段 Cell 分散竞争，高并发计数"
                       "更优。",
        }, {
            "title": "32-Nacos配置中心原理_2026-08-12",
            "content": "Nacos 动态刷新：长轮询挂起等待变更，配置变化拉取新值，@RefreshScope "
                       "重建 Bean。",
        }],
        "sufficient": False,
        "keywords": ["2PC", "协调者"],
        "category": "distributed",
        "note": "完全不沾边：问 2PC 恢复却检索到原子类/Nacos 文档",
    },
    {
        "question": "Seata 的 undo_log 是怎么生成和使用的？",
        "documents": [{
            "title": "21-HTTPS与TLS握手原理_2026-08-01",
            "content": "TLS 握手：ClientHello/ServerHello 交换随机数与证书，预主密钥加密传输，"
                       "双方计算会话密钥。",
        }, {
            "title": "38-JVM类加载过程详解_2026-08-18",
            "content": "类加载五阶段：加载、验证、准备（静态变量默认值）、解析、初始化"
                       "（static 块）。",
        }],
        "sufficient": False,
        "keywords": ["Seata", "undo_log"],
        "category": "distributed",
        "note": "完全不沾边：问 Seata 回滚日志却检索到 HTTPS/类加载文档",
    },
    {
        "question": "Redisson 看门狗续期失败的场景有哪些？",
        "documents": [{
            "title": "8-MySQL索引原理与B+树_2026-07-18",
            "content": "B+ 树：非叶子只存键高扇出，叶子有序串联；聚簇索引叶子存整行，"
                       "二级索引回表。",
        }, {
            "title": "43-ThreadLocal原理与内存泄漏_2026-08-23",
            "content": "ThreadLocalMap Entry 的 key 弱引用 value 强引用，ThreadLocal 回收后 "
                       "value 不释放导致泄漏，用完 remove()。",
        }],
        "sufficient": False,
        "keywords": ["看门狗", "续期"],
        "category": "distributed",
        "note": "完全不沾边：问分布式锁续期却检索到 MySQL 索引/ThreadLocal 文档",
    },
    {
        "question": "幂等唯一索引冲突时怎么处理？",
        "documents": [{
            "title": "2-ZGC超低停顿垃圾收集器原理_2026-07-12",
            "content": "ZGC 着色指针 + 读屏障，停顿不随堆增长；适合大堆低延迟场景。",
        }, {
            "title": "34-限流算法：令牌桶与漏桶_2026-08-14",
            "content": "令牌桶允许突发（桶容量），漏桶恒定速率；分布式限流用 Redis Lua 原子"
                       "计数。",
        }],
        "sufficient": False,
        "keywords": ["幂等", "唯一索引"],
        "category": "distributed",
        "note": "完全不沾边：问幂等索引冲突却检索到 ZGC/限流文档",
    },
    {
        "question": "MQ 消费端怎么保证消息不重复处理？",
        "documents": [{
            "title": "17-Spring事务管理与传播行为_2026-07-28",
            "content": "Spring 传播行为：REQUIRED 默认有则加入；事务失效：同类自调用、"
                       "异常被吞、非 public。",
        }, {
            "title": "44-Java内存泄漏排查实战_2026-08-24",
            "content": "泄漏排查：堆转储 + MAT 分析 Dominator Tree，对比快照定位增长对象。",
        }],
        "sufficient": False,
        "keywords": ["MQ", "重复消费"],
        "category": "mq",
        "note": "完全不沾边：问消费幂等却检索到 Spring 事务/内存泄漏文档",
    },
    {
        "question": "Kafka 相同 key 的消息怎么保证有序？",
        "documents": [{
            "title": "18-SpringBean生命周期与扩展点_2026-07-29",
            "content": "Bean 生命周期：实例化、属性填充、Aware 回调、BeanPostProcessor、"
                       "初始化、销毁；扩展点 BPP/BPFP。",
        }, {
            "title": "45-JVMOOM类型与应对_2026-08-25",
            "content": "OOM 类型：heap space、Metaspace、native thread、Direct buffer；"
                       "容器场景注意 cgroup 与 -Xmx 匹配。",
        }],
        "sufficient": False,
        "keywords": ["Kafka", "有序"],
        "category": "kafka",
        "note": "完全不沾边：问 Kafka 消息有序却检索到 Spring Bean/OOM 文档",
    },
    {
        "question": "重平衡期间消费者会暂停消费多久？",
        "documents": [{
            "title": "16-volatile与Java内存模型JMM_2026-07-27",
            "content": "volatile 可见性 + 有序性，写插入 StoreLoad 屏障；双检锁单例依赖此语义。",
        }, {
            "title": "35-Docker镜像分层与构建优化_2026-08-15",
            "content": "镜像构建优化：指令顺序影响缓存命中，多阶段构建分离编译与运行环境。",
        }],
        "sufficient": False,
        "keywords": ["重平衡", "暂停"],
        "category": "kafka",
        "note": "完全不沾边：问重平衡却检索到 volatile/Docker 文档",
    },
    {
        "question": "注册中心的健康检查心跳超时怎么调？",
        "documents": [{
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23",
            "content": "HashMap 1.7 头插法并发扩容可能死循环，1.8 尾插法 + 红黑树优化；"
                       "ConcurrentHashMap CAS+synchronized 桶级锁。",
        }, {
            "title": "47-JWT刷新与双Token机制_2026-08-27",
            "content": "Refresh Token 存 Redis 白名单版本号递增实现轮换失效，Access 短过期"
                       "自然失效。",
        }],
        "sufficient": False,
        "keywords": ["健康检查", "心跳"],
        "category": "microservice",
        "note": "完全不沾边：问注册中心心跳却检索到 HashMap/JWT 文档",
    },
    {
        "question": "Nacos 长轮询的超时时间设多少？",
        "documents": [{
            "title": "3-CMS垃圾收集器原理与缺陷分析_2026-07-13",
            "content": "CMS 并发标记 + 并发清除，缺陷：并发模式失败、碎片、浮动垃圾；"
                       "JDK 14 移除。",
        }, {
            "title": "27-接口幂等性设计实践_2026-08-07",
            "content": "幂等方案：唯一键约束、幂等表、状态机、SETNX；幂等键有效期覆盖重试窗口。",
        }],
        "sufficient": False,
        "keywords": ["长轮询", "超时"],
        "category": "microservice",
        "note": "完全不沾边：问 Nacos 长轮询却检索到 CMS/幂等文档",
    },
    {
        "question": "熔断器半开状态放多少探测流量？",
        "documents": [{
            "title": "43-ThreadLocal原理与内存泄漏_2026-08-23",
            "content": "ThreadLocal 线程隔离原理；线程池复用场景不 remove 会脏数据串线，"
                       "需 finally remove()。",
        }, {
            "title": "29-Kafka分区与路由机制_2026-08-09",
            "content": "Kafka 分区路由：key 哈希取模保同 key 有序，无 key 轮询/黏性分区；"
                       "分区数平衡并行度。",
        }],
        "sufficient": False,
        "keywords": ["半开", "探测"],
        "category": "microservice",
        "note": "完全不沾边：问熔断探测却检索到 ThreadLocal/Kafka 文档",
    },
    {
        "question": "令牌桶的突发能力由什么决定？",
        "documents": [{
            "title": "8-MySQL索引原理与B+树_2026-07-18",
            "content": "B+ 树索引优化：最左前缀、覆盖索引、区分度；like '%xx' 前缀模糊"
                       "无法用索引。",
        }, {
            "title": "25-Seata分布式事务方案_2026-08-05",
            "content": "Seata AT 模式：TC/TM/RM 三角色，undo_log 快照反向补偿回滚，业务零侵入。",
        }],
        "sufficient": False,
        "keywords": ["令牌桶", "突发"],
        "category": "microservice",
        "note": "完全不沾边：问限流算法却检索到 MySQL 索引/Seata 文档",
    },
    {
        "question": "Docker 的 Copy-on-Write 是怎么触发的？",
        "documents": [{
            "title": "15-AQS抽象队列同步器与ReentrantLock实现原理_2026-07-26",
            "content": "AQS：state + CLH 队列，独占/共享两种模式；ReentrantLock 重入计数，"
                       "公平/非公平。",
        }, {
            "title": "13-Redis缓存三大问题与解决方案_2026-07-24",
            "content": "缓存穿透布隆过滤 + 空值缓存，击穿互斥锁，雪崩随机过期；"
                       "Cache Aside 先更新 DB 再删缓存。",
        }],
        "sufficient": False,
        "keywords": ["Copy-on-Write", "容器"],
        "category": "devops",
        "note": "完全不沾边：问 Docker CoW 却检索到 AQS/Redis 缓存文档",
    },
    {
        "question": "K8s 污点容忍的调度优先级是怎样的？",
        "documents": [{
            "title": "39-Java内存模型JMM详解_2026-08-19",
            "content": "JMM：工作内存与主内存交互，Load/Store 八种操作；volatile 插入"
                       "StoreStore/StoreLoad 屏障。",
        }, {
            "title": "24-分布式事务：2PC与3PC详解_2026-08-04",
            "content": "2PC：准备阶段参与者返回 yes/no，提交阶段广播 commit/rollback；"
                       "缺点同步阻塞、协调者单点。",
        }],
        "sufficient": False,
        "keywords": ["污点", "容忍"],
        "category": "devops",
        "note": "完全不沾边：问 K8s 污点容忍却检索到 JMM/2PC 文档",
    },
    {
        "question": "应用类加载器之间的隔离是怎么实现的？",
        "documents": [{
            "title": "34-限流算法：令牌桶与漏桶_2026-08-14",
            "content": "令牌桶固定速率生成令牌，桶容量允许突发；漏桶恒定速率输出削峰；"
                       "分布式限流 Redis Lua。",
        }, {
            "title": "12-HashMap与ConcurrentHashMap底层原理_2026-07-23",
            "content": "HashMap 数组+链表+红黑树；扩容 2 倍 rehash；1.8 尾插法防死循环。",
        }],
        "sufficient": False,
        "keywords": ["类加载器", "隔离"],
        "category": "jvm",
        "note": "完全不沾边：问类加载器隔离却检索到限流/HashMap 文档",
    },
    {
        "question": "静态变量的初始化顺序怎么保证？",
        "documents": [{
            "title": "5-Kafka消息可靠性与高吞吐设计_2026-07-15",
            "content": "Kafka ISR 机制 + acks 级别 + 手动提交 offset；高吞吐靠顺序写、"
                       "页缓存、零拷贝。",
        }, {
            "title": "35-Docker镜像分层与构建优化_2026-08-15",
            "content": "镜像分层复用 + 构建缓存，指令顺序影响命中率；多阶段构建减小体积。",
        }],
        "sufficient": False,
        "keywords": ["静态变量", "初始化顺序"],
        "category": "jvm",
        "note": "完全不沾边：问静态初始化却检索到 Kafka/Docker 文档",
    },
    {
        "question": "什么场景下 volatile 的可见性会失效？",
        "documents": [{
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16",
            "content": "线程池核心参数与工作流程：核心线程→队列→扩容→拒绝；拒绝策略"
                       "AbortPolicy/CallerRunsPolicy 等。",
        }, {
            "title": "2-ZGC超低停顿垃圾收集器原理_2026-07-12",
            "content": "ZGC 着色指针编码 GC 状态进指针高位，读屏障修正，停顿恒定 10ms 级。",
        }],
        "sufficient": False,
        "keywords": ["volatile", "可见性"],
        "category": "jvm",
        "note": "完全不沾边：问 volatile 却检索到线程池/ZGC 文档",
    },
    {
        "question": "Happens-Before 的传递性怎么应用？",
        "documents": [{
            "title": "11-MySQL慢查询优化实践_2026-07-22",
            "content": "慢查询优化：慢日志定位、EXPLAIN 看 type/rows、加索引/覆盖索引、"
                       "改写 SQL。",
        }, {
            "title": "20-Netty编解码与粘包拆包_2026-07-31",
            "content": "Netty 解决粘包拆包：定长、分隔符、长度域解码器；自定义协议含魔数+"
                       "长度域。",
        }],
        "sufficient": False,
        "keywords": ["Happens-Before", "传递性"],
        "category": "jvm",
        "note": "完全不沾边：问 HB 规则却检索到慢查询/Netty 文档",
    },
    {
        "question": "CAS 在高竞争下的自旋开销怎么降？",
        "documents": [{
            "title": "32-Nacos配置中心原理_2026-08-12",
            "content": "Nacos 长轮询探测配置变更，变化后拉取新值刷新上下文；本地缓存兜底"
                       "不可用时用上次配置。",
        }, {
            "title": "45-JVMOOM类型与应对_2026-08-25",
            "content": "OOM 六类：heap、GC overhead、Metaspace、native thread、Direct "
                       "buffer、array size。",
        }],
        "sufficient": False,
        "keywords": ["CAS", "自旋"],
        "category": "java_concurrency",
        "note": "完全不沾边：问 CAS 自旋却检索到 Nacos/OOM 文档",
    },
    {
        "question": "LongAdder 的分段计数原理是什么？",
        "documents": [{
            "title": "17-Spring事务管理与传播行为_2026-07-28",
            "content": "Spring 传播行为七种；REQUIRES_NEW 挂起当前事务新建独立事务；"
                       "事务失效四个场景。",
        }, {
            "title": "22-JWT认证机制详解_2026-08-02",
            "content": "JWT 流程：签发、Bearer 携带、验签信任；无状态免查询，载荷明文"
                       "不放敏感信息。",
        }],
        "sufficient": False,
        "keywords": ["LongAdder", "分段"],
        "category": "java_concurrency",
        "note": "完全不沾边：问 LongAdder 却检索到 Spring 事务/JWT 文档",
    },
    {
        "question": "线程池场景下 ThreadLocal 脏数据怎么防？",
        "documents": [{
            "title": "8-MySQL索引原理与B+树_2026-07-18",
            "content": "B+ 树索引：聚簇/二级索引、回表、覆盖索引优化、最左前缀原则。",
        }, {
            "title": "33-熔断降级与Sentinel原理_2026-08-13",
            "content": "熔断打开快速失败给下游恢复时间，半开放探测；降级缓存兜底；"
                       "线程池隔离舱壁模式。",
        }],
        "sufficient": False,
        "keywords": ["ThreadLocal", "脏数据"],
        "category": "java_concurrency",
        "note": "完全不沾边：问 ThreadLocal 脏数据却检索到 MySQL 索引/熔断文档",
    },
    {
        "question": "MAT 的 Dominator Tree 怎么定位泄漏根？",
        "documents": [{
            "title": "30-Kafka消费者组与重平衡_2026-08-10",
            "content": "重平衡触发与 KIP-429 增量协调；重平衡风暴用调参缓解：max.poll.interval.ms、"
                       "静态成员。",
        }, {
            "title": "6-Java线程池ThreadPoolExecutor核心参数与工作原理_2026-07-16",
            "content": "ThreadPoolExecutor 六大参数与任务流转：核心线程→队列→扩容→拒绝。",
        }],
        "sufficient": False,
        "keywords": ["MAT", "DominatorTree"],
        "category": "jvm",
        "note": "完全不沾边：问 MAT 分析却检索到 Kafka 重平衡/线程池文档",
    },
    {
        "question": "Metaspace 溢出一般是什么原因？",
        "documents": [{
            "title": "14-Redis高可用架构：主从+哨兵_2026-07-25",
            "content": "哨兵监控心跳、多数同意判定客观下线、选举新主；哨兵 vs 集群："
                       "高可用 vs 分片扩展。",
        }, {
            "title": "15-AQS抽象队列同步器与ReentrantLock实现原理_2026-07-26",
            "content": "AQS 独占/共享模式，ReentrantLock 重入计数、公平锁 FIFO 排队、"
                       "Condition 多条件队列。",
        }],
        "sufficient": False,
        "keywords": ["Metaspace", "溢出"],
        "category": "jvm",
        "note": "完全不沾边：问 Metaspace OOM 却检索到 Redis 哨兵/AQS 文档",
    },
    {
        "question": "G1 的 IHOP 参数怎么调优？",
        "documents": [{
            "title": "21-HTTPS与TLS握手原理_2026-08-01",
            "content": "TLS 握手流程与 1.3 优化：1-RTT、会话复用、OCSP Stapling、ECC 证书。",
        }, {
            "title": "27-接口幂等性设计实践_2026-08-07",
            "content": "幂等：唯一索引、幂等表、状态机、SETNX；支付回调必须幂等防重复扣款。",
        }],
        "sufficient": False,
        "keywords": ["G1", "IHOP"],
        "category": "java_gc",
        "note": "完全不沾边：问 G1 调参却检索到 HTTPS/幂等文档",
    },
    {
        "question": "Refresh Token 轮换后旧的还能用吗？",
        "documents": [{
            "title": "46-FullGC频繁排查实践_2026-08-26",
            "content": "Full GC 排查：jstat -gcutil 看老年代高位，堆转储分析大对象/泄漏，"
                       "调整新生代比例与晋升阈值。",
        }, {
            "title": "34-限流算法：令牌桶与漏桶_2026-08-14",
            "content": "令牌桶允许突发、漏桶恒定速率；Redis Lua 原子计数实现分布式限流。",
        }],
        "sufficient": False,
        "keywords": ["RefreshToken", "轮换"],
        "category": "auth",
        "note": "完全不沾边：问 Token 轮换却检索到 Full GC/限流文档",
    },
    {
        "question": "延迟双删的第二次删除时机怎么定？",
        "documents": [{
            "title": "38-JVM类加载过程详解_2026-08-18",
            "content": "类加载五阶段：加载、验证、准备、解析、初始化；初始化触发条件："
                       "new、反射、静态访问。",
        }, {
            "title": "29-Kafka分区与路由机制_2026-08-09",
            "content": "Kafka 分区路由与分区数选择：max(生产吞吐, 消费并行度)，同 key "
                       "保序。",
        }],
        "sufficient": False,
        "keywords": ["延迟双删", "缓存"],
        "category": "redis",
        "note": "完全不沾边：问缓存双删却检索到类加载/Kafka 分区文档",
    },
    {
        "question": "雪花算法的时钟回拨一般怎么处理？",
        "documents": [{
            "title": "17-Spring事务管理与传播行为_2026-07-28",
            "content": "Spring 事务传播行为与失效场景：自调用、非 public、异常被吞、"
                       "rollbackFor 未配置。",
        }, {
            "title": "16-volatile与Java内存模型JMM_2026-07-27",
            "content": "volatile 内存屏障与可见性/有序性；不保证原子性，i++ 仍需锁。",
        }],
        "sufficient": False,
        "keywords": ["雪花算法", "时钟回拨"],
        "category": "distributed",
        "note": "完全不沾边：问雪花回拨却检索到 Spring 事务/volatile 文档",
    },
    {
        "question": "分布式事务和本地事务的区别是什么？",
        "documents": [{
            "title": "18-SpringBean生命周期与扩展点_2026-07-29",
            "content": "Spring Bean 生命周期：实例化、属性填充、Aware 回调、BeanPostProcessor、"
                       "初始化、销毁；AOP 代理挂在 BPP 阶段。",
        }, {
            "title": "40-Happens-Before规则_2026-08-20",
            "content": "Happens-Before 八条规则：程序次序、管程锁、volatile、线程启动/终止、"
                       "中断、对象终结、传递性。",
        }],
        "sufficient": False,
        "keywords": ["分布式事务", "本地事务"],
        "category": "distributed",
        "note": "完全不沾边：问分布式事务却检索到 Spring Bean/HB 规则文档",
    },
]


def load_sufficiency_dataset() -> list[dict]:
    """加载充分性标注集，校验结构

    Returns:
        样本列表，每项含 question / documents / sufficient（可含 keywords/category/note）

    Raises:
        ValueError: 样本 < 10、question 为空、documents 为空、sufficient 非 bool、两类不齐全
    """
    data = SUFFICIENCY_DATASET
    if len(data) < 10:
        raise ValueError(f"充分性评测集过小：需 ≥ 10 条，当前 {len(data)}")
    for item in data:
        if not item.get("question", "").strip():
            raise ValueError(f"充分性评测集存在空 question: {item}")
        if not item.get("documents"):
            raise ValueError(f"充分性评测集存在空 documents: {item.get('question', '')[:30]}")
        if not isinstance(item.get("sufficient"), bool):
            raise ValueError(f"sufficient 须为 bool: {item.get('question', '')[:30]}")
    counts = {s: sum(1 for i in data if i["sufficient"] == s) for s in (True, False)}
    if not all(counts.values()):
        raise ValueError(f"充分性评测集缺少类别（充分/不充分须都有）: {counts}")
    return data


def heuristic_judge(query: str, documents: list[dict], keywords: list[str]) -> bool:
    """fixture 启发式判断器：关键词命中判定充分性（确定性，不依赖 LLM/DB）

    问题核心术语（keywords）任一出现在文档内容中 → 充分；否则不充分。
    仅用于 fixture 模式演示评测管线，不代表真实判断能力。

    Args:
        query: 用户问题
        documents: 检索文档列表
        keywords: 该问题核心术语（样本标注字段）

    Returns:
        True=充分 / False=不充分
    """
    if not documents:
        return False
    text = "".join(d.get("content", "") for d in documents)
    return any(kw in text for kw in keywords)


async def judge_sufficiency(query: str, documents: list[dict]) -> bool:
    """真实模式：调用 reflector.check_sufficiency 判断充分性

    返回结构兼容 check_sufficiency 契约（sufficient/reason/rewritten_query）；
    失败降级由 reflector 内部兜底（默认充分，防死循环）。

    Args:
        query: 用户问题
        documents: 检索文档列表

    Returns:
        True=充分 / False=不充分
    """
    result = await reflector.check_sufficiency(query, documents)
    return bool(result.get("sufficient", True))


def extract_sufficiency_label(item: dict) -> bool:
    """取样本标注标签（bool），统一数据访问"""
    return item["sufficient"]


def label_str(sufficient: bool) -> str:
    """bool 标签 → 语义字符串（混淆矩阵/报告用）"""
    return "sufficient" if sufficient else "insufficient"


async def run_eval(judge=None, dataset=None) -> tuple[dict, list[dict], list[dict]]:
    """执行一次充分性评估

    Args:
        judge: 判断协程 (query, documents) -> bool 充分性；默认走 reflector（真实模式）
        dataset: 评测样本列表；默认 load_sufficiency_dataset()

    Returns:
        (scores, per_question, skipped)
        - scores: accuracy + 混淆矩阵 + per-class 指标 + 统计（含 insufficient_recall 重点项）
        - per_question: 每题明细（label/predicted/correct）
        - skipped: 判断失败的样本记录
    """
    items = dataset if dataset is not None else load_sufficiency_dataset()
    judge_fn = judge if judge is not None else judge_sufficiency
    per_question: list[dict] = []
    skipped: list[dict] = []

    for i, item in enumerate(items):
        question = item["question"]
        documents = item["documents"]
        label = extract_sufficiency_label(item)
        try:
            predicted = await judge_fn(question, documents)
        except Exception as e:
            logger.error("[%d/%d] 充分性判断失败: %s — %s", i + 1, len(items), question[:40], e)
            skipped.append({"question": question, "label": label, "reason": f"error: {e}"})
            continue
        per_question.append({
            "question": question,
            "label": label,
            "predicted": bool(predicted),
            "correct": bool(predicted) == label,
            "category": item.get("category", ""),
        })

    conf = compute_confusion_matrix(
        [label_str(q["label"]) for q in per_question],
        [label_str(q["predicted"]) for q in per_question],
    )
    # 重点项：不充分类的 Recall（漏判"不充分"→ 基于无关文档硬答，最致命）
    insufficient_recall = conf["per_class"].get("insufficient", {}).get("recall", 0.0)
    scores = {
        "dataset_size": len(items),
        "evaluated": len(per_question),
        "skipped": len(skipped),
        "accuracy": conf["accuracy"],
        "confusion_matrix": conf["matrix"],
        "per_class": conf["per_class"],
        "classes": conf["classes"],
        "insufficient_recall": insufficient_recall,
    }
    return scores, per_question, skipped


async def record_eval_run(scores: dict, per_question: list[dict]) -> tuple[str, int]:
    """版本化落库：git_commit + rag_config 快照 + eval_type='sufficiency'

    Args:
        scores: 整体指标 dict
        per_question: 每题明细 list

    Returns:
        (commit, saved_id)；落库失败 saved_id=0（save_eval_run 内部已捕获并警告）
    """
    commit = get_git_commit()
    config_snapshot = await load_rag_config()
    saved_id = await save_eval_run(
        eval_type="sufficiency",
        git_commit=commit,
        config_snapshot=config_snapshot,
        scores=scores,
        per_question=per_question,
    )
    return commit, saved_id


def print_report(scores: dict, per_question: list[dict], skipped: list[dict],
                 saved_id: int, commit: str, fixture: bool) -> None:
    """打印评估报告到控制台：混淆矩阵 + per-class 指标 + 误判明细

    重点标出 insufficient Recall（漏判"不充分"最致命）。
    """
    classes = scores["classes"]
    print("\n" + "=" * 60)
    print("Golden Sufficiency Eval" + ("  [fixture 模式：启发式判断器，非真实指标]" if fixture else ""))
    print("=" * 60)
    print(f"Dataset: {scores['dataset_size']} samples | Evaluated: {scores['evaluated']} | Skipped: {scores['skipped']}")
    print("-" * 60)
    print(f"Accuracy: {scores['accuracy']:.4f}")
    print(f"==> 重点: insufficient Recall = {scores['insufficient_recall']:.4f} "
          f"（漏判'不充分' → 基于无关文档硬答，最致命）")
    print("-" * 60)
    print("Confusion Matrix (row=label, col=predicted):")
    print(f"{'':<14}" + "".join(f"{c[:10]:>14}" for c in classes))
    for label in classes:
        row = scores["confusion_matrix"][label]
        print(f"{label:<14}" + "".join(f"{row[pred]:>14}" for pred in classes))
    print("-" * 60)
    print("Per-Class Precision/Recall/F1:")
    for cls in classes:
        pc = scores["per_class"][cls]
        print(f"  {cls:<14} precision={pc['precision']:.4f} recall={pc['recall']:.4f} "
              f"f1={pc['f1']:.4f} support={pc['support']}")
    mis = [q for q in per_question if not q["correct"]]
    if mis:
        print("-" * 60)
        print(f"Misclassified ({len(mis)}):")
        for q in mis:
            print(f"  label={'sufficient' if q['label'] else 'insufficient':<14} "
                  f"-> {'sufficient' if q['predicted'] else 'insufficient':<14} | {q['question'][:40]}")
    if skipped:
        print("-" * 60)
        print(f"Skipped ({len(skipped)}):")
        for s in skipped:
            print(f"  [{s['reason'][:30]}] {s['question'][:50]}")
    print("=" * 60)
    if saved_id:
        print(f"Saved to eval_runs (id={saved_id}, commit={commit[:8]})")
    else:
        print("Not saved to eval_runs")
    print()


async def main() -> None:
    """评测脚本入口"""
    parser = argparse.ArgumentParser(description="Golden sufficiency 评测：充分性判断混淆矩阵 + 版本化回归")
    parser.add_argument("--fixture", action="store_true",
                        help="fixture 模式：启发式判断器（确定性，不依赖 LLM/DB），仅演示管线")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs 表")
    args = parser.parse_args()

    load_sufficiency_dataset()

    if args.fixture:
        async def _fixture_judge(query, documents):
            item = next(i for i in SUFFICIENCY_DATASET if i["question"] == query)
            return heuristic_judge(query, documents, item["keywords"])
        judge = _fixture_judge
    else:
        judge = None  # 默认走 reflector（真实模式）

    scores, per_question, skipped = await run_eval(judge=judge)

    saved_id = 0
    commit = ""
    if not args.no_save:
        commit, saved_id = await record_eval_run(scores, per_question)
    print_report(scores, per_question, skipped, saved_id, commit, fixture=args.fixture)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
