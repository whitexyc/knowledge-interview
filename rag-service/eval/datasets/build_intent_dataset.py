"""
人造意图训练集构造脚本（module-056 / ADR-0003 L4 数据扩充）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.datasets.build_intent_dataset         # 构造 + 校验 + 落盘
    python -m eval.datasets.build_intent_dataset --dump  # 打印分类统计（人工复核用）

产出:
    eval/intent_train_dataset.json —— [{"query", "intent", "note"?}, ...]
        query  用户问题（非空、全库唯一）
        intent knowledge / casual_chat / realtime（三类契约）
        note   样本分型（可选）：边界易混 / 边界易混 E2E bug 类 / 专有术语 / 口语化

标注指南（本数据集的构造口径）:
    knowledge（知识库问答）——问技术概念/原理/排查/选型，答案依赖知识库文档。
      判定要点：不能仅凭句子外壳判断——"你们网站有什么功能""你知识库里有没有
      讲线程池的"看似闲聊/系统问答，实为对知识库内容的问题，标 knowledge。
      四类 knowledge 子型：
        1. 边界易混（note="边界易混"）：闲聊/系统问答外壳 + 知识库内核，
           LLM 分类高频误判区（module-054/055 E2E 实测）；含 E2E bug 类
           "G1垃圾收集器的核心创新是什么？"（曾被 LLM 高置信误判 casual_chat）
        2. 专有术语（note="专有术语"）：G1/JVM/Redis/GC/MySQL/Kafka/Spring
           等专有名词 + 疑问句（模块-055 边界测试样本同型）
        3. 口语化（note="口语化"）：无术语的口语化知识问题（"内存老是溢出
           咋办"）——模拟真实用户措辞，分类器须在语义层识别而非关键词层
        4. 常规：标准书面问法
    casual_chat（闲聊寒暄）——问候/情绪/夸赞/询问机器人自身，不依赖知识库
      与实时信息。
    realtime（实时数据）——时间/日期/天气/新闻/行情/票务，答案随时间变化，
      不依赖知识库文档。

诚实边界（写入 changelog，module-056 声明）:
  - 本数据集为人工构造（非真实用户对话），用于方向性验证分类器能力；
  - 真实飞轮数据（前端 👍/👎）积累后仍可并入重训（IntentClassifier.fit
    接口已预留），届时以真实数据为准；
  - 训练/评测分离：golden_intent 100 条评测集不进入本训练集（防泄漏），
    本文件所有 query 与评测集字符串零重复（build 时校验强制）。

校验（build 时强制，不满足报错退出）:
  - 总样本 ≥300；每类 ≥80
  - 边界易混（note 含"边界易混"）≥30；专有术语 ≥30；口语化 ≥20
  - 含 E2E bug 类样本"G1垃圾收集器的核心创新是什么？"
  - query 非空、无重复；intent 合法；与 golden_intent 评测集零字符串重叠
"""
import argparse
import json
import sys

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVAL_DIR / "intent_train_dataset.json"

INTENT_CLASSES = ("knowledge", "casual_chat", "realtime")


# ---------------- 样本定义（构造口径见上方标注指南） ----------------

# 边界易混：闲聊/系统问答外壳，实为知识库问答（LLM 高频误判区）
_BOUNDARY: list[tuple[str, str]] = [
    ("你们网站都有哪些功能？", "边界易混"),
    ("你能做什么？能帮我解决问题吗", "边界易混"),
    ("这个知识库都收录了哪些主题？", "边界易混"),
    ("你的知识库里有讲线程池的吗？", "边界易混"),
    ("给我推荐一篇入门文档呗", "边界易混"),
    ("你们最近新增了什么内容？", "边界易混"),
    ("你都会哪些技术话题？", "边界易混"),
    ("从哪篇开始学比较好？", "边界易混"),
    ("有没有讲 Redis 的文档？", "边界易混"),
    ("我该怎么用这个知识库？", "边界易混"),
    ("你能解释一下你们系统里的概念吗？", "边界易混"),
    ("这个平台主要收录了哪些方面的内容？", "边界易混"),
    ("你们的问答系统支持什么问题？", "边界易混"),
    ("我想了解 JVM，你能给我介绍吗？", "边界易混"),
    ("G1垃圾收集器的核心创新是什么？", "边界易混 E2E bug 类"),
    ("帮我看看这个知识库能不能解决我的问题", "边界易混"),
    ("你们是怎么做技术问答的？", "边界易混"),
    ("这个网站里有没有面试题？", "边界易混"),
    ("怎么才能学好分布式系统？", "边界易混"),
    ("你们的文档里有没有讲高并发的？", "边界易混"),
    ("这系统能查 Kafka 吗？", "边界易混"),
    ("帮我推荐学习 Spring 的顺序", "边界易混"),
    ("你们的知识库支持哪些内容？", "边界易混"),
    ("我想搜一下 JVM 相关的内容", "边界易混"),
    ("这个系统能帮我查技术问题吗？", "边界易混"),
    ("有没有关于 MySQL 优化的文档？", "边界易混"),
    ("你们这里能解答哪些问题？", "边界易混"),
    ("我该从哪里开始了解这个系统？", "边界易混"),
    ("你的知识库和搜索引擎有什么区别？", "边界易混"),
    ("有没有讲消息队列的文档？", "边界易混"),
    ("你们有关于微服务的文档吗？", "边界易混"),
    ("你知识库里最火的话题是什么？", "边界易混"),
]

# 专有术语 + 疑问句（G1/JVM/Redis/GC/MySQL/Kafka 等）
_TERM: list[tuple[str, str]] = [
    ("G1 垃圾收集器的 Region 分区机制是怎样的？", "专有术语"),
    ("JVM 内存溢出怎么排查？", "专有术语"),
    ("Redis 的持久化机制有哪些？", "专有术语"),
    ("GC Roots 有哪些？", "专有术语"),
    ("CMS 和 G1 在停顿上的区别？", "专有术语"),
    ("Full GC 频繁怎么排查？", "专有术语"),
    ("JVM 类加载的双亲委派模型是什么？", "专有术语"),
    ("HashMap 1.7 与 1.8 的实现有什么不同？", "专有术语"),
    ("ConcurrentHashMap 为什么线程安全？", "专有术语"),
    ("ThreadLocal 为什么会有内存泄漏问题？", "专有术语"),
    ("CAS 和 synchronized 怎么选？", "专有术语"),
    ("线程池的核心参数怎么设置？", "专有术语"),
    ("AQS 的原理是什么？", "专有术语"),
    ("Spring Bean 的生命周期是什么？", "专有术语"),
    ("Spring 事务什么时候会失效？", "专有术语"),
    ("Spring Boot 的自动配置原理是什么？", "专有术语"),
    ("MySQL 的索引为什么用 B+ 树？", "专有术语"),
    ("MySQL 的隔离级别有哪些？", "专有术语"),
    ("MySQL 索引在什么情况下会失效？", "专有术语"),
    ("Redis 缓存穿透、击穿、雪崩有什么区别？", "专有术语"),
    ("Redis 哨兵和集群有什么区别？", "专有术语"),
    ("Redis 的过期删除策略是什么？", "专有术语"),
    ("Kafka 的 ISR 机制是怎么保证可靠性的？", "专有术语"),
    ("Kafka 消息会丢失吗？怎么保证不丢？", "专有术语"),
    ("RocketMQ 和 Kafka 怎么选型？", "专有术语"),
    ("Netty 的 Reactor 线程模型怎么工作的？", "专有术语"),
    ("TCP 三次握手为什么是三次？", "专有术语"),
    ("HTTP/2 相比 HTTP/1.1 有哪些改进？", "专有术语"),
    ("分布式事务两阶段提交是什么？", "专有术语"),
    ("分布式锁用 Redis 还是 Zookeeper？", "专有术语"),
    ("CAP 定理怎么权衡？", "专有术语"),
    ("Nacos 做注册中心的原理是什么？", "专有术语"),
    ("Sentinel 限流熔断怎么工作的？", "专有术语"),
    ("JWT 和 Session 的区别是什么？", "专有术语"),
    ("接口幂等性怎么设计", "专有术语"),
    ("Docker 和虚拟机有什么不同？", "专有术语"),
    ("Kubernetes 有哪些核心组件？", "专有术语"),
    ("ES 的倒排索引是什么？", "专有术语"),
    ("B+ 树为什么适合做数据库索引？", "专有术语"),
    ("消息队列怎么保证顺序消费？", "专有术语"),
]

# 口语化无术语知识问题（真实用户措辞，语义层识别场景）
_COLLOQUIAL: list[tuple[str, str]] = [
    ("内存老是溢出咋办", "口语化"),
    ("服务动不动就卡死怎么回事", "口语化"),
    ("数据库查询越来越慢怎么办", "口语化"),
    ("系统老是宕机是什么原因", "口语化"),
    ("接口请求特别慢怎么优化", "口语化"),
    ("线程老是死锁怎么办", "口语化"),
    ("网站一上线就崩了咋整", "口语化"),
    ("缓存加了还是慢是为什么", "口语化"),
    ("项目启动特别慢正常吗", "口语化"),
    ("日志里全是报错怎么排查", "口语化"),
    ("高并发下服务扛不住怎么办", "口语化"),
    ("数据重复了怎么去重", "口语化"),
    ("订单老是超时怎么办", "口语化"),
    ("两个系统间消息老是丢怎么办", "口语化"),
    ("登录状态老丢怎么回事", "口语化"),
    ("数据库连不上了怎么查", "口语化"),
    ("代码跑着跑着内存就满了咋办", "口语化"),
    ("请求量一大就 502 是什么原因", "口语化"),
    ("服务之间怎么通信比较好", "口语化"),
    ("图片上传老是失败怎么处理", "口语化"),
    ("前端页面加载很慢是后端的问题吗", "口语化"),
    ("定时任务老是重复执行怎么办", "口语化"),
    ("数据经常不一致怎么解决", "口语化"),
    ("系统升级后老出 bug 怎么办", "口语化"),
]

# 常规 knowledge（标准书面问法）
_KNOWLEDGE_PLAIN: list[str] = [
    "G1 垃圾收集器和 CMS 的区别是什么",
    "Java 内存模型 JMM 是什么？",
    "volatile 关键字的作用是什么？",
    "synchronized 的锁升级过程是怎样的？",
    "深拷贝和浅拷贝有什么区别？",
    "equals 和 hashCode 为什么要一起重写？",
    "String 为什么是不可变的？",
    "Java 8 的 Stream 流怎么用？",
    "泛型擦除是什么？",
    "反射的原理是什么？",
    "动态代理有哪些实现方式？",
    "数据库事务的 ACID 特性是什么？",
    "数据库的乐观锁和悲观锁有什么区别？",
    "聚簇索引和非聚簇索引有什么区别？",
    "大表分页查询怎么优化？",
    "慢 SQL 怎么定位和优化？",
    "Redis 的数据类型有哪些？",
    "Redis 为什么快？",
    "缓存一致性问题怎么解决？",
    "限流算法有哪些？",
    "熔断和降级的区别是什么？",
    "负载均衡有哪些策略？",
    "服务注册和发现是怎么工作的？",
    "微服务拆分的原则是什么？",
    "幂等和去重的区别是什么？",
    "雪花算法生成的是什么？",
    "令牌桶和漏桶算法的区别？",
    "灰度发布是什么？",
    "蓝绿部署是什么？",
    "混沌工程是什么？",
    "可观测性包括哪些部分？",
    "链路追踪的原理是什么？",
    "容器和镜像的区别是什么？",
    "什么是冷启动和热启动？",
    "函数式编程的特点是什么？",
    "设计模式有哪些分类？",
]

# casual_chat（闲聊寒暄，不依赖知识库/实时信息）
_CASUAL: list[str] = [
    "你好", "嗨喽", "哈喽", "在吗", "早上好", "中午好", "下午好", "晚上好",
    "吃饭了没", "睡了吗", "忙不忙", "最近咋样", "想你了", "好久不见",
    "你叫什么名字", "你是谁", "介绍一下你自己", "你多大了", "你是哪里人",
    "你住在哪", "你有女朋友吗", "你会唱歌吗", "唱首歌听听", "讲个笑话",
    "讲个段子", "猜个谜语", "夸夸我", "你是真人吗", "你是机器人吗", "再见",
    "拜拜啦", "晚安", "明天聊", "下次再聊", "先忙去了", "回头见", "谢谢",
    "谢谢帮忙", "太感谢了", "辛苦啦", "你真好", "真棒", "厉害厉害", "给你点赞",
    "太牛了", "好厉害呀", "棒棒的", "无聊死了", "好无聊啊", "陪我聊聊",
    "说点什么吧", "随便聊聊", "今天心情不好", "今天好累", "周末过得好吗",
    "工作好辛苦", "生活好难", "好开心呀", "哈哈哈哈", "哈哈", "哈哈哈笑死我了",
    "嗯嗯", "好的", "没问题呀", "明白了", "是的", "对呀", "没错", "我也是",
    "那好吧", "行吧", "算了算了", "咋了", "怎么啦", "发生什么事了", "说真的",
    "我觉得你说得对", "你这个回答我很满意", "继续保持", "换个话题聊聊",
    "说点别的", "我们聊点别的吧", "你听过什么好听的歌", "推荐一部电影",
    "有什么好看的剧", "你玩游戏吗", "你爱吃什么", "周末打算干嘛",
    "你平时喜欢做什么", "你开心吗", "你累吗", "你心情怎么样", "我心情不错",
    "我最近在学东西", "你懂我意思吗", "你能听懂我说的话吗", "你在听吗",
    "你还在吗", "你没反应了吗", "你好笨呀", "你真聪明", "开玩笑的", "别当真哈",
    "好呀好呀", "晚点聊",
]

# realtime（实时数据：时间/日期/天气/新闻/行情/票务）
_REALTIME: list[str] = [
    "现在几点了", "现在几点钟", "现在是几点", "北京时间几点", "现在几点半了",
    "今天是几号", "今天是星期几", "今天什么日子", "今年是哪一年", "今年几几年",
    "现在是什么季节", "现在是上午还是下午啊", "现在是早上还是晚上",
    "今天天气怎么样", "明天天气咋样", "今天天气好吗", "今天下雨吗",
    "明天会下雨吗", "今天气温多少度", "现在室外多少度", "今天温度高不高",
    "明天冷不冷", "这几天热不热", "后天天气怎么样", "周末天气适合出去玩吗",
    "这周天气怎么样", "明天有雾霾吗", "今天风力大吗", "今天紫外线强吗",
    "现在湿度多少", "今天空气质量怎么样", "今天有啥新闻", "有什么大新闻",
    "今天热点新闻是什么", "最新科技新闻", "今天股市行情怎么样",
    "今天大盘涨还是跌", "现在上证指数多少点", "今天 A 股行情如何",
    "现在美股怎么样", "今天港股怎么样", "今天基金涨了吗",
    "今天比特币价格多少", "今天金价多少", "现在黄金多少钱一克",
    "今天汇率是多少", "美元兑人民币现在多少", "今天油价涨了吗",
    "今天有什么比赛", "今晚有什么球赛", "今天足球比分多少", "今晚有直播吗",
    "现在比分多少了", "最近有什么新电影", "今天上映什么电影",
    "现在电影院有什么好看的", "最近有什么热门电视剧", "今天有什么综艺",
    "现在流行啥", "今年流行什么发型", "现在流行什么歌", "最近有什么热梗",
    "今天微博热搜是什么", "现在哪里堵车", "今天限行吗", "今天高速免费吗",
    "现在航班情况怎么样", "今天火车票还有吗", "离周末还有几天",
    "今天放假吗", "这周放不放假", "明天是工作日吗", "今年国庆放假几天",
    "距离五一还有多久", "今天适合出门吗", "现在几点下班合适",
    "现在北京时间几点", "现在几点几分", "现在是几点钟了", "今天几月几号",
    "现在是农历几号", "今年润几月", "现在什么时辰", "今天天气怎么样啊",
    "明天天气怎么样呢", "今天会下雪吗", "明天台风来吗", "今天地震了吗",
    "现在日出了吗", "今天日落几点", "现在月亮出来了没", "今天太阳几点下山",
    "现在几点放学", "现在几点了该睡了", "现在能出门吗", "今天有什么活动",
    "现在流行语是什么", "现在几点开会", "今天会议几点开始", "现在几点了呀",
]

# E2E bug 类样本（module-054/055 实测：曾被 LLM 高置信误判 casual_chat）
E2E_BUG_QUERY = "G1垃圾收集器的核心创新是什么？"


def build_dataset() -> list[dict]:
    """组装样本集（知识库四子型 + 闲聊 + 实时），保持定义顺序"""
    samples: list[dict] = []
    for query, note in _BOUNDARY + _TERM + _COLLOQUIAL:
        samples.append({"query": query, "intent": "knowledge", "note": note})
    for query in _KNOWLEDGE_PLAIN:
        samples.append({"query": query, "intent": "knowledge"})
    for query in _CASUAL:
        samples.append({"query": query, "intent": "casual_chat"})
    for query in _REALTIME:
        samples.append({"query": query, "intent": "realtime"})
    return samples


def validate(samples: list[dict]) -> None:
    """结构/数量/平衡/边界/术语/口语化校验，不满足报错退出"""
    queries = [s["query"] for s in samples]
    intents = [s["intent"] for s in samples]
    notes = [s.get("note", "") for s in samples]

    errors: list[str] = []
    if len(samples) < 300:
        errors.append(f"总样本不足 300：{len(samples)}")
    for cls in INTENT_CLASSES:
        if intents.count(cls) < 80:
            errors.append(f"类别 {cls} 不足 80：{intents.count(cls)}")
    if sum(1 for n in notes if "边界易混" in n) < 30:
        errors.append(f"边界易混样本不足 30：{sum(1 for n in notes if '边界易混' in n)}")
    if notes.count("专有术语") < 30:
        errors.append(f"专有术语样本不足 30：{notes.count('专有术语')}")
    if notes.count("口语化") < 20:
        errors.append(f"口语化样本不足 20：{notes.count('口语化')}")
    if E2E_BUG_QUERY not in queries:
        errors.append(f"缺少 E2E bug 类样本：{E2E_BUG_QUERY}")
    if any(not q.strip() for q in queries):
        errors.append("存在空 query")
    if len(set(queries)) != len(queries):
        errors.append("存在重复 query")
    if any(i not in INTENT_CLASSES for i in intents):
        errors.append(f"存在非法 intent：{set(intents) - set(INTENT_CLASSES)}")

    # 训练/评测分离：与 golden_intent 评测集零字符串重叠（防泄漏）
    try:
        from eval.golden.golden_intent import INTENT_DATASET
        eval_queries = {item["query"] for item in INTENT_DATASET}
        overlap = eval_queries & set(queries)
        if overlap:
            errors.append(f"与 golden_intent 评测集存在字符串重叠（防泄漏）: {sorted(overlap)[:5]}")
    except Exception as e:  # 评测集不可用时跳过该校验（不阻塞构造）
        print(f"[warn] golden_intent 评测集不可用，跳过分离校验: {e}")

    if errors:
        for err in errors:
            print(f"[error] {err}")
        sys.exit(1)


def print_stats(samples: list[dict]) -> None:
    notes = [s.get("note", "") for s in samples]
    print("=" * 50)
    print(f"Total: {len(samples)}")
    for cls in INTENT_CLASSES:
        print(f"  {cls:<12} {sum(1 for s in samples if s['intent'] == cls)}")
    print(f"  边界易混     {sum(1 for n in notes if '边界易混' in n)}"
          f"（含 E2E bug 类 {sum(1 for n in notes if 'E2E bug' in n)}）")
    print(f"  专有术语     {notes.count('专有术语')}")
    print(f"  口语化       {notes.count('口语化')}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="人造意图训练集构造（module-056）")
    parser.add_argument("--dump", action="store_true", help="只打印统计，不落盘")
    args = parser.parse_args()

    samples = build_dataset()
    if args.dump:
        print_stats(samples)
        return

    validate(samples)
    OUTPUT_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print_stats(samples)
    print(f"已落盘: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
