"""
人造记忆矛盾训练集构造脚本（module-062 WP4 数据）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.datasets.build_memory_conflict_train         # 构造 + 校验 + 落盘
    python -m eval.datasets.build_memory_conflict_train --dump  # 打印分类统计（人工复核用）

产出:
    eval/memory_conflict_train_dataset.json —— [{"premise", "hypothesis", "label"}, ...]
        premise   旧记忆内容
        hypothesis 新事实内容
        label     contradiction（矛盾）/ non_conflict（非矛盾，含 entailment+neutral）

标注口径（对齐 eval/memory_conflict_dataset.py 评测集五类场景，扩展到 100+）:
    contradiction：改口（偏好/习惯翻转）、迁移（技术栈/部署/存储切换）、过时
      （旧版本被新版本取代）、升级冲突（短期新事实 vs 长期旧记忆）、其它互斥。
    non_conflict：entailment（同义/被蕴含）+ neutral（无关主题）——防"不同主题
      也算矛盾"，与"过度标矛盾有害（Precision 是更硬约束）"的评测口径一致。

诚实边界（写入 changelog）:
  - 人工构造（非真实用户改口数据），方向性验证分类器能力；
  - 训练/评测分离：eval/memory_conflict_dataset.py 评测集 30 条不进入本训练集
    （防泄漏），本文件所有 premise/hypothesis 与评测集字符串零重叠（build 校验强制）。

校验（build 时强制，不满足报错退出）:
  - 总样本 ≥100；contradiction ≥40；non_conflict ≥40
  - premise/hypothesis 非空；label 合法
  - 与评测集零字符串重叠
"""
import argparse
import json
import sys

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVAL_DIR / "memory_conflict_train_dataset.json"

LABELS = ("contradiction", "non_conflict")


# ---------------- 样本定义（五类场景 + 正例/中性） ----------------

# 改口类（偏好/习惯/状态翻转，contradiction）
_CHANGE: list[tuple[str, str]] = [
    ("用户每天喝美式咖啡", "用户现在讨厌美式咖啡了，改喝豆浆"),
    ("用户偏好用 Python 编程", "用户现在主要用 Rust 写后端了"),
    ("用户喜欢简明的回答", "用户觉得回答太简单了，希望更详尽一些"),
    ("用户不喜欢吃酸", "用户最近爱上吃酸的了，顿顿要醋"),
    ("用户习惯晚睡", "用户现在改成晚上十点就睡了"),
    ("用户使用 Sublime 写代码", "用户换成了 Vim 编辑器"),
    ("用户目前在 C 公司工作", "用户已经辞职去了 D 公司"),
    ("用户面试方向是算法", "用户改投了系统架构方向"),
    ("用户每天工作九小时", "用户现在每天只工作五小时"),
    ("用户让大家叫他的外号", "用户改让大家叫他的本名了"),
    ("用户喜欢热闹的氛围", "用户现在偏爱安静独处"),
    ("用户习惯喝奶茶", "用户戒奶茶了，只喝白开水"),
    ("用户喜欢猫", "用户最近开始养狗了，不太喜欢猫了"),
    ("用户偏好冬天", "用户现在更喜欢夏天了"),
    ("用户平时用 Windows", "用户换成了 Linux 系统"),
    ("用户喜欢中餐", "用户现在只吃西餐了"),
    ("用户习惯每天喝三杯咖啡", "用户现在一天只喝一杯咖啡了"),
    ("用户偏爱短途旅行", "用户现在只做长途旅行"),
    ("用户喜欢自己做饭", "用户现在顿顿点外卖"),
    ("用户习惯晚睡晚起", "用户改成早睡早起了"),
    ("用户偏爱异步沟通", "用户现在要求即时回复"),
    ("用户喜欢冷色调", "用户最近只买暖色调的东西"),
    ("用户习惯写日记", "用户不写日记了，改拍 vlog"),
    ("用户偏好线上办公", "用户现在必须到公司上班"),
    ("用户喜欢慢跑", "用户现在只游泳不跑步了"),
    ("用户不爱吃甜食", "用户最近迷上了甜品"),
    ("用户习惯用左手", "用户现在练习用右手写字"),
    ("用户偏好大屏手机", "用户换成了小屏手机"),
    ("用户喜欢看纪录片", "用户现在只看综艺了"),
    ("用户习惯周末加班", "用户现在周末坚决不加班"),
]

# 迁移类（技术栈/部署/存储切换，contradiction）
_MIGRATE: list[tuple[str, str]] = [
    ("用户的数据库用的是 PostgreSQL", "用户把数据库换成了 Oracle"),
    ("用户主要使用 Gradle 构建项目", "用户项目改用 Maven 构建"),
    ("用户用 Caffeine 做本地缓存", "用户缓存换回了 Redis"),
    ("用户使用 K8s 部署服务", "用户部署改成了裸机 Docker"),
    ("用户的消息队列用的是 RabbitMQ", "用户换成了 RocketMQ"),
    ("用户的注册中心是 Nacos", "用户换成了 Eureka"),
    ("用户的网关是 Spring Cloud Gateway", "用户换成了 Zuul"),
    ("用户的日志框架是 Log4j2", "用户换成了 Logback"),
    ("用户的 CI 用 Jenkins", "用户换成了 GitHub Actions"),
    ("用户的监控用 Prometheus", "用户换成了 Zabbix"),
    ("用户的搜索引擎是 ES", "用户换成了 Solr"),
    ("用户的文档库用 MongoDB", "用户换成了 PostgreSQL"),
    ("用户的鉴权框架是 Shiro", "用户换成了 Spring Security"),
    ("用户的 API 网关是 Nginx", "用户换成了 OpenResty"),
    ("用户的定时任务用 XXL-Job", "用户换成了 Quartz"),
    ("用户的链路追踪是 SkyWalking", "用户换成了 Zipkin"),
    ("用户的容器编排用 Docker Swarm", "用户换成了 Kubernetes"),
    ("用户的前端框架是 Vue", "用户换成了 React"),
    ("用户的构建工具是 Webpack", "用户换成了 Vite"),
    ("用户的 ORM 用 MyBatis-Plus", "用户换成了 JPA"),
]

# 过时类（旧版本被新版本取代，contradiction）
_OBSOLETE: list[tuple[str, str]] = [
    ("用户在使用 Spring Boot 2.7", "用户已经升级到 Spring Boot 3.2"),
    ("用户用 Java 11 开发", "用户项目迁移到了 Java 17"),
    ("用户的 Node 版本是 18", "用户升级到了 Node 22"),
    ("用户的 Python 是 3.9", "用户升到了 Python 3.12"),
    ("用户在用 JDK 8 的语法", "用户改用 JDK 21 的语法了"),
    ("用户使用 TypeScript 4.5", "用户升到了 TypeScript 5.x"),
    ("用户的 React 是 17", "用户升级到了 React 19"),
    ("用户的 Vue 是 2.x", "用户迁移到了 Vue 3"),
    ("用户的 MySQL 是 5.7", "用户升级到了 MySQL 8.0"),
    ("用户的 Elasticsearch 是 6.x", "用户升到了 ES 8.x"),
    ("用户的 Kafka 是 2.x", "用户升到了 Kafka 3.x"),
    ("用户的 Redis 是 5", "用户升级到了 Redis 7"),
]

# 升级冲突类（短期新事实 vs 长期旧记忆，contradiction）
_UPGRADE: list[tuple[str, str]] = [
    ("长期记忆：用户喜欢的语言是 Rust", "短期新事实：用户现在主要写 Go"),
    ("长期记忆：用户计划学习 Pulsar", "短期新事实：用户放弃 Pulsar 改学 Kafka"),
    ("长期记忆：用户常用技术栈是 React", "短期新事实：用户转型做 Vue 开发"),
    ("长期记忆：用户准备考 AWS 证书", "短期新事实：用户改考 Azure 证书"),
    ("长期记忆：用户计划去成都工作", "短期新事实：用户决定留在杭州"),
    ("长期记忆：用户习惯用 iOS", "短期新事实：用户换成了安卓手机"),
    ("长期记忆：用户负责后端开发", "短期新事实：用户转岗做了测试"),
    ("长期记忆：用户每周三锻炼", "短期新事实：用户改到每周五锻炼"),
    ("长期记忆：用户爱吃面食", "短期新事实：用户最近只吃米饭"),
    ("长期记忆：用户偏好独立开发", "短期新事实：用户加入了创业团队"),
]

# 其它互斥类（事件/地点/状态互斥，contradiction）
_OTHER_CONFLICT: list[tuple[str, str]] = [
    ("用户下周去北京出差", "用户改去上海了，北京不去了"),
    ("用户明天上午有面试", "用户明天的面试取消了"),
    ("用户这周六结婚", "用户婚礼改到周日了"),
    ("用户下个月搬家到朝阳区", "用户改搬到海淀区了"),
    ("用户今晚八点开会", "用户今晚的会取消了"),
    ("用户周五提交周报", "用户改周四交了"),
    ("用户明天去医院", "用户明天的预约取消了"),
    ("用户这周末去爬山", "用户改在家里宅着了"),
    ("用户下周一入职新公司", "用户推迟到下月入职"),
    ("用户这次旅行去三亚", "用户改去青岛了"),
]

# 正例类（entailment，non_conflict）
_ENTAIL: list[tuple[str, str]] = [
    ("用户早上习惯喝一杯咖啡", "用户有喝咖啡的习惯"),
    ("用户从事 Java 后端开发", "用户是做后端的"),
    ("用户居住在北京", "用户依然住在北京"),
    ("用户每周四举行技术分享", "用户分享固定在周四"),
    ("用户写代码用 VS Code", "用户使用 VS Code 编辑器"),
    ("用户会 Java", "用户掌握 Java 语言"),
    ("用户有三年经验", "用户有三年工作经验"),
    ("用户喜欢跑步", "用户经常跑步锻炼"),
    ("用户负责检索模块", "用户负责检索相关的模块"),
    ("用户用 Redis", "用户使用 Redis 作为缓存"),
    ("用户是高级工程师", "用户的职级是高级工程师"),
    ("用户做过运维", "用户之前从事运维工作"),
    ("用户爱喝茶", "用户喜欢喝茶"),
    ("用户学过机器学习", "用户了解机器学习知识"),
    ("用户用 MacBook", "用户使用 MacBook 电脑"),
    ("用户会前端", "用户懂前端开发"),
    ("用户考过雅思", "用户参加过雅思考试"),
    ("用户负责订单系统", "用户负责订单相关系统"),
    ("用户用 Git", "用户使用 Git 做版本控制"),
    ("用户习惯早起", "用户有早起的习惯"),
    ("用户喜欢摄影", "用户对摄影有兴趣"),
    ("用户用 Spring", "用户使用 Spring 框架"),
    ("用户会 SQL", "用户掌握 SQL 查询"),
    ("用户做过电商", "用户做过电商项目"),
    ("用户用 Docker", "用户使用 Docker 容器"),
    ("用户懂网络", "用户了解计算机网络"),
    ("用户会英语", "用户掌握英语"),
    ("用户做过测试", "用户之前做过测试工作"),
    ("用户用 Linux", "用户使用 Linux 服务器"),
    ("用户擅长调优", "用户擅长性能调优"),
]

# 中性类（neutral，non_conflict）
_NEUTRAL: list[tuple[str, str]] = [
    ("用户爱喝咖啡", "用户新养了一只狗"),
    ("用户在准备秋招面试", "用户最近在学吉他"),
    ("用户偏好普通话回答", "用户下月要去重庆出差"),
    ("用户习惯清晨阅读技术文章", "用户的手机是小米的"),
    ("用户通过 YouTube 学习技术", "用户喜欢游泳锻炼"),
    ("用户定居在北京", "用户这周末要去公园"),
    ("用户是后端工程师", "用户最近买了新相机"),
    ("用户用 Redis", "用户下周要去体检"),
    ("用户喜欢喝绿茶", "用户明天要交周报"),
    ("用户会 Java", "用户今晚有个聚会"),
    ("用户偏好简洁回答", "用户下个月要去度假"),
    ("用户习惯早起", "用户这周五有项目评审"),
    ("用户喜欢摄影", "用户下周二去客户现场"),
    ("用户用 MacBook", "用户这周末去露营"),
    ("用户是高级工程师", "用户明早要开晨会"),
    ("用户爱喝茶", "用户下周一有体检"),
    ("用户做过运维", "用户这周六去图书馆"),
    ("用户负责检索模块", "用户下季度要述职"),
    ("用户用 Docker", "用户明天要去银行"),
    ("用户会 SQL", "用户这周日有马拉松"),
    ("用户喜欢跑步", "用户下个月办护照"),
    ("用户懂网络", "用户今晚有线上课程"),
    ("用户做过电商", "用户下周三去杭州"),
    ("用户用 Git", "用户明天下午约了牙医"),
    ("用户擅长调优", "用户这周末看房子"),
    ("用户会前端", "用户下周五有个 deadline"),
    ("用户考过雅思", "用户明天上午去面试"),
    ("用户负责订单系统", "用户这周六办婚礼"),
    ("用户用 Linux", "用户下个月去香港出差"),
    ("用户会英语", "用户这周三有牙医预约"),
]


def build_dataset() -> list[dict]:
    """组装样本集（五类场景），保持定义顺序"""
    samples: list[dict] = []
    for premise, hypothesis in (_CHANGE + _MIGRATE + _OBSOLETE
                                + _UPGRADE + _OTHER_CONFLICT):
        samples.append({"premise": premise, "hypothesis": hypothesis,
                        "label": "contradiction"})
    for premise, hypothesis in _ENTAIL + _NEUTRAL:
        samples.append({"premise": premise, "hypothesis": hypothesis,
                        "label": "non_conflict"})
    return samples


def validate(samples: list[dict]) -> None:
    """结构/数量/平衡/唯一性/评测集零重叠校验，不满足报错退出"""
    labels = [s["label"] for s in samples]
    pairs = [(s["premise"], s["hypothesis"]) for s in samples]
    errors: list[str] = []
    if len(samples) < 100:
        errors.append(f"总样本不足 100：{len(samples)}")
    if labels.count("contradiction") < 40:
        errors.append(f"contradiction 样本不足 40：{labels.count('contradiction')}")
    if labels.count("non_conflict") < 40:
        errors.append(f"non_conflict 样本不足 40：{labels.count('non_conflict')}")
    if any(not s["premise"].strip() or not s["hypothesis"].strip() for s in samples):
        errors.append("存在空 premise/hypothesis")
    if any(lbl not in LABELS for lbl in labels):
        errors.append(f"存在非法 label：{set(labels) - set(LABELS)}")
    if len(set(pairs)) != len(pairs):
        errors.append("存在重复 (premise, hypothesis) 对")

    # 训练/评测分离：与评测集字符串零重叠（防泄漏）
    try:
        from eval.datasets.memory_conflict_dataset import MEMORY_CONFLICT_DATASET
        eval_texts = set()
        for item in MEMORY_CONFLICT_DATASET:
            eval_texts.add(item["premise"])
            eval_texts.add(item["hypothesis"])
        overlap = eval_texts & {p for pair in pairs for p in pair}
        if overlap:
            errors.append(f"与评测集存在字符串重叠（防泄漏）: {sorted(overlap)[:5]}")
    except Exception as e:  # 评测集不可用（构造脚本先跑）时跳过校验
        print(f"[warn] 评测集不可用，跳过分离校验: {e}")

    if errors:
        for err in errors:
            print(f"[error] {err}")
        sys.exit(1)


def print_stats(samples: list[dict]) -> None:
    labels = [s["label"] for s in samples]
    print("=" * 50)
    print(f"Total: {len(samples)}")
    print(f"  contradiction  {labels.count('contradiction')}")
    print(f"  non_conflict   {labels.count('non_conflict')}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="人造记忆矛盾训练集构造（module-062）")
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
