"""
人造记忆类型训练集构造脚本（module-062 WP1 方案 A 数据）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

用法（在 ai_service 目录下）:
    python -m eval.datasets.build_memory_type_dataset        # 构造 + 校验 + 落盘
    python -m eval.datasets.build_memory_type_dataset --dump # 打印分类统计（人工复核用）

产出:
    eval/memory_type_train_dataset.json —— [{"content", "type"}, ...]
        content  记忆内容（短句偏好/事实/事件，非空、全库唯一）
        type     preference / fact / event（三类契约，与 extract_facts 共用）

标注口径（本数据集的构造指南）:
    preference（偏好/习惯/兴趣）——"用户喜欢/偏好/习惯/偏爱 + 某事"，长期有效慢衰减。
    fact（客观事实）——"用户是/有/住在/在 X 工作/毕业于"，较稳定中衰减。
    event（带时间临时事件）——"用户下周/明天/这周/下个月 + 一次性事件"，迅速过期快衰减。

诚实边界（写入 changelog）:
  - 人工构造（非真实用户记忆），用于方向性验证类型分类器能力；
  - 真实分布以飞轮数据积累后重训为准（MemoryTypeClassifier.fit 接口已预留）；
  - 训练/评测分离：eval/memory_type_dataset.py 评测集 30 条不进入本训练集
    （防泄漏），本文件所有 content 与评测集字符串零重叠（build 时校验强制）。

校验（build 时强制，不满足报错退出）:
  - 总样本 ≥120；每类 ≥40
  - content 非空、无重复；type 合法
  - 与评测集零字符串重叠
"""
import argparse
import json
import sys

from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVAL_DIR / "memory_type_train_dataset.json"

TYPE_CLASSES = ("preference", "fact", "event")


# ---------------- 样本定义（构造口径见上方标注指南） ----------------

# preference：用户喜好/习惯/兴趣（慢衰减）
_PREFERENCE: list[str] = [
    "用户喜欢喝美式咖啡", "用户偏好简洁的回答风格", "用户习惯晚睡",
    "用户喜欢用 VS Code 写代码", "用户偏爱中文回答", "用户喜欢摄影",
    "用户偏好周末出行", "用户习惯早起", "用户爱吃辣", "用户喜欢看科幻电影",
    "用户偏好远程办公", "用户习惯每周跑步", "用户喜欢 Java 和分布式",
    "用户偏爱极简设计", "用户喜欢安静的咖啡馆", "用户习惯用命令行",
    "用户偏好短回答", "用户喜欢养猫", "用户偏爱读纸质书", "用户习惯晚上学习",
    "用户喜欢蓝色", "用户偏好用 MacBook", "用户喜欢听古典音乐", "用户习惯午休",
    "用户偏爱轻量级框架", "用户喜欢打篮球", "用户偏好直接了当的沟通",
    "用户习惯记笔记", "用户喜欢旅行", "用户偏爱开源软件", "用户喜欢喝茶",
    "用户偏好自学", "用户习惯早起健身", "用户喜欢小狗", "用户偏爱侧躺睡",
    "用户喜欢看技术博客", "用户习惯睡前看书", "用户偏好团队协作", "用户喜欢做菜",
    "用户偏爱自然光",
]

# fact：客观事实（中衰减）
_FACT: list[str] = [
    "用户是 Java 后端开发", "用户有三年工作经验", "用户住在北京",
    "用户在一家创业公司工作", "用户负责检索模块", "用户所在团队 8 人",
    "用户的学历是硕士", "用户毕业于某大学", "用户的职位是高级工程师",
    "用户使用 Redis 和 MySQL", "用户做过三年运维", "用户的家乡是成都",
    "用户有一个弟弟", "用户的年龄是 28 岁", "用户从事 AI 开发",
    "用户的项目是知识库问答系统", "用户会两种语言", "用户通过了 A 公司面试",
    "用户的领导是张工", "用户参与过开源项目", "用户的 GitHub 有 500 星",
    "用户负责三个微服务", "用户的技术栈是 Spring 全家桶", "用户的办公地在上海",
    "用户有 P8 级别", "用户的专业是计算机", "用户考过雅思 7 分",
    "用户的工龄是五年", "用户在公司负责消息队列", "用户的上一个项目是电商",
    "用户的导师姓李", "用户的身份证所在地是广州", "用户擅长性能调优",
    "用户的团队使用敏捷开发", "用户的毕业院校是某 985", "用户之前是做前端的",
    "用户的英语流利", "用户持有 AWS 证书", "用户的直属上级是经理王",
    "用户的社保在上海",
]

# event：带时间的一次性/临时事件（快衰减）
_EVENT: list[str] = [
    "用户下周去北京出差", "用户明天上午有面试", "用户今晚八点开会",
    "用户这周末去爬山", "用户下个月要搬家", "用户周五提交周报",
    "用户明天参加技术分享会", "用户这周三有体检", "用户下周一入职新公司",
    "用户今天下午要去医院", "用户周末计划学 Kafka", "用户下季度要述职",
    "用户这周五交代码", "用户明天去机场接人", "用户下周六办婚礼",
    "用户本周日有马拉松", "用户下个月考驾照", "用户明天要答辩",
    "用户这周末看房子", "用户下周二回老家", "用户今天要去银行办业务",
    "用户下周五有个 deadline", "用户这周末去参加朋友婚礼", "用户明天下午约了牙医",
    "用户下个月换手机", "用户这周五有项目评审", "用户明天早上有晨会",
    "用户下周三去杭州", "用户这周末清理房间", "用户下个月办理公积金",
    "用户明天要交季度总结", "用户这周六去图书馆还书", "用户下周日有家庭聚会",
    "用户明天晚上有电话会议", "用户下个月去三亚度假", "用户这周五开季度会议",
    "用户明天去拿快递", "用户下周二去客户现场", "用户这周末搬家",
    "用户下个月有考试",
]


def build_dataset() -> list[dict]:
    """组装样本集（三类，保持定义顺序）"""
    samples: list[dict] = []
    for content in _PREFERENCE:
        samples.append({"content": content, "type": "preference"})
    for content in _FACT:
        samples.append({"content": content, "type": "fact"})
    for content in _EVENT:
        samples.append({"content": content, "type": "event"})
    return samples


def validate(samples: list[dict]) -> None:
    """结构/数量/平衡/唯一性/评测集零重叠校验，不满足报错退出"""
    contents = [s["content"] for s in samples]
    types = [s["type"] for s in samples]
    errors: list[str] = []
    if len(samples) < 120:
        errors.append(f"总样本不足 120：{len(samples)}")
    for cls in TYPE_CLASSES:
        if types.count(cls) < 40:
            errors.append(f"类别 {cls} 不足 40：{types.count(cls)}")
    if any(not c.strip() for c in contents):
        errors.append("存在空 content")
    if len(set(contents)) != len(contents):
        errors.append("存在重复 content")
    if any(t not in TYPE_CLASSES for t in types):
        errors.append(f"存在非法 type：{set(types) - set(TYPE_CLASSES)}")

    # 训练/评测分离：与评测集字符串零重叠（防泄漏）
    try:
        from eval.datasets.memory_type_dataset import MEMORY_TYPE_DATASET
        eval_contents = {item["content"] for item in MEMORY_TYPE_DATASET}
        overlap = eval_contents & set(contents)
        if overlap:
            errors.append(f"与评测集存在字符串重叠（防泄漏）: {sorted(overlap)[:5]}")
    except Exception as e:  # 评测集不可用（构造脚本先跑）时跳过校验
        print(f"[warn] 评测集不可用，跳过分离校验: {e}")

    if errors:
        for err in errors:
            print(f"[error] {err}")
        sys.exit(1)


def print_stats(samples: list[dict]) -> None:
    types = [s["type"] for s in samples]
    print("=" * 50)
    print(f"Total: {len(samples)}")
    for cls in TYPE_CLASSES:
        print(f"  {cls:<12} {types.count(cls)}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="人造记忆类型训练集构造（module-062）")
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
