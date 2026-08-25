# 矛盾样本标注指南（module-054 / ADR-0010 P1-③ 复测）

## 1. "什么是矛盾"判定标准（NLI 三分类定义）

对每一对 (doc, claim) 做三分类人工标注，verdict ∈ {entailment, neutral, contradiction}：

| verdict | 判定标准 | 通俗讲 |
|---------|----------|--------|
| **contradiction** | claim 的断言与 doc 的陈述**直接冲突**——若 doc 为真，claim 必为假（反之亦然） | 文档说黑，答案说白 |
| **entailment** | claim 的断言可由 doc **直接推出**（doc 支持 claim） | 文档就是答案的依据 |
| **neutral** | claim 与 doc **既不冲突也不支持**——主题无关或信息不足 | 文档和答案各说各的 |

判定要点：
1. **以 doc 为真**为前提判断 claim：doc 说 X，claim 说 not-X → contradiction；
   doc 说 X，claim 也说 X → entailment；doc 无关 → neutral。
2. 只依据**给定 doc 片段**判断，不引入外部知识补全（外部事实即使为真，
   若 doc 不支持/不冲突，仍标 neutral）。
3. 复合 claim（含多个子断言）：任一子断言与 doc 冲突即 contradiction；
   全部子断言被支持才 entailment；无冲突也无支持 → neutral。
4. **阶段/段数"穷尽性"陈述互斥**：claim 与 doc 对**同一事物**的阶段数/
   组成部分数做数字互斥的穷尽性断言 → 直接冲突判 contradiction（真实检索
   对同口径）。例：claim"类加载分加载/验证/准备/解析/初始化五个阶段" vs doc
   "类加载的生命周期是 7 个阶段"；claim"雪花 ID 由时间戳、机器 ID、序列号
   三部分组成" vs doc"切成四段（含最高位符号位）"。
   注意区分主语：**不同主语**（不同配置/不同对象）的陈述互不否定——如 claim
   "AOF 关闭 fsync 后数据更安全一点也不会丢，而默认每秒 fsync 反而会丢 1 秒
   数据"两半句主语为不同配置（no-fsync vs everysec），非同一主语的 X/not-X，
   且 doc 未覆盖 no-fsync → 按规则 3 判 neutral（信息不足）。

## 2. 两类矛盾（contradiction 细分）

### ① claim_vs_doc（断言与文档矛盾，30 条）
doc 支持 X，claim 声称 not-X。构造方法：取知识库真实文档中的明确陈述 X，
把 X 反转成 not-X 作 claim（注意反转要语义精确，不能模糊成"不完全一样"）。

例：doc="G1 是 JDK 9 之后的默认垃圾收集器" → claim="G1 是 JDK 8 及之前的
默认垃圾收集器，JDK 9 之后已被 CMS 取代"。

### ② internal_contradiction（claim 内部自相矛盾，23 条）

**单句混合（15 条，module-054 首版）**：claim 单句内同时出现 X 与 not-X
（或互相排斥的两种状态）。构造方法：把"X（真）"与"not-X（构造）"拼进
同一句断言，让句子在逻辑上不可能为真。

例："G1 是 JDK 9 之后的默认垃圾收集器，但它自 JDK 9 起就不再被使用"。

**多句混合"前真后假"（8 条，module-057 扩充）**：以句号（。！？；）分隔的
多句 claim，前句为文档真断言（X），后句为反断言（not-X）。此类样本是句级
拆解的目标场景：整句看混合断言 mDeBERTa 倾向判 neutral，拆成子句后后句
单独与文档矛盾应判 contradiction。

例："G1 是 JDK 9 之后的默认垃圾收集器。从 JDK 10 开始 G1 已经被完全移除了。"

## 3. 正例对照（一致样本）

entailment 22 条（module-054 首版 16 + module-057 扩充 6 条多句正例）：
claim 为 doc 原文陈述（逐字或近义）；扩充的多句正例两个子句均被文档支持，
验证"无矛盾有 entailment → entailment"聚合不会误杀一致样本。

neutral 11 条（module-054 首版 9 + module-057 扩充 2 条多句无关对照）：
claim 与 doc 主题无关或信息不足（含 AOF fsync 改标样本——两半句主语为
不同配置 no-fsync vs everysec，非同一主语 X/not-X，按规则 3 判 neutral），
验证模型不会把无关/不冲突当矛盾（防"一切不符即矛盾"的过激倾向）。

## 4. 构造方法

1. **文档来源**：知识库真实文档段落（SUFFICIENCY_DATASET 内嵌片段，与
   module-050/052 同源同构），非虚构文档。
2. **claim 来源**：人工构造（反转断言 / 拼接互斥断言 / 原文陈述），
   模拟 LLM 生成答案中出现的幻觉句式。
3. **人工复核**：Developer 构造 + Reviewer 抽查标注一致性（非多人独立标注，
   方向性验证）。
4. **JSON 结构**（与 golden_factcheck 兼容）：
   ```json
   {"question": "...", "claim": "...", "doc": "...", "doc_title": "...",
    "verdict": "contradiction|entailment|neutral",
    "contradiction_type": "claim_vs_doc|internal_contradiction|positive|neutral",
    "note": "...", "part": "constructed|real_retrieval"}
   ```
   与 golden_factcheck（question/documents/label）映射：
   question ↔ question；doc+doc_title ↔ documents[0]；
   verdict ↔ label（entailment→supported / neutral→inferred /
   contradiction→unsupported，与 module-052 三态映射一致）。

## 5. 诚实边界

- 样本为人工构造（非真实用户对话），方向性验证非最终结论；
- 复测结论以 eval/retest_nli.py 输出 kappa 为准（三分类 ≥0.7 放行替换 /
  未达降级双轨：NLI 只做矛盾扫描），不因样本可观而替代门槛判定。
