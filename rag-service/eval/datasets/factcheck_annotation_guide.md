# factcheck 样本标注指南（module-071 / ADR-0010 P1-④ 阈值校准）

> 三态判定口径（golden_factcheck 数据集），module-071 写死 inferred"部分覆盖"
> 边界——module-051 归因②（HHEM 对"部分覆盖"判一致偏乐观）的对症定义。

## 1. 三态定义（对每一对 (doc, claim)，claim = 问题或真实答案句子）

| label | 判定标准 | 通俗讲 |
|-------|----------|--------|
| **supported** | 文档内容**直接支持 claim 的全部核心断言**（claim=问题时 = 文档能完整回答问题） | 文档就是答案的全部依据 |
| **inferred（部分覆盖，边界写死）** | claim 的**至少一个核心断言**被文档直接支持，且**至少一个核心断言未被文档覆盖（无冲突）** | 文档答了一半，另一半没提 |
| **unsupported** | 文档**不包含支持 claim 任何核心断言的内容**（含矛盾内容——按 module-052 三态映射口径 contradiction→unsupported） | 文档答非所问 / 相关背景不算数 / 文档说反话 |

## 2. 核心断言拆解（每条的强制依据）

1. **每条标注必须在 note 字段给出核心断言拆解依据**：把 claim 拆成 2-3 个可独立
   验证的子断言，逐条对照 doc 说明"被支持 / 未被覆盖且无冲突 / 冲突"。
2. **"直接被支持" ≠ "主题相关"**：doc 必须直接陈述了该子断言的相同或部分内容。
   只提供相关背景（如问"调优参数怎么设置"，doc 只讲 GC 机制）不算支持——
   落 unsupported，不落 inferred。
3. 只依据**给定 doc 片段**判断，不引入外部知识补全。
4. **复合 claim**：全部子断言被支持才 supported；≥1 支持 + ≥1 未覆盖 → inferred；
   任一子断言与 doc 冲突 → unsupported（矛盾口径同 contradiction_annotation_guide.md，
   阶段/段数穷尽性陈述互斥直接判冲突）。
5. **无冲突**是 inferred 的必要条件——未被覆盖但 doc 陈述与之矛盾，判 unsupported
   而非 inferred。

## 3. 数据来源与 part 字段

- `part=real_retrieval`：real_retrieval_pairs.json 转换（claim=deepseek-v4-flash
  真实答案句子，doc=DB golden 112 题 hybrid 检索 top 片段），verdict 人工标注
  按 to_factcheck_item 口径映射：entailment→supported / neutral→inferred /
  contradiction→unsupported（与 module-052 三态映射一致）。
- `part=constructed`：人工构造"部分覆盖"样本（文档为知识库真实段落，问题借
  golden 题/面试题，核心断言拆解写死边界）。
- `part=sufficiency`：SUFFICIENCY_DATASET 代理标注（充分→supported /
  不充分→unsupported，claim=问题，module-044 人工充分性标注继承）。

## 4. 评审流程

1. Developer 构造/转换 + 按本指南逐条写 note 拆解；
2. 标注变更（含 module-071 口径复核的改判）记录变更清单（样本 / 旧→新 / 理由）
   入 changelog——可审计；
3. Reviewer 抽查标注一致性（方向性验证，非多人独立标注）。

## 5. 诚实边界

- 代理标注（sufficiency 继承）与人工标注（real/constructed）混编，kappa 是
  混合口径的方向性指标；
- 阈值最优基于当前标注集，可能过拟合标注集本身——标注集扩充后需复扫。
