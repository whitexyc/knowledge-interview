# Graph 补跑进度记录

> 更新: 2026-08-01 02:12 (用户手动停止)

## 原因

Qwen 触达今日配额上限（ModelScope 429: `You have exceeded today's quota for model Qwen/Qwen3.5-35B-A3B`），用户决定暂停补跑，等配额恢复或换模型后再续。

## 进度

- **已完成**: 50 篇（doc_id 75 → 124）
- **失败**: 0
- **跳过**: 0
- **总待处理**: 68 篇父块文档
- **剩余**: ~18 篇（doc_id 125 之后，含 2026-07-24 ~ 07-30 的文档）

## 最后成功条目

```
doc_id=124 12-分布式训练与ZeRO优化_2026-07-23 | entities=20 relations=24
```

## 幂等性（为什么可安全续跑）

`backfill_graph.py` 对每篇文档先调用 `_doc_has_entities(doc_id)` 检查图中是否已有该 doc_id 关联实体：
- 已有 → 跳过（不重复提取，不消耗 Qwen 额度）
- 没有 → 才提取并写入

所以 **明天直接重跑 `python backfill_graph.py` 即可**，已完成部分会自动跳过，只处理剩余的 ~18 篇。

## 已产生的图数据

| 项 | 值 |
|----|-----|
| 节点数 | 221+（补跑前 221，随补跑持续增长） |
| 边数 | 250+ |
| ag_label | Entity + RELATED_TO |

## 续跑命令

```bash
cd ai_service
python backfill_graph.py          # 全量（幂等，自动跳过已完成的）
python backfill_graph.py --limit 5   # 先试 5 篇
```
