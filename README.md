# knowledge-interview

知识库 × 面试系统闭环（ADR-0019）：RAG 检索增强出题 → 面试系统消费，双向演进的 monorepo。

## 结构

```
knowledge-interview/
├── interview-admin/    # Java 面试系统（Spring Boot 3.2 / Java 17，源自 AI-Meeting 上游 + 闭环模块）
│   └── admin/          #   出题注入点：flow/extraction/InterviewQuestionExtractionService
│   └── admin/src/main/java/.../kb/   #   KnowledgeBaseClient + ResumeKeywordExtractor（module-074）
└── rag-service/        # Python RAG 服务（FastAPI :8001，pgvector + bge-m3 本地嵌入）
    ├── main.py         #   POST /ai/rag/search 契约端
    └── rag/            #   混合检索 / 重排 / 反思 / 记忆
```

## 闭环链路（ADR-0019 阶段 1，已联调通过）

```
简历 PDF → 关键词抽取(中英 Top-8) → POST /ai/rag/search (8001)
        → 混合检索(FTS+向量) + 重排(bge-reranker-v2-m3 int8)
        → <kb_reference> 参考知识点注入出题 prompt（防提示注入守卫）
        → fail-open：RAG 故障时退化为纯简历出题，主链路不阻断
```

- 实测：真实检索 432ms 注入 4030 字符上下文；top1 相关性 score=1.0
- 单测：module-074 相关 16/16 绿；全量回归 104/106（2 个失败为上游基线遗留）

## 环境要点

- RAG 服务：`rag-service/.venv/Scripts/python -m uvicorn main:app --port 8001`
  - 依赖 PostgreSQL(+pgvector) 与本地模型 `models/`（不入库，见 .gitignore）
  - `.env` 不入库：需 PW_DATABASE_URL / PW_DEEPSEEK_API_KEY / PW_JWT_SECRET（与 Java Sa-Token jwt-secret-key 同值）/ PW_MCP_TOKEN
- Java 侧配置：`admin/src/main/resources/application.yaml` 的 `kb.base-url`（默认 http://localhost:8001，可被 KB_BASE_URL 覆盖）
- 测试注意：本机 surefire 未配 JaCoCo，跑测试需 `-DforkCount=0`

## 模块记录

| 模块 | 内容 | 状态 |
|------|------|------|
| module-074 | KnowledgeBaseClient + ResumeKeywordExtractor + 出题 prompt 注入（ADR-0019 阶段 1） | ✅ 已联调 |
| 阶段 2（规划） | @Scheduled 抓取流水线、白/黑名单、审查节点复用反思双判 | ⬜ |
| 阶段 3（规划） | 增量 append 语料、无全量重嵌验证 | ⬜ |

> 敏感说明：`interview-admin/admin/src/main/resources/sql/*.sql` 中的 agent api_key/api_secret 为上游公开仓库自带的种子数据，与本仓库同步保留；生产环境请通过环境变量覆盖。
