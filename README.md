<div align="center">

# knowledge-interview

**知识库 × 面试系统闭环 —— 基于RAG检索增强的智能面试出题平台**

让每一次技术面试都建立在真实知识库之上：简历理解 → 知识检索 → 个性化出题。

[![License](https://img.shields.io/github/license/whitexyc/knowledge-interview?color=blue&label=license)](LICENSE)
[![Stars](https://img.shields.io/github/stars/whitexyc/knowledge-interview?style=flat&logo=github&color=yellow)](https://github.com/whitexyc/knowledge-interview/stargazers)
[![Forks](https://img.shields.io/github/forks/whitexyc/knowledge-interview?style=flat&logo=github)](https://github.com/whitexyc/knowledge-interview/network/members)
[![Contributors](https://img.shields.io/github/contributors/whitexyc/knowledge-interview?color=green)](https://github.com/whitexyc/knowledge-interview/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/whitexyc/knowledge-interview?logo=git&color=orange)](https://github.com/whitexyc/knowledge-interview/commits/main)
[![Issues](https://img.shields.io/github/issues/whitexyc/knowledge-interview?color=red)](https://github.com/whitexyc/knowledge-interview/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/whitexyc/knowledge-interview/pulls)

![Java](https://img.shields.io/badge/Java-17-orange?logo=openjdk)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2-6DB33F?logo=springboot)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)

</div>

---

## ✨ 项目简介

knowledge-interview 将企业私有知识库与 AI 面试系统打通，形成「知识检索 → 出题增强 → 面试反馈 → 知识沉淀」的双向闭环：

- **检索增强出题**：上传简历后，系统抽取技术关键词，从知识库混合检索（全文 + 向量）相关知识点，重排后注入出题 Prompt——面试题不再凭空生成，而是围绕候选人的技术栈与知识库的真实内容。
- **防注入设计**：知识库内容以 `<kb_reference>` 数据块注入，配「仅数据、无指令」守卫与清洗截断，杜绝知识库内容反向操纵出题模型。
- **fail-open 熔断**：知识库服务不可用时自动退化为纯简历出题，面试主链路永不阻断。
- **本地化嵌入**：bge-m3（GGUF 量化）本地推理 + pgvector 向量检索，知识不出内网。

## 🏗️ 系统架构

```
                        ┌─────────────────────────────────────────┐
                        │            rag-service (:8001)          │
                        │   FastAPI + pgvector + 本地嵌入/重排      │
  简历 PDF              │                                         │
      │  关键词抽取      │   query ──→ 混合检索(FTS+向量)           │
      ▼                 │              ──→ CrossEncoder 重排       │
┌──────────────────┐    │              ──→ 编号知识点列表           │
│ interview-admin  │    └────────────────┬────────────────────────┘
│ (:8002)          │                     │ POST /ai/rag/search
│ Spring Boot 3.2  │ ◄───────────────────┘
│                  │
│  出题 Prompt 组装 │──→ <kb_reference> 参考知识点 + 防注入守卫
│  LLM 出题        │──→ 结构化面试题（JSON）
└──────────────────┘
```

## 🚀 快速开始

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | Java 面试服务 |
| Maven | 3.8+ | 构建工具 |
| Python | 3.11 | RAG 服务 |
| PostgreSQL | 15+ | 需安装 [pgvector](https://github.com/pgvector/pgvector) 扩展 |
| Redis | 5+ | 缓存 |

### 1. 启动 RAG 服务

```bash
cd rag-service
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux/macOS

# 配置环境变量（参考 .env.example）
cp .env.example .env

# 下载本地模型到 models/（bge-m3 嵌入 + bge-reranker-v2-m3 重排，见下方模型说明）

.venv/Scripts/python -m uvicorn main:app --port 8001
```

> 模型文件不入库，需自行下载至 `rag-service/models/`：
> - `bge-m3-gguf/bge-m3-q8_0.gguf`（嵌入，~634MB）
> - `bge-reranker-v2-m3/`（重排，~2.3GB，含 tokenizer）
> 可从 HuggingFace（或 hf-mirror.com 镜像）下载对应模型仓库。
>
> Windows 下 `llama-cpp-python==0.3.34` 建议安装预编译 wheel（源码编译需 MSVC）：
> `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34/llama_cpp_python-0.3.34-py3-none-win_amd64.whl`
### 2. 启动 Java 面试服务

```bash
cd interview-admin
mvn spring-boot:run -pl admin
```

### 3. 验证闭环

```bash
curl -X POST http://localhost:8001/ai/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Redis 持久化 高并发", "top_k": 5}'
```

## ⚙️ 配置说明

**RAG 服务（`rag-service/.env`，不入库）**

| 变量 | 说明 |
|------|------|
| `PW_DATABASE_URL` | PostgreSQL 连接串（asyncpg） |
| `PW_LLM_PROVIDER` | LLM 供应商（deepseek / modelscope） |
| `PW_DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `PW_JWT_SECRET` | JWT 共享密钥（与 Java 侧 Sa-Token `jwt-secret-key` 同值） |
| `PW_MCP_TOKEN` | MCP HTTP 访问令牌 |

**Java 服务（`interview-admin/admin/src/main/resources/application.yaml`）**

| 配置 | 说明 |
|------|------|
| `kb.base-url` | RAG 服务地址（默认 `http://localhost:8001`，可用 `KB_BASE_URL` 覆盖） |
| `spring.datasource.*` | 数据库连接 |

## 📁 目录结构

```
knowledge-interview/
├── interview-admin/        # Java 面试系统（Spring Boot 多模块）
│   └── admin/              #   管理后台 + 面试核心服务
└── rag-service/            # Python RAG 服务
    ├── main.py             #   FastAPI 入口（/ai/rag/search 等）
    ├── rag/                #   检索 / 重排 / 反思 / 记忆
    ├── llm/                #   多供应商 LLM 客户端
    └── models/             #   本地模型（不入库）
```

## 🗺️ Roadmap

- [ ] 知识抓取流水线：定时抓取 + 白名单/黑名单过滤
- [ ] 面试反馈回流：低分题自动沉淀为待学笔记
- [ ] 增量语料接入：新增文档追加索引，无需全量重建
- [ ] 反向闭环：待学笔记 → 自动抓取任务优先级排序

## 🤝 贡献

欢迎 Issue 与 PR！提交前请确保：

- 新增依赖与现有版本约束兼容
- 涉密配置一律走环境变量，严禁硬编码

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。
