"""
应用配置管理
使用 pydantic-settings 从环境变量读取配置
"""
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 应用
    app_name: str = "Personal Website AI Service"
    app_version: str = "0.1.0"
    debug: bool = False

    # 数据库
    database_url: str = "postgresql+asyncpg://postgres:postgres123@localhost:5432/personal_website"

    # Redis 缓存
    redis_url: str = "redis://localhost:6379/0"

    # LLM 供应商
    # fallback: 按 fallback_chain 顺序自动降级（默认 qwen → zhipu → deepseek）
    # 单供应商: claude | deepseek | qwen | zhipu | modelscope
    llm_provider: str = "fallback"

    # 降级链（逗号分隔，仅 llm_provider=fallback 时生效）
    fallback_chain: str = "qwen,zhipu,deepseek"

    # Claude
    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-5-20251001"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # Qwen (通过 ModelScope API，默认首选)
    qwen_model: str = "Qwen/Qwen3.5-35B-A3B"

    # ZhipuAI GLM (通过 ModelScope API，Qwen 降级备用)
    zhipu_model: str = "ZhipuAI/GLM-5.2"

    # ModelScope（魔搭）
    modelscope_api_key: str = ""
    modelscope_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    modelscope_base_url: str = "https://api-inference.modelscope.cn/v1"

    # 文本嵌入（默认使用 ModelScope 云端 API）
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "OllmOne/bge-m3-GGUF"

    # JWT 登录（module-032）：HS256 共享密钥，与 Java 后端一致
    # 环境变量：PW_JWT_SECRET（.env 本地配置，不进仓库）
    jwt_secret: str = ""

    # MCP 集成（module-067 / ADR-0018）：MCP HTTP 模式（/ai/mcp）访问 token，
    # 认证头 Authorization: Bearer <token>。fail-closed：未设置时 HTTP 模式
    # 拒绝启动（宁可不用不能裸奔）；stdio 本地模式零认证是设计（本地进程）。
    # 环境变量：PW_MCP_TOKEN（.env 本地配置，不进仓库）
    mcp_token: str = ""

    # 混合检索
    hybrid_search_alpha: float = 0.3  # BM25 权重，向量权重为 1-alpha

    # 检索融合模式（module-053 三通道融合验证，module-055 切默认 rrf）：
    #   hybrid   —— 两通道 min-max 加权（FTS+向量），module-055 起为回退开关
    #   rrf      —— 三通道（FTS/向量/图谱）Reciprocal Rank Fusion，
    #                score(d) = Σ 1/(k + rank_i(d))，k=60 业界默认；
    #                图谱通道仅 round 0 语义参与融合（引擎层 round 1/2 单路混合）
    #   weighted —— 三通道 min-max 归一化 + 权重加权（retrieval_fusion_weights）
    # module-053 实测（golden 112 题同口径，见 specs/module-053-rrf-fusion/
    # changelog.md 对比表）：rrf Hit@5=0.9905 > 基线 hybrid 两通道 0.9714
    # （+0.0191，0 回退）> 加权两组 = 基线 —— **rrf 放行**。
    # module-054 清障：向量化降级方案 A/B（rrf 向量路失败 = FTS+图谱照常、
    # 引擎补图兜底）+ 引擎 rrf 真实 HTTP E2E 通过（chat/stream 全链路）。
    # module-055 决策：默认 hybrid→rrf（前置清障全部完成；存量 2 项降级
    # 用例在 module-054 方案 A/B 落地后已消解，零断言改动）。回退方式：
    # PW_RETRIEVAL_FUSION_MODE=hybrid 一键回退（保留开关）。
    # minor 修复（module-053）：Literal 枚举校验——非法字符串（拼写错误）
    # 启动即抛 ValidationError（fail-fast），防静默落入 rrf 分支（rrf 每次
    # 知识库查询 +1 次 LLM 实体提取调用，静默走错分支代价高）。
    retrieval_fusion_mode: Literal["hybrid", "rrf", "weighted"] = "rrf"
    # 加权融合权重（逗号分隔：FTS,向量,图谱；仅 retrieval_fusion_mode=weighted 生效）
    retrieval_fusion_weights: str = "0.3,0.6,0.1"
    # RRF 常数 k（业界默认 60；本模块不做 k 扫描，扫 k 留后续）
    rrf_constant_k: int = 60

    # 重排性能优化（非 module 批）：
    #   reranker_quantize_enabled —— bge-reranker-v2-m3 int8 动态量化（CPU 实测
    #     约 2x 提速：6 pair/250 字符 0.89s/pair → 0.42s/pair）。弱相关文档分数
    #     有漂移但相关文档排序保持。量化失败 fail-open 回退原始模型。false 回退
    #     fp32（逃生口）。
    #   rerank_max_candidates —— 粗筛上限：候选超过该值先按现有融合分
    #     （hybrid_score/rrf_score）截断再进 CrossEncoder 精排（粗筛后精排，
    #     降交叉对数）。上限不低于 top_k（保证返回足够结果）。实测 Hit@5 不降
    #     才采纳，见性能优化记录。
    reranker_quantize_enabled: bool = True
    rerank_max_candidates: int = 6

    # Agent 工具化（module-028）：ReAct 循环工具总调用次数预算（防空转烧钱）
    # module-068：默认 4→5——总预算 = 检索阶段 ≤3 + 生成阶段 ≤2 的兜底和
    # （旧环境显式 PW_MAX_AGENT_TOOLS=4 时总预算仍 4，阶段预算让位取 min，
    # 行为正确）。测试环境由 conftest autouse fixture 钉住 4（存量断言保持）。
    max_agent_tools: int = 5

    # 工具阶段切分（module-058 / ADR-0012 方案 A，原 module-059 并入）：
    # 按 ctx.phase 状态机只暴露当前阶段工具（检索组 7 / 生成组 4，re_search
    # 双组）——省 schema token + 结构性防误调（检索阶段调不到 generate/
    # verify，不再靠工具内部字符串防御）。默认 true；false 回退全量 10 个
    # 零回归（逃生口）。测试环境由 conftest autouse fixture 钉住 false。
    tool_phase_split: bool = True

    # Agent 阶段推进死锁修复（module-068）：
    #   agent_retrieval_max_rounds —— 检索阶段轮次 ≥ 该值且始终未命中 →
    #     强制切 generation（防空转兜底；066 实测 4 轮预算耗尽，取 3 = 预算-1）
    #   agent_retrieval_budget / agent_generation_budget —— 阶段预算：检索阶段
    #     累计工具调用 ≤3、生成阶段 ≤2（总 5 = max_agent_tools；生成 2 轮留
    #     一次 re_search 补检余量）。仅 tool_phase_split=true 生效（false 回退
    #     纯总预算，存量行为逐字）；总预算仍为硬上限（截断取 min）。
    agent_retrieval_max_rounds: int = 3
    agent_retrieval_budget: int = 3
    agent_generation_budget: int = 2

    # 工具失败自动重试（module-073）：AgentTool.run 捕获异常后对同一 func
    # 自动重试 1 次（只读检索类 + note_to_self；generate_answer/verify_answer
    # 排除在 _NO_RETRY_TOOLS——15s 超时是常态，重试无意义）。超时（15s）永不
    # 重试（超时=慢不是抖动，重试翻倍墙钟）。重试发生在 run 内部，不增加
    # tool_count / phase_count（预算语义不变）。**默认 true（task-brief 指定，
    # 少数默认开的新开关）**；PW_TOOL_AUTO_RETRY=false 回退存量"失败即空"。
    # 测试环境由 conftest autouse fixture 钉住 false（hermetic）。
    tool_auto_retry: bool = True

    # 请求可观测性（module-058 WP-C）：trace_id + 阶段计时 + token 用量 +
    # 缓存命中 → request_logs 落库（init_db 自愈幂等 DDL）。默认 true；
    # false 时零埋点零落库（中间件不初始化观测上下文、helper 直接返回）。
    # 测试环境由 conftest autouse fixture 钉住 false（测试不污染落库）。
    request_logs_enabled: bool = True

    # 工具调用明细落库（module-066 / ADR-0017 决策 2）：react 循环每次实际
    # 执行工具落一行 tool_call_logs（trace_id/工具名/参数/成败/预览/耗时），
    # 补 request_logs 缺工具调用明细的核心缺口。默认 true（与 request_logs
    # 同生命周期）；false 零开销跳过（不构造记录）。测试环境由 conftest
    # autouse fixture 钉住 false（测试不污染落库）。
    tool_call_logs_enabled: bool = True

    # 长期记忆（module-033/035）：提取 / 去重 / 动态K 阈值（参考 llm-push/19-Agent记忆管理）
    memory_importance_threshold: float = 0.6    # 提取事实 importance < 0.6 丢弃
    # module-035 校准：真实 bge-m3 同义改写 cosine≈0.88，0.95 太严导致漏去重 → 下调 0.85
    memory_dedup_threshold: float = 0.85        # 语义去重：与本身份现有记忆 cosine > 0.85 视为重复
    memory_recall_high_threshold: float = 0.85  # 候选平均绝对余弦 > 0.85 → 召回 5 条
    memory_recall_mid_threshold: float = 0.75   # 0.75-0.85 → 召回 3 条；<0.75 → 1 条（宁缺毋滥）
    memory_max_recall: int = 5                  # 动态 K 上限
    # module-035：低分过滤阈值（绝对余弦口径）——低于该值的候选丢弃，防"本批相对高但绝对烂"注入
    memory_recall_min_score: float = 0.4

    # 短期记忆 + 会话记忆（module-034）
    memory_short_ttl_days: int = 7              # 短期记忆 TTL（天）：module-046 起由衰减+硬上限替代，保留兼容
    memory_session_max_messages: int = 50       # 每 identity 会话持久化消息上限（超限滚动删除最旧）
    memory_session_history_limit: int = 20      # 会话恢复注入生成的最近消息数

    # 短期记忆进化（module-046 / ADR-0007 问题 2）：强化/衰减/升级可配
    memory_short_half_life: float = 3.0         # 平滑衰减半衰期（天）：decay = 0.5**(age_days/half_life)
    memory_short_max_days: int = 30             # 硬上限（天）：last_mentioned_at/created_at 超上限不参与召回
    memory_mention_boost_alpha: float = 0.2     # 提及加权系数：最终分 = 语义分×decay×(1+α×mention_count)
    memory_promote_mentions: int = 2            # 短期→长期升级：mention_count ≥ 该值
    memory_promote_window_days: int = 7         # 升级窗口（天）：最近提及在窗口内才升级

    # 记忆类型化衰减（module-062 / ADR-0007 P2）：记忆按类型差异化半衰期（A-MAC 参考——
    # 偏好慢衰减、事件快过期），替换"所有短期记忆同一半衰期"的一刀切。
    #   memory_type_mode 定生产注入（类型从哪来，winner 决定，见 eval/memory_type_dataset.py）：
    #     clf  —— bge-m3+逻辑回归分类模型判型（复用 module-056 intent 基建，落盘
    #            models/memory_type_clf.joblib；推理失败回退 llm_type/默认 fact）
    #     llm  —— extract_facts 输出 type（_EXTRACT_PROMPT few-shot，缺失/非法默认 fact）
    #     none —— 不判型，全部按默认 fact 存储（类型化衰减零生效 = 零回归回退，
    #            不预设成功：Accuracy<0.8 谁都不上则保持 none）
    #   memory_type_decay_enabled（PW_MEMORY_TYPE_DECAY）默认 true：_evolve_recall 按
    #     type 选半衰期；false 回退全局 memory_short_half_life（现状行为）。
    #   半衰期（天）：preference 30（偏好长期有效）/ event 1（临时事件迅速过期）/
    #     其余（fact/未知/存量无 type）→ memory_short_half_life=3（存量零回归口径）。
    #   升级阈值未按类型区分（保持 ≥2 次/7 天），如实声明。
    #   实测（module-062 WP1，eval_runs id=32/33，同 30 条评测集）：clf 1.0000 /
    #   LLM 1.0000 双达标且同分 → 取 clf（零成本/确定性/离线，对齐 module-056 L4
    #   分类器替代 LLM 哲学；评测集小且与训练集同模式，1.0 含一定"同分布"成分，
    #   如实声明）。clf 推理失败自动回退 llm_type（extract_facts 输出）→ 默认 fact。
    memory_type_mode: Literal["clf", "llm", "none"] = "clf"
    memory_type_decay_enabled: bool = True
    memory_type_half_life_preference: float = 30.0
    memory_type_half_life_event: float = 1.0

    # 冷记忆降权（module-062 / ADR-0007 P3）：长期层久未召回的旧记忆检索时降权
    # （Memory Decay 参考 ×0.3-1.0，温和不删除）。recall 长期层检索命中后按
    # 距上次召回（last_recalled_at or created_at）天数加权：< memory_cold_decay_days
    # → ×1.0（最近召回）；此后平滑渐降（30→100 天 1.0→0.3），下限
    # memory_cold_decay_min（默认 0.3，不删旧可回溯）。召回命中 fire-and-forget
    # 刷新 last_recalled_at=now（冷记忆升温）。短期层不降权（已有衰减），如实声明。
    #   memory_cold_decay_enabled（PW_MEMORY_COLD_DECAY）默认 true；false 回退现状。
    memory_cold_decay_enabled: bool = True
    memory_cold_decay_days: int = 30
    memory_cold_decay_min: float = 0.3

    # 记忆冲突消解（module-061 / ADR-0007 P1）：true 时 _merge_duplicate 去重
    # 命中后走矛盾判定（contradiction → 旧父块标 superseded=true + 新内容按正常新增
    # 入库，替代"拼接共存"）；false 完全旧行为（追加拼接，零回归）。
    # module-061 原默认 false（旧双门槛 Recall≥0.8 且 Precision≥0.8 未达标）；
    # module-062 WP4 用户决策改为 **Precision≥0.8 者启用**（Recall 后续提升不阻塞，
    # 保守方向：宁可漏检也不错标）。实测（module-062 WP4 同 30 条评测集）：
    #   clf（bge-m3+LR，142 案例训练）：Precision 0.9048 / Recall 0.9500（eval_runs id=34）
    #   nli（mDeBERTa）：Precision 1.0000 / Recall 0.5000（eval_runs id=35，module-061 复现）
    # 双达标取 Precision 高者 → **nli 启用**（PW_MEMORY_CONFLICT=true，judge=nli）；
    # clf Recall 更高（0.95 vs 0.5），产品如需更全召回可 PW_MEMORY_CONFLICT_JUDGE=clf
    # 一键切换（已在 config 预置）。矛盾判定器不可用/超时 → 返回 None → 旧行为（零回归）。
    # module-070：双判共识——nli+clf 双确认 contradiction 才标 superseded（Precision
    # 极保守，冤枉=误标 superseded=用户记忆消失），单判 contradiction → conflict_hint
    # 新旧并存；任一裁判不可用对称回退单判（clf 缺失→nli 单判=现状零回归）。
    # **默认值决策（module-070 WP-A 70 条真实跑分，eval_runs id=46/47/48）**：dual
    # Precision 0.9412（fp=1）为三方案最高，符合用户"宁漏检也不错标"哲学；clf
    # Recall 最高（0.775）但 fp=7（误标 7 条用户记忆，最贵失败模式）；nli 30 条
    # 口径 1.0000 Precision 为"窄而准"假象（70 条跌至 0.9167）——详见 changelog。
    memory_conflict_enabled: bool = True
    memory_conflict_judge: Literal["clf", "nli", "dual"] = "dual"

    # 意图分类（module-043 L4）：true 时 router 尝试加载 bge-m3+逻辑回归分类器
    #（模型缺失/加载失败自动回退 LLM 分类，零影响）。
    # module-056 达标启用：人造标注集 337 条重训 + golden_intent 100 条真实
    # 评测（LLM 1.0000 / 分类器 1.0000，eval_runs id=23/24）→ 默认开；
    # 回退开关：PW_INTENT_CLASSIFIER_ENABLED=false 保持 LLM 路径
    intent_classifier_enabled: bool = True
    # 意图分类多轮拼接（module-063 / WP-A）：true 时 router 给 L4 分类器传
    # 最近一轮 user query 向量拼接（2048 维，训练时同构）。**当前落盘模型
    # intent_clf.joblib 为单 query 1024 维训练**——置 true 需先重训（eval/
    # train/train_intent_classifier.py 构造含 prev 的配对样本，模型维度对齐
    # 2048）；未重训置 true 会令 L4 推理维度不匹配抛异常 → router 捕获回退
    # LLM（fail-open 零回归，仅损失多轮场景 L4 成本优势）。默认 false =
    # 存量模型零回归（多轮路由走 LLM 上下文 + 短句继承，见 ADR-0015）。
    # （2026-08-16 架构评估：多轮拼接已降级不做——能力已被 LLM 路径覆盖
    # + WP-B 规则层兜底，性价比不足，见 METRICS 待办区 #8；恒 false 不启用）
    intent_classifier_multi_turn: bool = False

    # 反思充分性自洽性检查（module-044 层 2）：true 时 check_sufficiency 对
    # 同一 query 用两个不同温度各判一次，两次不一致 → 保守判充分（防漏检）；
    # 默认 false = 零额外 LLM 调用（成本翻倍，按需开启）
    sufficiency_self_check_enabled: bool = False

    # 反思充分性硬闸门阈值（module-048，module-047 实测数据结论）：
    # check_sufficiency 层 1 top-1 abs_cosine < 该值 → 直接判不充分（零 LLM）。
    # module-047 阈值扫描：0.4 漏判 60% 不充分；0.55 切在分布间隙上缘
    #（充分 min 0.490 / 不充分 max 0.550），F1=0.98 最优且误杀与 0.5 相同
    #（1/50）。不得改回 0.4（红线）。
    sufficiency_gate_threshold: float = 0.55

    # 分诊式 Query 改写（module-049 / ADR-0009）：
    # 静态分诊（FTS 术语命中 → 精确 query 直接检索，零成本不走改写）+ 模糊
    # query 走 LLM 改写 + 保真预检（改写 vs 原 query 余弦 < 阈值 → 回退原话，
    # 省一次检索）+ 并行检索择优（改写检索 top-1 绝对余弦 > 原检索 → 用改写
    # 结果，否则回退原结果）。改写链路任何一环失败 → 回退原 query（零回归）。
    # 回退开关：PW_QUERY_REWRITE_ENABLED=false。
    # **默认值决策（module-072 WP-C 四跑实测达标，2026-08-19）**：golden_intent
    # Accuracy on 1.0000 ≥ off 1.0000 − 0.01 + 短路触发 50/100 判对率 100% +
    # golden_multi_turn 意图保持 on 1.0 ≥ off 1.0 − 0.01 / 检索 +0.60 ≥ +0.60 − 0.01
    # → 全达标改默认 true（短路路由零误杀省 LLM 调用；短路 = 分诊 precise 且非
    # 规则词 → knowledge，纯确定性零 LLM）。测试环境 conftest autouse 钉住 false。
    query_rewrite_enabled: bool = True
    rewrite_fidelity_threshold: float = 0.6  # 保真预检阈值：改写与原 query 余弦低于该值 → 回退原话

    # 上下文改写（module-072，PW_CONTEXTUAL_REWRITE_ENABLED）：
    # true 时 engine 给分诊式改写链传对话历史（history），当前句为省略句/
    # 指代句（分诊 vague）时结合上一轮问题改写为自包含 query，修复多轮
    # "为什么"检索落空（04 文档 #1）。复用 module-049 保真预检，锚点 =
    # f"{prev} {query}" 拼接双锚（裸省略句作锚无信息量会系统性误杀）。
    # 与 query_rewrite_enabled 独立生效（prepare 调用条件 OR，两开关独立
    # 评测独立决策——本开关是检索侧增益，短路路由是路由侧成本收益）。
    # 回退开关：PW_CONTEXTUAL_REWRITE_ENABLED=false。
    # **默认值决策（module-072 WP-A 接入前后对比，2026-08-19）**：意图保持
    # 12/12 = 接入前不降 + 检索提升 +0.60 ≥ 接入前 +0.4363 + vague 句改写
    # 能力 0/1 → 1/1（"为什么"检索 0.00→0.60 实测修复）→ 达标改默认 true。
    # self_contained 全量 0.0833 系 triage-precise 度量口径（11/12 含术语自
    # 包含句不改写，plan 实现要点 4/5 已预言；意图/检索零回归），详见 changelog。
    contextual_rewrite_enabled: bool = True

    # 答案验证裁判（module-051 / ADR-0010 P0-②）：
    # verify_answer 的 verdict 判定模型——"hhem"（默认）：LLM 拆句 + HHEM-2.1-Open
    # 批量判分（module-050 实测中文 Accuracy 0.77 显著胜出 MiniCheck 0.51，选型已定）；
    # "llm"：完全不加载 HHEM，直走旧逻辑（零回归开关）。HHEM 不可用（缺失/加载失败/
    # 推理异常）自动降级 LLM 判分，降级链保证默认 "hhem" 零风险。
    verify_judge_model: str = "hhem"
    # HHEM 三态映射阈值（经验值，标注集可校准）：每 claim 对每文档打分取 max →
    # max_score ≥ high → supported；low ≤ max_score < high → inferred；< low → unsupported
    verify_hhem_threshold_high: float = 0.7
    verify_hhem_threshold_low: float = 0.3

    # verify 异步化（module-060）：true（默认）——chat_stream 流式生成完不再
    # 同步 await verify（15-50s 阻塞主链路尾部），改 submit 后台任务 + done 事件
    # 带 verify_task_id + 前端轮询 GET /ai/rag/chat/verify/{task_id} 补结果，
    # 结果落 verify_results 表持久化（done 不因重启丢失）。false 回退现状同步
    # 路径（verified→done 事件逐字一致，逃生口）。测试环境由 conftest autouse
    # fixture 钉住 false（存量 chat_stream 测试零漂移）。
    verify_async_enabled: bool = True

    # 多格式文档解析与清洗（module-064 / ADR-0014）：
    #   image_ocr_enabled（PW_IMAGE_OCR）—— L1 PDF 内嵌图片 OCR（图内文字，
    #       PaddleOCR/RapidOCR），默认关（重工具不默认启用，只路由复杂文档）；
    #       组件缺失时该层自动降级（fail-open，图片标记未解析）。
    #   image_caption_enabled（PW_IMAGE_CAPTION）—— L2 本地轻量 VLM 图片描述
    #       插回 Markdown（显式占位符替换），默认关；模型缺失降级关（fail-open）。
    #   pdf_engine（PW_PDF_ENGINE）—— PDF 解析引擎："anydoc"（默认，统一 GFM
    #       Markdown 输出）；"mineru"（L3 复杂版面独立通道，MinerU 未安装时
    #       fail-open 回退 anydoc/PyMuPDF）。三层全默认关 + 分层路由见
    #       rag/retrieval/image_pipeline.py。
    #   upload_dir（PW_UPLOAD_DIR）—— 上传原始文件落盘目录（WP5 原件留存，重灌
    #       依赖；相对路径相对 ai_service 运行目录解析）。
    #   doc_dedup_semantic_enabled（PW_DOC_DEDUP_SEMANTIC）—— 文档级语义去重
    #       开关（WP6 L2：embedding 余弦≥doc_dedup_threshold 不删，标簇+canonical）。
    #   doc_dedup_threshold（PW_DOC_DEDUP_THRESHOLD）—— 语义重复判定阈值（默认
    #       0.95，对齐 module-035 记忆去重同款绝对余弦口径，复用 bge-m3）。
    #   doc_dedup_boilerplate_enabled（PW_DOC_DEDUP_BOILERPLATE）—— 相似度计算
    #       前剥离 Boilerplate（共同页脚/免责声明），防套话主导相似度。
    image_ocr_enabled: bool = False
    image_caption_enabled: bool = False
    pdf_engine: Literal["anydoc", "mineru"] = "anydoc"
    upload_dir: str = "uploads"
    doc_dedup_semantic_enabled: bool = True
    doc_dedup_threshold: float = 0.95
    doc_dedup_boilerplate_enabled: bool = True

    #   doc_dedup_candidate_top_k（PW_DOC_DEDUP_CANDIDATE_TOP_K）—— L2 语义去重
    #       向量候选上限（pgvector top-K，ORDER BY embedding <=> :vec LIMIT :k）：
    #       默认 50 远超语义重复量级；O(N) 全表余弦 → O(log N + K)（module-079）。
    doc_dedup_candidate_top_k: int = 50

    # PDF 回退路径 Markdown 升级（module-069）：
    #   true（默认）—— PyMuPDF 回退路径用 pymupdf4llm.to_markdown() 输出
    #     Markdown（标题/列表/表格恢复），双栏页面先走中线重组再出 MD。
    #   false —— 走旧路径 page.get_text() 裸文本（存量行为零回归，逃生口）。
    #   pymupdf4llm 仍 AGPL-3.0（与 PyMuPDF 同许可），输出仍过清洗层
    #   （document_cleaner.clean()）——pymupdf4llm 解决结构，清洗层解决格式噪声。
    pdf_fallback_md: bool = True

    # 知识抓取流水线（module-075 / ADR-0019 阶段2）：
    #   crawl_enabled —— 抓取功能总开关（false 时不启动调度器 + run_crawl 直接返回空）。
    #   crawl_interval_minutes —— 定时抓取间隔（默认 1440=24h，config 可调）。
    #   crawl_max_pages_per_run —— 单次抓取最大页数上限（防失控）。
    # 测试环境由 conftest autouse fixture 钉住 crawl_enabled=false（hermetic）。
    crawl_enabled: bool = False
    crawl_interval_minutes: int = 1440
    crawl_max_pages_per_run: int = 10

    # 递归爬取（module-076）：深度控制 + 黑名单 + 单页链接上限
    # crawl_max_depth —— 全局深度上限（min(source.max_depth, crawl_max_depth) 生效）
    # crawl_blacklist_patterns —— 逗号分隔 URL 前缀黑名单（PW_CRAWL_BLACKLIST_PATTERNS）
    # crawl_max_links_per_page —— 单页提取链接数上限（防导航页爆量）
    crawl_max_depth: int = 2
    crawl_blacklist_patterns: str = ""
    crawl_max_links_per_page: int = 20
    # 审查节点增强（module-078 / ADR-0019 阶段2）：
    #   crawl_review_policy —— 审查策略三档：fail-open（默认，module-075 零回归：
    #     审查异常/矛盾命中仅记录不改变 status）/ lenient（矛盾命中 rejected，异常
    #     仍放行）/ strict（fail-closed：审查异常 rejected + 阈值更严）。
    #     PW_CRAWL_REVIEW_POLICY 切换，Literal 校验非法值启动即抛 ValidationError。
    #   crawl_hhem_threshold —— HHEM 质量分拒绝阈值（0.3 = module-075 硬编码值，
    #     零回归）；crawl_hhem_threshold_strict —— strict 档更严阈值（0.45）。
    #   crawl_conflict_top_k / crawl_conflict_min_cosine —— 矛盾候选向量查询参数：
    #     根父块 top-K + cosine 下限过滤（不相干文档无矛盾语义，过滤防误报）。
    crawl_review_policy: Literal["fail-open", "lenient", "strict"] = "fail-open"
    crawl_hhem_threshold: float = 0.3
    crawl_hhem_threshold_strict: float = 0.45
    crawl_conflict_top_k: int = 3
    crawl_conflict_min_cosine: float = 0.6

    # 反爬绕过 + 代理池（module-077 / ADR-0019 阶段2 第三片）：
    #   crawl_request_delay_seconds —— 同源请求间隔（秒），_recursive_crawl 每个
    #     子链接 fetch 前 await asyncio.sleep，防频率封禁。0 = 不限速。
    #   crawl_retry_max —— 429/5xx 指数退避最大重试次数（0 = 不重试）。
    #   crawl_retry_base_seconds —— 重试退避基础延迟（秒），实际 = base × 2^attempt + jitter。
    #   crawl_proxies —— 逗号分隔 HTTP 代理列表（http://host:port），空 = 直连。
    #     round-robin 轮换，失败自动切下一个，全部失败 → fail-open 直连。
    #   crawl_robots_cache_ttl —— robots.txt 解析结果缓存 TTL（秒），0 = 不缓存。
    #   crawl_user_agents —— 逗号分隔自定义 UA 列表，空 = 使用内置 ~10 个浏览器 UA 池。
    crawl_request_delay_seconds: float = 1.0
    crawl_retry_max: int = 3
    crawl_retry_base_seconds: float = 2.0
    crawl_proxies: str = ""
    crawl_robots_cache_ttl: int = 3600
    crawl_user_agents: str = ""

    # 反向闭环（module-080）：待学笔记优先级加权
    # 待学笔记主题关键词匹配源 url_pattern/name 时，动态提升该源的内存态 priority
    #（不写回 DB，每次 run_crawl 动态算）。默认 10，PW_WEAK_TOPIC_PRIORITY_BOOST 可覆盖。
    weak_topic_priority_boost: int = 10

    # 反向闭环（module-080）：低分题→待学笔记→优先级抓取。环境变量 PW_FEEDBACK_*可覆盖；
    # reverse_enabled 总开关 / low_score_threshold=60 对齐 Java / scan_interval_minutes=1440
    # 次日一次 / internal_token 调 weak-points（空=不带头，失败在 Java 侧 fail-closed）
    feedback_reverse_enabled: bool = True
    feedback_java_base_url: str = "http://localhost:8002"
    feedback_low_score_threshold: int = 60
    feedback_scan_interval_minutes: int = 1440
    feedback_http_timeout_s: float = 10
    feedback_learning_identity: str = "learning"
    feedback_search_url_template: str = "https://www.bing.com/search?q={query}"
    feedback_priority_crawl_depth: int = 1
    feedback_priority_max_per_run: int = 5
    feedback_internal_token: str = ""

    model_config = {"env_prefix": "PW_", "env_file": ".env"}


settings = Settings()
