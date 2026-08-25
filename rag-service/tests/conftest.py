"""
pytest 全局配置 — 测试环境统一取消 IP 限流
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景：本地限流器（src/ratelimit.py，滑动窗口 20 次/60s）对所有非 health
请求生效。全量测试用同一 IP（127.0.0.1）高频调用 HTTP 端点会触发 429，
历史表现为 test_rerank_langgraph.py 的 2 项预存失败（assert 429 == 200）。

本文件在测试进程内把 main 模块绑定的 check_rate_limit 替换为恒放行，
生产行为完全不受影响（patch 只存在于 pytest 进程）。IP 注入
（request.state.client_ip / user_id）是中间件独立步骤，不受影响。
"""
import pytest
from unittest.mock import AsyncMock


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch):
    """全量测试取消 IP 限流（仅测试进程内生效）

    main.py 通过 from-import 把 check_rate_limit 绑定到自身命名空间，
    需 patch main 模块属性而非 src.ratelimit 源函数。
    """
    import main as main_module

    monkeypatch.setattr(
        main_module,
        "check_rate_limit",
        lambda client_ip, **kwargs: (True, 0),
    )


@pytest.fixture(autouse=True)
def default_intent_classifier_disabled(monkeypatch):
    """测试环境统一钉住 L4 分类器开关=关闭（module-056）

    生产默认已开启（PW_INTENT_CLASSIFIER_ENABLED 默认 true），但单测需
    hermetic：不依赖真实模型文件 models/intent_clf.joblib（非仓库产物），
    存量 LLM 路径用例不受影响。测试体内显式 setattr True 的用例
    （如 test_intent_dataset.py 的 L4 回退用例）后写覆盖本钉住值。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "intent_classifier_enabled", False)


@pytest.fixture(autouse=True)
def default_reranker_quantize_disabled(monkeypatch):
    """测试环境统一钉住 reranker int8 量化开关=关闭（非 module 性能优化批）

    生产默认开启（PW_RERANKER_QUANTIZE_ENABLED 默认 true，CPU 实测 ~2x 提速），
    但单测需 hermetic：不加载真实模型（存量用例全 mock CrossEncoder），钉住
    false 防任何真实加载路径触发 11s 量化；新用例体内显式 setattr True 可覆盖。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "reranker_quantize_enabled", False)


@pytest.fixture(autouse=True)
def default_tool_phase_split_disabled(monkeypatch):
    """测试环境统一钉住工具阶段切分开关=关闭（module-058，对齐 056 模式）

    生产默认已开启（PW_TOOL_PHASE_SPLIT 默认 true），但存量 react 层 agent
    测试以全量 10 个工具 schema 为准——默认 true 会漂移走 react 层的存量
    测试（检索阶段 schema 只有 7 个）。钉住 false 是"存量测试全绿"的真正
    保证；新测试（test_tool_phase_split.py）体内显式 setattr True 验证切分。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "tool_phase_split", False)


@pytest.fixture(autouse=True)
def default_max_agent_tools_4(monkeypatch):
    """测试环境钉住总预算=4（module-068）

    生产默认 max_agent_tools 已 4→5（总预算 = 检索阶段 ≤3 + 生成阶段 ≤2
    的兜底和），但存量 react/langgraph 测试断言 budget==4
    （test_agent_tools.py:585/679/705、test_rerank_langgraph.py:456）——
    钉住 4 使存量断言逐字保持（对齐 056/058/066 模式）；新测试
    （test_agent_phase_fix.py）体内显式 setattr 覆盖验证阶段预算。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "max_agent_tools", 4)


@pytest.fixture(autouse=True)
def tool_call_logs_disabled(monkeypatch):
    """测试环境统一钉住 tool_call_logs 落库开关=关闭（module-066，对齐 056/058 模式）

    生产默认已开启（PW_TOOL_CALL_LOGS 默认 true），但单测需 hermetic：不
    依赖真实 DB 落库、不污染 tool_call_logs 表（存量 react 循环测试全量
    覆盖执行路径，默认 true 会触发真实连接）。新测试（test_tool_call_logs.py）
    体内显式开启验证落库（配合假 session 打桩）。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "tool_call_logs_enabled", False)


@pytest.fixture(autouse=True)
def default_tool_auto_retry_disabled(monkeypatch):
    """测试环境统一钉住工具失败自动重试开关=关闭（module-073，对齐 056/058/066 模式）

    生产默认已开启（PW_TOOL_AUTO_RETRY 默认 true，task-brief 指定），但存量
    test_agent_tools.py 以"失败一次返回空"为基准断言——钉住 false 保证存量
    测试 hermetic（不实际触发重试路径/不改变 func 执行次数），重试行为由新
    测试（test_tool_retry_dedup.py）体内显式 setattr True 验证。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "tool_auto_retry", False)


@pytest.fixture(autouse=True)
def request_logs_disabled(monkeypatch):
    """测试环境统一钉住 request_logs 落库开关=关闭（module-058 WP-C）

    生产默认已开启（PW_REQUEST_LOGS 默认 true），但单测需 hermetic：不
    依赖真实 DB 落库、不污染 request_logs 表。新测试（test_observability.py）
    体内显式开启验证埋点/落库（配合假 session 打桩）。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "request_logs_enabled", False)


@pytest.fixture(autouse=True)
def default_verify_async_disabled(monkeypatch):
    """测试环境统一钉住 verify 异步开关=关闭（module-060，对齐 056/058 模式）

    生产默认已开启（PW_VERIFY_ASYNC 默认 true），但存量 chat_stream 测试以
    现状同步路径（verified→done 顺序）为准——默认 true 会漂移走异步路径
    （不再发 verified 事件）导致存量断言失败。钉住 false 是"存量测试全绿"
    的真正保证；新测试（test_verify_tasks.py）体内显式 setattr True 验证
    异步行为（配合 mock DB，不依赖真实 PG 落库）。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "verify_async_enabled", False)


@pytest.fixture(autouse=True)
def default_memory_conflict_disabled(monkeypatch):
    """测试环境统一钉住记忆冲突消解开关=关闭（module-061，对齐 056/058/060 模式）

    生产默认已关闭（PW_MEMORY_CONFLICT 默认 false，评测达标才启用），但存量
    记忆测试（module-033/034/035/046）以旧行为（去重命中 → 追加拼接）为准——
    显式钉住 false 是"存量测试全绿 + 开关 false 完全旧行为零回归"的双重保证
    （即使生产误开也不会漂移存量测试）。新测试（test_memory_correction.py）
    体内显式 setattr True 验证冲突分流（配合 mock NLI，不依赖真实模型）。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "memory_conflict_enabled", False)


@pytest.fixture(autouse=True)
def default_document_pipeline_switches(monkeypatch):
    """测试环境钉住 module-064 文档管线新开关=保守值（对齐 056/058/060/061/062 模式）

    生产默认 doc_dedup_semantic_enabled=True（WP6 L2 语义去重），但单测需
    hermetic：语义去重要真实 bge-m3 embedding + DB 查询（慢/依赖模型）。钉住
    false 防意外触发；新测试（test_document_ingest.py 等）体内显式 setattr True
    + mock 各层验证。图片三层（image_ocr/image_caption）默认已关无需钉住，
    pdf_engine 默认 anydoc 不触发额外通道。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "doc_dedup_semantic_enabled", False)


@pytest.fixture(autouse=True)
def default_pdf_fallback_md_disabled(monkeypatch):
    """测试环境统一钉住 PDF 回退 Markdown 升级开关=关闭（module-069）

    生产默认已开启（PW_PDF_FALLBACK_MD 默认 true），但存量 _parse_pdf_pymupdf
    测试以旧行为（page.get_text() 裸文本）为准——默认 true 会漂移走 pymupdf4llm
    路径导致存量断言失败。钉住 false 是"存量测试全绿"的真正保证；新测试体内
    显式 setattr True 验证 pymupdf4llm 行为（配合 mock）。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "pdf_fallback_md", False)


@pytest.fixture(autouse=True)
def default_memory_evolution2_disabled(monkeypatch):
    """测试环境统一钉住 module-062 新开关=保守值（对齐 056/058/060/061 模式）

    生产默认：memory_type_mode='none'（类型注入待 WP1 达标后由 winner 决定）、
    memory_type_decay_enabled=true（存量无 type → 走 memory_short_half_life=3 零
    回归，故无需钉住）、memory_cold_decay_enabled=true（长期层冷降权）、
    memory_conflict_judge='clf'（WP4 达标后 PW_MEMORY_CONFLICT=true 才生效）。
    钉住原因：
      - memory_type_mode='none'：类型注入不依赖真实分类器/LLM 判型（hermetic），
        存量 _persist_memory 测试以 type 默认 fact 为准零漂移
      - memory_cold_decay_enabled=False：存量长期 recall 测试不触发冷降权额外
        DB 查询/刷新任务（冷降权测试体内显式 setattr True + mock）
      - memory_conflict_judge='nli'：module-061 矛盾测试 mock NLI 断言 NLI 路径
        语义，钉住 nli 保持其 hermetic；clf 裁判测试体内显式设 'clf' + mock
    新测试（test_memory_evolution2.py）体内显式 setattr 覆盖验证各开关行为。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "memory_type_mode", "none")
    monkeypatch.setattr(settings, "memory_cold_decay_enabled", False)
    monkeypatch.setattr(settings, "memory_conflict_judge", "nli")


@pytest.fixture(autouse=True)
def default_rewrite_switches_disabled(monkeypatch):
    """测试环境统一钉住两改写开关=关闭（module-072，对齐 056/058 模式）

    module-072 WP-C 四跑达标后两开关生产默认均为 true（PW_QUERY_REWRITE_
    ENABLED / PW_CONTEXTUAL_REWRITE_ENABLED），存量引擎测试以改写前行为为
    准——钉住 false 保证存量测试不触发分诊改写/历史透传/真实 LLM 调用
    （hermetic）；新测试（test_query_rewrite_history.py 等）体内显式
    setattr True + mock 验证各开关行为。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(settings, "contextual_rewrite_enabled", False)


@pytest.fixture(autouse=True)
def default_crawl_disabled(monkeypatch):
    """测试环境统一钉住抓取功能开关=关闭（module-075，对齐 056/058 模式）

    生产默认已关闭（PW_CRAWL_ENABLED 默认 false），但显式钉住是"存量测试
    全绿"的双重保证。新测试（test_crawler.py）体内显式 setattr True 验证。
    """
    from src.config import settings

    monkeypatch.setattr(settings, "crawl_enabled", False)


@pytest.fixture(autouse=True)
def default_antibot_mocks(monkeypatch):
    """测试环境钉住 module-077 反爬功能=安全值（对齐 056/058 模式）

    robots 检查恒放行（不触发真实 HTTP）；限速恒 0（不延迟）；代理恒空（直连）；
    重试恒 0（不重试，保持存量行为）。
    """
    from src.config import settings
    from rag.crawl.crawler import _check_robots_allowed

    monkeypatch.setattr(settings, "crawl_retry_max", 0)
    monkeypatch.setattr(settings, "crawl_proxies", "")
    monkeypatch.setattr(settings, "crawl_robots_cache_ttl", 0)
    monkeypatch.setattr(settings, "crawl_user_agents", "")
    monkeypatch.setattr(
        "rag.crawl.crawler._check_robots_allowed",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(settings, "crawl_request_delay_seconds", 0)  # module-077: 防测试限速延迟
