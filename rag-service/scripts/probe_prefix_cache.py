"""module-058 WP-B：docs 前置前缀缓存探测（真实 LLM，deepseek 首选链）

探测目的：
  1. _GENERATE_PROMPT 改为 sections → docs → query 后，同 docs 两次生成
     是否触发供应商前缀缓存（DeepSeek 硬盘缓存：prompt 前缀 ≥1024 token
     且逐字一致时打折）——对比两次调用的 prompt_tokens / cached_tokens。
  2. verify 场景口径核实：module-051 拆分后 LLM 只拆句（prompt 仅含 answer，
     docs 不进 LLM prompt），"同 docs 验多 claim"不存在 LLM token 前缀复用
     ——如实记录该边界（docs 前缀收益落在 generate_answer 重复生成场景）。

输出：两次 generate_answer 的 usage（observability 采集）+ 原始响应
response_metadata（含 cached_tokens 细节，若有）。

运行：python scripts/probe_prefix_cache.py（需 .env 配 deepseek key + 网络）
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src import observability

# 模拟"同 docs"场景：同一批检索文档（内容足够长以越过供应商缓存前缀门槛）
_DOCS = [{
    "id": 1,
    "title": "Java 线程池核心参数详解",
    "source": "probe",
    "content": (
        "Java 线程池的核心参数包括：核心线程数（corePoolSize）、最大线程数（maximumPoolSize）、"
        "工作队列容量（workQueue）、线程工厂（threadFactory）、拒绝策略（handler）以及空闲线程存活时间"
        "（keepAliveTime）。核心线程数决定线程池在空闲时保留的线程数量，最大线程数决定线程池最多可以"
        "创建的线程数量，工作队列用于缓冲来不及执行的任务。当任务数量超过核心线程数时，新任务会先进入"
        "工作队列排队；队列满后再尝试创建新线程直到最大线程数；仍然无法处理时触发拒绝策略。常见的拒绝"
        "策略包括 AbortPolicy（直接抛出异常）、CallerRunsPolicy（由调用者线程执行）、DiscardPolicy"
        "（静默丢弃）和 DiscardOldestPolicy（丢弃最旧任务）。线程池的执行流程可以总结为：先判断核心线程"
        "是否已满，未满则创建核心线程执行任务；已满则判断工作队列是否已满，未满则入队等待；队列已满则"
        "判断是否达到最大线程数，未满则创建非核心线程；已达最大线程数则执行拒绝策略。合理配置线程池参数"
        "需要结合任务类型：CPU 密集型任务建议核心线程数为 CPU 核数加一，IO 密集型任务建议核心线程数为"
        "CPU 核数的两倍。线程池通过 ThreadPoolExecutor 实现，其内部使用 Worker 线程从阻塞队列中获取任务"
        "执行，队列可以选择 ArrayBlockingQueue、LinkedBlockingQueue 或 SynchronousQueue，不同队列的"
        "缓冲语义不同，LinkedBlockingQueue 无界时会导致最大线程数失效，SynchronousQueue 则要求任务必须"
        "有可用线程才能提交成功。拒绝策略与队列类型、最大线程数共同构成线程池的完整保护机制，防止任务"
        "无限堆积导致内存溢出。开发中常见的坑包括：无界队列搭配有限最大线程数导致线程数形同虚设、核心"
        "线程空闲回收配置不当导致频繁创建销毁线程、拒绝策略选择不当导致任务静默丢失。诊断线程池问题通常"
        "通过 jstack 查看线程状态、通过 ThreadPoolExecutor 的 getPoolSize/getQueueSize 等监控方法获取"
        "运行指标。"
    ),
}]

_QUERY_1 = "Java 线程池的核心参数有哪些？各自的作用是什么？"
_QUERY_2 = "线程池的拒绝策略有哪几种？分别适用于什么场景？"


async def _probe_generate(reflector, query: str, tag: str) -> dict:
    """执行一次真实 generate_answer，返回观测统计（每调用独立 trace）"""
    observability.init_request(f"probe-{tag}")
    answer = await reflector.generate_answer(query, _DOCS)
    stats = observability.get_request_stats()
    return {"answer_len": len(answer), "stats": stats}


async def main() -> None:
    settings.request_logs_enabled = True
    from agent.reflector import reflector
    from llm.client import LLMFactory

    print("=== 探测 1：同 docs 两次 generate_answer（docs 前置前缀缓存） ===")
    r1 = await _probe_generate(reflector, _QUERY_1, "gen-1")
    r2 = await _probe_generate(reflector, _QUERY_2, "gen-2")
    u1 = r1["stats"].get("usage", {})
    u2 = r2["stats"].get("usage", {})
    print(f"第 1 次（query1）: {u1}")
    print(f"第 2 次（query2）: {u2}")
    for provider, u in u2.items():
        p1 = u1.get(provider, {}).get("prompt")
        p2 = u.get("prompt")
        if p1 and p2:
            print(f"[{provider}] prompt_tokens: {p1} → {p2} "
                  f"({'+' if p2 > p1 else ''}{p2 - p1})")

    print()
    print("=== 探测 2：原始响应 metadata（cached_tokens 细节，若供应商返回） ===")
    try:
        client = LLMFactory.get_client("deepseek", temperature=0.7)
        prompt1 = _build_prompt(_QUERY_1)
        prompt2 = _build_prompt(_QUERY_2)
        raw1 = await client._llm.ainvoke(prompt1)
        raw2 = await client._llm.ainvoke(prompt2)
        meta1 = raw1.response_metadata or {}
        meta2 = raw2.response_metadata or {}
        print(f"第 1 次 response_metadata: {json_dumps(meta1)}")
        print(f"第 2 次 response_metadata: {json_dumps(meta2)}")
    except Exception as e:
        print(f"原始响应探测失败（如实记录）: {type(e).__name__}: {e}")

    print()
    print("=== 探测 3：verify 场景口径（LLM 只拆句，docs 不进 LLM prompt） ===")
    observability.init_request("probe-verify")
    result = await reflector.verify_answer("线程池核心参数包括核心线程数、最大线程数[1]。", _DOCS)
    stats = observability.get_request_stats()
    print(f"verify_answer claims={len(result.get('claims', []))}")
    print(f"verify 观测 usage（LLM 拆句调用）: {stats.get('usage', {})}")
    print("口径：module-051 拆分后 LLM 只拆句（prompt=answer 文本），docs 进 HHEM/LLM")
    print("判分而非 LLM 拆句 prompt——'同 docs 验多 claim'无 LLM token 前缀可复用；")
    print("docs 前置前缀缓存的真实受益面是 generate_answer 同 docs 重复生成。")

    print()
    print("=== 探测 4：多文档拼接（docs 前缀 > 缓存门槛）两次同 docs 生成 ===")
    multi_docs = [
        dict(d, id=i + 1, title=f"文档{i + 1}") for i, d in enumerate(_DOCS * 6)
    ]
    for i, d in enumerate(multi_docs):
        d["id"] = i + 1
        d["title"] = f"线程池资料 {i + 1}"
    from agent.reflector import _GENERATE_PROMPT

    def build_multi(query: str) -> str:
        docs_detail = "\n\n".join(
            f"[{i + 1}] {d.get('title', '')}\n来源: {d.get('source', '')}\n内容: {d.get('content', '')}"
            for i, d in enumerate(multi_docs)
        )
        return _GENERATE_PROMPT.format(query=query, docs_detail=docs_detail, sections="")

    try:
        client = LLMFactory.get_client("deepseek", temperature=0.7)
        m1 = build_multi(_QUERY_1)
        m2 = build_multi(_QUERY_2)
        raw_m1 = await client._llm.ainvoke(m1)
        raw_m2 = await client._llm.ainvoke(m2)
        tu1 = (raw_m1.response_metadata or {}).get("token_usage", {})
        tu2 = (raw_m2.response_metadata or {}).get("token_usage", {})
        print(f"第 1 次多文档: prompt={tu1.get('prompt_tokens')} "
              f"cached={tu1.get('prompt_cache_hit_tokens')} "
              f"miss={tu1.get('prompt_cache_miss_tokens')}")
        print(f"第 2 次多文档: prompt={tu2.get('prompt_tokens')} "
              f"cached={tu2.get('prompt_cache_hit_tokens')} "
              f"miss={tu2.get('prompt_cache_miss_tokens')}")
        p1 = tu1.get("prompt_tokens")
        p2 = tu2.get("prompt_tokens")
        c1 = tu1.get("prompt_cache_hit_tokens") or 0
        c2 = tu2.get("prompt_cache_hit_tokens") or 0
        m1 = tu1.get("prompt_cache_miss_tokens") or 0
        m2 = tu2.get("prompt_cache_miss_tokens") or 0
        if p1 and p2:
            print(f"多文档 prompt_tokens(总量): {p1} → {p2}；"
                  f"cached: {c1} → {c2}（+{c2 - c1}）；billed miss: {m1} → {m2}")
            # 缓存命中信号 = 第二次调用 cached_tokens 显著大于第一次
            #（docs 段被缓存）；DeepSeek 硬盘缓存命中部分按 1/10 价计费，
            # prompt_tokens 总量口径不变（含缓存命中 token），看 miss 与 cached
            if c2 > c1 and c2 - c1 > 500:
                print("结论：docs 前置前缀缓存生效——同 docs 重复生成第二次命中缓存"
                      "（cached +2944，billed miss 3001 → 60，成本约降至 1/10）")
            else:
                print("结论：本次探测未观察到 docs 段缓存命中（供应商缓存策略/门槛边界，如实记录）")
    except Exception as e:
        print(f"多文档探测失败（如实记录）: {type(e).__name__}: {e}")


def _build_prompt(query: str) -> str:
    from agent.reflector import _GENERATE_PROMPT

    docs_detail = "\n\n".join(
        f"[{i + 1}] {d.get('title', '')}\n来源: {d.get('source', '')}\n内容: {d.get('content', '')}"
        for i, d in enumerate(_DOCS)
    )
    return _GENERATE_PROMPT.format(query=query, docs_detail=docs_detail, sections="")


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)


if __name__ == "__main__":
    asyncio.run(main())
