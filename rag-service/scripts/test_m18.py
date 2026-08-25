"""M18 test script: verify Rerank fix (Qwen3-Reranker-0.6B).

覆盖验收标准:
  - 加载 + 排序: rerank() 返回按 rerank_score 降序的 top_k 条
  - 边界: 空 documents / 单文档 / top_k 大于文档数 / 缺 content 字段
  - 异常: 本地模型目录不存在或缺权重文件 → RerankerException（不回退 HF）
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from rag.retrieval.reranker import CrossEncoderReranker, RerankerException


async def main():
    print("=== Test 0: 目录存在但缺权重文件 → 明确报错 ===")
    with tempfile.TemporaryDirectory() as tmp:
        # 只放一个无关文件，无 model.safetensors / pytorch_model.bin
        open(os.path.join(tmp, "tokenizer.json"), "w").close()
        no_weight = CrossEncoderReranker(model_name=tmp)
        try:
            await no_weight.rerank("test", [{'id': 1, 'content': 'x'}])
            print("FAIL: 应抛 RerankerException")
            return 1
        except RerankerException as e:
            print(f"PASS: raised RerankerException: {e}")
    print("PASS\n")

    print("=== Test 1: 目录不存在明确报错（不回退 HF）===")
    bad = CrossEncoderReranker(model_name="/nonexistent/model/path")
    try:
        await bad.rerank("test", [{'id': 1, 'content': 'x'}])
        print("FAIL: 应抛 RerankerException")
        return 1
    except RerankerException as e:
        print(f"PASS: raised RerankerException: {e}")
    print("PASS\n")

    print("=== Test 2: 加载真实模型 + 排序 ===")
    rr = CrossEncoderReranker()
    docs = [
        {'id': 1, 'content': 'Java 线程池的核心参数包括核心线程数、最大线程数'},
        {'id': 2, 'content': 'Redis 缓存穿透是指查询不存在的数据'},
        {'id': 3, 'content': '线程池的拒绝策略有 AbortPolicy、CallerRunsPolicy'},
    ]
    result = await rr.rerank('Java 线程池参数', docs, top_k=3)
    for d in result:
        print(d['id'], round(float(d.get('rerank_score', 0)), 4))
    assert len(result) == 3
    assert all('rerank_score' in d for d in result)
    assert all(isinstance(d['rerank_score'], float) for d in result)
    scores = [d['rerank_score'] for d in result]
    assert scores == sorted(scores, reverse=True), "应降序"
    # 原字段保留
    assert result[0]['content'] is not None and 'id' in result[0]
    print("PASS\n")

    print("=== Test 3: 边界情况 ===")
    empty = await rr.rerank('q', [])
    assert empty == [], "空 documents 应返回 []"
    print("PASS: 空 documents -> []")

    single = await rr.rerank('q', [{'id': 9, 'content': '只有一篇文档'}])
    assert len(single) == 1 and 'rerank_score' in single[0]
    print("PASS: 单文档带 rerank_score")

    many = await rr.rerank('q', docs, top_k=99)
    assert len(many) == len(docs), "top_k 大于文档数应返回全部"
    print("PASS: top_k 大于文档数 -> 全部")

    missing_content = await rr.rerank('q', [{'id': 7}, {'id': 8, 'content': '有内容'}])
    assert len(missing_content) == 2, "缺 content 不抛异常"
    print("PASS: 缺 content 字段不抛异常")
    print("PASS\n")

    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
