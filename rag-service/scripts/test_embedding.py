"""测试 ModelScope 云端嵌入模型是否可用"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from rag.retrieval.embeddings import embedding_service


async def main():
    # 1. 单条
    vec = await embedding_service.embed_text("你好，世界")
    print(f"embed_text 成功, dim={len(vec)}, 前5维={vec[:5]}")

    # 2. 批量
    embs = await embedding_service.embed_documents(["今天天气很好", "RAG 检索测试", "Java 线程池"])
    print(f"embed_documents 成功, count={len(embs)}, dim={len(embs[0])}")

    # 3. 验证归一化
    norm = sum(v * v for v in vec) ** 0.5
    print(f"向量 L2 范数={norm:.4f} (≈1 表示已归一化)")


if __name__ == "__main__":
    asyncio.run(main())
