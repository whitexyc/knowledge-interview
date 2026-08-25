"""测试 Qwen 和 ZhipuAI GLM 模型是否可用（通过 ModelScope API）"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("PW_MODELSCOPE_API_KEY", "")
if not TOKEN:
    print("❌ PW_MODELSCOPE_API_KEY 未配置，请在 .env 中设置后重试")
    sys.exit(1)

BASE_URL = os.getenv("PW_MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")

from langchain_openai import ChatOpenAI

MODELS = [
    ("Qwen", os.getenv("PW_QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B")),
    ("ZhipuAI GLM", os.getenv("PW_ZHIPU_MODEL", "ZhipuAI/GLM-5.2")),
]

TEST_PROMPT = "用一句话介绍你自己"


async def test_model(label: str, model_id: str) -> bool:
    print(f"\n{'='*60}")
    print(f"测试 {label}: {model_id}")
    print(f"{'='*60}")

    llm = ChatOpenAI(
        model=model_id,
        api_key=TOKEN,
        base_url=BASE_URL,
        temperature=0.1,
        timeout=30,
    )

    # 1. 同步生成
    try:
        print("  [generate] 调用中...")
        response = await llm.ainvoke(TEST_PROMPT)
        print(f"  ✅ generate 成功: {response.content[:100]}...")
    except Exception as e:
        print(f"  ❌ generate 失败: {e}")
        return False

    # 2. 流式生成
    try:
        print("  [stream] 调用中...")
        chunks = []
        async for chunk in llm.astream(TEST_PROMPT):
            if chunk.content:
                chunks.append(chunk.content)
        print(f"  ✅ stream 成功 ({len(chunks)} chunks): {''.join(chunks)[:100]}...")
    except Exception as e:
        print(f"  ❌ stream 失败: {e}")
        return False

    return True


async def main():
    print(f"API Base: {BASE_URL}")
    print(f"Token: {TOKEN[:10]}...{TOKEN[-4:]}")

    results = {}
    for label, model_id in MODELS:
        results[label] = await test_model(label, model_id)

    print(f"\n{'='*60}")
    print("结果汇总:")
    for label, ok in results.items():
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {label}: {status}")

    all_ok = all(results.values())
    if all_ok:
        print("\n🎉 全部通过，降级链可用")
    else:
        print("\n⚠️  部分模型不可用，检查错误信息")


if __name__ == "__main__":
    asyncio.run(main())
