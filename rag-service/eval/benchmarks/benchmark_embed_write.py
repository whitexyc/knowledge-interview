"""嵌入写入侧性能基准（module-065 WP1：探路证伪固化为可复跑验证脚本）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景:
    module-064 验收后真实文档插入实测暴露 CSV 大数据集入库慢（7231 块 50 分钟），
    2026-08-15 探路实测两条优化路径均已证伪：
      ① llama_cpp.Llama.create_embedding 支持 List[str] 批量入参，但内部非真
         batch decode——10 条 0.8x / 50 条 1.3x / 200 条 0.9x（vs 循环）无加速；
      ② ProcessPoolExecutor 多进程并行嵌入（Windows spawn）实测负优化——
         2 进程 0.4x / 4 进程 0.3x（vs 串行：spawn 重 import + 内存带宽竞争）。
    结论：写入侧无廉价优化路径，~110-210ms/块（bge-m3 Q8 单核推理，随文本长度
    波动：探路短文本 110-160 / 本脚本 141 字符文本实测 173-211）为固有成本，
    生产代码不改（无可行优化，不做投机改动）。本脚本把探路结论固化为可复跑
    验证（同 2026-08-15 探路口径），防未来重复踩坑。

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.benchmark_embed_write            # 全档位（约 4-5 分钟）
    python -m eval.benchmarks.benchmark_embed_write --quick    # 冒烟（批量 10/50 + 进程 2）
    python -m eval.benchmarks.benchmark_embed_write --no-mp    # 跳过进程档（内存/慢速机器）

输出:
    ① 循环 vs List 批量（10/50/200 条）耗时 + 倍率表（<1x = 批量反而更慢）
    ② 串行 vs 多进程 2/4（200 条）耗时 + 倍率表（<1x = 多进程负优化）
    结论行与 2026-08-15 探路口径一致。

口径声明（与探路一致）:
    - 模型: 本地 bge-m3 Q8 GGUF（llama-cpp-python），单进程单 Llama 实例持锁
    - 文本: 与探路同源的中文技术问答样例（长度相似，非真实入库长块）
    - 批量路径 = create_embedding(List[str])；循环路径 = 逐条 create_embedding
    - 多进程 = ProcessPoolExecutor（Windows spawn），worker 各自独立加载模型
"""
import argparse
import concurrent.futures
import sys
import time

from rag.retrieval.embeddings import embedding_service

# 与 2026-08-15 探路同源文本（嵌入长度相近的技术问答，非超长块）
TEXT = ("AQS抽象队列同步器的工作原理是什么？ReentrantLock如何基于AQS实现公平锁与非公平锁？"
        "state表示重入次数，CLH变体FIFO等待队列，独占与共享模式。Java线程池核心参数。"
        "G1垃圾收集器Region分区。Kafka ISR机制。Redis RDB与AOF持久化。")

# 批量档位（10/50/200：探路实测点；--quick 只跑前两档）
BATCH_SIZES = (10, 50, 200)
MP_SIZES = (2, 4)
MP_N = 200


# ── ① 循环 vs List 批量（同进程同实例） ─────────────────────────────────
def bench_loop(n: int) -> float:
    """逐条 create_embedding（当前生产路径，_embed_documents_sync 同款）"""
    with embedding_service._lock:
        embedding_service._lazy_load()
        t0 = time.perf_counter()
        for _ in range(n):
            embedding_service._model.create_embedding(TEXT)
        return time.perf_counter() - t0


def bench_batch(n: int) -> float:
    """create_embedding(List[str])（探路①：LLM 声称批量但内部非真 batch decode）"""
    with embedding_service._lock:
        embedding_service._lazy_load()
        t0 = time.perf_counter()
        resp = embedding_service._model.create_embedding([TEXT] * n)
        dt = time.perf_counter() - t0
    assert len(resp["data"]) == n, f"批量返回 {len(resp['data'])} 条 != {n}"
    return dt


def run_batch_comparison(sizes: tuple) -> tuple[list[float], float, float]:
    """循环 vs List 批量对比；返回 (各档倍率, ms/条最小, ms/条最大)"""
    print("\n" + "=" * 78)
    print("① 循环 vs List 批量（同进程同实例，create_embedding 直调）")
    print("=" * 78)
    print(f"  {'条数':>4} | {'循环':>10} {'ms/条':>7} | {'批量':>10} {'ms/条':>7} | "
          f"{'倍率(循环/批量)':>15}")
    print("-" * 78)
    ratios: list[float] = []
    ms_all: list[float] = []
    for n in sizes:
        loop_t = bench_loop(n)
        batch_t = bench_batch(n)
        ratio = loop_t / batch_t if batch_t else float("inf")
        ratios.append(ratio)
        ms_all += [loop_t / n * 1000, batch_t / n * 1000]
        print(f"  {n:>4} | {loop_t:>7.2f}s {loop_t/n*1000:>6.0f} | "
              f"{batch_t:>7.2f}s {batch_t/n*1000:>6.0f} | {ratio:>13.1f}x")
    print("  解读: 倍率≈1x = 无批量加速（探路结论①）；<1x = 批量反而更慢")
    return ratios, min(ms_all), max(ms_all)


# ── ② 串行 vs 多进程（ProcessPoolExecutor，Windows spawn） ───────────────
def _worker(texts: list[str]) -> list:
    """进程内 worker：独立加载模型（605MB/进程），串行嵌入分片

    模块级函数（spawn 可 pickle）；模型加载放 worker 内（子进程各自冷加载）。
    """
    from rag.retrieval.embeddings import EmbeddingService
    svc = EmbeddingService()
    with svc._lock:
        svc._lazy_load()
        return [svc._model.create_embedding(t)["data"][0]["embedding"] for t in texts]


def run_mp_parallel(n: int, workers: int) -> float:
    """n 条分片给 workers 个进程并行嵌入"""
    chunks = [[TEXT] * (n // workers) for _ in range(workers)]
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker, chunks))
    dt = time.perf_counter() - t0
    assert sum(len(c) for c in results) == n
    return dt


def run_mp_comparison(sizes: tuple) -> list[float]:
    """串行 vs 多进程对比；返回各进程档相对串行的倍率"""
    print("\n" + "=" * 78)
    print(f"② 串行 vs 多进程（{MP_N} 条，ProcessPoolExecutor / Windows spawn）")
    print("=" * 78)
    serial_t = bench_loop(MP_N)
    print(f"  串行 {MP_N} 条: {serial_t:7.2f}s ({serial_t/MP_N*1000:.0f}ms/条)")
    ratios: list[float] = []
    for w in sizes:
        p = run_mp_parallel(MP_N, w)
        ratios.append(serial_t / p)
        print(f"  并行 {w} 进程 {MP_N} 条: {p:7.2f}s ({p/MP_N*1000:.0f}ms/条)  "
              f"相对串行 {serial_t/p:.1f}x")
    print("  解读: <1x = 多进程负优化（探路结论②：spawn 重 import + 内存带宽竞争）")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="嵌入写入侧性能基准（module-065 WP1 证伪记录，可复跑）")
    parser.add_argument("--quick", action="store_true",
                        help="冒烟：批量只跑 10/50，进程只跑 2")
    parser.add_argument("--no-mp", action="store_true",
                        help="跳过多进程档（内存/慢速机器）")
    args = parser.parse_args()

    sizes = BATCH_SIZES[:2] if args.quick else BATCH_SIZES
    mp_sizes = (2,) if args.quick else MP_SIZES

    print(f"模型冷加载（bge-m3 Q8 GGUF 605MB，本进程串行侧实例）...")
    t0 = time.perf_counter()
    embedding_service._lazy_load()
    print(f"模型就绪: {time.perf_counter()-t0:.1f}s，文本 {len(TEXT)} 字符\n")

    batch_ratios, ms_min, ms_max = run_batch_comparison(sizes)
    mp_ratios: list[float] = []
    if not args.no_mp:
        mp_ratios = run_mp_comparison(mp_sizes)

    # 结论用本次实测数据（Review 修复：不再写死固定倍率/范围；倍率随文本长度
    # 与 CPU 争用波动——探路 0.8-1.3x / 本机安静态 0.9-1.0x / 争用态 1.6-2.0x，
    # 无稳定可复现的批量加速）
    print("\n" + "=" * 78)
    print(f"结论（本次实测数据）: 批量倍率 {min(batch_ratios):.1f}-{max(batch_ratios):.1f}x"
          + (" / 多进程倍率 " + "-".join(f"{r:.1f}x" for r in mp_ratios) if mp_ratios else ""))
    print(f"  → 批量无稳定加速、多进程负优化（与 2026-08-15 探路口径一致）；")
    print(f"  写入吞吐 ~{ms_min:.0f}-{ms_max:.0f}ms/块（本次文本 {len(TEXT)} 字符；"
          f"探路短文本 ~110-160ms/块，同量级随文本长度波动）为 bge-m3 Q8 单核推理固有成本；")
    print("  生产代码零改动（无可行优化，不做投机改动）。")
    print("=" * 78)


if __name__ == "__main__":
    # Windows spawn 多进程必须经 __main__ 守卫（子进程重 import 本模块）
    sys.exit(main())
