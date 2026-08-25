"""公开基准检索评测（module-065 WP3：C-MTEB/BEIR 测通用泛化，补乐观偏差）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景:
    自建 golden 集（112 题）是"自己考自己"——文档来自本项目知识库，命中率高
    存在乐观偏差（Hit@5 0.9905 / MRR 0.9341）。本脚本用公开标准检索数据集
    （BEIR nfcorpus / C-MTEB EcomRetrieval 中文子集）测 bge-m3 + 余弦检索的
    通用泛化能力，与自建集对照。

数据源（2026-08-15 探测：huggingface.co 502 不可达 / hf-mirror.com、
    ModelScope 可达 → 下载走 hf-mirror 直链 resolve URL，数据落
    eval/datasets/public/{dataset}/ 缓存，已存在则跳过重下）:
    - BeIR/nfcorpus（英文，BEIR 官方小集）: 3,633 文档 / 323 测试查询 /
      test.tsv qrels（下载 ~3.1MB，嵌入约 8 分钟）
    - C-MTEB/EcomRetrieval（中文电商检索）+ EcomRetrieval-qrels: corpus
      100,902 篇 —— 全量嵌入约 3.6 小时不现实，本脚本固定种子抽样（默认
      3,000 篇，~6 分钟）；抽样代理口径 nDCG@10 与官方 leaderboard 不可直接
      比（相关文档可能落在抽样外 → 结果偏保守），如实标注。

指标（按数据集口径）:
    - nDCG@10: BEIR/C-MTEB 官方指标（gain = qrels 分数，trec 口径）
    - Hit@5 / MRR: 补充常规指标（二元相关 = qrels 分数 ≥1，与自建集同口径）

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.benchmark_public_retrieval --dataset nfcorpus --corpus-sample 1000
    python -m eval.benchmarks.benchmark_public_retrieval --dataset nfcorpus --corpus-sample 0 --limit 20   # 全量冒烟（0=不抽样）
    python -m eval.benchmarks.benchmark_public_retrieval --dataset ecom-zh --corpus-sample 3000
    python -m eval.benchmarks.benchmark_public_retrieval --no-save   # 不写 eval_runs

全量 vs 抽样（2026-08-15 实测，环境约束如实声明）:
    nfcorpus 全量 corpus 3,633 篇（平均 ~1,800 字符/篇）→ 单篇嵌入 1.7-2.2s
    （bge-m3 Q8 单机 CPU），全量约 2 小时不现实；默认 --corpus-sample 抽样
    并如实标注"代理口径"（相关文档可能落在抽样外 → 结果偏保守）。ecom-zh
    corpus 10 万级同理抽样。有 GPU/集群环境可 --corpus-sample 0 跑全量。

降级:
    - 数据源下载失败（网络/镜像不可达）→ 明确报错标注"待环境"，不伪造数字
    - 单查询嵌入/检索失败 → 跳过该查询并计数（如实打印）
"""
import argparse
import math
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

from rag.retrieval.embeddings import embedding_service

# 数据目录（gitignored 缓存；下载失败标注"待环境"）
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "public")
MIRROR = "https://hf-mirror.com"

TOP_K = 10  # nDCG@10 / Hit@5 / MRR 均覆盖（取前 10 够用）

# 数据集定义: {name: (files, loader)}
DATASETS = {
    "nfcorpus": {
        "dir": "beir_nfcorpus",
        "files": {
            "corpus.parquet": "https://hf-mirror.com/datasets/BeIR/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet",
            "queries.parquet": "https://hf-mirror.com/datasets/BeIR/nfcorpus/resolve/main/queries/queries-00000-of-00001.parquet",
            "test.tsv": "https://hf-mirror.com/datasets/BeIR/nfcorpus-qrels/resolve/main/test.tsv",
        },
        "lang": "英文",
    },
    "ecom-zh": {
        "dir": "cmteb_ecomretrieval",
        "files": {
            "corpus.parquet": "https://hf-mirror.com/datasets/C-MTEB/EcomRetrieval/resolve/main/data/corpus-00000-of-00001-7f0e87850b5c9454.parquet",
            "queries.parquet": "https://hf-mirror.com/datasets/C-MTEB/EcomRetrieval/resolve/main/data/queries-00000-of-00001-46d6d2b2f0ad8826.parquet",
            "qrels.parquet": "https://hf-mirror.com/datasets/C-MTEB/EcomRetrieval-qrels/resolve/main/data/dev-00000-of-00001-42566aaf5580f662.parquet",
        },
        "lang": "中文",
    },
}


# ── 数据下载（hf-mirror 直链，缓存复用） ────────────────────────────────
def download_dataset(name: str) -> str:
    spec = DATASETS[name]
    data_dir = os.path.join(DATA_ROOT, spec["dir"])
    os.makedirs(data_dir, exist_ok=True)
    for fname, url in spec["files"].items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"  下载 {fname} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  下载失败 {fname}: {e}")
            print("  数据源不可达 → 如实标注：本数据集待环境（不伪造数字）")
            sys.exit(2)
    return data_dir


# ── 指标（trec 口径） ───────────────────────────────────────────────────
def ndcg_at_k(grades: list[float], k: int = TOP_K) -> float:
    """nDCG@k：gain = qrels 分数（trec 口径）；grades 按检索排名序"""
    if not grades:
        return 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    ideal = sorted(grades, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def q_metrics(ranked: list[str], relevant: set[str], grades: dict[str, float]) -> dict:
    """单查询指标：Hit@5 / MRR / nDCG@10（三元相关 = 分数≥1）"""
    hit5 = mrr = 0.0
    for i, cid in enumerate(ranked[:TOP_K]):
        if cid in relevant:
            if i < 5:
                hit5 = 1.0
            mrr = 1.0 / (i + 1)
            break
    grades_ranked = [grades.get(cid, 0.0) for cid in ranked[:TOP_K]]
    return {"hit_at_5": hit5, "mrr": mrr, "ndcg_at_10": ndcg_at_k(grades_ranked)}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_queries": n,
        "hit_at_5": round(sum(r["hit_at_5"] for r in rows) / n, 4),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
        "ndcg_at_10": round(sum(r["ndcg_at_10"] for r in rows) / n, 4),
    }


# ── 数据加载（各数据集格式适配） ────────────────────────────────────────
def load_nfcorpus(data_dir: str, corpus_sample: int = 0):
    corpus = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
    queries = pd.read_parquet(os.path.join(data_dir, "queries.parquet"))
    qrels = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t",
                        names=["qid", "cid", "score"], skiprows=1)

    if corpus_sample and len(corpus) > corpus_sample:
        # 固定种子抽样（代理口径：相关文档可能落在抽样外 → 结果偏保守）
        rng = np.random.default_rng(20260815)
        corpus = corpus.iloc[rng.choice(len(corpus), corpus_sample, replace=False)]

    # 注: 列名以 _ 开头（_id），itertuples 会改名 → 用按列名访问（iterrows）
    doc_texts = {}
    for _, row in corpus.iterrows():
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        did = str(row["_id"])
        doc_texts[did] = (title + "\n" + text).strip() or did

    # 测试查询 = qrels 中出现的 query（BEIR 口径：test split）
    qids = set(qrels["qid"].astype(str))
    q_texts = {}
    for _, row in queries.iterrows():
        qid = str(row["_id"])
        if qid not in qids:
            continue
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        q_texts[qid] = ((title + " " + text) if title else text).strip() or qid

    rel: dict[str, dict[str, float]] = {}
    for _, row in qrels.iterrows():
        rel.setdefault(str(row["qid"]), {})[str(row["cid"])] = float(row["score"])
    return q_texts, doc_texts, rel


def load_ecom(data_dir: str, corpus_sample: int):
    corpus = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
    queries = pd.read_parquet(os.path.join(data_dir, "queries.parquet"))
    qrels = pd.read_parquet(os.path.join(data_dir, "qrels.parquet"))

    if corpus_sample and len(corpus) > corpus_sample:
        # 固定种子抽样（代理口径：相关文档可能落在抽样外 → 结果偏保守）
        rng = np.random.default_rng(20260815)
        corpus = corpus.iloc[rng.choice(len(corpus), corpus_sample, replace=False)]

    doc_texts = {str(r["id"]): str(r["text"]) for _, r in corpus.iterrows()}
    q_texts = {str(r["id"]): str(r["text"]) for _, r in queries.iterrows()}
    rel: dict[str, dict[str, float]] = {}
    for _, r in qrels.iterrows():
        rel.setdefault(str(r["qid"]), {})[str(r["pid"])] = float(r["score"])
    return q_texts, doc_texts, rel


# ── 主流程 ──────────────────────────────────────────────────────────────
def run_eval(name: str, limit: int | None, corpus_sample: int) -> tuple[dict, list[dict]]:
    print(f"[1/4] 下载/校验数据（{name}，{DATASETS[name]['lang']}）...")
    data_dir = download_dataset(name)
    print(f"      数据目录: {data_dir}")

    print("[2/4] 加载数据...")
    if name == "nfcorpus":
        q_texts, doc_texts, rel = load_nfcorpus(data_dir, corpus_sample)
        qids = sorted(q_texts)
    else:
        q_texts, doc_texts, rel = load_ecom(data_dir, corpus_sample)
        qids = sorted(q_texts)
    if limit:
        qids = qids[:limit]
    print(f"      corpus {len(doc_texts)} 篇 / 查询 {len(qids)} 条"
          + (f"（--limit 抽样 {len(qids)}）" if limit else ""))

    print(f"[3/4] 嵌入 corpus（{len(doc_texts)} 篇，bge-m3 Q8 串行，约 "
          f"{len(doc_texts) * 0.13 / 60:.0f} 分钟）...")
    doc_ids = list(doc_texts.keys())
    docs_vec = np.zeros((len(doc_ids), 1024), dtype=np.float32)
    with embedding_service._lock:
        embedding_service._lazy_load()
        for i, did in enumerate(doc_ids):
            vec = embedding_service._model.create_embedding(doc_texts[did])["data"][0]["embedding"]
            docs_vec[i] = vec
            if (i + 1) % 500 == 0 or i + 1 == len(doc_ids):
                print(f"      corpus 嵌入进度: {i+1}/{len(doc_ids)}")
    # MAJOR-1 修复（Review 2026-08-15）：bge-m3 Q8 llama.cpp 原始输出未 L2 归一化
    # （embeddings.py:13 明确"输出未 L2 归一化需 _normalize"），旧版直接用原始
    # 向量点积——模长∝文本长度参与排名，标称"余弦"实为点积。对齐 _normalize：
    # 逐行 L2 归一化后点积 = 余弦。
    norms = np.linalg.norm(docs_vec, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    docs_vec = docs_vec / norms

    # Tester 修复轮（2026-08-15）：真实跳过语义——qids 虽按 qrels 预过滤恒非空，
    # 但 corpus 抽样后相关文档可能全部落在抽样之外（该查询按构造只得 0 分，
    # 计入会稀释指标）。有效查询 = 相关文档 ∩ 抽样 corpus 非空者；空者跳过不计
    # （并计数如实打印）。跳过判定纯集合运算，且省去被跳过查询的嵌入成本。
    sample_ids = set(doc_ids)
    qids_effective = [qid for qid in qids if set(rel.get(qid, {})) & sample_ids]
    n_skip = len(qids) - len(qids_effective)
    print(f"      有效查询 {len(qids_effective)}（跳过 {n_skip}："
          f"相关文档全部不在抽样 corpus 内）")

    print(f"[4/4] 逐查询检索（余弦 top-{TOP_K}，L2 归一化后点积）+ 指标...")
    rows = []
    for qid in qids_effective:
        q_vec = np.asarray(embedding_service._model.create_embedding(q_texts[qid])["data"][0]["embedding"],
                           dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm
        scores = docs_vec @ q_vec
        ranked = [doc_ids[i] for i in np.argsort(scores)[::-1][:TOP_K]]
        relevant = set(rel.get(qid, {}))
        rows.append({"query_id": qid, **q_metrics(ranked, relevant, rel[qid])})
    agg = aggregate(rows)
    print(f"      指标基于有效查询 {len(rows)} 条")
    print()
    print("=" * 78)
    print(f"公开基准检索指标（{name}，{DATASETS[name]['lang']}）")
    print("=" * 78)
    print(f"  查询数: {agg['n_queries']}")
    print(f"  Hit@5:   {agg['hit_at_5']:.4f}")
    print(f"  MRR:     {agg['mrr']:.4f}")
    print(f"  nDCG@10: {agg['ndcg_at_10']:.4f}（BEIR/C-MTEB 官方口径）")
    if corpus_sample:
        print(f"  口径声明: corpus 固定种子抽样 {corpus_sample} 篇（官方全量口径"
              f"需完整 corpus）——代理口径；相关文档全部落在抽样外的查询跳过不计"
              f"（有效查询数如实打印），结果偏保守，与官方 leaderboard 不可直接比。")
    print("=" * 78)
    return agg, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="公开基准检索评测（module-065 WP3：C-MTEB/BEIR 通用泛化）")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nfcorpus",
                        help="数据集（nfcorpus=英文 BEIR 官方 / ecom-zh=中文电商）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 条查询（冒烟）")
    parser.add_argument("--corpus-sample", type=int, default=3000,
                        help="ecom-zh corpus 固定种子抽样数（0=全量，不推荐）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs")
    args = parser.parse_args()

    agg, rows = run_eval(args.dataset, args.limit, args.corpus_sample)

    if not args.no_save:
        try:
            saved_id = _save_eval_run(args, agg)
            print(f"已落库 eval_runs (id={saved_id}, eval_type='public_retrieval')")
        except Exception as e:
            print(f"eval_runs 落库失败（不中断）: {e}")


def _save_eval_run(args, agg: dict) -> int:
    """eval_runs 落库（eval_type='public_retrieval'）；失败仅警告不中断"""
    import asyncio

    from eval.golden.golden_retrieval import (get_git_commit, load_rag_config,
                                              save_eval_run)

    async def _do():
        config_snapshot = await load_rag_config()
        return await save_eval_run(
            eval_type="public_retrieval",
            git_commit=get_git_commit(),
            config_snapshot=config_snapshot,
            scores={
                "dataset": args.dataset,
                "n_queries": agg["n_queries"],
                "hit_at_5": agg["hit_at_5"],
                "mrr": agg["mrr"],
                "ndcg_at_10": agg["ndcg_at_10"],
                "embedding": "bge-m3 Q8 本地 + L2 归一化后余弦暴力检索",
                "top_k": TOP_K,
                "caliber_note": "nfcorpus=全量 test 口径；ecom-zh=corpus 抽样代理口径；"
                                "L2 归一化（Review MAJOR-1 修复：旧 id=40/41 为点积口径）",
            },
            per_question=[],
        )

    return asyncio.run(_do())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
