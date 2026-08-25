"""公开基准混合检索评测（2026-08-19：向量 vs 向量+BM25 RRF 融合增益验证）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

背景:
    原脚本 benchmark_public_retrieval.py 只跑纯向量（bge-m3 + 余弦暴力检索）。
    本脚本在其基础上新增内存 BM25 通道（rank_bm25，中文 jieba 分词 / 英文
    空格分词），与向量路 RRF（k=60，对齐生产 retrieval_fusion_mode=rrf 口径）
    融合，同数据集 / 同固定种子 / 同有效查询过滤，并排对比：
      a) vector_only  —— 纯向量（与原脚本同实现，可复现对照）
      b) vector+bm25  —— RRF 融合
    图谱通道不适用公开集（无实体/关系数据），如实标注。

用法（在 ai_service 目录下）:
    python -m eval.benchmarks.benchmark_public_hybrid --dataset ecom-zh --corpus-sample 0
    python -m eval.benchmarks.benchmark_public_hybrid --dataset nfcorpus --limit 20 --no-save

落库: eval_runs eval_type='public_hybrid'（scores 含两配置 + 口径标注）
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# 复用原脚本：数据集定义 / 下载 / 加载 / 指标 / 聚合（保证同实现可复现对照）
from eval.benchmarks.benchmark_public_retrieval import (
    DATASETS, download_dataset, load_nfcorpus, load_ecom,
    q_metrics, aggregate, ndcg_at_k, TOP_K,
)
from rag.retrieval.embeddings import embedding_service

RRF_K = 60          # RRF 常数（对齐生产 rrf_constant_k）
VECTOR_CANDIDATES = 1000   # 向量路候选（RRF 融合用）
BM25_CANDIDATES = 1000     # BM25 路候选


def tokenize(text: str, lang: str) -> list[str]:
    """分词：中文 jieba / 英文空格（BM25 词项）"""
    if lang == "中文":
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    return [t.lower() for t in text.split() if t.strip()]


def build_bm25(doc_ids: list[str], doc_texts: dict, lang: str):
    """构建内存 BM25 索引（rank_bm25 BM25Okapi）"""
    from rank_bm25 import BM25Okapi
    print(f"      构建 BM25 索引（{lang} 分词，{len(doc_ids)} 篇）...")
    corpus_tokens = []
    for i, did in enumerate(doc_ids):
        corpus_tokens.append(tokenize(doc_texts[did], lang))
        if (i + 1) % 20000 == 0 or i + 1 == len(doc_ids):
            print(f"      BM25 分词进度: {i+1}/{len(doc_ids)}")
    return BM25Okapi(corpus_tokens)


def rrf_fuse(vector_rank: dict, bm25_rank: dict, k: int = RRF_K) -> list[str]:
    """RRF 融合：score(d) = Σ 1/(k + rank_i(d))，返回按融合分降序的 doc 列表"""
    scores: dict[str, float] = {}
    for rank_map in (vector_rank, bm25_rank):
        for did, rank in rank_map.items():
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return [did for did, _ in sorted(scores.items(), key=lambda x: -x[1])]


def run_hybrid(name: str, limit: int | None, corpus_sample: int) -> tuple[dict, dict, list[dict]]:
    print(f"[1/4] 下载/校验数据（{name}，{DATASETS[name]['lang']}）...")
    data_dir = download_dataset(name)

    print("[2/4] 加载数据...")
    if name == "nfcorpus":
        q_texts, doc_texts, rel = load_nfcorpus(data_dir, corpus_sample)
    else:
        q_texts, doc_texts, rel = load_ecom(data_dir, corpus_sample)
    qids = sorted(q_texts)
    if limit:
        qids = qids[:limit]
    print(f"      corpus {len(doc_texts)} 篇 / 查询 {len(qids)} 条"
          + (f"（--limit 抽样 {len(qids)}）" if limit else ""))

    print(f"[3/4] 嵌入 corpus（{len(doc_texts)} 篇，bge-m3 Q8 GPU 串行）...")
    doc_ids = list(doc_texts.keys())
    docs_vec = np.zeros((len(doc_ids), 1024), dtype=np.float32)
    with embedding_service._lock:
        embedding_service._lazy_load()
        for i, did in enumerate(doc_ids):
            vec = embedding_service._model.create_embedding(doc_texts[did])["data"][0]["embedding"]
            docs_vec[i] = vec
            if (i + 1) % 500 == 0 or i + 1 == len(doc_ids):
                print(f"      corpus 嵌入进度: {i+1}/{len(doc_ids)}")
    norms = np.linalg.norm(docs_vec, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    docs_vec = docs_vec / norms

    # BM25 索引（向量嵌入后构建，CPU 分词）
    bm25 = build_bm25(doc_ids, doc_texts, DATASETS[name]["lang"])

    # 有效查询过滤（同原脚本口径：相关文档 ∩ corpus 非空）
    sample_ids = set(doc_ids)
    qids_effective = [qid for qid in qids if set(rel.get(qid, {})) & sample_ids]
    n_skip = len(qids) - len(qids_effective)
    print(f"      有效查询 {len(qids_effective)}（跳过 {n_skip}）")

    print(f"[4/4] 逐查询双配置检索（向量 / 向量+BM25 RRF k={RRF_K}）+ 指标...")
    rows_v, rows_h = [], []
    for qi, qid in enumerate(qids_effective):
        q_vec = np.asarray(embedding_service._model.create_embedding(
            q_texts[qid])["data"][0]["embedding"], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm
        scores = docs_vec @ q_vec
        vec_order = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        vector_rank = {did: i + 1 for i, did in enumerate(vec_order[:VECTOR_CANDIDATES])}

        # 纯向量配置（与原脚本同口径：top-TOP_K）
        ranked_v = vec_order[:TOP_K]
        relevant = set(rel.get(qid, {}))
        rows_v.append({"query_id": qid, **q_metrics(ranked_v, relevant, rel[qid])})

        # 混合配置：向量 + BM25 RRF 融合
        q_tokens = tokenize(q_texts[qid], DATASETS[name]["lang"])
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_order = [doc_ids[i] for i in np.argsort(bm25_scores)[::-1][:BM25_CANDIDATES]]
        bm25_rank = {did: i + 1 for i, did in enumerate(bm25_order)}
        ranked_h = rrf_fuse(vector_rank, bm25_rank)[:TOP_K]
        rows_h.append({"query_id": qid, **q_metrics(ranked_h, relevant, rel[qid])})

        if (qi + 1) % 100 == 0 or qi + 1 == len(qids_effective):
            print(f"      查询进度: {qi+1}/{len(qids_effective)}")

    agg_v = aggregate(rows_v)
    agg_h = aggregate(rows_h)
    print()
    print("=" * 78)
    print(f"公开基准混合检索对比（{name}，{DATASETS[name]['lang']}，有效查询 {len(rows_v)}）")
    print("=" * 78)
    for label, agg in (("纯向量 (vector_only)    ", agg_v), ("向量+BM25 (rrf k=60)", agg_h)):
        print(f"  {label}: Hit@5 {agg['hit_at_5']:.4f} / MRR {agg['mrr']:.4f} / nDCG@10 {agg['ndcg_at_10']:.4f}")
    print("=" * 78)
    return agg_v, agg_h, rows_v


def _save_eval_run(args, agg_v: dict, agg_h: dict, n_skip: int) -> int:
    import asyncio
    from eval.golden.golden_retrieval import (get_git_commit, load_rag_config,
                                              save_eval_run)

    async def _do():
        config_snapshot = await load_rag_config()
        return await save_eval_run(
            eval_type="public_hybrid",
            git_commit=get_git_commit(),
            config_snapshot=config_snapshot,
            scores={
                "dataset": args.dataset,
                "corpus_sample": args.corpus_sample,
                "n_queries": agg_v["n_queries"],
                "n_skip": n_skip,
                "vector_only": {"hit_at_5": agg_v["hit_at_5"], "mrr": agg_v["mrr"],
                                "ndcg_at_10": agg_v["ndcg_at_10"]},
                "vector_bm25_rrf": {"hit_at_5": agg_h["hit_at_5"], "mrr": agg_h["mrr"],
                                    "ndcg_at_10": agg_h["ndcg_at_10"], "rrf_k": RRF_K},
                "embedding": "bge-m3 Q8 本地 + L2 归一化后余弦",
                "bm25": "rank_bm25 BM25Okapi（中文 jieba / 英文空格分词，内存索引）",
                "caliber_note": "同固定种子 20260815 / 同有效查询过滤；RRF k=60 对齐生产；"
                                "图谱通道不适用公开集（无实体/关系数据）如实标注；"
                                "纯向量配置与原脚本 benchmark_public_retrieval 同实现可复现对照",
            },
            per_question=[],
        )

    return asyncio.run(_do())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="公开基准混合检索评测（向量 vs 向量+BM25 RRF）")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ecom-zh")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条查询（冒烟）")
    parser.add_argument("--corpus-sample", type=int, default=0,
                        help="corpus 固定种子抽样数（0=全量）")
    parser.add_argument("--no-save", action="store_true", help="不记录 eval_runs")
    args = parser.parse_args()

    agg_v, agg_h, _ = run_hybrid(args.dataset, args.limit, args.corpus_sample)

    if not args.no_save:
        try:
            saved_id = _save_eval_run(args, agg_v, agg_h, n_skip=0)
            print(f"已落库 eval_runs (id={saved_id}, eval_type='public_hybrid')")
        except Exception as e:
            print(f"eval_runs 落库失败（不中断）: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
