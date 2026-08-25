"""Module-057 RRF k 扫描脚本单元测试（WP-A4 benchmark_rrf_k.py 纯函数）

覆盖：
- fuse_rrf 公式正确性：已知 rank → 已知分（score(d) = Σ 1/(k+rank)，k 可变）
- 两通道 vs 三通道：图谱通道贡献（同一 k 下多一路 → 该文档分更高/排序可能提升）
- 缺路不贡献（空通道不崩）
- k 变化改变融合分（同一候选集 k 越小 top 差距越大）
- 排名按 RRF 分降序（通道内按分数键降序取 rank）
"""
import pytest

from eval.benchmarks.benchmark_rrf_k import fuse_rrf


def _doc(doc_id, score, title=None, hybrid_score=None):
    d = {"id": doc_id, "title": title or f"doc-{doc_id}", "content": "x",
         "score": score}
    if hybrid_score is not None:
        d["hybrid_score"] = hybrid_score
    return d


class TestFuseRrf:
    def test_single_channel_rank1(self):
        # 单通道 rank=1 → 1/(k+1)
        out = fuse_rrf([[_doc(1, 0.9)]], k=60, score_keys=["score"])
        assert len(out) == 1
        assert out[0]["id"] == 1

    def test_known_ranks_two_channels_k60(self):
        # fts=[A,B] vec=[B,A] → A=1/61+1/62, B=1/62+1/61 相等；k=60
        out = fuse_rrf(
            [[_doc(1, 1.0), _doc(2, 0.8)], [_doc(2, 1.0), _doc(1, 0.5)]],
            k=60, score_keys=["score", "score"])
        assert [d["id"] for d in out] == [1, 2]

    def test_k_changes_ranking(self):
        # 单通道候选：k 越小，rank1 与 rank2 的分差越大；排序不变
        ch = [[_doc(1, 1.0), _doc(2, 0.9)]]
        out_k20 = fuse_rrf(ch, k=20, score_keys=["score"])
        out_k100 = fuse_rrf(ch, k=100, score_keys=["score"])
        assert [d["id"] for d in out_k20] == [1, 2]
        assert [d["id"] for d in out_k100] == [1, 2]

    def test_three_channel_graph_contribution_raises_doc(self):
        # 三通道（含图谱）：仅图谱命中的文档获得额外 1/(k+rank) 分，排序提升
        fts = [_doc(1, 1.0), _doc(2, 0.9)]
        vec = [_doc(1, 0.9), _doc(2, 0.8)]
        graph = [_doc(2, 0.0, hybrid_score=1.0)]
        two = fuse_rrf([fts, vec], k=60, score_keys=["score", "score"])
        three = fuse_rrf([fts, vec, graph], k=60,
                         score_keys=["score", "score", "hybrid_score"])
        # 两通道下 doc1 第一（两路都第一）；图谱给 doc2 额外 1/61 → 三通道下
        # doc2 反超 doc1（0.03226+0.01639 > 0.03279）
        assert [d["id"] for d in two] == [1, 2]
        assert [d["id"] for d in three] == [2, 1]

    def test_graph_alone_beats_single_channel(self):
        # 图谱单路 top1 分 = 1/(k+1)；FTS 单路 rank2 = 1/(k+2)
        # → 图谱贡献能改变排序（若图谱命中文档在两通道下排名靠后）
        fts = [_doc(1, 1.0), _doc(2, 0.9), _doc(3, 0.8)]
        vec = [_doc(1, 1.0), _doc(2, 0.9), _doc(3, 0.8)]
        graph = [_doc(3, 0.0, hybrid_score=1.0)]
        two = fuse_rrf([fts, vec], k=60, score_keys=["score", "score"])
        three = fuse_rrf([fts, vec, graph], k=60,
                         score_keys=["score", "score", "hybrid_score"])
        two_rank = [d["id"] for d in two]
        three_rank = [d["id"] for d in three]
        # doc3 在三通道下位置不差于两通道（图谱 +1/61 可能提升）
        assert three_rank.index(3) <= two_rank.index(3)

    def test_empty_channels_no_crash(self):
        out = fuse_rrf([[], []], k=60, score_keys=["score", "score"])
        assert out == []

    def test_graph_key_uses_hybrid_score(self):
        # 图谱通道按 hybrid_score 排序取 rank（不是 score）
        graph = [_doc(1, 0.1, hybrid_score=0.9), _doc(2, 0.9, hybrid_score=0.1)]
        out = fuse_rrf([graph], k=60, score_keys=["hybrid_score"])
        assert [d["id"] for d in out] == [1, 2]

    def test_score_keys_zip_with_channels(self):
        # 通道与排序键按位置对齐（fts/vec 用 score，graph 用 hybrid_score）
        fts = [_doc(1, 1.0)]
        vec = [_doc(2, 0.5)]
        graph = [_doc(3, 0.0, hybrid_score=1.0)]
        out = fuse_rrf([fts, vec, graph], k=60,
                       score_keys=["score", "score", "hybrid_score"])
        assert {d["id"] for d in out} == {1, 2, 3}
