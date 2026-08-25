"""
Rerank 重排服务 — 本地 Cross-Encoder 精排
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置：
  HybridRetriever (Top N) → [Reranker] Cross-Encoder 逐对评分 → Top K

为什么使用 bge-reranker-v2-m3（module-030）？
  module-018 曾切换为 Qwen3-Reranker-0.6B（生成式模型），但 CPU 每对约 6s
  （自回归生成慢），top-20 重排需 120s，真实链路被阻塞。
  现改用 BAAI/bge-reranker-v2-m3（分类式 CrossEncoder，实测约 515ms/对，
  快约 12 倍）：
  - sentence-transformers CrossEncoder 原生支持，predict 传 (query, doc)
    裸 pair 即可，无需 chat template 适配
  - 分类式打分（sigmoid），分数接近 1.0 时排序仍正确，区分度低是已知特性，
    校准留待后续（不阻塞）
  - 权衡：首次加载 2.17GB 入内存较慢（预热后复用实例）

缺权重策略（决策，module-018 保留）：
  本地模型目录缺少权重文件时**明确报错**（抛 RerankerException），
  不回退 HuggingFace 在线加载。让问题可见而非静默降级。
"""
import asyncio
import logging
import os
import threading
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from src.config import settings

logger = logging.getLogger(__name__)

# 本地模型路径（必须完整：含 model.safetensors / pytorch_model.bin）
# module-050 目录细分后本文件位于 rag/retrieval/ 下，需三级 dirname 才回到
# ai_service/ 根（对齐 embeddings.py:27-32 同款修法）；二级 dirname 会落在
# rag/ 下解析出 rag/models/... 导致模型缺失——module-053 曾致向量通道全断，
# 本模块（module-054）修复重排通道同款回归。
_LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "bge-reranker-v2-m3",
)
# 权重文件名候选（safetensors 优先，兼容 pytorch bin）
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_DEFAULT_MODEL = _LOCAL_MODEL_DIR
# 重排内容截断阈值（性能修复）：CrossEncoder 按 batch 内最长序列填充，
# 知识库父块可达数万字符（无 ## 标题的文档整篇入库），fp32 CPU 下近满长
# 上下文使单次 rerank 从 ~0.5s 飙到 ~200s（实测 2 pair 201s）。重排只需
# 判断相关度，截断阈值越小整体越快，代价是丢失文档中后段的匹配信号
# （对已检索候选的重排影响小）。
# 选数依据（2026-08-09 九档拐点扫描，6 pair，本地 bge-reranker-v2-m3，
# 见 eval/benchmark_rerank.py --sweep 与 ADR-0004）：
#   2000 字符: 45.4s（7.57s/pair）  相关分数 0.977~0.999
#   1000 字符: 19.7s（3.28s/pair）  相关分数 0.983~1.000
#   500 字符:   8.9s（1.48s/pair）  相关分数 0.867~0.999
#   250 字符:   5.0s（0.83s/pair）  相关分数 0.724~0.999  （历史生产值）
#   200 字符:   4.2s（0.70s/pair）  相关分数 0.700~0.999  ← 采纳
#   150 字符:   3.5s（0.58s/pair）  相关分数 0.388~0.999  ← 分数拐点（弱相关跌破 0.4）
#   100 字符:   2.6s（0.44s/pair）  相关分数 0.103~0.998
#    75 字符:   2.3s（0.38s/pair）  相关分数 0.079~0.992  （主文档开始掉至 0.858）
#    50 字符:   2.0s（0.34s/pair）  相关分数 0.001~0.992  （主文档崩溃至 0.157）
# 结论：250→200 分数/排序均稳定（差 ≤0.003、6/6 一致），拐点在 150。
# 2026-08-14 五主题×4 文档（强/中/弱相关+干扰）精细扫描（240→150 步长 10）：
#   250→170 top-2 排序与 250 全一致；160 出现漂移（次/弱相关互换）；中相关
#   文档（信号在后段）170 以下分数崩塌（AQS doc-b 0.93→0.47@170→0.26@150），
#   干扰文档恒 0.000。采纳 200：省时 18%（0.83→0.70s/pair）且排序/中相关分数
#   双稳，是"分数语义 + 排序稳定"的安全下限；170 省时 30% 但中相关已现崩塌。
_MAX_PAIR_CHARS = 200


class RerankerException(Exception):
    """重排异常"""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.__cause__ = cause


class Reranker:
    """重排器抽象基类"""

    async def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """对检索结果重排，返回按相关性降序的 top_k 条"""
        ...


class CrossEncoderReranker(Reranker):
    """本地 CrossEncoder 重排器

    使用 bge-reranker-v2-m3（本地，分类式 CrossEncoder）逐对计算
    (query, doc_content) 的相关性分数，返回按相关性降序的结果。

    CrossEncoder 比 Bi-Encoder（向量检索）精度更高，因为它让 query 和 doc
    做完整的交叉注意力计算。但速度慢，所以只对 Top N 做精排。

    缺权重策略：本地目录必须包含权重文件（model.safetensors / pytorch_model.bin），
    缺失时抛 RerankerException 明确报错，不回退 HuggingFace 在线加载。
    """

    def __init__(self, model_name: str = ""):
        self._model_name = model_name or _DEFAULT_MODEL
        self._model: Optional[CrossEncoder] = None
        # 单 CrossEncoder 实例访问串行化：to_thread 在真线程执行，模型推理非线程安全
        self._lock = threading.Lock()

    def _validate_model_dir(self):
        """校验本地模型目录完整性

        要求：
        1. 目录存在（否则提示下载）
        2. 包含权重文件（model.safetensors 或 pytorch_model.bin）

        缺任一条件即抛 RerankerException，明确报错而非静默降级。
        """
        if not os.path.isdir(self._model_name):
            raise RerankerException(
                f"重排模型目录不存在: {self._model_name}，请先下载 bge-reranker-v2-m3"
            )
        missing = [f for f in _WEIGHT_FILES if not os.path.isfile(os.path.join(self._model_name, f))]
        if len(missing) == len(_WEIGHT_FILES):
            raise RerankerException(
                f"重排模型缺少权重文件: {self._model_name}（需包含 {_WEIGHT_FILES[0]} 或 {_WEIGHT_FILES[1]}）"
            )

    def _lazy_load(self):
        if self._model is None:
            # 缺权重校验：目录不存在或缺权重文件时明确报错
            self._validate_model_dir()
            logger.info("加载 reranker 模型: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            # P0 性能优化：int8 动态量化（本机 CPU 无 GPU，fp16 不加速）
            if settings.reranker_quantize_enabled:
                self._apply_quantization()
            logger.info("reranker 模型就绪")

    def _apply_quantization(self):
        """对 CrossEncoder 应用 int8 动态量化（P0，CPU 推理提速）

        torch.quantization.quantize_dynamic 递归量化所有 Linear 层（注意力/
        FFN/分类头）。实测 6 pair/250 字符 0.89s/pair → 0.42s/pair（约 2x）。
        弱相关文档分数有漂移（如 0.05→0.17）但相关文档排序保持；若后续发现
        排序被量化破坏，可 PW_RERANKER_QUANTIZE_ENABLED=false 回退 fp32。
        量化失败 fail-open 回退原始模型（不阻断加载）；非 torch.nn.Module
        （如测试 mock）直接跳过。
        """
        if not isinstance(self._model, torch.nn.Module):
            logger.debug("跳过量化：模型非 torch.nn.Module（如测试 mock）")
            return
        try:
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            logger.info("reranker 模型已应用 int8 动态量化")
        except Exception as e:
            logger.warning("reranker 量化失败，回退原始模型: %s", e)

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """同步执行重排打分（由 to_thread 调用）

        与 embeddings.py 的 module-027 模式一致：lazy_load + predict 均为同步
        CPU 密集调用，直接放在 async 函数里会阻塞事件循环；锁保证单实例访问
        完全串行（模型推理非线程安全）。
        """
        with self._lock:
            self._lazy_load()
            return self._model.predict(pairs)

    def _coarse_filter(self, documents: list[dict], top_k: int) -> list[dict]:
        """粗筛候选：超过上限时按现有融合分截断（P1 性能优化）

        CrossEncoder 逐对打分 0.4-0.9s/对，候选越多越慢（检索 10 候选全打分
        是 TTFT 最大头）。检索层已按融合分（hybrid_score / rrf_score / score）
        降序返回，此处再按 settings.rerank_max_candidates 截断——只保留相关
        性最高的 N 个候选进精排，淘汰明显低相关。上限不低于 top_k（保证返回
        足够结果）。粗筛依据是检索融合分而非 CrossEncoder 精排分，理论上可能
        丢掉"融合分低但精排分高"的候选——采纳前实测 Hit@5 不降（见性能优化
        记录），若 6 候选掉命中则保守上调。
        """
        cap = max(settings.rerank_max_candidates, top_k)
        if len(documents) <= cap:
            return documents
        ranked = sorted(
            documents,
            key=lambda d: d.get("hybrid_score", d.get("rrf_score", d.get("score", 0.0))),
            reverse=True,
        )
        return ranked[:cap]

    async def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """执行 CrossEncoder 重排

        将 query 与每个文档的 content 拼为 (query, doc) 裸 pair，
        用 CrossEncoder 模型逐对打分，按分数降序返回 top_k 条。

        bge-reranker-v2-m3 是分类式标准 CrossEncoder，predict 直接接受
        (query, doc) 裸 pair（不同于 Qwen3 生成式模型需要的 chat message 适配，
        已移除）。
        """
        if not documents:
            return []

        # P1 性能优化：粗筛候选——候选超过上限时按现有融合分截断，
        # 只对相关性最高的 N 个进 CrossEncoder 精排（粗筛后精排，降交叉对数）。
        documents = self._coarse_filter(documents, top_k)

        try:
            # 截断超长文档内容（性能修复）：CrossEncoder 按 batch 最长序列填充，
            # 超长父块（数千~数万字符）会把单次 rerank 拖到 ~200s。重排只需
            # 判断相关度，截断到 _MAX_PAIR_CHARS=200（module-044 九档实测 250→200
            # 同档安全；2026-08-14 20 文档精细扫描确认 200 为安全下限，见头部注释）。
            pairs = [
                (query, (d.get("content") or "")[:_MAX_PAIR_CHARS])
                for d in documents
            ]

            # 批量预测相关性分数：CPU 密集推理挪到线程池（to_thread），
            # 避免阻塞事件循环导致 rerank 期间整个服务无响应
            scores = await asyncio.to_thread(self._predict_sync, pairs)

            # 将分数附加到文档上，按分数降序排列
            ranked = []
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
                ranked.append(doc)

            ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            logger.info("Rerank 完成: %d → %d", len(documents), len(ranked[:top_k]))
            return ranked[:top_k]

        except Exception as e:
            logger.error("Rerank 失败: %s", e)
            raise RerankerException("重排服务暂时不可用", cause=e)


# 全局单例
reranker = CrossEncoderReranker()
