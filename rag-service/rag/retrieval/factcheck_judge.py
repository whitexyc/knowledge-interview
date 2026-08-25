"""HHEM 幻觉检测裁判（ADR-0010 P0-②，module-051）

职责：LLM 拆好 claims 后，用 HHEM-2.1-Open 批量判定每条 claim 与各文档的一致性分数。

设计（对齐 embeddings.py 模式）：
    - 延迟加载：首次 predict 时才加载模型（模块导入零开销，模型缺失不影响服务启动）
    - threading.Lock：asyncio.to_thread 在真线程执行，asyncio.Lock 无法跨线程互斥
      （module-027 嵌入并发修复同款经验）
    - CPU 推理经 asyncio.to_thread 不阻塞事件循环

降级契约（WP1 AC）：模型缺失/加载失败/推理异常/20s 超时 → predict 返回
None（不抛异常），由 verify_answer 上层回退 LLM 判分。

加载复用：rag/retrieval/hhem_loader.load_hhem_model（module-050 已验证路径，单一来源）。
"""
import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# 本地 HHEM 模型目录（gitignored 环境文件，module-050 下载）
_HHEM_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "hhem-2.1-open",
)

# 推理超时（秒）：对齐 LLM 拆句超时哲学（module-051 minor#1 修复，15s；
# module-055 提至 20s）——交叉打分随 claims×docs 增长（reflector 已限对数
# 上限：每 claim ≤2 文档、≤8 claims，典型 10 对），HHEM 推理 hang 时不无限
# 阻塞（超时降级 LLM 判分）。提预算依据（module-055 changelog 实测）：15 对
# 冷启动（含 438MB 模型加载）≈9s、E2E 服务负载下 12s+ 贴近旧 15s 上限致
# 级联超时 verified_claims=0；对数上限后冷启动 ≈6s，20s = 3 倍余量。
_PREDICT_TIMEOUT = 20


class HHEMJudge:
    """HHEM-2.1-Open 幻觉检测裁判（批量打分，延迟加载 + 线程安全）

    predict(docs, claims) 返回与 claims 长度无关的交叉打分数组
    （每个 (doc, claim) 对一分数，0-1，class 1 = consistent，
    官方 predict() 内部拼 prompt + softmax）。
    任何失败（模型缺失/加载失败/推理异常/超时）→ 返回 None，由上层降级 LLM 判分。
    """

    def __init__(self, model_dir: str = ""):
        self._model_dir = model_dir or _HHEM_MODEL_DIR
        self._model = None
        self._load_failed = False
        # to_thread 在真线程执行，asyncio.Lock 无法跨线程互斥 → threading.Lock
        self._lock = threading.Lock()

    def _lazy_load(self):
        """首次调用时加载；加载失败记 flag 避免每次请求重试 438MB 加载"""
        if self._model is None:
            if self._load_failed:
                raise RuntimeError("HHEM 模型此前加载失败，不再重试")
            from rag.retrieval.hhem_loader import load_hhem_model
            try:
                self._model = load_hhem_model(self._model_dir)
            except Exception:
                self._load_failed = True
                raise

    def _predict_sync(self, docs: list[str], claims: list[str]) -> list[float]:
        """同步批量打分（由 to_thread 调用，整批持锁串行访问模型）"""
        with self._lock:
            self._lazy_load()
            scores = self._model.predict(list(zip(docs, claims)))
        return [float(s) for s in scores]

    async def predict(self, docs: list[str], claims: list[str]) -> Optional[list[float]]:
        """批量打分（CPU 推理走 to_thread，不阻塞事件循环）

        Args:
            docs: 文档文本列表（与 claims 逐对配对）
            claims: claim 文本列表

        Returns:
            与 (docs, claims) 交叉对等长的 0-1 分数数组；
            模型缺失/加载失败/推理异常/超时 → None（降级信号，不抛异常）
        """
        if not docs or not claims:
            return None
        try:
            # 超时（对齐 LLM 拆句超时哲学）：交叉打分随 claims×docs 增长
            # （5×5≈9s），推理 hang 时不无限阻塞——超时返回 None 走降级 LLM 判分
            return await asyncio.wait_for(
                asyncio.to_thread(self._predict_sync, docs, claims),
                timeout=_PREDICT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("HHEM 推理超时 (%ds)，降级 LLM 判分", _PREDICT_TIMEOUT)
            return None
        except Exception as e:
            logger.warning("HHEM 推理失败，降级 LLM 判分: %s", e)
            return None


# 全局单例
hhem_judge = HHEMJudge()
