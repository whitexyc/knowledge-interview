"""mDeBERTa 记忆冲突 NLI 裁判（module-061 / ADR-0007 P1 写路径冲突消解）

职责：`_merge_duplicate` 语义去重命中（cosine>memory_dedup_threshold）后，
判新事实 vs 旧父块 content 是否矛盾（contradiction）——矛盾则旧记忆标
SUPERSEDED（不删除）+ 新内容按正常新增入库，替代旧"拼接共存"。

设计（对齐 module-050/051 hhem_loader + factcheck_judge 模式）：
    - 延迟加载：首次 predict 才加载 557MB 模型（模块导入零开销，模型缺失
      不影响服务启动与记忆写入主链路）
    - threading.Lock：asyncio.to_thread 在真线程执行，asyncio.Lock 无法跨
      线程互斥（module-027 嵌入并发修复同款经验）
    - CPU 推理经 asyncio.to_thread 不阻塞事件循环
    - 加载复用 rag/memory/nli_loader.load_nli_model（单一来源，镜像 eval
      compare_nli_models 已验证路径）

降级契约（AC §3/§5）：模型缺失/加载失败/推理异常/20s 超时 → predict 返回
None（不抛异常），由 _merge_duplicate 上层回退旧行为（追加拼接，零回归）。
"""
import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# 推理超时（秒）：对齐 factcheck_judge._PREDICT_TIMEOUT 哲学——单对 CPU 推理
# 秒级（25 对批量约 5s），hang 时不无限阻塞，超时降级旧行为（追加拼接）。
_PREDICT_TIMEOUT = 20


class MemoryNLIJudge:
    """mDeBERTa 三分类 NLI 裁判（entailment/neutral/contradiction）

    predict(premise, hypothesis) 返回三分类字符串标签；任何失败（模型缺失/
    加载失败/推理异常/超时）→ 返回 None，由上层降级旧行为。
    """

    def __init__(self, model_dir: str = ""):
        self._model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "mdeberta-nli",
        )
        self._payload = None
        self._load_failed = False
        # to_thread 在真线程执行，asyncio.Lock 无法跨线程互斥 → threading.Lock
        self._lock = threading.Lock()

    def _lazy_load(self):
        """首次调用时加载；加载失败记 flag 避免每次请求重试 557MB 加载"""
        if self._payload is None:
            if self._load_failed:
                raise RuntimeError("NLI 模型此前加载失败，不再重试")
            from rag.memory.nli_loader import load_nli_model
            try:
                self._payload = load_nli_model(self._model_dir)
            except Exception:
                self._load_failed = True
                raise

    def _predict_sync(self, premise: str, hypothesis: str) -> str:
        """同步单对三分类（由 to_thread 调用，整批持锁串行访问模型）"""
        with self._lock:
            self._lazy_load()
            from rag.memory.nli_loader import nli_score
            labels, _probs = nli_score(
                self._payload, [premise], [hypothesis])
            idx = int(labels[0])
            return str(self._payload["id2label"][idx])

    async def predict(self, premise: str, hypothesis: str) -> Optional[str]:
        """判 premise（旧记忆）与 hypothesis（新事实）的关系

        Args:
            premise: 旧记忆内容（父块 content）
            hypothesis: 新事实内容

        Returns:
            "entailment" / "neutral" / "contradiction" 之一；
            模型缺失/加载失败/推理异常/超时 → None（降级信号，不抛异常）
        """
        if not premise or not premise.strip() or not hypothesis or not hypothesis.strip():
            return None
        try:
            # 超时（对齐 factcheck_judge 哲学）：CPU 推理 hang 时不无限阻塞
            return await asyncio.wait_for(
                asyncio.to_thread(self._predict_sync, premise, hypothesis),
                timeout=_PREDICT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("NLI 推理超时 (%ds)，降级旧行为", _PREDICT_TIMEOUT)
            return None
        except Exception as e:
            logger.warning("NLI 推理失败，降级旧行为: %s", e)
            return None


# 全局单例 — 整个应用共享一个 MemoryNLIJudge 实例（无状态，延迟加载）
nli_judge = MemoryNLIJudge()
