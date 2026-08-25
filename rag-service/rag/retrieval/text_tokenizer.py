"""
jieba 中文分词工具 — Module-020 中文 FTS 复活
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

为什么需要：
  PostgreSQL 'simple' 全文检索配置对无空格连续文本按整个字符串作为单个
  lexeme（如 'Java线程池核心参数' 是一个词元），多字查询必然空召回
  （module-019 基线 FTS Hit@5=0）。用 jieba 预分词后以空格连接写入
  search_tokens 列，'simple' 按空格切分即得到中文词元，可精确匹配。

使用：
  from rag.retrieval.text_tokenizer import tokenize
  tokenize("Java线程池核心参数")   # -> "Java 线程 池 核心 参数"（空格连接）
"""
import logging
import re

try:
    import jieba
except ImportError as e:  # pragma: no cover
    raise RuntimeError("jieba 未安装，请先执行: pip install jieba") from e

logger = logging.getLogger(__name__)

# 含中文字符/英文字母/数字的 token 视为有效词（用于过滤纯标点/空白）
_WORD_RE = re.compile(r"[一-龥a-zA-Z0-9]")


class Tokenizer:
    """jieba 分词器（带结果缓存）

    缓存设计：同一文本（子块 content、查询串）可能被重复分词，
    用 dict 缓存避免重复 CPU 计算。jieba 的 cut 是纯本地 CPU 计算，
    GIL 保护下并发安全。
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    def tokenize(self, text: str) -> str:
        """将文本分词为空格连接的词元串

        Args:
            text: 原始文本（可为空/None）

        Returns:
            空格连接的分词结果；空文本返回空串

        Raises:
            RuntimeError: jieba 未安装时（模块导入期即抛出，见顶部 try）
        """
        if not text:
            return ""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # jieba.cut 默认 HMM 模式，对技术文档词边界识别更准；
        # 过滤纯标点/空白 token（'simple' 配置按空格切分，标点留无益）
        tokens = [
            tok for tok in jieba.cut(text.strip())
            if _WORD_RE.search(tok)
        ]
        result = " ".join(tokens)
        self._cache[text] = result
        return result

    def clear_cache(self) -> None:
        """清空分词结果缓存（测试用）"""
        self._cache.clear()


# 全局单例 — 整个应用共享同一缓存
tokenizer = Tokenizer()


def tokenize(text: str) -> str:
    """便捷函数：对文本做 jieba 分词并返回空格连接结果

    Args:
        text: 原始文本

    Returns:
        空格连接的分词串；空文本返回空串
    """
    return tokenizer.tokenize(text)
