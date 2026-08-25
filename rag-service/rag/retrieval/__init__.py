"""rag/retrieval 子包（module-050 WP5 目录细分）

职责：检索领域（retriever/reranker/chunker/embeddings/text_tokenizer/query_rewrite）。

旧路径兼容（rag.retriever 等 → 同一模块对象）由 rag/__init__.py 统一注册；
本文件 re-export 兜底（存量 tests 的 from rag.retriever import X 经 sys.modules
别名命中真实模块，新路径 from rag.retrieval.retriever import X 为规范写法）。
"""
from rag.retrieval.retriever import *  # noqa: F401,F403  (re-export 兜底)
from rag.retrieval.reranker import *  # noqa: F401,F403
from rag.retrieval.chunker import *  # noqa: F401,F403
from rag.retrieval.embeddings import *  # noqa: F401,F403
from rag.retrieval.text_tokenizer import *  # noqa: F401,F403
from rag.retrieval.query_rewrite import *  # noqa: F401,F403
