"""rag 包（module-050 WP5 目录细分）

按职责拆分子包：
    rag.retrieval —— retriever/reranker/chunker/embeddings/text_tokenizer/query_rewrite
    rag.graph     —— graph/graph_extractor/graph_store
    rag.memory    —— memory/memory_extractor/session_memory
    rag 根        —— state/models/schemas/engine/migrate_parent_child（跨领域共享）

import 兼容（红线：存量 tests 的 from rag.retriever import X / from rag.memory import Y
与 mock.patch("rag.retriever.hybrid_retriever") 必须不破）：
    旧路径模块名 → 同一模块对象：既注册 sys.modules 别名（覆盖 from rag.xxx import X），
    也挂到 rag 包属性（覆盖 rag.xxx 属性访问）。生产代码一律用新路径。
"""
import sys

from rag import retrieval  # noqa: F401
from rag import graph  # noqa: F401
from rag import memory  # noqa: F401

# 旧路径 → 同一模块对象（sys.modules + 包属性双注册）
# 注意：不用 retrieval.xxx 属性访问——子包 __init__ 的 import * 会把模块级同名
# 单例（如 reranker = CrossEncoderReranker(...)）覆盖为属性，须走 sys.modules。
_OLD_PATHS = {
    "retriever": sys.modules["rag.retrieval.retriever"],
    "reranker": sys.modules["rag.retrieval.reranker"],
    "chunker": sys.modules["rag.retrieval.chunker"],
    "embeddings": sys.modules["rag.retrieval.embeddings"],
    "text_tokenizer": sys.modules["rag.retrieval.text_tokenizer"],
    "query_rewrite": sys.modules["rag.retrieval.query_rewrite"],
    "graph": sys.modules["rag.graph.graph"],
    "graph_extractor": sys.modules["rag.graph.graph_extractor"],
    "graph_store": sys.modules["rag.graph.graph_store"],
    "memory": sys.modules["rag.memory.memory"],
    "memory_extractor": sys.modules["rag.memory.memory_extractor"],
    "session_memory": sys.modules["rag.memory.session_memory"],
}
for _name, _module in _OLD_PATHS.items():
    sys.modules[f"rag.{_name}"] = _module
    globals()[_name] = _module
