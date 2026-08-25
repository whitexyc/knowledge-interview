"""rag/memory 子包（module-050 WP5 目录细分）

职责：记忆领域（memory/memory_extractor/session_memory）。

旧路径兼容（rag.memory 等 → 同一模块对象）由 rag/__init__.py 统一注册；
本文件 re-export 兜底（存量 tests 的 from rag.memory import X 经 sys.modules
别名命中真实模块，新路径 from rag.memory.memory import X 为规范写法）。
"""
from rag.memory.memory import *  # noqa: F401,F403  (re-export 兜底)
from rag.memory.memory_extractor import *  # noqa: F401,F403
from rag.memory.session_memory import *  # noqa: F401,F403
# module-061：nli_loader/nli_judge（记忆冲突消解）也须在子包 __init__ 导入一次，
# 使 'rag.memory.nli_judge' 等全路径注册进 sys.modules（rag.memory 被旧路径别名
# 覆盖为普通模块后，未在 init 导入的子模块无法经子包路径导入——module-050 兼容机制）。
# 两个模块顶层只 import stdlib（torch 在函数内延迟导入），包导入零开销。
from rag.memory.nli_loader import *  # noqa: F401,F403
from rag.memory.nli_judge import *  # noqa: F401,F403
# module-062：memory_type_clf（类型判断分类器）/memory_conflict_clf（矛盾检测
# 分类器）同款注册——顶层仅 import 已加载的 embedding_service/numpy/内存提取器
# （sklearn/joblib 惰性导入），包导入零模型加载。
from rag.memory.memory_type_clf import *  # noqa: F401,F403
from rag.memory.memory_conflict_clf import *  # noqa: F401,F403
from rag.memory.weak_topics import *  # noqa: F401,F403

