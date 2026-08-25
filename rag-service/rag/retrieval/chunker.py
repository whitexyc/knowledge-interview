"""
Markdown 文档分块器 — 预处理层（父子分块）
===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

在整个 RAG 链路中的位置：
  文档入库 → [Chunker] 两级分块 → [EmbeddingService] 子块向量化 → 入库

为什么需要两级分块？
  父块（回答单元）：按 ##/### 标题层级分割，保持完整的段落语义，无向量，
     检索时不参与，但作为最终返回给用户的粒度（尺寸上限 4000 字符）。
  子块（~300 字符）：对每个父块内容二次分割，携带向量嵌入，
     参与混合检索（FTS + 向量），命中后通过 parent_id 映射回父块。

  这种方式结合了检索精度（小块更聚焦）和展示完整性（父块语义完整）。

实现：
  使用 LangChain 的 MarkdownHeaderTextSplitter（一级：按 ##/### 标题层级）
  和 RecursiveCharacterTextSplitter（二级：父块尺寸上限 + 三级：子块按语义边界）。
"""
import logging
from typing import Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class MarkdownChunker:
    """Markdown 文档分块器（父子两级）

    第一级：MarkdownHeaderTextSplitter 按 ##/### 标题层级分割 → 父块候选
    第二级：RecursiveCharacterTextSplitter 对超 max_parent_chars 的父块二次切分 → 子父块
    第三级：RecursiveCharacterTextSplitter 对每父块按 ~300 字符切 → 子块

    为什么用 LangChain 而不是手写正则？
    1. 成熟的分割逻辑，处理了标题嵌套、代码块等边界情况
    2. 保留标题层级 metadata（如 {"section": "板块6", "subsection": "题目2"}）
    3. RecursiveCharacterTextSplitter 按语义边界（段落、句子）分割
    """

    def __init__(
        self,
        headers_to_split_on: Optional[list[tuple[str, str]]] = None,
        min_chars: int = 50,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 50,
        max_parent_chars: int = 4000,
    ):
        """
        Args:
            headers_to_split_on: 按哪些标题分割，默认 [("##", "section"), ("###", "subsection")]
                ## 为小节，### 为子小节（父块粒度=最小标题单元，如"板块6 > 题目2"）
            min_chars: 最小块字符数，低于此值的父块被过滤
            child_chunk_size: 子块目标字符数
            child_chunk_overlap: 相邻子块重叠字符数
            max_parent_chars: 父块字符数上限，超过的父块再按段落二次切分为多个子父块
                （父块是返回给 LLM 的回答单元，尺寸需有界；默认 4000）
        """
        self._headers_to_split_on = headers_to_split_on or [("##", "section"), ("###", "subsection")]
        self._min_chars = min_chars
        self._max_parent_chars = max_parent_chars
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers_to_split_on,
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_parent_chars,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def chunk(self, text: str, source: str = "") -> dict:
        """将 Markdown 文本两级分割为父块和子块

        Args:
            text: Markdown 原文
            source: 来源标识（仅用于日志）

        Returns:
            {
                "parents": [{"title": str, "content": str}, ...],
                "children": [{"title": str, "content": str, "parent_index": int}, ...]
            }
        """
        if not text or not text.strip():
            return {"parents": [], "children": []}

        # ===== 第一级：按标题层级分割 → 父块候选 =====
        # headers_to_split_on 默认 [("##", "section"), ("###", "subsection")]：
        #   ## 小节 / ### 子小节都成为分割点，父块粒度=最小标题单元，
        #   标题路径如 "板块6 > 题目2"（MarkdownHeaderTextSplitter 从 metadata 组装）。
        langchain_docs = self._splitter.split_text(text)

        raw_parents = []
        for doc in langchain_docs:
            content = doc.page_content.strip()
            if len(content) < self._min_chars:
                continue
            raw_parents.append({"title": self._build_title(doc.metadata), "content": content})

        # 无有效父块：无 ##/### 标题，或所有标题块都短于 min_chars 被过滤。
        # 此时整篇作为单一父块（title 空），子块仍按 child_splitter 二次分割，
        # 避免整篇退化为单一子块（会导致 embedding 截断 + rerank 过慢）。
        # 内容过短（< min_chars）时返回空，由引擎兜底存 1 父 + 1 子。
        if not raw_parents:
            whole = text.strip()
            if len(whole) < self._min_chars:
                logger.debug("分块: source=%s, 内容过短，返回空（由引擎兜底）", source)
                return {"parents": [], "children": []}
            raw_parents = [{"title": "", "content": whole}]
            logger.debug("分块 fallback: source=%s, input=%d chars", source, len(whole))

        # ===== 第二级：父块尺寸上限 =====
        # 标题块可能仍超大（如"板块6 面试题"整节 12k+），父块是返回给 LLM 的
        # 回答单元，需按段落二次切分为多个 ≤ max_parent_chars 的子父块。
        parents = []
        for p in raw_parents:
            if len(p["content"]) <= self._max_parent_chars:
                parents.append(p)
                continue
            pieces = self._parent_splitter.split_text(p["content"])
            for piece in pieces:
                piece = piece.strip()
                if piece and len(piece) >= self._min_chars:
                    parents.append({"title": p["title"], "content": piece})

        # ===== 第三级：对每父块用 RecursiveCharacterTextSplitter 分割 → 子块 =====
        children = []
        for pi, parent in enumerate(parents):
            child_texts = self._child_splitter.split_text(parent["content"])
            for ct in child_texts:
                child_content = ct.strip()
                if not child_content:
                    continue
                children.append({
                    "title": parent["title"],
                    "content": child_content,
                    "parent_index": pi,
                })

        logger.debug("分块: source=%s, input=%d chars, parents=%d, children=%d",
                      source, len(text), len(parents), len(children))
        return {"parents": parents, "children": children}

    def _build_title(self, metadata: dict) -> str:
        """从 LangChain metadata 构建标题路径（如 "板块6 > 题目2"）"""
        title_parts = []
        for _, header_name in self._headers_to_split_on:
            val = metadata.get(header_name, "")
            if val:
                title_parts.append(val)
        return " > ".join(title_parts) if title_parts else ""


# 全局单例
chunker = MarkdownChunker()
