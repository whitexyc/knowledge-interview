"""Module-031 chunker 分块回归测试

覆盖：
- 正常 ## 多节分割：每 ## 小节一个父块，子块二次分割
- ### 子小节成为独立父块（标题 "小节 > 子小节"）
- 超大父块尺寸上限（>4000 再切分为多个子父块）
- 无 ## 长文档：整篇单一父块 + 多个子块（不退化单一子块），超上限也切分
- 所有 ## 小节短于 min_chars：整篇兜底为单一父块 + 多子块
- 内容短于 min_chars：返回空，由引擎兜底（1 父 + 1 子）
- 空文本 / 纯空白：返回空

同步用例内直接调用，不依赖 DB / 模型。
"""
from rag.chunker import MarkdownChunker


def _chunker(**kwargs):
    return MarkdownChunker(**kwargs)


def test_split_multi_section():
    """多个 ## 小节 → 多个父块 + 子块二次分割"""
    text = "## 第一节\n\n" + "内容A。" * 200 + "\n\n## 第二节\n\n" + "内容B。" * 200
    res = _chunker().chunk(text, source="t")
    assert len(res["parents"]) == 2
    assert res["parents"][0]["title"] == "第一节"
    assert res["parents"][1]["title"] == "第二节"
    # 每个父块内容较长，应被切成多个子块
    assert len(res["children"]) > 4
    assert all(c["parent_index"] in (0, 1) for c in res["children"])


def test_h3_subsection_splits():
    """### 子小节成为独立父块，标题为 "小节 > 子小节" """
    text = "## 板块\n\n### 题目1\n\n" + "内容A。" * 300 + \
           "\n\n### 题目2\n\n" + "内容B。" * 300
    res = _chunker().chunk(text, source="t")
    titles = [p["title"] for p in res["parents"]]
    assert "板块 > 题目1" in titles
    assert "板块 > 题目2" in titles
    # 两个 ### 子小节各为独立父块，无合并
    assert len(res["parents"]) == 2


def test_parent_size_cap():
    """超大 ## 小节（> max_parent_chars）被切分为多个子父块，每个 ≤ 上限"""
    text = "## 大节\n\n" + ("这一段文字用于测试父块尺寸上限切分。" * 300)  # ~5400 字符
    res = _chunker().chunk(text, source="t")
    assert len(res["parents"]) >= 2
    # 默认 max_parent_chars=4000，RecursiveCharacterTextSplitter 输出 ≤ ~4100
    assert all(len(p["content"]) <= 4100 for p in res["parents"])
    assert len(res["children"]) >= 2


def test_no_heading_long_text_splits_children():
    """无 ## 的长文本 → 单一父块 + 多个子块（不退化单一子块）"""
    text = "\n\n".join(f"这是第{i}段内容，讲某个主题的细节说明。" for i in range(80))
    res = _chunker().chunk(text, source="t")
    assert len(res["parents"]) == 1
    assert res["parents"][0]["title"] == ""
    # 300 字符子块，3000+ 字符应产出多个子块
    assert len(res["children"]) >= 5
    assert all(len(c["content"]) <= 350 for c in res["children"])


def test_no_heading_big_text_size_capped():
    """无 ## 的超长文本（> max_parent_chars）被切分为多个子父块"""
    text = "\n\n".join(f"这是第{i}段无标题的长文本内容，用于验证尺寸上限切分行为。" for i in range(300))
    res = _chunker().chunk(text, source="t")
    assert len(res["parents"]) >= 2
    assert all(len(p["content"]) <= 4100 for p in res["parents"])


def test_all_sections_below_min_chars_fallback():
    """所有 ## 小节都短于 min_chars → 整篇兜底为单一父块 + 多子块"""
    text = "## A\n" + "短内容。" * 5 + "\n## B\n" + "也短。" * 5 + "\n## C\n" + "都短。" * 5
    res = _chunker().chunk(text, source="t")
    # 每个小节 < 50 字符，全部被过滤；整篇兜底
    assert len(res["parents"]) == 1
    assert len(res["children"]) >= 1


def test_tiny_text_returns_empty():
    """内容短于 min_chars → 返回空，由引擎兜底"""
    res = _chunker().chunk("太短了。", source="t")
    assert res["parents"] == []
    assert res["children"] == []


def test_empty_and_blank():
    """空文本 / 纯空白 → 返回空"""
    assert _chunker().chunk("", source="t") == {"parents": [], "children": []}
    assert _chunker().chunk("   \n  ", source="t") == {"parents": [], "children": []}
