"""Module-020 分词工具单元测试

覆盖：
- 中文分词正确（技术文档专有词/代码片段）
- 英文/数字分词
- 空文本/纯空白返回空串
- 特殊字符（标点）被过滤
- 缓存一致性（同一文本重复调用返回相同结果）
- tokenizer 单例与便捷函数行为

说明：tokenize 为同步纯函数，pytest 可直接收集（无 pytest-asyncio 依赖，
规避测试环境缺 pytest-asyncio 的既有问题）。
"""
from rag.text_tokenizer import tokenize, tokenizer


class TestTokenizerChinese:
    """中文分词正确性"""

    def test_chinese_phrase(self):
        # 验收场景：tokenize('Java线程池核心参数') 应含"线程/池/核心/参数"
        t = tokenize("Java线程池核心参数")
        assert "线程" in t and "池" in t and "核心" in t and "参数" in t

    def test_chinese_whole_words(self):
        # 分词后是空格连接的多词串，而非单个汉字串（FTS 命中的关键）
        t = tokenize("Java线程池核心参数")
        assert len(t.split(" ")) >= 4

    def test_english_words(self):
        t = tokenize("Java G1 GC")
        assert "java" in t.lower() and "g1" in t.lower() and "gc" in t.lower()

    def test_mixed_with_sentence(self):
        # 技术文档混合语句：中英文都可分词
        t = tokenize("Kafka Consumer Group 的 Rebalance 机制")
        low = t.lower()
        assert "kafka" in low and "rebalance" in low and "的" in t


class TestTokenizerEdgeCases:
    """边界条件"""

    def test_empty_string(self):
        assert tokenize("") == ""

    def test_whitespace_only(self):
        assert tokenize("   ") == ""

    def test_punctuation_only(self):
        # 纯标点 → 分词后为空串（不崩溃）
        assert tokenize("？？？！！！---") == ""

    def test_punctuation_filtered_from_mixed(self):
        # 标点被过滤，只保留中英文词
        t = tokenize("你好，世界！Hello, world!")
        assert "你好" in t and "世界" in t
        low = t.lower()
        assert "hello" in low and "world" in low
        assert "，" not in t and "！" not in t


class TestTokenizerCache:
    """结果缓存"""

    def test_cache_consistency(self):
        # 同一文本重复调用返回相同结果
        a = tokenize("Java线程池核心参数")
        b = tokenize("Java线程池核心参数")
        assert a == b

    def test_clear_cache_recomputes(self):
        # clear_cache 后重新分词结果一致（可复现）
        tokenizer.clear_cache()
        a = tokenize("线程池")
        tokenizer.clear_cache()
        b = tokenize("线程池")
        assert a == b
