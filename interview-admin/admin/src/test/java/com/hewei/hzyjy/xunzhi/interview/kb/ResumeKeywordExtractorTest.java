package com.hewei.hzyjy.xunzhi.interview.kb;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ResumeKeywordExtractor 单测（module-074）。
 */
class ResumeKeywordExtractorTest {

    private final ResumeKeywordExtractor extractor = new ResumeKeywordExtractor();

    @Test
    void buildQuery_returnsEmpty_forNullText() {
        assertEquals("", extractor.buildQuery(null));
    }

    @Test
    void buildQuery_returnsEmpty_forBlankText() {
        assertEquals("", extractor.buildQuery("   \n  "));
    }

    @Test
    void buildQuery_extractsFrequentEnglishTechTerms() {
        String resume = ("Java Redis Kafka Java Redis Kafka Java Redis high concurrency "
                + "Java Redis Kafka JVM tuning Kafka consumer group");
        String query = extractor.buildQuery(resume);
        assertTrue(query.contains("java"), "应含高频词 java: " + query);
        assertTrue(query.contains("redis"), "应含高频词 redis: " + query);
        assertTrue(query.contains("kafka"), "应含高频词 kafka: " + query);
    }

    @Test
    void buildQuery_filtersStopwords() {
        String resume = "responsible for system design responsible for system design responsible";
        String query = extractor.buildQuery(resume);
        assertFalse(query.contains("responsible"), "停用词应被过滤: " + query);
        assertFalse(query.contains("system"), "停用词应被过滤: " + query);
    }

    @Test
    void buildQuery_limitsToTopKeywords() {
        StringBuilder many = new StringBuilder();
        for (int i = 0; i < 6; i++) {
            for (char c = 'a'; c <= 'j'; c++) {
                many.append("t0ken").append(c).append(' ');
            }
        }
        String[] parts = extractor.buildQuery(many.toString()).split(" ");
        assertTrue(parts.length <= 10, "关键词数量应有上限: " + parts.length);
    }

    @Test
    void extractText_returnsEmpty_forNullBytes() {
        assertEquals("", extractor.extractText(null));
        assertEquals("", extractor.extractText(new byte[0]));
    }
}
