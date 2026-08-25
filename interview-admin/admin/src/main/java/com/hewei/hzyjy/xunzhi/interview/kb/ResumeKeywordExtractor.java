package com.hewei.hzyjy.xunzhi.interview.kb;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 简历关键词抽取器（ADR-0019 阶段 1）：PDF 文本提取 + 中英关键词 Top-N。
 *
 * <p>零分词器依赖的启发式方案：英文按非字母数字切词取词频，
 * 中文取连续汉字串的 2-gram 词频，停用词过滤后取 Top-N 拼接为 KB 检索 query。
 * 抽取质量是启发式的——只影响检索 query 质量，不影响主链路正确性（fail-open）。
 */
@Slf4j
@Component
public class ResumeKeywordExtractor {

    private static final int MAX_KEYWORDS = 8;
    /** 英文 token 最小长度（与实际过滤条件一致：小于该值的丢弃）。 */
    private static final int MIN_TOKEN_LENGTH = 3;
    /** 最小出现频次：1 以保召回（重要技术词常仅出现一次），Top-N 截断兜底精度。 */
    private static final int MIN_FREQUENCY = 1;
    private static final Pattern CJK_RUN = Pattern.compile("[\\u4e00-\\u9fa5]{2,}");

    private static final Set<String> STOPWORDS = Set.of(
            "the", "and", "for", "with", "from", "this", "that", "have", "has", "are",
            "was", "were", "will", "been", "their", "them", "they", "your", "you",
            "our", "not", "but", "can", "all", "any", "out", "use", "using", "used",
            "work", "working", "project", "projects", "experience", "experienced",
            "familiar", "proficient", "master", "understand", "responsible",
            "development", "develop", "design", "designing", "system", "systems",
            "application", "applications", "company", "team", "years", "year",
            "month", "months",
            "以上", "熟悉", "掌握", "了解", "精通", "负责", "参与", "开发", "设计",
            "实现", "使用", "技术", "项目", "系统", "经验", "相关", "进行", "完成",
            "工作", "公司", "团队");

    /** 提取 PDF 文本；失败返回空串。 */
    public String extractText(byte[] pdfBytes) {
        if (pdfBytes == null || pdfBytes.length == 0) {
            return "";
        }
        try (PDDocument document = PDDocument.load(pdfBytes)) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document);
        } catch (IOException | SecurityException e) {
            log.warn("Failed to extract resume text, error={}", e.getMessage());
            return "";
        }
    }

    /**
     * 从简历文本构建 KB 检索 query（Top 关键词空格连接）。
     *
     * @return 空格分隔的关键词串；无可抽取关键词时返回空串
     */
    public String buildQuery(String resumeText) {
        if (resumeText == null || resumeText.isBlank()) {
            return "";
        }
        Map<String, Integer> freq = new LinkedHashMap<>();
        collectEnglishTokens(resumeText, freq);
        collectChineseBigrams(resumeText, freq);
        String keywords = freq.entrySet().stream()
                .filter(e -> e.getValue() >= MIN_FREQUENCY)
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(MAX_KEYWORDS)
                .map(Map.Entry::getKey)
                .collect(Collectors.joining(" "));
        log.debug("Resume keywords extracted, count={}", keywords.isBlank() ? 0 : keywords.split(" ").length);
        return keywords;
    }

    private void collectEnglishTokens(String text, Map<String, Integer> freq) {
        for (String token : text.split("[^A-Za-z0-9+#]+")) {
            String t = token.toLowerCase();
            if (t.length() < MIN_TOKEN_LENGTH || STOPWORDS.contains(t) || t.matches("\\d+")) {
                continue;
            }
            freq.merge(t, 1, Integer::sum);
        }
    }

    private void collectChineseBigrams(String text, Map<String, Integer> freq) {
        Matcher matcher = CJK_RUN.matcher(text);
        while (matcher.find()) {
            String run = matcher.group();
            for (int i = 0; i + 2 <= run.length(); i++) {
                String gram = run.substring(i, i + 2);
                if (!STOPWORDS.contains(gram)) {
                    freq.merge(gram, 1, Integer::sum);
                }
            }
        }
    }
}
