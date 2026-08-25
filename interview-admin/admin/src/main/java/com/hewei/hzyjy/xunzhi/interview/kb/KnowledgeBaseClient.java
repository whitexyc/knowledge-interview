package com.hewei.hzyjy.xunzhi.interview.kb;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 知识库检索客户端（ADR-0019 决策 2）。
 *
 * <p>调用 Agentic RAG 服务（FastAPI，默认 http://localhost:8001）的
 * POST /ai/rag/search 接口，把检索到的知识点拼接为可注入出题 prompt 的文本。
 *
 * <p>熔断策略：fail-open——任何失败（连接拒绝/超时/非 200/解析异常）一律
 * 返回空串，出题主链路退化为纯简历出题，绝不因知识库故障阻断面试流程
 * （对齐项目既有决策「Java 端实现熔断降级」）。
 */
@Slf4j
@Component
public class KnowledgeBaseClient {

    private static final MediaType JSON_MEDIA = MediaType.parse("application/json");

    /** 响应体大小上限：防 KB 服务异常返回超大响应拖垮内存（64KB，覆盖 top_k=5 正常量级）。 */
    private static final int MAX_RESPONSE_BYTES = 64 * 1024;

    /** KB 上下文缓存上限：超过后整体清空（低频资源，粗粒度足够）。 */
    private static final int CACHE_MAX_ENTRIES = 500;

    /** KB 上下文缓存：按简历内容哈希键控，避免同一简历重复同步检索。 */
    private final ConcurrentHashMap<String, String> contextCache = new ConcurrentHashMap<>();

    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(Duration.ofSeconds(3))
            .callTimeout(Duration.ofSeconds(8))
            .build();

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final String baseUrl;

    public KnowledgeBaseClient(@Value("${kb.base-url:http://localhost:8001}") String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
    }

    /**
     * 检索知识库上下文。
     *
     * @param query 检索查询（如简历关键词）
     * @param topK  召回条数
     * @return 编号列表文本（每行一条知识点）；任何失败返回空串（fail-open）
     */
    public String retrieveContext(String query, int topK) {
        if (query == null || query.isBlank()) {
            return "";
        }
        try {
            String body = objectMapper.writeValueAsString(Map.of("query", query, "top_k", topK));
            Request request = new Request.Builder()
                    .url(baseUrl + "/ai/rag/search")
                    .post(RequestBody.create(body, JSON_MEDIA))
                    .build();
            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful() || response.body() == null) {
                    // 不记录原始 query（可能含简历个人信息），只记状态与长度
                    log.warn("KB search failed, status={}, queryChars={}", response.code(), query.length());
                    return "";
                }
                long declared = response.body().contentLength();
                if (declared > MAX_RESPONSE_BYTES) {
                    log.warn("KB response too large, declaredBytes={}, fail-open", declared);
                    return "";
                }
                // 流式读取并硬性限制读取量，防超大/畸形响应拖垮内存
                byte[] buf = new byte[MAX_RESPONSE_BYTES + 1];
                int total = 0;
                int n;
                try (InputStream in = response.body().byteStream()) {
                    while (total <= MAX_RESPONSE_BYTES
                            && (n = in.read(buf, total, buf.length - total)) > 0) {
                        total += n;
                    }
                }
                if (total > MAX_RESPONSE_BYTES) {
                    log.warn("KB response too large, actualBytes={}, fail-open", total);
                    return "";
                }
                return parseResults(new String(buf, 0, total, StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            log.warn("KB search unavailable, fail-open. queryChars={}, error={}", query.length(), e.getMessage());
            return "";
        }
    }

    /**
     * 带缓存的检索：同一缓存键（简历内容哈希）在 CACHE_MAX_ENTRIES 内直接命中，
     * 避免同一简历在单次提取流程中重复同步调用 KB（KB 服务 rerank 延迟约 2-3s）。
     * 缓存键为空（如文件为空）时退化为无缓存直查。
     */
    public String retrieveContextCached(String cacheKey, String query, int topK) {
        if (cacheKey != null && !cacheKey.isBlank()) {
            String cached = contextCache.get(cacheKey);
            if (cached != null) {
                return cached;
            }
        }
        String context = retrieveContext(query, topK);
        if (cacheKey != null && !cacheKey.isBlank()) {
            if (contextCache.size() >= CACHE_MAX_ENTRIES) {
                contextCache.clear();
            }
            contextCache.put(cacheKey, context);
        }
        return context;
    }

    /** 解析 /ai/rag/search 响应为编号知识点文本；包私有便于单测。 */
    String parseResults(String json) throws java.io.IOException {
        JsonNode results = objectMapper.readTree(json).path("results");
        if (!results.isArray()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        int index = 1;
        for (JsonNode item : results) {
            String content = item.path("content").asText("");
            if (content.isBlank()) {
                continue;
            }
            sb.append(index++).append(". ").append(content.trim()).append('\n');
        }
        return sb.toString().trim();
    }
}
