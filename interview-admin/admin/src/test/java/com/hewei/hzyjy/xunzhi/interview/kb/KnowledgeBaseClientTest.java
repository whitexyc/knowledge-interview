package com.hewei.hzyjy.xunzhi.interview.kb;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
/**
 * KnowledgeBaseClient 单测（module-074）：解析逻辑 + fail-open 行为。
 */
class KnowledgeBaseClientTest {

    private final KnowledgeBaseClient client = new KnowledgeBaseClient("http://127.0.0.1:8001");

    @Test
    void parseResults_joinsContentsWithIndex() throws Exception {
        String json = "{\"message\":\"ok\",\"results\":["
                + "{\"content\":\"Redis 持久化 RDB 与 AOF\",\"source\":\"redis.md\",\"score\":0.9},"
                + "{\"content\":\"Kafka 分区与消费组\",\"source\":\"kafka.md\",\"score\":0.8}"
                + "]}";
        String parsed = client.parseResults(json);
        assertTrue(parsed.contains("1. Redis 持久化 RDB 与 AOF"));
        assertTrue(parsed.contains("2. Kafka 分区与消费组"));
    }

    @Test
    void parseResults_skipsBlankContent() throws Exception {
        String json = "{\"results\":[{\"content\":\"\"},{\"content\":\"有效知识点\"}]}";
        String parsed = client.parseResults(json);
        assertEquals("1. 有效知识点", parsed);
    }

    @Test
    void parseResults_emptyResults_returnsEmptyString() throws Exception {
        assertEquals("", client.parseResults("{\"results\":[]}"));
    }

    @Test
    void retrieveContext_failOpen_whenServiceUnreachable() {
        // 端口 9（discard）通常无监听，连接立即拒绝 → fail-open 返回空串
        KnowledgeBaseClient unreachable = new KnowledgeBaseClient("http://127.0.0.1:9");
        String result = unreachable.retrieveContext("Java Redis", 3);
        assertEquals("", result);
    }

    @Test
    void retrieveContext_blankQuery_returnsEmptyWithoutCall() {
        assertEquals("", new KnowledgeBaseClient("http://127.0.0.1:8001").retrieveContext("  ", 3));
    }

    @Test
    @Timeout(15)
    void retrieveContext_neverThrows_onUnreachableNetwork() {
        // fail-open 契约：任何情况下不抛异常
        KnowledgeBaseClient c = new KnowledgeBaseClient("http://127.0.0.1:1");
        assertEquals("", c.retrieveContext("any", 1));
    }

    @Test
    void parseResults_nonArrayResults_returnsEmpty() throws Exception {
        assertEquals("", client.parseResults("{\"results\":{\"content\":\"not an array\"}}"));
    }

    @Test
    void retrieveContextCached_cachesByKey() throws Exception {
        AtomicInteger hits = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/ai/rag/search", exchange -> {
            hits.incrementAndGet();
            byte[] resp = "{\"results\":[{\"content\":\"Redis 持久化\"}]}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, resp.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(resp);
            }
        });
        server.start();
        try {
            KnowledgeBaseClient c = new KnowledgeBaseClient("http://127.0.0.1:" + server.getAddress().getPort());
            String first = c.retrieveContextCached("hash-1", "Redis", 5);
            String second = c.retrieveContextCached("hash-1", "Redis", 5);
            assertEquals(first, second);
            assertEquals(1, hits.get(), "同键第二次应命中缓存");
            c.retrieveContextCached("hash-2", "Kafka", 5);
            assertEquals(2, hits.get(), "不同键应重新检索");
        } finally {
            server.stop(0);
        }
    }

    @Test
    void retrieveContext_oversizedResponse_failsOpen() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/ai/rag/search", exchange -> {
            byte[] resp = new byte[70 * 1024];
            Arrays.fill(resp, (byte) 'a');
            exchange.sendResponseHeaders(200, resp.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(resp);
            }
        });
        server.start();
        try {
            KnowledgeBaseClient c = new KnowledgeBaseClient("http://127.0.0.1:" + server.getAddress().getPort());
            assertEquals("", c.retrieveContext("Java", 5), "超大响应应 fail-open 返回空串");
        } finally {
            server.stop(0);
        }
    }
}
