package com.hewei.hzyjy.xunzhi.streamingMedia.service;

import com.alibaba.fastjson2.JSONObject;
import com.hewei.hzyjy.xunzhi.common.config.storage.ApplicationStorageProperties;
import com.hewei.hzyjy.xunzhi.common.config.xunfei.XunfeiLatProperties;
import com.hewei.hzyjy.xunzhi.media.infrastructure.integration.XunfeiAudioService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

class XunfeiAudioServiceTest {

    private XunfeiAudioService service;

    @BeforeEach
    void setUp() {
        ApplicationStorageProperties storageProperties = new ApplicationStorageProperties();
        storageProperties.setBaseDir("target/xunzhi-test");
        storageProperties.setAudioTempDir("target/xunzhi-test/audio");
        storageProperties.setUploadTempDir("target/xunzhi-test/upload");
        storageProperties.setLogDir("target/xunzhi-test/logs");
        service = new XunfeiAudioService(new XunfeiLatProperties(), storageProperties);
    }

    @Test
    void resolveSegmentId_ShouldPreferDataSegIdOverFallbackCounter() throws Exception {
        JSONObject root = new JSONObject();
        JSONObject data = new JSONObject();
        data.put("seg_id", 7);
        root.put("data", data);

        Integer segId = invoke("resolveSegmentId",
                new Class[]{JSONObject.class, JSONObject.class, AtomicInteger.class},
                root, null, new AtomicInteger(0));

        assertEquals(7, segId);
    }

    @Test
    void applyAstSegment_ShouldOverwriteSameSegmentForApdSnapshots() throws Exception {
        TreeMap<Integer, Object> sentencePool = new TreeMap<>();

        invokeVoid("applyAstSegment",
                new Class[]{TreeMap.class, int.class, String.class, int[].class, String.class, boolean.class},
                sentencePool, 0, "apd", null, "Hello", false);
        invokeVoid("applyAstSegment",
                new Class[]{TreeMap.class, int.class, String.class, int[].class, String.class, boolean.class},
                sentencePool, 0, "apd", null, "Hello there", true);

        String merged = invoke("buildFinalResult", new Class[]{java.util.Map.class}, sentencePool);
        assertEquals("Hello there", merged);
    }

    @Test
    void applyAstSegment_ShouldMergeDifferentSegmentsWithoutDuplicatingPriorSnapshots() throws Exception {
        TreeMap<Integer, Object> sentencePool = new TreeMap<>();

        invokeVoid("applyAstSegment",
                new Class[]{TreeMap.class, int.class, String.class, int[].class, String.class, boolean.class},
                sentencePool, 0, null, null, "Hello there", false);
        invokeVoid("applyAstSegment",
                new Class[]{TreeMap.class, int.class, String.class, int[].class, String.class, boolean.class},
                sentencePool, 1, null, null, ", general kenobi", true);

        String merged = invoke("buildFinalResult", new Class[]{java.util.Map.class}, sentencePool);
        assertEquals("Hello there, general kenobi", merged);
    }

    @Test
    void applyAstSegmentWithoutPgs_ShouldPreserveEarlierSegment_WhenOnlySlightTailOverlapExists() throws Exception {
        TreeMap<Integer, Object> sentencePool = new TreeMap<>();

        invokeVoid("applyAstSegmentWithoutPgs",
                new Class[]{TreeMap.class, int.class, Integer.class, Integer.class, String.class, boolean.class},
                sentencePool, 10, 10000, 14900, "数据结构，主要用来快速、准确地完成", false);
        invokeVoid("applyAstSegmentWithoutPgs",
                new Class[]{TreeMap.class, int.class, Integer.class, Integer.class, String.class, boolean.class},
                sentencePool, 29, 14880, 16980, "用户商品能不能命中某个优惠", false);

        String merged = invoke("buildFinalResult", new Class[]{java.util.Map.class}, sentencePool);
        assertEquals("数据结构，主要用来快速、准确地完成用户商品能不能命中某个优惠", merged);
    }

    @Test
    void applyAstSegmentWithoutPgs_ShouldReuseExistingSegment_WhenIncomingPacketIsSameSentenceEvolution() throws Exception {
        TreeMap<Integer, Object> sentencePool = new TreeMap<>();

        invokeVoid("applyAstSegmentWithoutPgs",
                new Class[]{TreeMap.class, int.class, Integer.class, Integer.class, String.class, boolean.class},
                sentencePool, 29, 14880, 17330, "能不能够命中某个优惠活动", false);
        invokeVoid("applyAstSegmentWithoutPgs",
                new Class[]{TreeMap.class, int.class, Integer.class, Integer.class, String.class, boolean.class},
                sentencePool, 30, 14890, 17770, "能不能够命中某个优惠活动的", false);

        String merged = invoke("buildFinalResult", new Class[]{java.util.Map.class}, sentencePool);
        assertEquals("能不能够命中某个优惠活动的", merged);
        assertEquals(1, sentencePool.size());
    }

    @Test
    void extractAstText_ShouldUseOnlyFirstRtAndTopCandidateWord() throws Exception {
        JSONObject root = JSONObject.parseObject("""
                {
                  "data": {
                    "cn": {
                      "st": {
                        "rt": [
                          {
                            "ws": [
                              {
                                "cw": [
                                  {"w": "Hello"},
                                  {"w": "hello"}
                                ]
                              },
                              {
                                "cw": [
                                  {"w": " there"},
                                  {"w": " their"}
                                ]
                              }
                            ]
                          },
                          {
                            "ws": [
                              {
                                "cw": [
                                  {"w": "ignored"}
                                ]
                              }
                            ]
                          }
                        ]
                      }
                    }
                  }
                }
                """);

        String text = invoke("extractAstText", new Class[]{JSONObject.class}, root);
        assertEquals("Hello there", text);
    }

    @SuppressWarnings("unchecked")
    private <T> T invoke(String methodName, Class<?>[] parameterTypes, Object... args) throws Exception {
        Method method = XunfeiAudioService.class.getDeclaredMethod(methodName, parameterTypes);
        method.setAccessible(true);
        return (T) method.invoke(service, args);
    }

    private void invokeVoid(String methodName, Class<?>[] parameterTypes, Object... args) throws Exception {
        Method method = XunfeiAudioService.class.getDeclaredMethod(methodName, parameterTypes);
        method.setAccessible(true);
        method.invoke(service, args);
    }
}
