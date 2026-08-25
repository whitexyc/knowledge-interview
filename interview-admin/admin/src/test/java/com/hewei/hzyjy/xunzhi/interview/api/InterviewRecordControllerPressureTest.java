package com.hewei.hzyjy.xunzhi.interview.api;

import com.hewei.hzyjy.xunzhi.common.convention.annotation.CurrentUser;
import com.hewei.hzyjy.xunzhi.common.convention.context.UserContext;
import com.hewei.hzyjy.xunzhi.common.convention.exception.ClientException;
import com.hewei.hzyjy.xunzhi.common.web.GlobalExceptionHandler;
import com.hewei.hzyjy.xunzhi.interview.service.InterviewRecordService;
import com.hewei.hzyjy.xunzhi.interview.testing.PressureTestReportUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.core.MethodParameter;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.method.support.ModelAndViewContainer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class InterviewRecordControllerPressureTest {

    private MockMvc mockMvc;
    private InterviewRecordService interviewRecordService;

    @BeforeEach
    void setUp() {
        interviewRecordService = mock(InterviewRecordService.class);
        InterviewRecordController controller = new InterviewRecordController(interviewRecordService);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setCustomArgumentResolvers(new MockCurrentUserResolver())
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    void shouldHandleConcurrentSaveFromRedisRequestsWithLockContentionShape() throws Exception {
        AtomicBoolean inProgress = new AtomicBoolean(false);
        AtomicInteger successExec = new AtomicInteger(0);
        AtomicInteger rejectedExec = new AtomicInteger(0);
        doAnswer(invocation -> {
            if (!inProgress.compareAndSet(false, true)) {
                rejectedExec.incrementAndGet();
                throw new ClientException("finalize is processing, please retry");
            }
            try {
                Thread.sleep(60L);
                successExec.incrementAndGet();
                return null;
            } finally {
                inProgress.set(false);
            }
        }).when(interviewRecordService).saveInterviewRecordFromRedis(anyString(), anyLong());

        int concurrency = 24;
        ExecutorService pool = Executors.newFixedThreadPool(10);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<HttpTaskResult>> futures = new ArrayList<>();

        for (int i = 0; i < concurrency; i++) {
            futures.add(pool.submit(() -> {
                start.await(2, TimeUnit.SECONDS);
                long begin = System.nanoTime();
                MvcResult result = mockMvc.perform(
                                MockMvcRequestBuilders.post("/api/xunzhi/v1/interview/interview/record/save-from-redis/session-record-1"))
                        .andReturn();
                return new HttpTaskResult(
                        result.getResponse().getContentAsString(),
                        TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin)
                );
            }));
        }

        start.countDown();

        int successResponses = 0;
        int failedResponses = 0;
        List<Long> latencies = new ArrayList<>();
        for (Future<HttpTaskResult> future : futures) {
            HttpTaskResult result = future.get(8, TimeUnit.SECONDS);
            latencies.add(result.latencyMs());
            String body = result.body();
            if (body.contains("\"code\":\"0\"")) {
                successResponses++;
            } else {
                failedResponses++;
            }
        }
        pool.shutdownNow();
        PressureTestReportUtil.printSummary(
                "controller.save-from-redis.concurrent",
                concurrency,
                successResponses,
                failedResponses,
                latencies,
                Map.of(
                        "sessionId", "session-record-1",
                        "executionSuccess", successExec.get(),
                        "executionRejected", rejectedExec.get()
                )
        );

        assertTrue(successResponses > 0);
        assertTrue(failedResponses > 0);
        assertEquals(concurrency, successResponses + failedResponses);
        assertEquals(concurrency, successExec.get() + rejectedExec.get());
        verify(interviewRecordService, times(concurrency)).saveInterviewRecordFromRedis("session-record-1", 7007L);
    }

    private record HttpTaskResult(String body, long latencyMs) {
    }

    private static final class MockCurrentUserResolver implements HandlerMethodArgumentResolver {
        @Override
        public boolean supportsParameter(MethodParameter parameter) {
            return parameter.hasParameterAnnotation(CurrentUser.class);
        }

        @Override
        public Object resolveArgument(
                MethodParameter parameter,
                ModelAndViewContainer mavContainer,
                NativeWebRequest webRequest,
                WebDataBinderFactory binderFactory) {
            return new UserContext(7007L, "record-pressure-user");
        }
    }
}
