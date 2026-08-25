package com.hewei.hzyjy.xunzhi.interview.api;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.hewei.hzyjy.xunzhi.common.convention.annotation.CurrentUser;
import com.hewei.hzyjy.xunzhi.common.convention.context.UserContext;
import com.hewei.hzyjy.xunzhi.common.web.GlobalExceptionHandler;
import com.hewei.hzyjy.xunzhi.interview.api.io.req.InterviewAnswerReqDTO;
import com.hewei.hzyjy.xunzhi.interview.api.io.resp.InterviewAnswerRespDTO;
import com.hewei.hzyjy.xunzhi.interview.api.io.resp.InterviewQuestionRespDTO;
import com.hewei.hzyjy.xunzhi.interview.flow.session.InterviewSessionFacade;
import com.hewei.hzyjy.xunzhi.interview.testing.PressureTestReportUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.core.MethodParameter;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.method.support.ModelAndViewContainer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class InterviewSessionControllerPressureTest {

    private MockMvc mockMvc;
    private InterviewSessionFacade interviewSessionFacade;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        interviewSessionFacade = mock(InterviewSessionFacade.class);
        InterviewSessionController controller = new InterviewSessionController(interviewSessionFacade);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setCustomArgumentResolvers(new MockCurrentUserResolver())
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        objectMapper = new ObjectMapper();
    }

    @Test
    void shouldHandleConcurrentAnswerJsonRequestsWithStableResponse() throws Exception {
        when(interviewSessionFacade.answerInterviewQuestion(anyString(), any(InterviewAnswerReqDTO.class), anyLong()))
                .thenAnswer(invocation -> {
                    InterviewAnswerReqDTO req = invocation.getArgument(1);
                    return InterviewAnswerRespDTO.init()
                            .withCurrentQuestion(req.getQuestionNumber(), "Q-" + req.getQuestionNumber())
                            .withEvaluation(88, "mock-feedback", 88)
                            .withNextQuestion("2", "next-q", false, 0)
                            .success();
                });

        int concurrency = 40;
        ExecutorService pool = Executors.newFixedThreadPool(10);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<Long>> futures = new ArrayList<>();

        for (int i = 0; i < concurrency; i++) {
            int idx = i;
            futures.add(pool.submit(() -> {
                InterviewAnswerReqDTO req = new InterviewAnswerReqDTO();
                req.setQuestionNumber("1");
                req.setAnswerContent("answer-" + idx);
                req.setRequestId("rid-" + idx);

                start.await(2, TimeUnit.SECONDS);
                long begin = System.nanoTime();
                mockMvc.perform(MockMvcRequestBuilders.post("/api/xunzhi/v1/interview/sessions/session-http-1/interview/answer-json")
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(objectMapper.writeValueAsString(req)))
                        .andExpect(status().isOk())
                        .andExpect(jsonPath("$.code").value(0))
                        .andExpect(jsonPath("$.data.isSuccess").value(true))
                        .andExpect(jsonPath("$.data.questionNumber").value("1"))
                        .andExpect(jsonPath("$.data.questionContent").value("Q-1"))
                        .andExpect(jsonPath("$.data.totalScore").value(88));
                return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            }));
        }

        start.countDown();
        List<Long> latencies = new ArrayList<>();
        for (Future<Long> future : futures) {
            latencies.add(future.get(6, TimeUnit.SECONDS));
        }
        pool.shutdownNow();
        PressureTestReportUtil.printSummary(
                "controller.answer-json.concurrent",
                concurrency,
                concurrency,
                0,
                latencies,
                Map.of("sessionId", "session-http-1")
        );

        verify(interviewSessionFacade, times(concurrency))
                .answerInterviewQuestion(anyString(), any(InterviewAnswerReqDTO.class), anyLong());
    }

    @Test
    void shouldHandleConcurrentExtractionRequestsWithMockedWorkflowData() throws Exception {
        when(interviewSessionFacade.extractInterviewQuestions(anyString(), any(), anyLong(), anyString()))
                .thenAnswer(invocation -> {
                    InterviewQuestionRespDTO resp = new InterviewQuestionRespDTO();
                    resp.setSessionId(invocation.getArgument(0));
                    resp.setUserName("pressure-user");
                    resp.setIsSuccess(1);
                    resp.setInterviewType("backend");
                    resp.setResumeFileUrl("https://mock/resume.pdf");
                    return resp;
                });

        int concurrency = 24;
        ExecutorService pool = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<Long>> futures = new ArrayList<>();

        for (int i = 0; i < concurrency; i++) {
            int idx = i;
            futures.add(pool.submit(() -> {
                MockMultipartFile file = new MockMultipartFile(
                        "resumePdf",
                        "resume-" + idx + ".pdf",
                        "application/pdf",
                        ("resume-content-" + idx).getBytes(StandardCharsets.UTF_8)
                );
                start.await(2, TimeUnit.SECONDS);
                long begin = System.nanoTime();
                mockMvc.perform(multipart("/api/xunzhi/v1/interview/sessions/session-http-2/interview-questions")
                                .file(file))
                        .andExpect(status().isOk())
                        .andExpect(jsonPath("$.code").value(0))
                        .andExpect(jsonPath("$.data.isSuccess").value(1))
                        .andExpect(jsonPath("$.data.interviewType").value("backend"));
                return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            }));
        }

        start.countDown();
        List<Long> latencies = new ArrayList<>();
        for (Future<Long> future : futures) {
            latencies.add(future.get(6, TimeUnit.SECONDS));
        }
        pool.shutdownNow();
        PressureTestReportUtil.printSummary(
                "controller.extract-questions.concurrent",
                concurrency,
                concurrency,
                0,
                latencies,
                Map.of("sessionId", "session-http-2")
        );

        verify(interviewSessionFacade, times(concurrency))
                .extractInterviewQuestions(anyString(), any(), anyLong(), anyString());
    }

    @Test
    void shouldHandleConcurrentAnswerFormRequestsWithStableResponse() throws Exception {
        when(interviewSessionFacade.answerInterviewQuestion(anyString(), any(InterviewAnswerReqDTO.class), anyLong()))
                .thenAnswer(invocation -> {
                    InterviewAnswerReqDTO req = invocation.getArgument(1);
                    return InterviewAnswerRespDTO.init()
                            .withCurrentQuestion(req.getQuestionNumber(), "Q-" + req.getQuestionNumber())
                            .withEvaluation(90, "form-ok", 90)
                            .withNextQuestion("2", "next-q", false, 0)
                            .success();
                });

        int concurrency = 36;
        ExecutorService pool = Executors.newFixedThreadPool(9);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<Long>> futures = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            int idx = i;
            futures.add(pool.submit(() -> {
                start.await(2, TimeUnit.SECONDS);
                long begin = System.nanoTime();
                mockMvc.perform(MockMvcRequestBuilders.post("/api/xunzhi/v1/interview/sessions/session-http-3/interview/answer")
                                .param("questionNumber", "1")
                                .param("answerContent", "form-answer-" + idx)
                                .param("requestId", "form-rid-" + idx))
                        .andExpect(status().isOk())
                        .andExpect(jsonPath("$.code").value(0))
                        .andExpect(jsonPath("$.data.isSuccess").value(true))
                        .andExpect(jsonPath("$.data.questionNumber").value("1"))
                        .andExpect(jsonPath("$.data.totalScore").value(90));
                return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            }));
        }
        start.countDown();
        List<Long> latencies = new ArrayList<>();
        for (Future<Long> future : futures) {
            latencies.add(future.get(6, TimeUnit.SECONDS));
        }
        pool.shutdownNow();
        PressureTestReportUtil.printSummary(
                "controller.answer-form.concurrent",
                concurrency,
                concurrency,
                0,
                latencies,
                Map.of("sessionId", "session-http-3")
        );

        verify(interviewSessionFacade, times(concurrency))
                .answerInterviewQuestion(anyString(), any(InterviewAnswerReqDTO.class), anyLong());
    }

    @Test
    void shouldReturnConsistentPayloadForAnswerFormAndJsonEndpoints() throws Exception {
        when(interviewSessionFacade.answerInterviewQuestion(anyString(), any(InterviewAnswerReqDTO.class), anyLong()))
                .thenAnswer(invocation -> {
                    InterviewAnswerReqDTO req = invocation.getArgument(1);
                    return InterviewAnswerRespDTO.init()
                            .withCurrentQuestion(req.getQuestionNumber(), "Q-" + req.getQuestionNumber())
                            .withEvaluation(92, "consistent", 92)
                            .withNextQuestion("2", "next-q", false, 0)
                            .success();
                });

        InterviewAnswerReqDTO jsonReq = new InterviewAnswerReqDTO();
        jsonReq.setQuestionNumber("1");
        jsonReq.setAnswerContent("json-answer");
        jsonReq.setRequestId("json-rid");

        mockMvc.perform(MockMvcRequestBuilders.post("/api/xunzhi/v1/interview/sessions/session-http-4/interview/answer-json")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(jsonReq)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value(0))
                .andExpect(jsonPath("$.data.questionNumber").value("1"))
                .andExpect(jsonPath("$.data.questionContent").value("Q-1"))
                .andExpect(jsonPath("$.data.totalScore").value(92));

        mockMvc.perform(MockMvcRequestBuilders.post("/api/xunzhi/v1/interview/sessions/session-http-4/interview/answer")
                        .param("questionNumber", "1")
                        .param("answerContent", "form-answer")
                        .param("requestId", "form-rid"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value(0))
                .andExpect(jsonPath("$.data.questionNumber").value("1"))
                .andExpect(jsonPath("$.data.questionContent").value("Q-1"))
                .andExpect(jsonPath("$.data.totalScore").value(92));
    }

    @Test
    void shouldHandleConcurrentCurrentQuestionReadsWhileAnswering() throws Exception {
        when(interviewSessionFacade.answerInterviewQuestion(anyString(), any(InterviewAnswerReqDTO.class), anyLong()))
                .thenReturn(InterviewAnswerRespDTO.init()
                        .withCurrentQuestion("1", "Q-1")
                        .withEvaluation(80, "ok", 80)
                        .withNextQuestion("2", "Q-2", false, 0)
                        .success());
        when(interviewSessionFacade.getCurrentQuestion(anyString(), anyLong()))
                .thenReturn(InterviewAnswerRespDTO.init()
                        .withCurrentQuestion("1", "Q-1")
                        .withNextQuestion("1", "Q-1", false, 0)
                        .success());

        int concurrency = 30;
        ExecutorService pool = Executors.newFixedThreadPool(10);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<MixedTaskResult>> futures = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            int idx = i;
            Callable<MixedTaskResult> task = idx % 2 == 0
                    ? () -> {
                        start.await(2, TimeUnit.SECONDS);
                        long begin = System.nanoTime();
                        mockMvc.perform(MockMvcRequestBuilders.post("/api/xunzhi/v1/interview/sessions/session-http-5/interview/answer-json")
                                        .contentType(MediaType.APPLICATION_JSON)
                                        .content("{\"questionNumber\":\"1\",\"answerContent\":\"a\",\"requestId\":\"r-" + idx + "\"}"))
                                .andExpect(status().isOk())
                                .andExpect(jsonPath("$.code").value(0))
                                .andExpect(jsonPath("$.data.isSuccess").value(true));
                        return new MixedTaskResult("answer-json", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin));
                    }
                    : () -> {
                        start.await(2, TimeUnit.SECONDS);
                        long begin = System.nanoTime();
                        mockMvc.perform(MockMvcRequestBuilders.get("/api/xunzhi/v1/interview/sessions/session-http-5/current-question"))
                                .andExpect(status().isOk())
                                .andExpect(jsonPath("$.code").value(0))
                                .andExpect(jsonPath("$.data.questionNumber").value("1"));
                        return new MixedTaskResult("current-question", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin));
                    };
            futures.add(pool.submit(task));
        }
        start.countDown();
        List<Long> latencies = new ArrayList<>();
        int answerOps = 0;
        int currentOps = 0;
        for (Future<MixedTaskResult> future : futures) {
            MixedTaskResult result = future.get(6, TimeUnit.SECONDS);
            latencies.add(result.latencyMs());
            if ("answer-json".equals(result.operation())) {
                answerOps++;
            } else {
                currentOps++;
            }
        }
        pool.shutdownNow();
        PressureTestReportUtil.printSummary(
                "controller.current-question-with-answer.concurrent",
                concurrency,
                concurrency,
                0,
                latencies,
                Map.of(
                        "sessionId", "session-http-5",
                        "answerOps", answerOps,
                        "currentQuestionOps", currentOps
                )
        );
    }

    @Test
    void shouldHandleConcurrentFinishAndEndRequests() throws Exception {
        doNothing().when(interviewSessionFacade).finishSession(anyString(), anyLong());
        doNothing().when(interviewSessionFacade).endConversation(anyString(), anyLong());

        int concurrency = 20;
        ExecutorService pool = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger finishCalls = new AtomicInteger();
        AtomicInteger endCalls = new AtomicInteger();
        List<Future<Long>> futures = new ArrayList<>();
        for (int i = 0; i < concurrency; i++) {
            int idx = i;
            futures.add(pool.submit(() -> {
                start.await(2, TimeUnit.SECONDS);
                long begin = System.nanoTime();
                if (idx % 2 == 0) {
                    finishCalls.incrementAndGet();
                    mockMvc.perform(put("/api/xunzhi/v1/interview/sessions/session-http-6/finish"))
                            .andExpect(status().isOk())
                            .andExpect(jsonPath("$.code").value(0));
                } else {
                    endCalls.incrementAndGet();
                    mockMvc.perform(put("/api/xunzhi/v1/interview/conversations/session-http-6/end"))
                            .andExpect(status().isOk())
                            .andExpect(jsonPath("$.code").value(0));
                }
                return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            }));
        }
        start.countDown();
        List<Long> latencies = new ArrayList<>();
        for (Future<Long> future : futures) {
            latencies.add(future.get(6, TimeUnit.SECONDS));
        }
        pool.shutdownNow();
        PressureTestReportUtil.printSummary(
                "controller.finish-end.concurrent",
                concurrency,
                concurrency,
                0,
                latencies,
                Map.of(
                        "sessionId", "session-http-6",
                        "finishCalls", finishCalls.get(),
                        "endCalls", endCalls.get()
                )
        );

        verify(interviewSessionFacade, times(finishCalls.get())).finishSession("session-http-6", 5005L);
        verify(interviewSessionFacade, times(endCalls.get())).endConversation("session-http-6", 5005L);
    }

    private record MixedTaskResult(String operation, long latencyMs) {
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
            return new UserContext(5005L, "pressure-user");
        }
    }
}
