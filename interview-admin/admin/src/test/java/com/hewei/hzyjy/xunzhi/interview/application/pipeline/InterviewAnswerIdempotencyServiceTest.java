package com.hewei.hzyjy.xunzhi.interview.application.pipeline;

import com.hewei.hzyjy.xunzhi.interview.api.io.resp.InterviewAnswerRespDTO;
import com.hewei.hzyjy.xunzhi.interview.config.InterviewAnswerGuardConfiguration;
import com.hewei.hzyjy.xunzhi.interview.flow.answer.InterviewAnswerIdempotencyService;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class InterviewAnswerIdempotencyServiceTest {

    @Test
    void shouldTransitFromNewToProcessingToSucceededReplay() {
        InterviewAnswerIdempotencyService service = newService();

        InterviewAnswerIdempotencyService.TryStartResult first = service.tryStart("s-1", "r-1");
        assertEquals(InterviewAnswerIdempotencyService.TryStartStatus.NEW, first.getStatus());

        InterviewAnswerIdempotencyService.TryStartResult inFlight = service.tryStart("s-1", "r-1");
        assertEquals(InterviewAnswerIdempotencyService.TryStartStatus.PROCESSING, inFlight.getStatus());

        InterviewAnswerRespDTO success = InterviewAnswerRespDTO.init()
                .withCurrentQuestion("1", "question-1")
                .withEvaluation(85, "good", 85)
                .withNextQuestion("2", "question-2", false, 0)
                .success();
        service.markSucceeded("s-1", "r-1", success);

        InterviewAnswerIdempotencyService.TryStartResult replay = service.tryStart("s-1", "r-1");
        assertEquals(InterviewAnswerIdempotencyService.TryStartStatus.SUCCEEDED, replay.getStatus());
        assertNotNull(replay.getReplayResponse());
        assertTrue(Boolean.TRUE.equals(replay.getReplayResponse().getIsSuccess()));
        assertEquals(85, replay.getReplayResponse().getTotalScore());
        assertEquals("2", replay.getReplayResponse().getNextQuestionNumber());
    }

    @Test
    void shouldAllowRetryAfterClearProcessing() {
        InterviewAnswerIdempotencyService service = newService();

        InterviewAnswerIdempotencyService.TryStartResult first = service.tryStart("s-2", "r-2");
        assertEquals(InterviewAnswerIdempotencyService.TryStartStatus.NEW, first.getStatus());

        InterviewAnswerIdempotencyService.TryStartResult inFlight = service.tryStart("s-2", "r-2");
        assertEquals(InterviewAnswerIdempotencyService.TryStartStatus.PROCESSING, inFlight.getStatus());

        service.clearProcessing("s-2", "r-2");

        InterviewAnswerIdempotencyService.TryStartResult retry = service.tryStart("s-2", "r-2");
        assertEquals(InterviewAnswerIdempotencyService.TryStartStatus.NEW, retry.getStatus());
    }

    private InterviewAnswerIdempotencyService newService() {
        Map<String, String> redisStore = new ConcurrentHashMap<>();
        StringRedisTemplate stringRedisTemplate = mock(StringRedisTemplate.class);
        @SuppressWarnings("unchecked")
        ValueOperations<String, String> valueOperations = mock(ValueOperations.class);
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get(anyString())).thenAnswer(invocation -> redisStore.get(invocation.getArgument(0)));
        when(valueOperations.setIfAbsent(anyString(), anyString(), anyLong(), any(TimeUnit.class)))
                .thenAnswer(invocation -> redisStore.putIfAbsent(invocation.getArgument(0), invocation.getArgument(1)) == null);
        doAnswer(invocation -> {
            redisStore.put(invocation.getArgument(0), invocation.getArgument(1));
            return null;
        }).when(valueOperations).set(anyString(), anyString(), anyLong(), any(TimeUnit.class));
        doAnswer(invocation -> {
            redisStore.remove(invocation.getArgument(0));
            return Boolean.TRUE;
        }).when(stringRedisTemplate).delete(anyString());

        InterviewAnswerGuardConfiguration configuration = new InterviewAnswerGuardConfiguration();
        configuration.setProcessingExpireSeconds(120L);
        configuration.setReplayExpireHours(24L);
        return new InterviewAnswerIdempotencyService(stringRedisTemplate, configuration);
    }
}
