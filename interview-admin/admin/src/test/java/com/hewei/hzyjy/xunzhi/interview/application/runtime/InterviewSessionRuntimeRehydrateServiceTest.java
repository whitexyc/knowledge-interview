package com.hewei.hzyjy.xunzhi.interview.application.runtime;

import com.hewei.hzyjy.xunzhi.interview.dao.mapper.InterviewRecordMapper;
import com.hewei.hzyjy.xunzhi.interview.service.InterviewQuestionCacheService;
import com.hewei.hzyjy.xunzhi.interview.service.InterviewQuestionService;
import com.hewei.hzyjy.xunzhi.interview.service.InterviewSessionService;
import com.hewei.hzyjy.xunzhi.interview.service.model.InterviewRuntimeConfidence;
import com.hewei.hzyjy.xunzhi.interview.service.model.InterviewRuntimeLoadMode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.HashOperations;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class InterviewSessionRuntimeRehydrateServiceTest {

    @Mock
    private InterviewSessionRuntimeSnapshotService runtimeSnapshotService;

    @Mock
    private InterviewSessionRuntimeLockService runtimeLockService;

    @Mock
    private InterviewSessionService interviewSessionService;

    @Mock
    private InterviewQuestionService interviewQuestionService;

    @Mock
    private InterviewQuestionCacheService interviewQuestionCacheService;

    @Mock
    private InterviewRecordMapper interviewRecordMapper;

    @Mock
    private StringRedisTemplate stringRedisTemplate;

    @Mock
    private HashOperations<String, Object, Object> hashOperations;

    private InterviewSessionRuntimeRehydrateService service;

    @BeforeEach
    void setUp() {
        service = new InterviewSessionRuntimeRehydrateService(
                runtimeSnapshotService,
                runtimeLockService,
                interviewSessionService,
                interviewQuestionService,
                interviewQuestionCacheService,
                interviewRecordMapper,
                stringRedisTemplate
        );
    }

    @Test
    void shouldNotRebuildRuntimeWhenRehydrateLockCannotBeAcquired() throws InterruptedException {
        String sessionId = "session-1";

        when(stringRedisTemplate.opsForHash()).thenReturn(hashOperations);
        when(hashOperations.size(anyString())).thenReturn(0L);
        when(runtimeLockService.acquire(sessionId)).thenReturn(null);
        when(runtimeLockService.acquire(sessionId, 80L)).thenReturn(null);
        when(runtimeSnapshotService.findSnapshot(sessionId)).thenReturn(Optional.empty());

        InterviewSessionRuntimeView result = service.ensureRuntime(
                sessionId,
                InterviewRuntimeLoadMode.READ_WRITE_REQUIRED,
                InterviewRuntimeRehydrateScope.FULL_RUNTIME
        );

        assertAll(
                () -> assertEquals(InterviewRuntimeLoadMode.READ_WRITE_REQUIRED, result.getLoadMode()),
                () -> assertEquals(InterviewRuntimeRestoreSource.NONE, result.getRestoreSource()),
                () -> assertEquals(InterviewRuntimeConfidence.READ_ONLY, result.getConfidence()),
                () -> assertFalse(result.isCacheRebuilt()),
                () -> assertFalse(result.canWrite()),
                () -> assertNull(result.getSnapshot())
        );

        verify(runtimeLockService).acquire(sessionId);
        verify(runtimeLockService).acquire(sessionId, 80L);
        verify(runtimeLockService).release(isNull());
        verify(interviewRecordMapper, never()).selectOne(any());
        verify(interviewSessionService, never()).getBySessionId(anyString());
        verify(interviewQuestionService, never()).getBySessionId(anyString());
    }
}
