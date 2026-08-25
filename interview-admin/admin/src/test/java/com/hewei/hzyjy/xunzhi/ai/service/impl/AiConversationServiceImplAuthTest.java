package com.hewei.hzyjy.xunzhi.ai.service.impl;

import com.hewei.hzyjy.xunzhi.ai.dao.entity.AiConversation;
import com.hewei.hzyjy.xunzhi.ai.dao.repository.AiConversationRepository;
import com.hewei.hzyjy.xunzhi.ai.service.AiPropertiesService;
import com.hewei.hzyjy.xunzhi.common.convention.exception.ClientException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AiConversationServiceImplAuthTest {

    @Mock
    private AiConversationRepository aiConversationRepository;

    @Mock
    private AiPropertiesService aiPropertiesService;

    private AiConversationServiceImpl aiConversationService;

    @BeforeEach
    void setUp() {
        aiConversationService = new AiConversationServiceImpl(aiConversationRepository, aiPropertiesService);
    }

    @Test
    void requireOwnedConversation_ShouldThrow_WhenUsernameMismatch() {
        AiConversation conversation = new AiConversation();
        conversation.setSessionId("ai-session-1");
        conversation.setUsername("owner");

        when(aiConversationRepository.findBySessionIdAndDelFlag("ai-session-1", 0))
                .thenReturn(Optional.of(conversation));

        assertThrows(
                ClientException.class,
                () -> aiConversationService.requireOwnedConversation("ai-session-1", "intruder")
        );
    }

    @Test
    void requireOwnedConversation_ShouldPass_WhenUsernameMatches() {
        AiConversation conversation = new AiConversation();
        conversation.setSessionId("ai-session-2");
        conversation.setUsername("owner");

        when(aiConversationRepository.findBySessionIdAndDelFlag("ai-session-2", 0))
                .thenReturn(Optional.of(conversation));

        assertDoesNotThrow(() -> aiConversationService.requireOwnedConversation("ai-session-2", "owner"));
    }
}
