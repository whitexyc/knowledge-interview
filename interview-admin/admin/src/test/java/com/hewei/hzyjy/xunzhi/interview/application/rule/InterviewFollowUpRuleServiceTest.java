package com.hewei.hzyjy.xunzhi.interview.application.rule;

import com.hewei.hzyjy.xunzhi.interview.config.InterviewRuleEngineConfiguration;
import com.yomahub.liteflow.core.FlowExecutor;
import com.yomahub.liteflow.flow.LiteflowResponse;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class InterviewFollowUpRuleServiceTest {

    @Test
    void shouldAlwaysUseDefaultChainAndUseLiteFlowDecision() {
        FlowExecutor flowExecutor = mock(FlowExecutor.class);
        InterviewRuleEngineConfiguration configuration = defaultConfiguration();
        InterviewFollowUpRuleService service = spy(new InterviewFollowUpRuleService(flowExecutor, configuration));

        LiteflowResponse response = mock(LiteflowResponse.class);
        when(response.isSuccess()).thenReturn(true);
        doAnswer(invocation -> {
            InterviewFollowUpRuleContext context = invocation.getArgument(1);
            context.markNeedFollowUp("AI_SUGGESTED", "follow-up suggested by ai result");
            return response;
        }).when(service).executeChain(anyString(), any(InterviewFollowUpRuleContext.class));

        InterviewFollowUpRuleContext context = new InterviewFollowUpRuleContext();
        context.setSessionId("s-1");
        context.setInterviewType("backend");
        context.setMaxFollowUp(2);
        context.setFollowUpCount(0);
        context.setFollowUpNeededFromAi(true);

        InterviewFollowUpRuleDecision decision = service.decide(context);

        assertTrue(decision.isNeedFollowUp());
        assertEquals("default_followup_chain", decision.getChainId());
        assertEquals("AI_SUGGESTED", decision.getReasonCode());
        assertFalse(decision.isFallback());
    }

    @Test
    void shouldFallbackWhenLiteFlowThrowsAndFailOpenEnabled() {
        FlowExecutor flowExecutor = mock(FlowExecutor.class);
        InterviewRuleEngineConfiguration configuration = defaultConfiguration();
        configuration.setFailOpen(true);
        InterviewFollowUpRuleService service = spy(new InterviewFollowUpRuleService(flowExecutor, configuration));
        doThrow(new RuntimeException("boom")).when(service).executeChain(anyString(), any(InterviewFollowUpRuleContext.class));

        InterviewFollowUpRuleContext context = new InterviewFollowUpRuleContext();
        context.setSessionId("s-2");
        context.setInterviewType("backend");
        context.setMaxFollowUp(2);
        context.setFollowUpCount(0);
        context.setFollowUpNeededFromAi(true);

        InterviewFollowUpRuleDecision decision = service.decide(context);

        assertTrue(decision.isFallback());
        assertTrue(decision.isNeedFollowUp());
        assertEquals("AI_SUGGESTED", decision.getReasonCode());
    }

    @Test
    void shouldUseDefaultChainWhenInterviewTypeMissing() {
        FlowExecutor flowExecutor = mock(FlowExecutor.class);
        InterviewRuleEngineConfiguration configuration = defaultConfiguration();
        InterviewFollowUpRuleService service = spy(new InterviewFollowUpRuleService(flowExecutor, configuration));

        LiteflowResponse response = mock(LiteflowResponse.class);
        when(response.isSuccess()).thenReturn(true);
        doAnswer(invocation -> {
            InterviewFollowUpRuleContext context = invocation.getArgument(1);
            context.markNoFollowUp("NO_RULE_TRIGGER", "no follow-up trigger matched");
            return response;
        }).when(service).executeChain(anyString(), any(InterviewFollowUpRuleContext.class));

        InterviewFollowUpRuleContext context = new InterviewFollowUpRuleContext();
        context.setSessionId("s-3");
        context.setInterviewType(null);
        context.setMaxFollowUp(2);
        context.setFollowUpCount(0);

        InterviewFollowUpRuleDecision decision = service.decide(context);

        assertEquals("default_followup_chain", decision.getChainId());
        assertFalse(decision.isNeedFollowUp());
        assertFalse(decision.isFallback());
    }

    @Test
    void shouldThrowWhenLiteFlowThrowsAndFailOpenDisabled() {
        FlowExecutor flowExecutor = mock(FlowExecutor.class);
        InterviewRuleEngineConfiguration configuration = defaultConfiguration();
        configuration.setFailOpen(false);
        InterviewFollowUpRuleService service = spy(new InterviewFollowUpRuleService(flowExecutor, configuration));
        doThrow(new RuntimeException("boom")).when(service).executeChain(anyString(), any(InterviewFollowUpRuleContext.class));

        InterviewFollowUpRuleContext context = new InterviewFollowUpRuleContext();
        context.setSessionId("s-4");
        context.setInterviewType("backend");
        context.setMaxFollowUp(2);
        context.setFollowUpCount(0);
        context.setFollowUpNeededFromAi(true);

        assertThrows(IllegalStateException.class, () -> service.decide(context));
    }

    private InterviewRuleEngineConfiguration defaultConfiguration() {
        InterviewRuleEngineConfiguration configuration = new InterviewRuleEngineConfiguration();
        configuration.setEnable(true);
        configuration.setFailOpen(true);
        configuration.setRuleVersion("v1.0.0");
        configuration.setDefaultChainId("default_followup_chain");
        configuration.setDefaultMaxFollowUp(2);
        configuration.setDefaultLowScoreThreshold(60);
        return configuration;
    }

}
