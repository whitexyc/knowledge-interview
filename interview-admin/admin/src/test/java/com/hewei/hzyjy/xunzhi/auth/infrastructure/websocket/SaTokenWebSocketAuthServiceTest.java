package com.hewei.hzyjy.xunzhi.auth.infrastructure.websocket;

import com.hewei.hzyjy.xunzhi.auth.application.CurrentUserService;
import jakarta.websocket.Session;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class SaTokenWebSocketAuthServiceTest {

    @Test
    void shouldAuthorizeWhenTokenMapsToSameUserId() {
        CurrentUserService currentUserService = mock(CurrentUserService.class);
        Session session = mock(Session.class);
        when(session.getRequestParameterMap()).thenReturn(Map.of("token", List.of("Bearer token-1")));
        when(currentUserService.isValidToken("token-1")).thenReturn(true);
        when(currentUserService.getUsernameByToken("token-1")).thenReturn("alice");
        when(currentUserService.getUserIdByUsername("alice")).thenReturn(9L);

        SaTokenWebSocketAuthService authService = new SaTokenWebSocketAuthService(currentUserService);

        assertTrue(authService.isAuthorized(session, "9"));
    }

    @Test
    void shouldRejectWhenTokenIsMissingOrInvalid() {
        CurrentUserService currentUserService = mock(CurrentUserService.class);
        Session session = mock(Session.class);
        when(session.getRequestParameterMap()).thenReturn(Map.of("token", List.of("bad-token")));
        when(currentUserService.isValidToken("bad-token")).thenReturn(false);

        SaTokenWebSocketAuthService authService = new SaTokenWebSocketAuthService(currentUserService);

        assertFalse(authService.isAuthorized(session, "9"));
    }
}
