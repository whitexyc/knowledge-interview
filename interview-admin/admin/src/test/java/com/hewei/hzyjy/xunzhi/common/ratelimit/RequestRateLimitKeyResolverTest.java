package com.hewei.hzyjy.xunzhi.common.ratelimit;

import com.hewei.hzyjy.xunzhi.auth.application.CurrentUserService;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RequestRateLimitKeyResolverTest {

    @Test
    void shouldResolveUserKeyWhenTokenMapsToUsername() {
        CurrentUserService currentUserService = mock(CurrentUserService.class);
        RequestRateLimitKeyResolver resolver = new RequestRateLimitKeyResolver(currentUserService);
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer token-1");

        when(currentUserService.extractToken(request)).thenReturn("token-1");
        when(currentUserService.getUsernameByToken("token-1")).thenReturn("alice");

        assertEquals("user:alice", resolver.resolve(request));
    }

    @Test
    void shouldResolveAnonymousKeyFromForwardedIpWhenNoUserToken() {
        CurrentUserService currentUserService = mock(CurrentUserService.class);
        RequestRateLimitKeyResolver resolver = new RequestRateLimitKeyResolver(currentUserService);
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Forwarded-For", "10.0.0.8, 10.0.0.9");
        request.setRemoteAddr("127.0.0.1");

        when(currentUserService.extractToken(request)).thenReturn(null);

        assertEquals("ip:10.0.0.8", resolver.resolve(request));
    }
}
