package com.hewei.hzyjy.xunzhi.interview.api;

import com.hewei.hzyjy.xunzhi.common.convention.result.Result;
import com.hewei.hzyjy.xunzhi.common.convention.result.Results;
import com.hewei.hzyjy.xunzhi.interview.api.io.resp.WeakPointRespDTO;
import com.hewei.hzyjy.xunzhi.interview.service.WeakPointService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;

/**
 * 低分题查询端点（module-080 反向闭环）。免登录（SaToken notMatch）但校验
 * 内部 token（X-Internal-Token，fail-closed 403，token 来自环境变量）。
 */
@Slf4j
@RestController
@RequestMapping("/api/xunzhi/v1/interview")
@RequiredArgsConstructor
public class WeakPointController {

    private static final String INTERNAL_TOKEN_HEADER = "X-Internal-Token";

    private final WeakPointService weakPointService;

    @Value("${xunzhi-agent.security.internal-token:}")
    private String internalToken;

    /**
     * 查询低分题列表（内部只读）；token 校验失败 → 403。
     */
    @GetMapping("/weak-points")
    public ResponseEntity<Result<List<WeakPointRespDTO>>> listWeakPoints(
            @RequestParam(defaultValue = "60") int threshold,
            @RequestParam(defaultValue = "7") int days,
            @RequestParam(defaultValue = "50") int limit,
            @RequestHeader(value = INTERNAL_TOKEN_HEADER, required = false) String token) {
        if (!verifyInternalToken(token)) {
            log.warn("weak-points 内部 token 校验失败（fail-closed 403）");
            Result<List<WeakPointRespDTO>> error = new Result<>();
            error.setCode("403").setMessage("内部接口 token 无效");
            return ResponseEntity.status(403).body(error);
        }
        return ResponseEntity.ok(Results.success(weakPointService.listWeakPoints(threshold, days, limit)));
    }

    private boolean verifyInternalToken(String token) {
        if (internalToken == null || internalToken.isEmpty() || token == null) {
            return false;
        }
        // MessageDigest.isEqual：常量时间比较，防时序侧信道
        return MessageDigest.isEqual(
                internalToken.getBytes(StandardCharsets.UTF_8),
                token.getBytes(StandardCharsets.UTF_8));
    }
}
