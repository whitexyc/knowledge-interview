package com.hewei.hzyjy.xunzhi.interview.api.io.resp;

import lombok.Data;

/**
 * 低分题响应 DTO（module-080 反向闭环）：字段与 InterviewTurnLog 逐一对齐。
 */
@Data
public class WeakPointRespDTO {

    /** 面试会话 ID */
    private String sessionId;

    /** 题号 */
    private String questionNumber;

    /** 题目内容 */
    private String questionContent;

    /** 本题得分 */
    private Integer score;

    /** 本题满分 */
    private Integer totalScore;

    /** 面试反馈 */
    private String feedback;

    /** 会话结束时间（毫秒时间戳字符串，可空） */
    private String endTime;
}
