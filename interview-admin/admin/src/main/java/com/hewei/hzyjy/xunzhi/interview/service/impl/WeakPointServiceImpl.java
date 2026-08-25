package com.hewei.hzyjy.xunzhi.interview.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.hewei.hzyjy.xunzhi.interview.api.io.resp.WeakPointRespDTO;
import com.hewei.hzyjy.xunzhi.interview.application.runtime.InterviewSessionRuntimeSnapshotService;
import com.hewei.hzyjy.xunzhi.interview.dao.entity.InterviewRecordDO;
import com.hewei.hzyjy.xunzhi.interview.dao.mapper.InterviewRecordMapper;
import com.hewei.hzyjy.xunzhi.interview.service.WeakPointService;
import com.hewei.hzyjy.xunzhi.interview.service.model.InterviewTurnLog;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * 低分题查询服务实现（module-080 反向闭环）。数据源：interview_record（FINISHED/
 * EVALUATED）+ 每题轮次聚合（复用 loadPersistedTurns）。单轮异常 fail-open。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class WeakPointServiceImpl implements WeakPointService {

    private static final int DEFAULT_DAYS = 7;
    private static final int DEFAULT_LIMIT = 50;

    private final InterviewRecordMapper interviewRecordMapper;
    private final InterviewSessionRuntimeSnapshotService runtimeSnapshotService;

    @Override
    public List<WeakPointRespDTO> listWeakPoints(int threshold, int days, int limit) {
        int effectiveDays = days > 0 ? days : DEFAULT_DAYS;
        int effectiveLimit = limit > 0 ? limit : DEFAULT_LIMIT;
        List<WeakPointRespDTO> result = new ArrayList<>();
        for (InterviewRecordDO record : loadFinishedRecords(effectiveDays)) {
            for (InterviewTurnLog turn : runtimeSnapshotService.loadPersistedTurns(record.getSessionId())) {
                if (!isLowScore(turn, threshold)) {
                    continue;
                }
                result.add(toDto(record, turn));
                if (result.size() >= effectiveLimit) {
                    return result;
                }
            }
        }
        return result;
    }

    private List<InterviewRecordDO> loadFinishedRecords(int days) {
        Date since = new Date(System.currentTimeMillis() - days * 86400_000L);
        return interviewRecordMapper.selectList(new LambdaQueryWrapper<InterviewRecordDO>()
                .in(InterviewRecordDO::getInterviewStatus, "FINISHED", "EVALUATED")
                .ge(InterviewRecordDO::getEndTime, since)
                .eq(InterviewRecordDO::getDelFlag, 0)
                .orderByDesc(InterviewRecordDO::getEndTime));
    }

    private boolean isLowScore(InterviewTurnLog turn, int threshold) {
        return turn != null && turn.getScore() != null
                && turn.getScore() < threshold
                && !Boolean.TRUE.equals(turn.getIsFollowUp());
    }

    private WeakPointRespDTO toDto(InterviewRecordDO record, InterviewTurnLog turn) {
        WeakPointRespDTO dto = new WeakPointRespDTO();
        dto.setSessionId(record.getSessionId());
        dto.setQuestionNumber(turn.getQuestionNumber());
        dto.setQuestionContent(turn.getQuestionContent());
        dto.setScore(turn.getScore());
        dto.setTotalScore(turn.getTotalScore());
        dto.setFeedback(turn.getFeedback());
        dto.setEndTime(record.getEndTime() == null ? null : String.valueOf(record.getEndTime().getTime()));
        return dto;
    }
}
