package com.hewei.hzyjy.xunzhi.interview.service;

import com.hewei.hzyjy.xunzhi.interview.api.io.resp.WeakPointRespDTO;

import java.util.List;

/**
 * 低分题查询服务（module-080 反向闭环）：聚合 Redis/快照/归档轮次后返回低分题。
 */
public interface WeakPointService {

    /**
     * 查询最近 days 天内 score &lt; threshold 的低分题（endTime 倒序，最多 limit 条）。
     *
     * @param threshold 低分阈值
     * @param days      查询窗口（天，非正数按 7 处理）
     * @param limit     条数上限（非正数按 50 处理）
     * @return 低分题列表；无低分题时返回空列表（不抛异常）
     */
    List<WeakPointRespDTO> listWeakPoints(int threshold, int days, int limit);
}
