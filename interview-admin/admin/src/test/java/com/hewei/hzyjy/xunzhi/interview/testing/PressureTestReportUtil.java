package com.hewei.hzyjy.xunzhi.interview.testing;

import com.alibaba.fastjson2.JSON;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Helper utility for printing readable pressure test summaries.
 */
public final class PressureTestReportUtil {

    private PressureTestReportUtil() {
    }

    public static void printSummary(
            String scenario,
            int concurrency,
            int success,
            int failed,
            List<Long> latencyMs,
            Map<String, Object> extra) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("scenario", scenario);
        report.put("concurrency", concurrency);
        report.put("success", success);
        report.put("failed", failed);
        report.put("successRate", concurrency <= 0 ? 0D : roundPercent((double) success * 100D / concurrency));
        report.put("latency", buildLatencyStats(latencyMs));
        if (extra != null && !extra.isEmpty()) {
            report.put("extra", extra);
        }
        System.out.println("PRESSURE_REPORT " + JSON.toJSONString(report));
    }

    private static Map<String, Object> buildLatencyStats(List<Long> latencyMs) {
        Map<String, Object> latency = new LinkedHashMap<>();
        if (latencyMs == null || latencyMs.isEmpty()) {
            latency.put("unit", "ms");
            latency.put("count", 0);
            return latency;
        }
        List<Long> sorted = new ArrayList<>(latencyMs);
        Collections.sort(sorted);
        long sum = 0L;
        for (Long value : sorted) {
            if (value != null) {
                sum += value;
            }
        }
        latency.put("unit", "ms");
        latency.put("count", sorted.size());
        latency.put("min", sorted.get(0));
        latency.put("p50", percentile(sorted, 0.50D));
        latency.put("p95", percentile(sorted, 0.95D));
        latency.put("p99", percentile(sorted, 0.99D));
        latency.put("max", sorted.get(sorted.size() - 1));
        latency.put("avg", round2((double) sum / sorted.size()));
        return latency;
    }

    private static long percentile(List<Long> sorted, double quantile) {
        if (sorted == null || sorted.isEmpty()) {
            return 0L;
        }
        int index = (int) Math.ceil(quantile * sorted.size()) - 1;
        if (index < 0) {
            index = 0;
        }
        if (index >= sorted.size()) {
            index = sorted.size() - 1;
        }
        return sorted.get(index);
    }

    private static double round2(double value) {
        return Math.round(value * 100D) / 100D;
    }

    private static double roundPercent(double value) {
        return Math.round(value * 1000D) / 1000D;
    }
}

