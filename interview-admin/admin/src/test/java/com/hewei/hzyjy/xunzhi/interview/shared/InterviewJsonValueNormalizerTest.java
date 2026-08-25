package com.hewei.hzyjy.xunzhi.interview.shared;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class InterviewJsonValueNormalizerTest {

    @Test
    void shouldNormalizeSelfReferentialJsonObjectWithoutStackOverflow() {
        JSONObject root = new JSONObject();
        JSONArray questions = new JSONArray();
        questions.add("Tell me about Spring transactions");
        root.put("questions", questions);
        root.put("self", root);

        Map<String, Object> normalized = InterviewJsonValueNormalizer.asMap(root);

        assertNotNull(normalized);
        assertEquals(List.of("Tell me about Spring transactions"), normalized.get("questions"));
        assertNull(normalized.get("self"));
    }
}
