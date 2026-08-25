import javax.net.ssl.HttpsURLConnection;
import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class WorkflowRequest {

    public static void main(String[] args) throws Exception {
        String apiKey = "e8565c438f59b301616e0498a86ad95d";
        String apiSecret = "OGZkZGQ5ZDY0Yzc4MTllZWI3ZmU2MDU4";

        String urlString = "https://xingchen-api.xf-yun.com/workflow/v1/chat/completions";
        URL url = new URL(urlString);
        HttpsURLConnection conn = (HttpsURLConnection) url.openConnection();

        // 配置 SSL（如果需要跳过证书验证，可以配置 TrustManager）
//        conn.setSSLSocketFactory((HttpsURLConnection.getDefaultSSLSocketFactory()));
//        conn.setRequestMethod("POST");
//        conn.setConnectTimeout(120_000);
//        conn.setReadTimeout(120_000);

        // 设置 headers
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "text/event-stream");
        conn.setRequestProperty("Authorization", "Bearer " + apiKey + ":" + apiSecret);
        conn.setDoOutput(true);

        // 构造 JSON 数据
        String payload = """
                {
                    "flow_id": "7332665061114077184",
                    "parameters": {"AGENT_USER_INPUT": "你好,你是谁"},
                    "stream": false
                }
                """;

        // 发送请求体
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = payload.getBytes(StandardCharsets.UTF_8);
            os.write(input);
        }

        // 读取 stream 参数，决定是否分段读取
        boolean stream = false; // 你可以改成 true 看效果
        if (stream) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    System.out.println(line);
                }
            }
        } else {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                String response = br.readLine();
                System.out.println(response);
            }
        }

        conn.disconnect();
    }
}