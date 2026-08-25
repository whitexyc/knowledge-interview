public class SqlMake {
    public static void main(String[] args) {
        long baseAutoIncrement = 1716344345346543L; // 起始自增值
        int tableCount = 16; // 生成16张表
        long incrementStep = 10000000000L; // 自增值步长

        for (int i = 0; i < tableCount; i++) {
            String tableName = "agent_message_" + i; // 动态表名
            long currentAutoIncrement = baseAutoIncrement + i * incrementStep;

            System.out.println(
                    "CREATE TABLE `" + tableName + "` (\n" +
                            "    `id`            bigint(20) NOT NULL AUTO_INCREMENT COMMENT 'ID',\n" +
                            "    `agent_id`      bigint(20) DEFAULT NULL COMMENT '智能体id',\n" +
                            "    `chat_message`  varchar(256) DEFAULT NULL COMMENT '智能体消息',\n" +
                            "    `user_id`       bigint(20) DEFAULT NULL COMMENT '用户id',\n" +
                            "    `user_message`  varchar(256) DEFAULT NULL COMMENT '用户消息',\n" +
                            "    `is_success`    tinyint(1) DEFAULT 1 COMMENT '是否完成对话',\n" +
                            "    `create_time`   datetime DEFAULT NULL COMMENT '创建时间',\n" +
                            "    `del_flag`      tinyint(1) DEFAULT NULL COMMENT '删除标识 0：未删除 1：已删除',\n" +
                            "    PRIMARY KEY (`id`),\n" +  // 注意此处有逗号
                            "    INDEX idx_user_id (`user_id`)\n" +  // 新增索引
                            ") ENGINE=InnoDB AUTO_INCREMENT=" + currentAutoIncrement + " DEFAULT CHARSET=utf8mb4;\n"
            );
        }
    }
}