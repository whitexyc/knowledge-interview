"""
创建 eval_runs 表（评估运行记录，module-019 §3.2）

用法（在 ai_service 目录下）:
  python create_eval_runs_table.py

幂等：CREATE TABLE IF NOT EXISTS，重复执行安全。
DDL 单源定义在 eval/golden_retrieval.py 的 EVAL_RUNS_DDL，
评估脚本在写库前也会自动执行同一 DDL（自愈建表）。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

from eval.golden.golden_retrieval import EVAL_RUNS_DDL, ensure_eval_runs_table


async def main() -> None:
    await ensure_eval_runs_table()
    print("✅ eval_runs 表已就绪（CREATE TABLE IF NOT EXISTS）")


if __name__ == "__main__":
    asyncio.run(main())
