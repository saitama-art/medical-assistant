"""真实 LLM 验收 harness（入库版，2026-10-20）。

把此前散落在仓库外 scratchpad 的 `_acceptance_rest.py`（21 条用例）收进仓库，
并补齐缺口：按测试项选择、结构化记录（输入/状态/耗时/检索信息/回答/引用/失败原因）、
密钥脱敏、API 不可用明确标记（blocked，不得伪造成通过）、FR-01 诊断样本。

用法（ai_agent 环境，cwd=backend/）：

    python -m acceptance --list                    # 列出全部验收项
    python -m acceptance --only T-03               # 只跑急症组
    python -m acceptance --only T-02b-1,T-12-二甲双胍   # 按编号单挑
    python -m acceptance --fr01 --out fr01.json    # 只跑 FR-01 诊断样本
    python -m acceptance                           # 全量（真实 LLM，注意额度）

设计约束：
- 优先复用业务入口（TestClient + 真实 /agent/chat 编排链路），不为测试另造 Agent；
- 不打印/不落盘任何密钥（redact() 兜底）；
- 失败项原因写进记录的 error 字段，pass 只能是显式断言结果；
- 与 unittest 单测的关系：单测（Mock/FakeLLM）管确定性行为，本 harness 管真实 LLM 验收。
"""

from .runner import redact, select_cases, run_report  # noqa: F401
