"""病例分析 Agent 的预算与限额配置。

复用主 Agent 的预算分配规则（medical_agent.config.node_budget，全局唯一
事实来源）与工具层拦截阈值；本模块只定义病例分析链路自己的全局时限、
环节预算与委派上限，不改写主 Agent 的任何常量。
"""

from __future__ import annotations

from ..medical_agent import config as _agent_cfg

CASE_ANALYSIS_DEADLINE_S = 60.0    # 单次病例分析整体硬上限（分析比单轮问答重）
BUDGET_NARRATIVE_MS = 12000        # 可选的 LLM 叙述生成环节预算
MAX_KNOWLEDGE_QUESTIONS = 2        # 单次分析最多委派给知识子 Agent 的问题数

# 直接复用主 Agent 的分配规则与启动门槛，保证两条链路的预算语义一致。
node_budget = _agent_cfg.node_budget
MIN_LAUNCH_BUDGET_MS = _agent_cfg.MIN_LAUNCH_BUDGET_MS
