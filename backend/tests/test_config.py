"""T-配置收敛守卫：预算常量与 _node_budget 规则只允许 config.py 一个事实来源。

背景（docs/开发进度.md 2026-10-20）：此前 REQUEST_DEADLINE/各环节预算散在
orchestrator.py 与 subagents/knowledge.py 两个模块顶部，且同一条
_node_budget 分配规则因循环导入在两处各存一份。现收敛至
app/medical_agent/config.py，两个模块改为导入并保留引用名。
本测试钉住这一结构，防止回退成各存一份。
"""

import unittest

from app.medical_agent import config, orchestrator
from app.medical_agent.subagents import knowledge


class TestConfigSingleSource(unittest.TestCase):
    def test_budget_constants_have_single_source(self):
        self.assertIs(orchestrator.REQUEST_DEADLINE, config.REQUEST_DEADLINE)
        self.assertIs(orchestrator.BUDGET_FOLLOWUP_MS, config.BUDGET_FOLLOWUP_MS)
        self.assertIs(orchestrator.BUDGET_CLASSIFY_MS, config.BUDGET_CLASSIFY_MS)
        self.assertIs(knowledge.TOOL_BUDGET_MS, config.TOOL_BUDGET_MS)
        self.assertIs(knowledge.BUDGET_JUDGE_MS, config.BUDGET_JUDGE_MS)
        self.assertIs(knowledge.BUDGET_GENERATE_MS, config.BUDGET_GENERATE_MS)
        self.assertIs(knowledge.MIN_LAUNCH_BUDGET_MS, config.MIN_LAUNCH_BUDGET_MS)

    def test_node_budget_single_rule(self):
        # 父/子 Agent 的 _node_budget 必须是同一个函数对象，不再是两份拷贝
        self.assertIs(orchestrator._node_budget, config.node_budget)
        self.assertIs(knowledge._node_budget, config.node_budget)

    def test_min_launch_budget_derives_from_tool_budget(self):
        from app.medical_agent import tools
        self.assertEqual(
            config.MIN_LAUNCH_BUDGET_MS,
            tools.MIN_USABLE_BUDGET_MS + config.TOOL_BUDGET_MS,
        )


if __name__ == "__main__":
    unittest.main()
