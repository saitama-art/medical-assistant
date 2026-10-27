"""主 Agent 链路的全局预算与环节预算配置（设计 5.6）。

收敛说明（2026-10-20）：此前这些常量散在 orchestrator.py 与 subagents/knowledge.py
两个模块顶部，且 `_node_budget` 分配规则在两个模块各存一份（后者因循环导入
无法引用前者）。本模块为唯一事实来源，两处改为从这里导入，不再各存一份。

归属边界（按"不做大范围重构"约定，以下仍留在原模块、不收进来）：
- 检索条数 / 证据阈值 / 问题长度上限：rag/service.py（检索侧口径，N06 校验直接用）；
- 工具拦截阈值（MAX_TOOL_ROUNDS / MIN_USABLE_BUDGET_MS）：tools.py（工具层）；
- 安全词表参数：safety.py；会话参数：session.py；全部 Prompt 与文案：各模块原地。
"""

from . import tools

# ---- 全局 ----
REQUEST_DEADLINE = 45.0        # 整体硬上限（5.6 选项 A），单位：秒

# ---- 父 Agent 环节 ----
BUDGET_FOLLOWUP_MS = 6000      # N02 上下文补全
BUDGET_CLASSIFY_MS = 6000      # N04 意图识别
CATEGORY_CONFIDENCE_MIN = 0.6  # 低于此值只澄清，不改写为业务未知

# ---- 复杂处理层恢复预算 ----
MAX_MODEL_REPAIRS = 1          # N02/N04/N08 通用修复：单请求合计最多一次

# ---- 知识子 Agent 环节 ----
TOOL_BUDGET_MS = 5000          # 单次检索
BUDGET_JUDGE_MS = 8000         # 第 3 段模型判定
BUDGET_GENERATE_MS = 12000     # 回答生成

# 子 Agent 启动门槛：低于此剩余预算不启动（防假动作）
MIN_LAUNCH_BUDGET_MS = tools.MIN_USABLE_BUDGET_MS + TOOL_BUDGET_MS  # 8000


def node_budget(state, config_ms: int) -> int | None:
    """按 5.6 分配规则算本环节预算：min(配置预算, 剩余全局预算)；
    低于 tools.MIN_USABLE_BUDGET_MS 时返回 None（该环节不得启动）。
    父 Agent（orchestrator）与子 Agent（knowledge）共用这一条规则。"""
    remaining = state.budget.remaining_ms()
    node = min(config_ms, remaining)
    if node < tools.MIN_USABLE_BUDGET_MS:
        return None
    state.budget.node_budget_ms = node
    return node
