"""纵向病例分析 Agent：面向医生复核的病例主流程（任务书 §二）。

本包是 ``app.case_analysis.service`` 的唯一上游调用方（Agent 侧）。纪律：

- 数据访问只走 ``app.case_analysis.service``，不碰 ``db.connect()``、不写 SQL；
  患者归属校验由 service/repository 层在 SQL 上强制，本包绕不过也不试图绕；
- 确定性计算只走 ``app.case_analysis.service`` 暴露的分析函数，本包不重新实现；
- 医学知识解释只在确有需要时委派给主 Agent 既有的知识子 Agent
  （``medical_agent.subagents.knowledge``），不复刻一套 RAG；
- 权限门（owner_user_id）、安全门（safety.check_emergency）、预算
  （BudgetState + config.node_budget）、日志追踪（AgentState.add_trace）
  全部复用主 Agent 既有机制；
- 任何时间关联都不得升级为因果结论；不输出诊断、处方或治疗决策；
  无法确认的内容标注"未知"，不自行补全。
"""

from .agent import run_case_analysis
from .outputs import (
    CaseAnalysisResult,
    CaseAnalysisStatus,
    DoctorQuestion,
    KnowledgeReference,
)

__all__ = [
    "CaseAnalysisResult",
    "CaseAnalysisStatus",
    "DoctorQuestion",
    "KnowledgeReference",
    "run_case_analysis",
]
