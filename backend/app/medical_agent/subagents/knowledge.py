"""知识子 Agent：承接父 Agent 委派的知识问答任务（原编排器 N06–N10 循环）。

委派契约（父分析 → 创建子任务 → 子独立执行 → 结构化返回 → 父校验）：

    任务目标        把 question 用知识库证据回答出来
    必要上下文      仅问题本身 + 父分配的全局 deadline（不传会话历史——上下文隔离）
    允许的工具      白名单 ``["search_medical_kb"]``（复用 tools 注册表/校验/拦截）
    完成标准        三段证据判定 sufficient 且生成成功
    结构化返回      ``SubagentResult{status, answer, sources, evidence, retrieval,
                    tool_rounds, tool_calls, trace, detail}``

职责边界：父 Agent 决定**是否委派**（路由表写死，不由模型决定）、做启动预算检查、
校验返回值并映射终止原因与文案；本子 Agent 不复用父的 AgentState——在本地 work
状态上执行，全部产出经返回值交还。本函数不抛异常：一切失败归一为
``SubagentResult.status`` + 机器可读 detail。
"""

from __future__ import annotations

from .. import tools
from ..state import (
    AgentState,
    EvidenceJudgement,
    RetrievalOutcome,
    SubagentResult,
    SubagentStatus,
    ToolStatus,
)
from ...rag import service
from ...rag.chain import get_llm

# ==================== 配置（设计文档 5.8，随子 Agent 收敛；最终进 config.py） ====================

TOOL_BUDGET_MS = 5000          # 单次检索
BUDGET_JUDGE_MS = 8000         # 第 3 段模型判定
BUDGET_GENERATE_MS = 12000     # 回答生成
MAX_CONSECUTIVE_FAILURES = 2   # 检索连续失败达此值即终止
_MAX_RETRIES = 1               # 证据不足回边最多触发 1 次
_K_STEP = 2                    # 重试时 k 的增量（改写手段：扩大候选集）

RETRIEVAL_TOOL = "search_medical_kb"

# 父 Agent 启动委派的最低剩余预算（min(配置, 剩余) < 此值时不启动子 Agent，
# 避免"启动了但必然预算失败"的假动作）。后续环节预算不足由内部守卫各自降级。
MIN_LAUNCH_BUDGET_MS = tools.MIN_USABLE_BUDGET_MS + TOOL_BUDGET_MS  # 8000


def run(question: str, *, deadline: float) -> SubagentResult:
    """执行委派的知识问答任务。

    Args:
        question: 父 Agent 补全后的独立问题（resolved_question）。
        deadline: 父 Agent 的全局 deadline（time.monotonic() 绝对时刻），子 Agent
            与之共享，不另设预算——父的硬上限对子同样硬。

    Returns:
        SubagentResult。status 为 OK 时 answer/sources 有效；其余 status 由父映射
        为终止原因与固定文案。
    """
    work = AgentState(raw_question=question, resolved_question=question)
    work.budget.deadline = deadline

    retries = 0
    k = service.DEFAULT_TOP_K

    while True:
        # —— N06/N07：参数校验 + 工具执行（校验与拦截在 tools 层完成） ——
        node_budget = _node_budget(work, TOOL_BUDGET_MS)
        invocation = tools.call_tool(
            RETRIEVAL_TOOL,
            {"question": work.resolved_question, "k": k},
            state=work,
            # node_budget 为 None 时传剩余量，由 tools 层归一成 BLOCKED_BUDGET
            budget_ms=node_budget if node_budget is not None else work.budget.remaining_ms(),
        )
        work.tool_calls.append(invocation.record)
        work.add_trace("N07", "tool_call", {
            "tool": invocation.record.tool,
            "status": invocation.record.status.value,
            "round": invocation.record.round,
            "detail": invocation.detail,
        }, latency_ms=invocation.record.latency_ms,
           node_budget_ms=invocation.record.budget_ms)

        status = invocation.status
        if status is ToolStatus.INVALID_ARGS:
            return _fail(work, SubagentStatus.ARGUMENT_INVALID, invocation.detail)
        if status is ToolStatus.BLOCKED_REPEAT:
            return _fail(work, SubagentStatus.REPEATED_CALL_BLOCKED, invocation.detail)
        if status is ToolStatus.BLOCKED_MAX_ROUNDS:
            return _fail(work, SubagentStatus.MAX_ROUNDS_REACHED, invocation.detail)
        if status is ToolStatus.BLOCKED_BUDGET:
            return _fail(work, SubagentStatus.BUDGET_EXHAUSTED, invocation.detail)
        if status in (ToolStatus.TIMEOUT, ToolStatus.RUNTIME_ERROR):
            work.consecutive_failures += 1
            if work.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                return _fail(work, SubagentStatus.RETRIEVAL_FAILED, invocation.detail)
            retries += 1
            k = min(service.MAX_TOP_K, k + _K_STEP)
            continue

        # ToolStatus.OK：检索实际执行完成，计一轮
        work.tool_rounds += 1
        work.consecutive_failures = 0
        work.retrieval = invocation.retrieval  # type: ignore[assignment]

        # EMPTY_RESULT 是成功执行的"返回 0 条"，与检索失败是不同语义
        if work.retrieval.outcome is RetrievalOutcome.EMPTY_RESULT:
            return SubagentResult(
                status=SubagentStatus.EMPTY_RESULT,
                retrieval=work.retrieval,
                tool_rounds=work.tool_rounds,
                tool_calls=work.tool_calls,
                trace=work.trace,
                detail="empty_result",
            )

        # —— N08：来源提取 + 三段证据判定（第 1 段未校准禁用，第 3 段条件调 LLM） ——
        work.sources = service.extract_sources(work.retrieval.chunks)
        node_budget = _node_budget(work, BUDGET_JUDGE_MS)
        if node_budget is None:
            work.evidence = EvidenceJudgement(reason="judge_skipped_budget")
            work.add_trace("N08", "budget", {"skipped": True})
        else:
            llm = get_llm(timeout=max(1.0, node_budget / 1000))
            work.evidence = service.judge_evidence(
                work.resolved_question, work.retrieval, llm=llm,
            )
            work.add_trace("N08", "judge", {
                "sufficient": work.evidence.sufficient,
                "stage1": work.evidence.stage1_gate1,
                "stage2": work.evidence.stage2_quote_verified,
                "stage3": work.evidence.stage3_answerable,
                "reason": work.evidence.reason,
            }, node_budget_ms=node_budget)

        # —— N09：继续或结束（唯一回边，守卫见设计文档 2.2） ——
        if work.evidence.sufficient:
            break
        can_retry = (
            work.evidence.stage1_gate1 is not False  # 第 1 段没过，重试无意义
            and work.tool_rounds < tools.MAX_TOOL_ROUNDS
            and work.budget.remaining_ms() >= tools.MIN_USABLE_BUDGET_MS + TOOL_BUDGET_MS
            and retries < _MAX_RETRIES
        )
        work.add_trace("N09", "decision", {
            "sufficient": False, "can_retry": can_retry, "tool_rounds": work.tool_rounds,
        })
        if not can_retry:
            return SubagentResult(
                status=SubagentStatus.INSUFFICIENT,
                sources=work.sources,
                evidence=work.evidence,
                retrieval=work.retrieval,
                tool_rounds=work.tool_rounds,
                tool_calls=work.tool_calls,
                trace=work.trace,
                detail=work.evidence.reason,
            )
        retries += 1
        k = min(service.MAX_TOP_K, k + _K_STEP)  # 改写后的 query：扩大候选集

    # —— N10：回答生成（只有证据充分才会到达这里） ——
    node_budget = _node_budget(work, BUDGET_GENERATE_MS)
    if node_budget is None:
        return _fail(work, SubagentStatus.BUDGET_EXHAUSTED, "generate_skipped_budget")
    try:
        llm = get_llm(timeout=max(1.0, node_budget / 1000))
        answer = service.generate_answer(
            work.resolved_question, work.retrieval.chunks, llm=llm,
        )
    except service.RagServiceError as exc:
        work.add_trace("N10", "error", {"detail": str(exc)}, node_budget_ms=node_budget)
        return _fail(work, SubagentStatus.GENERATION_FAILED, str(exc))
    work.answer_text = answer
    return SubagentResult(
        status=SubagentStatus.OK,
        answer=answer,
        sources=work.sources,
        evidence=work.evidence,
        retrieval=work.retrieval,
        tool_rounds=work.tool_rounds,
        tool_calls=work.tool_calls,
        trace=work.trace,
    )


def _fail(work: AgentState, status: SubagentStatus, detail: str) -> SubagentResult:
    """失败出口统一封装：带上截至当前的记录与轨迹，父合并后仍可完整复盘。"""
    return SubagentResult(
        status=status,
        tool_rounds=work.tool_rounds,
        tool_calls=work.tool_calls,
        trace=work.trace,
        detail=detail,
    )


def _node_budget(work: AgentState, config_ms: int) -> int | None:
    """min(配置预算, 剩余全局预算)；低于 MIN_USABLE_BUDGET 返回 None（不启动该环节）。

    与 orchestrator._node_budget 同一条分配规则；因 orchestrator 导入本模块
    （本模块不能再导入 orchestrator），暂时各存一份，收敛 config.py 时一并收掉。
    """
    remaining = work.budget.remaining_ms()
    node = min(config_ms, remaining)
    if node < tools.MIN_USABLE_BUDGET_MS:
        return None
    work.budget.node_budget_ms = node
    return node


__all__ = [
    "TOOL_BUDGET_MS",
    "BUDGET_JUDGE_MS",
    "BUDGET_GENERATE_MS",
    "MAX_CONSECUTIVE_FAILURES",
    "RETRIEVAL_TOOL",
    "MIN_LAUNCH_BUDGET_MS",
    "run",
]
