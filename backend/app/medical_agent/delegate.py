"""知识路径委派（N06–N10）：从 orchestrator 拆出（2026-10-29 解耦）。

父 Agent 只负责：启动预算检查 → task_spec 授权 → 委派 → 合并产出 →
映射终止原因与固定文案（本模块是唯一解释点）。终态收口走 finalize.finish。
"""

from __future__ import annotations

from . import config as _cfg
from . import respond, task_spec
from .finalize import finish
from .state import (
    AgentError,
    AgentErrorKind,
    AgentState,
    SubagentStatus,
    TerminationReason,
)
from .subagents import knowledge

# 子 Agent 返回 → 终止原因/文案映射（父侧唯一解释点）
_SUBAGENT_TERMINATION: dict[SubagentStatus, tuple[TerminationReason, str]] = {
    SubagentStatus.OK: (TerminationReason.COMPLETED, ""),
    SubagentStatus.INSUFFICIENT: (TerminationReason.INSUFFICIENT_EVIDENCE, respond.EVIDENCE_INSUFFICIENT),
    SubagentStatus.EMPTY_RESULT: (TerminationReason.EMPTY_KNOWLEDGE_BASE, respond.EMPTY_RESULT),
    SubagentStatus.RETRIEVAL_FAILED: (TerminationReason.RETRIEVAL_FAILED, respond.RETRIEVAL_ERROR),
    SubagentStatus.GENERATION_FAILED: (TerminationReason.GENERATION_FAILED, respond.GENERATION_ERROR),
    SubagentStatus.BUDGET_EXHAUSTED: (TerminationReason.BUDGET_EXHAUSTED, respond.BUDGET_EXHAUSTED_REPLY),
    SubagentStatus.ARGUMENT_INVALID: (TerminationReason.ARGUMENT_INVALID, respond.INVALID_ARGS_REPLY),
    SubagentStatus.REPEATED_CALL_BLOCKED: (TerminationReason.REPEATED_CALL_BLOCKED, respond.REPEATED_CALL_REPLY),
    SubagentStatus.MAX_ROUNDS_REACHED: (TerminationReason.MAX_ROUNDS_REACHED, respond.ROUNDS_EXCEEDED_REPLY),
    SubagentStatus.PROCESSING_FAILED: (TerminationReason.PROCESSING_FAILED, respond.PROCESSING_ERROR),
    SubagentStatus.SAFETY_REJECTED: (TerminationReason.SAFETY_REJECTED, respond.SAFETY_REJECTED_REPLY),
}


def delegate_knowledge(state: AgentState, got_lock: bool) -> AgentState:
    """KNOWLEDGE 路由的委派入口：启动预算检查 → 委派 → 合并产出 → 映射终止。

    父 Agent 不介入子 Agent 的内部循环（重试/判定/生成），只消费
    SubagentResult 契约——这是"受控委派"与"模型自主循环"的分界。
    """
    launch_budget = _cfg.node_budget(state, knowledge.MIN_LAUNCH_BUDGET_MS)
    if launch_budget is None:
        # 剩余预算不足以启动一次有意义的检索：不启动子 Agent（防"假动作"）
        return finish(state, got_lock, TerminationReason.BUDGET_EXHAUSTED,
                      respond.BUDGET_EXHAUSTED_REPLY)
    spec = task_spec.build_task_spec(state)
    authorization = task_spec.authorize_task_spec(spec, state)
    if not authorization.ok:
        blocked_budget = authorization.detail == "budget_blocked"
        state.error = AgentError(
            "TASK_SPEC",
            AgentErrorKind.BUDGET_BLOCKED if blocked_budget else AgentErrorKind.SAFETY_REJECTED,
            authorization.detail,
        )
        state.add_trace("N06", "decision", {
            "delegate": False, "reason": authorization.detail,
        })
        return finish(
            state, got_lock,
            TerminationReason.BUDGET_EXHAUSTED if blocked_budget else TerminationReason.SAFETY_REJECTED,
            respond.BUDGET_EXHAUSTED_REPLY if blocked_budget else respond.SAFETY_REJECTED_REPLY,
        )

    state.add_trace("N06", "decision", {
        "delegate": "knowledge_subagent",
        "spec_id": spec.spec_id,
        "tool_whitelist": list(spec.constraints.allowed_tools),
        "question_len": len(state.resolved_question),
    }, node_budget_ms=launch_budget)
    state.recovery.worker_launched = True
    try:
        result = knowledge.run(spec, authorization.permit, recovery=state.recovery)
    except Exception as exc:
        state.error = AgentError(
            "WORKER", AgentErrorKind.WORKER_RESULT_INVALID,
            f"worker_exception:{type(exc).__name__}",
        )
        return finish(state, got_lock, TerminationReason.PROCESSING_FAILED,
                      respond.PROCESSING_ERROR)

    # 合并子 Agent 产出：seq 重排后并入父状态，trace 连续性不断（T-15）
    state.tool_rounds += result.tool_rounds
    for record in result.tool_calls:
        record.seq = len(state.tool_calls) + 1
        state.tool_calls.append(record)
    for trace in result.trace:
        trace.seq = len(state.trace) + 1
        state.trace.append(trace)
    if result.evidence is not None:
        state.evidence = result.evidence
    if result.retrieval is not None:
        state.retrieval = result.retrieval

    validation = task_spec.validate_worker_result(spec, result)
    if not validation.ok:
        state.error = AgentError(
            "WORKER", AgentErrorKind.WORKER_RESULT_INVALID, validation.detail,
        )
        state.add_trace("N11", "worker_result_invalid", {
            "reason": validation.detail,
        })
        return finish(state, got_lock, TerminationReason.PROCESSING_FAILED,
                      respond.PROCESSING_ERROR)

    if result.sources:
        state.sources = result.sources
    if result.error is not None:
        state.error = result.error

    # 父侧唯一解释点：status → 终止原因 + 固定文案
    reason, body = _SUBAGENT_TERMINATION[result.status]
    if result.status is SubagentStatus.OK:
        state.answer_text = result.answer
        return finish(state, got_lock, reason, result.answer)
    return finish(state, got_lock, reason, body)


__all__ = ["delegate_knowledge"]
