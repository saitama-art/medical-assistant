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

import hashlib
from collections.abc import Mapping

from .. import complex_handler, task_spec, tools
from ..state import (
    AgentError,
    AgentErrorKind,
    AgentState,
    EvidenceJudgement,
    NodeResult,
    NodeStatus,
    RecoveryState,
    RetrievalOutcome,
    SubagentResult,
    SubagentStatus,
    TerminationReason,
    ToolStatus,
    exception_result,
)
from ...rag import service
from ...rag.chain import get_llm
from .. import config as _cfg

# ==================== 配置 ====================
# 预算类常量的唯一事实来源已收敛至 config.py；此处导入以保持既有引用点不变。
# MAX_CONSECUTIVE_FAILURES / _MAX_RETRIES / _K_STEP 是子 Agent 内部策略，
# 单一出处即本模块，不收进 config.py。

TOOL_BUDGET_MS = _cfg.TOOL_BUDGET_MS
BUDGET_JUDGE_MS = _cfg.BUDGET_JUDGE_MS
BUDGET_GENERATE_MS = _cfg.BUDGET_GENERATE_MS
MAX_CONSECUTIVE_FAILURES = 2   # 检索连续失败达此值即终止
_MAX_RETRIES = 1               # 证据不足回边最多触发 1 次
_K_STEP = 2                    # 重试时 k 的增量（改写手段：扩大候选集）

RETRIEVAL_TOOL = tools.RETRIEVAL_TOOL

# 父 Agent 启动委派的最低剩余预算（min(配置, 剩余) < 此值时不启动子 Agent，
# 避免"启动了但必然预算失败"的假动作）。后续环节预算不足由内部守卫各自降级。
MIN_LAUNCH_BUDGET_MS = _cfg.MIN_LAUNCH_BUDGET_MS  # 8000

_node_budget = _cfg.node_budget


def run(
    spec: task_spec.TaskSpec,
    permit: task_spec.WorkerPermit,
    *,
    recovery: RecoveryState | None = None,
) -> SubagentResult:
    """在单次 WorkerPermit 内执行知识 Task Spec；无授权不执行工具。"""
    if not task_spec.consume_permit(spec, permit):
        return _rejected(spec)
    question = spec.inputs["question"]
    deadline = permit.deadline
    work = AgentState(raw_question=spec.constraints.raw_question, resolved_question=question)
    work.budget.deadline = deadline
    work.recovery = recovery if recovery is not None else RecoveryState()
    work.task_spec_id = spec.spec_id

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
            # k 已达上限时「扩大候选集」的 retries 无意义（k+_K_STEP 被钳回同值，
            # 同参数必被 tools 层 REPEAT 拦截），不再发第二次调用，直接判检索失败。
            if (work.consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                    or k >= service.MAX_TOP_K):
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
            return _result(
                work,
                status=SubagentStatus.EMPTY_RESULT,
                retrieval=work.retrieval,
                detail="empty_result",
            )

        # —— N08：来源提取 + 三段证据判定（第 1 段未校准禁用，第 3 段条件调 LLM） ——
        work.sources = service.extract_sources(work.retrieval.chunks)
        node_budget = _node_budget(work, BUDGET_JUDGE_MS)
        if node_budget is None:
            judgement_result = NodeResult(NodeStatus.BUDGET_BLOCKED)
        else:
            judgement_result = _judge_with_repair(work, node_budget)
        work.evidence = judgement_result.value or EvidenceJudgement()
        work.add_trace("N08", "judge", {
            "node_status": judgement_result.status.value,
            "sufficient": work.evidence.sufficient,
            "stage1": work.evidence.stage1_gate1,
            "stage2": work.evidence.stage2_quote_verified,
            "stage3": work.evidence.stage3_answerable,
            "reason": work.evidence.reason,
        }, node_budget_ms=node_budget)
        if judgement_result.status is not NodeStatus.OK:
            return _processing_failure(work, judgement_result)

        # —— N09：继续或结束（唯一回边，守卫见设计文档 2.2） ——
        if work.evidence.sufficient:
            break
        can_retry = (
            work.evidence.stage1_gate1 is not False  # 第 1 段没过，重试无意义
            and work.tool_rounds < tools.MAX_TOOL_ROUNDS
            and work.budget.remaining_ms() >= tools.MIN_USABLE_BUDGET_MS + TOOL_BUDGET_MS
            and retries < _MAX_RETRIES
            and k < service.MAX_TOP_K  # k 已达上限时重试只会同参数被 REPEAT 拦截
        )
        work.add_trace("N09", "decision", {
            "sufficient": False, "can_retry": can_retry, "tool_rounds": work.tool_rounds,
        })
        if not can_retry:
            return _result(
                work,
                status=SubagentStatus.INSUFFICIENT,
                sources=work.sources,
                evidence=work.evidence,
                retrieval=work.retrieval,
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
    return _result(
        work,
        status=SubagentStatus.OK,
        answer=answer,
        sources=work.sources,
        evidence=work.evidence,
        retrieval=work.retrieval,
    )


def _judge_once(
    work: AgentState, node_budget_ms: int, *, repair: bool = False,
) -> NodeResult:
    try:
        llm = get_llm(timeout=max(1.0, node_budget_ms / 1000))
    except Exception as exc:
        return exception_result(exc)
    result = service.judge_evidence(
        work.resolved_question, work.retrieval, llm=llm, repair=repair,
    )
    if result.status is NodeStatus.OK and not isinstance(
        result.value, EvidenceJudgement,
    ):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="invalid_judgement_payload")
    return result


def _judge_with_repair(work: AgentState, initial_budget_ms: int) -> NodeResult:
    """N08 无效输出最多修复一次，并与父节点共享全请求修复预算。"""
    result = _judge_once(work, initial_budget_ms, repair=False)
    decision = complex_handler.assess_node_result(work, "N08", result)
    if decision.action is not complex_handler.RecoveryAction.RETRY_NODE:
        return result
    retry_budget = _node_budget(work, BUDGET_JUDGE_MS)
    if retry_budget is None:
        return NodeResult(NodeStatus.BUDGET_BLOCKED)
    digest = hashlib.sha256(
        f"{work.resolved_question}\0{len(work.retrieval.chunks)}".encode("utf-8")
    ).hexdigest()
    if not complex_handler.record_retry(work, "N08", "repair", digest):
        return result
    work.add_trace("N08", "recovery", {
        "action": "retry", "repair_total": work.recovery.repair_total,
    }, node_budget_ms=retry_budget)
    return _judge_once(work, retry_budget, repair=True)


def _processing_failure(work: AgentState, result: NodeResult) -> SubagentResult:
    decision = complex_handler.assess_node_result(work, "N08", result)
    if decision.action is not complex_handler.RecoveryAction.TERMINATE:
        # 授权失败等边界不得重新进入循环；按无效输出终止。
        decision = complex_handler.RecoveryDecision(
            complex_handler.RecoveryAction.TERMINATE,
            TerminationReason.PROCESSING_FAILED,
            AgentError("N08", AgentErrorKind.MODEL_OUTPUT_INVALID, result.detail),
        )
    work.error = decision.error
    status = (
        SubagentStatus.BUDGET_EXHAUSTED
        if decision.termination is TerminationReason.BUDGET_EXHAUSTED
        else SubagentStatus.PROCESSING_FAILED
    )
    return _result(
        work,
        status=status,
        sources=work.sources,
        evidence=work.evidence,
        retrieval=work.retrieval,
        detail=decision.error.kind.value,
        error=decision.error,
    )


def _result(work: AgentState, status: SubagentStatus, **kwargs) -> SubagentResult:
    """统一补齐 Task Spec 身份、输入约束和执行记录。"""
    return SubagentResult(
        status=status,
        spec_id=work.task_spec_id,
        resolved_question=work.resolved_question,
        tool_rounds=work.tool_rounds,
        tool_calls=work.tool_calls,
        trace=work.trace,
        **kwargs,
    )


def _rejected(spec: task_spec.TaskSpec) -> SubagentResult:
    inputs = spec.inputs if isinstance(spec, task_spec.TaskSpec) and isinstance(
        spec.inputs, Mapping,
    ) else {}
    question = inputs.get("question", "")
    return SubagentResult(
        status=SubagentStatus.SAFETY_REJECTED,
        spec_id=spec.spec_id if isinstance(spec, task_spec.TaskSpec) else "",
        resolved_question=question if isinstance(question, str) else "",
        detail="worker_permit_invalid",
        error=AgentError("WORKER", AgentErrorKind.SAFETY_REJECTED, "worker_permit_invalid"),
    )


def _fail(work: AgentState, status: SubagentStatus, detail: str) -> SubagentResult:
    """失败出口统一封装：带上截至当前的记录与轨迹，父合并后仍可完整复盘。"""
    return _result(work, status=status, detail=detail)


__all__ = [
    "TOOL_BUDGET_MS",
    "BUDGET_JUDGE_MS",
    "BUDGET_GENERATE_MS",
    "MAX_CONSECUTIVE_FAILURES",
    "RETRIEVAL_TOOL",
    "MIN_LAUNCH_BUDGET_MS",
    "run",
]
