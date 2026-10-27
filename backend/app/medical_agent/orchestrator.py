"""主 Agent 编排器：N00–N12 的唯一控制流（设计文档 1.1 / 2.1 / 2.2）。

职责边界：本模块**唯一决定分支走向**，负责预算分配、trace 写入、终止原因落值、
文案组装（N11）与会话写回（N12）；不实现检索/判定/生成本身（那些在
``subagents/knowledge.py`` 与 ``rag/service.py``），不实现参数校验与拦截
（``tools.py``），不提供文案常量（``respond.py``）。

委派结构（2026-10-16 起）：KNOWLEDGE 路由的 N06–N10（检索→三段证据判定→生成）
委派给知识子 Agent 执行，父 Agent 只负责：
① 是否委派——由 N05 路由表写死，**不由模型决定**；
② 启动前预算检查（低于 MIN_LAUNCH_BUDGET_MS 不启动，防假动作）；
③ 校验 SubagentResult 并映射终止原因与固定文案；
④ 合并子 Agent 的工具记录与轨迹（重排 seq，trace 连续性不断）。
安全结构不委派：N01/N03 急症规则、N04/N05 分类路由、预算总账、N11/N12 永远在父 Agent。

本阶段与设计的两处已知偏差（记录在案，非疏漏）：

    1. triage 工具未注册（tools.py 注释已说明：设计全文未给出其参数 Schema），
       triage 路径直接返回 respond.TRIAGE_GUIDANCE 固定文案，0 次工具调用。
    2. N09 重试的"改写后 query"落地为：问题不变、检索条数 k 增大（k 不同即
       通过 tools 层的重复调用拦截，语义上是"扩大候选再判一次"）。
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid

from . import respond, router, safety, session, tools
from . import calculator, complex_handler, task_spec
from .state import (
    AgentError,
    AgentErrorKind,
    AgentState,
    ContextStatus,
    EmergencyHit,
    EmergencyLevel,
    NodeResult,
    NodeStatus,
    QuestionCategory,
    ResolutionResult,
    ResolutionStatus,
    Route,
    SubagentStatus,
    TerminationReason,
)
from .subagents import knowledge
from ..rag.chain import get_llm
from . import config as _cfg

# ==================== 配置 ====================
# 预算常量的唯一事实来源已收敛至 config.py；这里导入以保持
# orchestrator.XXX 引用点（含测试）不变。知识子 Agent 环节预算同处 config.py。

REQUEST_DEADLINE = _cfg.REQUEST_DEADLINE
BUDGET_FOLLOWUP_MS = _cfg.BUDGET_FOLLOWUP_MS
BUDGET_CLASSIFY_MS = _cfg.BUDGET_CLASSIFY_MS
_node_budget = _cfg.node_budget

_EMERGENCY_ROUTES = (Route.EMERGENCY,)


class AgentRequestError(ValueError):
    """N00 参数缺失。由接口层转 400，不进入编排（设计文档 2.1 N00）。消息只含
    机器可读标识，不含用户原文。"""


# ==================== 子 Agent 返回 → 终止原因/文案映射（父侧唯一解释点） ====================

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


# ==================== 入口 ====================


def run_agent(session_id: str, question: str) -> AgentState:
    """执行一次完整的 N00–N12 流程。不抛异常（AgentRequestError 除外）。"""
    state = AgentState(
        request_id=uuid.uuid4().hex[:12],
        session_id=session_id,
        raw_question=question,
    )

    # ---- N00 请求受理与会话装载 ----
    if not session_id or not isinstance(question, str) or not question.strip():
        raise AgentRequestError("missing_session_id_or_question")
    state.budget.deadline = time.monotonic() + REQUEST_DEADLINE
    got_lock, history, ctx_status = session.acquire(session_id)
    state.history = history
    state.context_status = ctx_status
    state.turn_id = sum(1 for role, _ in history if role == "user") + 1
    state.add_trace("N00", "decision", {
        "context_status": ctx_status.value,
        "history_msgs": len(history),
        "turn_id": state.turn_id,
    })

    # ---- N01 紧急风险检查（原始输入） ----
    try:
        state.emergency = safety.check_emergency(question)
    except safety.EmergencyRuleError:
        state.add_trace("N01", "error", {"reason": "emergency_rule_fault"})
        return _finish(state, got_lock, TerminationReason.EMERGENCY_RULE_FAULT,
                       respond.CONSERVATIVE_MEDICAL_NOTICE)
    state.add_trace("N01", "decision", {
        "level": state.emergency.level.value, "rule_id": state.emergency.rule_id,
    })
    if state.emergency.level is EmergencyLevel.STRONG:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)

    # ---- N02 上下文补全（显式与隐性追问使用同一依赖判断） ----
    state.resolved_question = question.strip()
    needs_context = (
        router.has_reference(state.resolved_question)
        or router.has_followup_lead(state.resolved_question)
    )
    if (
        state.history
        and state.context_status is not ContextStatus.MISSING
        and needs_context
    ):
        node_budget = _node_budget(state, BUDGET_FOLLOWUP_MS)
        if node_budget is None:
            resolution = NodeResult(NodeStatus.BUDGET_BLOCKED)
        else:
            resolution = _run_with_repair(
                state, "N02", BUDGET_FOLLOWUP_MS, node_budget,
                _resolve_reference,
                lambda st, result: complex_handler.assess_resolution(st, result),
            )
        decision = complex_handler.assess_resolution(state, resolution)
        state.add_trace("N02", "decision", {
            "node_status": resolution.status.value,
            "resolved": bool(
                resolution.value
                and resolution.value.status is ResolutionStatus.RESOLVED
            ),
            "resolved_len": len(state.resolved_question),
        }, node_budget_ms=node_budget)
        if decision.action is complex_handler.RecoveryAction.CONTINUE:
            state.resolved_question = resolution.value.question
            state.context_status = ContextStatus.OK
        else:
            if decision.termination is TerminationReason.CONTEXT_MISSING:
                state.context_status = ContextStatus.MISSING
            return _finish_recovery(state, got_lock, decision, "N02")

    # ---- N03 紧急风险检查（补全后），取两趟中更强者 ----
    try:
        second = safety.check_emergency(state.resolved_question)
    except safety.EmergencyRuleError:
        state.add_trace("N03", "error", {"reason": "emergency_rule_fault"})
        return _finish(state, got_lock, TerminationReason.EMERGENCY_RULE_FAULT,
                       respond.CONSERVATIVE_MEDICAL_NOTICE)
    state.emergency = _merge_hits(state.emergency, second)
    state.add_trace("N03", "decision", {"level": state.emergency.level.value})
    if state.emergency.level is EmergencyLevel.STRONG:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)

    # 弱紧急信号：回答顶部附加风险提示（不与知识库结论混写）
    if state.emergency.level is EmergencyLevel.WEAK:
        state.risk_notice = respond.RISK_NOTICE

    # 历史截断或锁降级时，含未解析指代的问题不得继续分类/检索。
    if needs_context and state.context_status in (
        ContextStatus.MISSING, ContextStatus.DEGRADED_NO_HISTORY,
    ):
        decision = complex_handler.RecoveryDecision(
            complex_handler.RecoveryAction.CLARIFY,
            TerminationReason.CONTEXT_MISSING,
        )
        return _finish_recovery(state, got_lock, decision, "N02")

    # ---- N04a 健康计算规则短路（确定性任务：0 LLM、0 工具）----
    # 急症判定（N01/N03）已先行，计算意图绝不绕过安全规则。
    calc_plan = calculator.detect(state.resolved_question)
    param_filled = False
    if (calc_plan is not None and calc_plan.status == "missing"
            and state.history
            and state.context_status is not ContextStatus.MISSING):
        # 只从现有上下文补全；补全后仍由确定性计算器解析和校验参数。
        filled_plan, fill_failure = _try_calc_param_fill(state)
        if fill_failure is not None:
            decision = complex_handler.assess_node_result(
                state, "N04a", fill_failure,
            )
            return _finish_recovery(state, got_lock, decision, "N04a")
        if filled_plan is not None:
            calc_plan = filled_plan
        param_filled = calc_plan.status == "ready"
    if calc_plan is not None:
        state.add_trace("N04a", "decision",
                        {"status": calc_plan.status, "kind": calc_plan.kind,
                         "param_filled": param_filled})
        if calc_plan.status == "missing":
            return _finish(state, got_lock, TerminationReason.CLARIFICATION_REQUIRED,
                           respond.CALC_NEED_HEIGHT_WEIGHT
                           if calc_plan.kind == "bmi" else respond.CALC_NEED_HEIGHT)
        calc_text, ok = _run_calc(calc_plan)
        if not ok:
            # 参数异常（如身高 300cm）：给追问文案，不返回虚假计算结果
            return _finish(state, got_lock, TerminationReason.CLARIFICATION_REQUIRED,
                           respond.CALC_NEED_HEIGHT_WEIGHT
                           if calc_plan.kind == "bmi" else respond.CALC_NEED_HEIGHT)
        note = f"\n\n{respond.CALC_PARAM_FROM_HISTORY}" if param_filled else ""
        if calc_plan.remaining and calc_plan.kind in ("bmi", "standard_weight"):
            # 组合问法：先完成确定性计算，剩余医学解释交知识子 Agent 检索
            state.resolved_question = calc_plan.remaining
            state.category = QuestionCategory.KNOWLEDGE
            state.route = Route.KNOWLEDGE
            state = _delegate_knowledge(state, got_lock)
            if state.answer_text:
                state.answer_text = calc_text + note + "\n\n---\n\n" + state.answer_text
            return state
        return _finish(state, got_lock, TerminationReason.COMPLETED,
                       calc_text + note + "\n\n" + respond.DISCLAIMER)

    # ---- N04 意图识别 ----
    node_budget = _node_budget(state, BUDGET_CLASSIFY_MS)
    if node_budget is None:
        classification = NodeResult(NodeStatus.BUDGET_BLOCKED)
    else:
        classification = _run_with_repair(
            state, "N04", BUDGET_CLASSIFY_MS, node_budget,
            _classify_once,
            lambda st, result: complex_handler.assess_intent(st, result),
        )
    decision = complex_handler.assess_intent(state, classification)
    if classification.value is not None:
        state.intent_assessment = classification.value
        state.category = classification.value.category
        state.category_confidence = classification.value.confidence
    state.add_trace("N04", "decision", {
        "node_status": classification.status.value,
        "category": state.category.value if state.category else None,
        "confidence": round(state.category_confidence, 3),
        "unresolved": (
            classification.value.unresolved.value
            if classification.value is not None else None
        ),
        "missing_input_count": (
            len(classification.value.missing_inputs)
            if classification.value is not None else 0
        ),
    }, node_budget_ms=node_budget)
    if decision.action is not complex_handler.RecoveryAction.CONTINUE:
        return _finish_recovery(state, got_lock, decision, "N04")

    # ---- N05 条件路由 ----
    state.route = router.decide_route(
        state.category, state.emergency.level,
        state.context_status, state.resolved_question,
    )
    state.add_trace("N05", "decision", {"route": state.route.value})

    if state.route is Route.EMERGENCY:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)
    if state.route is Route.CONTEXT_MISSING:
        return _finish(state, got_lock, TerminationReason.CONTEXT_MISSING,
                       respond.CONTEXT_MISSING_REPLY)
    if state.route is Route.CHITCHAT:
        return _finish(state, got_lock, TerminationReason.COMPLETED,
                       respond.CHITCHAT_REPLY)
    if state.route is Route.UNCLASSIFIABLE:
        return _finish(state, got_lock, TerminationReason.UNCLASSIFIED,
                       respond.UNCLASSIFIABLE_REPLY)
    if state.route is Route.TRIAGE:
        # 就医指引工具本阶段未注册（tools.py 注释）：直接给通用指引固定文案
        return _finish(state, got_lock, TerminationReason.COMPLETED,
                       respond.TRIAGE_GUIDANCE)
    # Route.KNOWLEDGE → 委派知识子 Agent 执行 N06–N10（下方）
    return _delegate_knowledge(state, got_lock)


# ==================== 知识路径：委派子 Agent（N06–N10） ====================


def _delegate_knowledge(state: AgentState, got_lock: bool) -> AgentState:
    """KNOWLEDGE 路由的委派入口：启动预算检查 → 委派 → 合并产出 → 映射终止。

    父 Agent 不介入子 Agent 的内部循环（重试/判定/生成），只消费
    SubagentResult 契约——这是"受控委派"与"模型自主循环"的分界。
    """
    launch_budget = _node_budget(state, knowledge.MIN_LAUNCH_BUDGET_MS)
    if launch_budget is None:
        # 剩余预算不足以启动一次有意义的检索：不启动子 Agent（防"假动作"）
        return _finish(state, got_lock, TerminationReason.BUDGET_EXHAUSTED,
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
        return _finish(
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
        return _finish(state, got_lock, TerminationReason.PROCESSING_FAILED,
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
        return _finish(state, got_lock, TerminationReason.PROCESSING_FAILED,
                       respond.PROCESSING_ERROR)

    if result.sources:
        state.sources = result.sources
    if result.error is not None:
        state.error = result.error

    # 父侧唯一解释点：status → 终止原因 + 固定文案
    reason, body = _SUBAGENT_TERMINATION[result.status]
    if result.status is SubagentStatus.OK:
        state.answer_text = result.answer
        return _finish(state, got_lock, reason, result.answer)
    return _finish(state, got_lock, reason, body)


# ==================== 内部工具 ====================


def _merge_hits(first: EmergencyHit, second: EmergencyHit) -> EmergencyHit:
    """取两趟检查中更强的一档（N01 原文 / N03 补全后）。"""
    order = {EmergencyLevel.NONE: 0, EmergencyLevel.WEAK: 1, EmergencyLevel.STRONG: 2}
    if order[second.level] > order[first.level]:
        return second
    return first


RESOLVE_SYSTEM_PROMPT = """你是对话上下文整理器。用户本轮的问题引用了此前的对话，
请把它改写成一个不依赖上下文、独立完整的问题。

要求：
1. 只输出 JSON 对象，不要输出解释、前后缀或 markdown 代码块；
2. 改写后的问题必须保留用户本轮的原始意图，补全被指代的药名、症状或事物；
3. **保留实体**：本轮问题中出现的或历史对话中已明确的具体实体（疾病名、症状、
   药品名、检查名）必须原样保留在改写后的问题里，只补全指代词所指的内容，
   不得把具体问题泛化成通用问题（如「腹泻要去医院的情况」不得泛化为
   「什么情况要去医院」）；
4. **保留否定词与限定语**：不、没有、别、未、无以及时间/人群限定
   （如「最近一周」「空腹」）必须原样保留，不得省略、反转或改写为反义；
5. 确实无法确定指代对象时，输出 {{"question": ""}}，不要猜测；
6. 输出格式：{{"question": "改写后的完整问题"}}"""

RESOLVE_USER_PROMPT = """【此前的对话】
{history}

【用户本轮的问题】
{question}

请输出改写后的完整问题。"""

RESOLVE_REPAIR_PROMPT = """

上一次输出未通过 JSON 或语义约束校验。请重新输出合法 JSON；question 必须保留原问题
的实体、否定词和限定语。无法可靠补全时只能输出空字符串，不得猜测。"""


def _try_calc_param_fill(
    state: AgentState,
) -> tuple[calculator.CalcPlan | None, NodeResult | None]:
    """返回（可用计算计划，技术失败）。业务上仍缺参时两者均为 None。"""
    node_budget = _node_budget(state, BUDGET_FOLLOWUP_MS)
    if node_budget is None:
        state.add_trace("N04a", "budget", {"param_fill": "skipped"})
        return None, NodeResult(NodeStatus.BUDGET_BLOCKED)
    resolution = _run_with_repair(
        state, "N04a", BUDGET_FOLLOWUP_MS, node_budget,
        _resolve_reference,
        lambda st, result: complex_handler.assess_resolution(st, result, "N04a"),
    )
    if resolution.status is not NodeStatus.OK:
        state.add_trace("N04a", "decision", {
            "param_fill": "node_failed", "node_status": resolution.status.value,
        })
        return None, resolution
    if resolution.value.status is ResolutionStatus.CONTEXT_REQUIRED:
        state.add_trace("N04a", "decision", {"param_fill": "context_required"})
        return None, None

    resolved = resolution.value.question
    plan = calculator.detect(resolved)
    if plan is None or plan.status != "ready":
        state.add_trace("N04a", "decision",
                        {"param_fill": "unresolved",
                         "resolved_status": plan.status if plan else "none"})
        return None, None
    state.resolved_question = resolved
    state.add_trace("N04a", "decision",
                    {"param_fill": "ok", "kind": plan.kind})
    return plan, None


def _dropped_negation(raw: str, resolved: str) -> bool:
    """改写产物是否丢弃了原句的否定词（S1-D2，程序校验先于 LLM 信任）。

    「没有胃肠道反应吗」被改写成「有胃肠道反应吗」是语义反转，医疗场景
    绝不能放行。校验不通过→调用方降级 missing 走"请重述"，不猜。
    保守取向：宁可误判漏改（多一次重述）也不放过反转（错误医疗语义）。
    """
    return any(m in raw and m not in resolved for m in _NEGATION_MARKERS)


_NEGATION_MARKERS = ("没有", "别", "不要", "未", "无")


def _classify_once(
    state: AgentState, node_budget_ms: int, *, repair: bool = False,
) -> NodeResult:
    try:
        llm = get_llm(timeout=max(1.0, node_budget_ms / 1000))
    except Exception as exc:
        return complex_handler.exception_result(exc)
    return router.classify(state.resolved_question, llm=llm, repair=repair)


def _run_with_repair(
    state: AgentState,
    node: str,
    config_budget_ms: int,
    initial_budget_ms: int,
    attempt,
    assessor,
) -> NodeResult:
    """执行节点一次；仅无效输出可申请全请求共享的一次修复。"""
    result = attempt(state, initial_budget_ms, repair=False)
    decision = assessor(state, result)
    if decision.action is not complex_handler.RecoveryAction.RETRY_NODE:
        return result

    retry_budget = _node_budget(state, config_budget_ms)
    if retry_budget is None:
        return NodeResult(NodeStatus.BUDGET_BLOCKED)
    digest = hashlib.sha256(state.resolved_question.encode("utf-8")).hexdigest()
    if not complex_handler.record_retry(state, node, "repair", digest):
        return result
    state.add_trace(node, "recovery", {
        "action": "retry", "repair_total": state.recovery.repair_total,
    }, node_budget_ms=retry_budget)
    return attempt(state, retry_budget, repair=True)


def _resolve_reference(
    state: AgentState, node_budget_ms: int, *, repair: bool = False,
) -> NodeResult[ResolutionResult]:
    """执行一次 N02 补全；合法无法补全与调用/输出失败分别返回。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate

        history_text = "\n".join(
            f"{'用户' if role == 'user' else '助手'}：{content}"
            for role, content in state.history
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESOLVE_SYSTEM_PROMPT + (RESOLVE_REPAIR_PROMPT if repair else "")),
            ("human", RESOLVE_USER_PROMPT),
        ])
        messages = prompt.format_messages(history=history_text, question=state.resolved_question)
        llm = get_llm(timeout=max(1.0, node_budget_ms / 1000))
        response = llm.invoke(messages)
    except Exception as exc:
        return complex_handler.exception_result(exc)

    content = getattr(response, "content", None)
    if not isinstance(content, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="content_not_text")
    import json
    import re

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")
    if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="question_invalid")

    resolved = payload["question"].strip()
    if _dropped_negation(state.resolved_question, resolved):
        return NodeResult(
            NodeStatus.INVALID_OUTPUT, detail="model_constraint_violation",
        )
    if not resolved:
        return NodeResult(
            NodeStatus.OK, ResolutionResult(ResolutionStatus.CONTEXT_REQUIRED),
        )
    return NodeResult(
        NodeStatus.OK, ResolutionResult(ResolutionStatus.RESOLVED, resolved),
    )


# 来源展示上限：同一文档的页码最多列 3 个、文档最多列 3 个（其余以「等」示意）。
# 背景：检索 k=10 的每个 chunk 都产出来源，同一文档常连续出现 7~9 条页码，
# 用户可读性差（2026-10-20 实测反馈）。
_SOURCE_MAX_PAGES_PER_DOC = 3
_SOURCE_MAX_DOCS = 3


def _format_sources(sources) -> list[str]:
    """把来源列表聚合成用户可读的展示串：按文档分组、页码升序去重、超限截断。

    返回每个文档一条（"文档 - 第2、7、12页"），保持检索相关度带来的首次出现顺序。
    """
    by_doc: dict[str, list[int]] = {}
    no_page: list[str] = []
    for s in sources:
        # 历史 Chroma 元数据可能是 Windows 路径；Linux 也需去掉盘符和目录。
        name = os.path.basename(s.document.replace("\\", "/")) if s.document else ""
        if not name:
            continue
        if s.page:
            pages = by_doc.setdefault(name, [])
            if s.page not in pages:
                pages.append(s.page)
        elif name not in no_page:
            no_page.append(name)

    out: list[str] = []
    for name, pages in by_doc.items():
        pages.sort()
        if len(pages) > _SOURCE_MAX_PAGES_PER_DOC:
            page_txt = "、".join(str(p) for p in pages[:_SOURCE_MAX_PAGES_PER_DOC]) + "页等"
        else:
            page_txt = "、".join(str(p) for p in pages) + "页"
        out.append(f"{name} - 第{page_txt}")
    for name in no_page:
        out.append(name)

    if len(out) > _SOURCE_MAX_DOCS:
        out = out[:_SOURCE_MAX_DOCS]
        out.append("等")
    return out


def _run_calc(plan) -> tuple[str, bool]:
    """执行计算工具并组装用户可读文案。ok=False 表示参数异常，由调用方给追问文案。"""
    try:
        if plan.kind == "bmi":
            r = calculator.calc_bmi(**plan.args)
            if r.error:
                return "", False
            h_m = round(r.extra["height_cm"] / 100, 2)
            return (
                "计算结果\n\n"
                f"- 公式：{r.formula}\n"
                f"- 代入：{r.extra['weight_kg']}kg ÷ ({h_m}m)²\n"
                f"- BMI = **{r.value}** {r.unit}\n"
                f"- 按中国成人参考分类（WS/T 428）：**{r.extra['label']}**\n\n"
                "注意：BMI 只是体重与身高的比值指标，不能直接等同于身体健康程度，"
                "也不能区分肌肉与脂肪的比例。"
            ), True
        if plan.kind == "standard_weight":
            r = calculator.calc_standard_weight(**plan.args)
            if r.error:
                return "", False
            low, high = r.extra["range"]
            return (
                "计算结果\n\n"
                f"- 公式：{r.formula}\n"
                f"- 代入：{r.extra['height_cm']}cm - 105\n"
                f"- 标准体重参考值 ≈ **{r.value}kg**（粗略参考区间 {low}~{high}kg）\n\n"
                "注意：该结果只是粗略估算，个体差异较大，不能替代个体化医学评估。"
            ), True
        if plan.kind == "convert":
            r = calculator.convert_unit(**plan.args)
            if r.error:
                return "", False
            v, fu = r.extra["from"]
            return f"换算结果：**{v} {fu} = {r.value} {r.unit}**", True
    except Exception:
        pass
    return "", False


def display_sources(state: AgentState) -> list[dict]:
    """接口层序列化用：聚合后的展示串，不含原始路径。"""
    return [{"document": t, "page": None} for t in _format_sources(state.sources)]


def _assemble(state: AgentState, body: str) -> str:
    """N11 回答组装：风险提示 / 无历史说明 → 正文 → 来源 → 免责声明，固定顺序。

    组装本身不抛异常；缺件按分支降级（来源为空则不拼来源段）。
    """
    parts: list[str] = []
    if state.risk_notice:
        parts.append(state.risk_notice)
    if state.context_status is ContextStatus.DEGRADED_NO_HISTORY:
        parts.append(respond.NO_HISTORY_NOTICE)
    parts.append(body)
    if state.termination_reason is TerminationReason.COMPLETED and state.sources:
        sources_text = "；".join(_format_sources(state.sources))
        if sources_text:
            parts.append(respond.SOURCE_HEADER.lstrip("\n") + sources_text)
    # 免责声明只跟在"含医学内容"的路径后：知识问答完成、紧急应答；
    # 闲聊/无法处理/重述等固定文案自身已完整，不再追加。
    show_disclaimer = (
        state.termination_reason is TerminationReason.EMERGENCY_HANDLED
        or (state.termination_reason is TerminationReason.COMPLETED
            and state.route is Route.KNOWLEDGE)
    )
    if show_disclaimer:
        parts.append(respond.DISCLAIMER)
    return "\n\n".join(parts)


def _finish_recovery(
    state: AgentState,
    got_lock: bool,
    decision: complex_handler.RecoveryDecision,
    node: str,
) -> AgentState:
    """把纯策略决策映射到唯一终态收口，不在此重试或调用 Worker。"""
    reason = decision.termination
    if reason is None:
        raise ValueError("recovery_decision_missing_termination")
    if decision.error is not None:
        state.error = decision.error
    if decision.action is complex_handler.RecoveryAction.CLARIFY:
        state.recovery.clarified = True
    if reason is TerminationReason.CONTEXT_MISSING:
        state.route = Route.CONTEXT_MISSING
    elif reason is TerminationReason.UNCLASSIFIED:
        state.route = Route.UNCLASSIFIABLE
    state.add_trace(node, "recovery", {
        "action": decision.action.value,
        "termination": reason.value,
        "error_kind": decision.error.kind.value if decision.error else None,
    })
    return _finish(state, got_lock, reason, respond.TERMINATION_TEXTS[reason])


def _finish(state: AgentState, got_lock: bool, reason: TerminationReason, body: str) -> AgentState:
    """终态收口：落终止原因、组装用户可见文本、写 trace、更新会话（N12）。

    会话更新失败不影响响应（session.complete 内部已兜底）。
    """
    state.termination_reason = reason
    state.answer_text = _assemble(state, body)
    state.add_trace("N12", "decision", {
        "termination": reason.value,
        "tool_rounds": state.tool_rounds,
        "tool_calls": len(state.tool_calls),
    })
    session.complete(
        state.session_id, state.raw_question, state.answer_text, got_lock=got_lock,
    )
    return state


__all__ = [
    "AgentRequestError",
    "run_agent",
    "display_sources",
    "REQUEST_DEADLINE",
]
