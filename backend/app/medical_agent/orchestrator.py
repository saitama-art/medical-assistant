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
import time
import uuid

from . import daily, delegate, resolver
from . import respond, router, safety, session
from . import calculator, complex_handler, context_engineering
from .finalize import (
    assemble as _assemble,
    display_sources,
    finish as _finish,
    finish_recovery as _finish_recovery,
    format_sources as _format_sources,
)
from .state import (
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
    TerminationReason,
)
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
    context_engineering.begin_turn(state)
    state.turn_id = sum(1 for role, _ in history if role == "user") + 1
    state.add_trace("N00", "decision", {
        "context_status": ctx_status.value,
        "history_msgs": len(history),
        "turn_id": state.turn_id,
    })

    # ---- N01 紧急风险检查（原始输入） ----
    try:
        state.emergency = safety.check_emergency(question)
        crisis = safety.check_crisis(question)
    except safety.EmergencyRuleError:
        state.add_trace("N01", "error", {"reason": "emergency_rule_fault"})
        return _finish(state, got_lock, TerminationReason.EMERGENCY_RULE_FAULT,
                       respond.CONSERVATIVE_MEDICAL_NOTICE)
    state.add_trace("N01", "decision", {
        "level": state.emergency.level.value, "rule_id": state.emergency.rule_id,
        "crisis": crisis.level.value, "crisis_rule": crisis.rule_id,
    })
    if crisis.level is EmergencyLevel.STRONG:
        # 心理危机优先：危机文案含急救引导，覆盖医疗急症同现的情况
        state.emergency = crisis
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.CRISIS_ANSWER)
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
        state.context_history
        and state.context_status is not ContextStatus.MISSING
        and needs_context
    ):
        if (state.context_package is not None
                and state.context_package.assembly_status
                is not context_engineering.ContextAssemblyStatus.OK):
            return _finish(state, got_lock, TerminationReason.CLARIFICATION_REQUIRED,
                           respond.CLARIFICATION_REPLY)
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
    context_engineering.refresh_context(state)
    try:
        second = safety.check_emergency(state.resolved_question)
        crisis = safety.check_crisis(state.resolved_question)
    except safety.EmergencyRuleError:
        state.add_trace("N03", "error", {"reason": "emergency_rule_fault"})
        return _finish(state, got_lock, TerminationReason.EMERGENCY_RULE_FAULT,
                       respond.CONSERVATIVE_MEDICAL_NOTICE)
    state.emergency = _merge_hits(state.emergency, second)
    state.add_trace("N03", "decision", {
        "level": state.emergency.level.value,
        "crisis": crisis.level.value, "crisis_rule": crisis.rule_id,
    })
    if crisis.level is EmergencyLevel.STRONG:
        state.emergency = crisis
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.CRISIS_ANSWER)
    if state.emergency.level is EmergencyLevel.STRONG:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)

    if (state.context_package is not None
            and state.context_package.assembly_status
            is not context_engineering.ContextAssemblyStatus.OK):
        return _finish(state, got_lock, TerminationReason.CLARIFICATION_REQUIRED,
                       respond.CLARIFICATION_REPLY)

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
    calc_plan = context_engineering.prepare_calculation(state, calc_plan)
    context_engineering.refresh_context(state)
    if calc_plan is not None:
        state.add_trace("N04a", "decision",
                        {"status": calc_plan.status, "kind": calc_plan.kind,
                         "param_filled": param_filled})
        if calc_plan.status in ("ambiguous", "conflict"):
            return _finish(state, got_lock, TerminationReason.CLARIFICATION_REQUIRED,
                           respond.CLARIFICATION_REPLY)
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
            state = delegate.delegate_knowledge(state, got_lock)
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
    if state.route is Route.DAILY:
        return daily.answer_daily(state, got_lock, llm_factory=get_llm)
    # Route.KNOWLEDGE → 委派知识子 Agent 执行 N06–N10（下方）
    return delegate.delegate_knowledge(state, got_lock)


# ==================== 内部工具 ====================


def _merge_hits(first: EmergencyHit, second: EmergencyHit) -> EmergencyHit:
    """取两趟检查中更强的一档（N01 原文 / N03 补全后）。"""
    order = {EmergencyLevel.NONE: 0, EmergencyLevel.WEAK: 1, EmergencyLevel.STRONG: 2}
    if order[second.level] > order[first.level]:
        return second
    return first


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




# ==================== 兼容包装（2026-10-29 解耦：实现已迁出，引用面不变） ====================
# _resolve_reference / _delegate_knowledge 的实现分别在 resolver.py / delegate.py；
# _finish/_finish_recovery/_assemble/_format_sources 在 finalize.py（顶部已 re-export）。
# 包装存在的唯一理由：LLM 注入点保持在 orchestrator.get_llm，测试 patch 面不变。


def _resolve_reference(state, node_budget_ms, *, repair: bool = False):
    return resolver.resolve_reference(
        state, node_budget_ms, repair=repair, llm_factory=get_llm,
    )


def _delegate_knowledge(state, got_lock):
    return delegate.delegate_knowledge(state, got_lock)


__all__ = [
    "AgentRequestError",
    "run_agent",
    "display_sources",
    "REQUEST_DEADLINE",
]
