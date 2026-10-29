"""健康指标解释 Agent 主流程：IA00–IA07 的唯一控制流。

职责边界（与 case_analysis_agent 同构，2026-10-28 实读对齐）：

    IA00 入参校验（身份/输入形态/文本长度）
    IA01 安全门：原文过主 Agent 急症+危机规则；命中直接透传，不走指标流程
    IA02 指标标准化：结构化输入直通；自由文本走 LLM 抽取（repair 一次，缺失不猜）
    IA03 确定性判读：仅对照用户**给定**的参考范围；范围缺失不套通用值
    IA04 趋势：同一指标多条且同单位才有 delta；方向不做医学解释
    IA05 知识需求判定（确定性规则）→ 委派知识子 Agent（task_spec 授权链）
    IA06 证据与不确定性汇总：单位/范围/检测条件/证据缺口逐项列出
    IA07 确定性渲染输出

本模块不抛异常：一切失败归一为 IndicatorResult.status + error_code/degraded。
不做诊断、不推荐药物、不下因果结论；紧急路径不进入本流程（IA01 已拦）。
"""

from __future__ import annotations

import time
import uuid

from ..case_analysis_agent.outputs import KnowledgeReference
from ..medical_agent import respond, safety, task_spec
from ..medical_agent.state import (
    AgentState,
    EmergencyLevel,
    NodeResult,
    NodeStatus,
    QuestionCategory,
    Route,
    SubagentStatus,
)
from ..medical_agent.subagents import knowledge
from ..rag.chain import get_llm
from ..rag.service import MAX_QUESTION_LEN
from ..medical_agent import config as _agent_cfg
from .context import IndicatorInput, IndicatorPayload, extract_indicators
from .outputs import (
    IndicatorReading,
    IndicatorResult,
    IndicatorStatus,
    RangeFlag,
    TrendNote,
    render_answer_text,
)

INDICATOR_DEADLINE_S = 45.0
BUDGET_EXTRACT_MS = 8000
MAX_KNOWLEDGE_QUESTIONS = 2

node_budget = _agent_cfg.node_budget
MIN_LAUNCH_BUDGET_MS = _agent_cfg.MIN_LAUNCH_BUDGET_MS

# 知识问题模板（确定性，不依赖模型决定要不要查资料）
_KNOWLEDGE_TEMPLATES = {
    RangeFlag.ABOVE: "{name}偏高的常见原因和注意事项是什么？",
    RangeFlag.BELOW: "{name}偏低的常见原因和注意事项是什么？",
    RangeFlag.WITHIN: "{name}这项检查指标的意义是什么？",
    RangeFlag.UNKNOWN: "{name}这项检查指标的意义和一般参考范围是什么？",
}


def run_indicator_explanation(
    owner_user_id: int,
    raw_text: str = "",
    *,
    indicators: list[IndicatorInput] | None = None,
    question: str = "",
) -> IndicatorResult:
    """健康指标解释的独立入口。不抛异常。

    Args:
        owner_user_id: 调用者身份（预算与 trace 归属；本流程不读患者数据）。
        raw_text: 自由文本输入（如"空腹血糖 7.2，参考 3.9-6.1，这个高吗"）。
        indicators: 已结构化的指标（优先于 raw_text 使用，直通不抽取）。
        question: 结构化输入时的关注点原文。
    """
    work = AgentState(
        request_id=f"ia-{uuid.uuid4().hex[:12]}",
        raw_question=raw_text or question or "",
    )
    work.budget.deadline = time.monotonic() + INDICATOR_DEADLINE_S
    try:
        return _run(work, owner_user_id, raw_text,
                    indicators=indicators, question=question)
    except Exception as exc:  # 兜底：未预期异常归一为结构化失败
        work.add_trace("IA00", "error", {"reason": f"internal:{type(exc).__name__}"})
        return _finish(work, IndicatorResult(
            status=IndicatorStatus.PROCESSING_FAILED,
            request_id=work.request_id, error_code="internal_error",
            trace=tuple(work.trace),
        ))


# ==================== 主流程 ====================


def _run(work, owner_user_id, raw_text, *, indicators, question) -> IndicatorResult:
    # ---- IA00 入参校验 ----
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) \
            or owner_user_id <= 0:
        return _finish(work, IndicatorResult(
            status=IndicatorStatus.PROCESSING_FAILED,
            request_id=work.request_id, error_code="owner_invalid",
            trace=tuple(work.trace)))
    if indicators is None and (not isinstance(raw_text, str)
                               or not raw_text.strip()):
        return _finish(work, IndicatorResult(
            status=IndicatorStatus.NEEDS_INPUT,
            request_id=work.request_id, error_code="input_empty",
            follow_up_questions=("请提供要解读的检验指标（名称和数值）",),
            trace=tuple(work.trace)))
    if isinstance(raw_text, str) and len(raw_text) > MAX_QUESTION_LEN:
        return _finish(work, IndicatorResult(
            status=IndicatorStatus.REFUSED,
            request_id=work.request_id, error_code="input_too_long",
            trace=tuple(work.trace)))

    # ---- IA01 安全门（与主 Agent 同一规则，不削弱） ----
    gate_text = raw_text or question or ""
    if gate_text.strip():
        try:
            emergency = safety.check_emergency(gate_text)
            crisis = safety.check_crisis(gate_text)
        except safety.EmergencyRuleError:
            work.add_trace("IA01", "error", {"reason": "emergency_rule_fault"})
            return _finish(work, IndicatorResult(
                status=IndicatorStatus.EMERGENCY_HANDLED,
                request_id=work.request_id,
                answer_text=respond.CONSERVATIVE_MEDICAL_NOTICE,
                trace=tuple(work.trace)))
        work.add_trace("IA01", "decision", {
            "emergency": emergency.level.value, "crisis": crisis.level.value,
        })
        if crisis.level is EmergencyLevel.STRONG:
            return _finish(work, IndicatorResult(
                status=IndicatorStatus.EMERGENCY_HANDLED,
                request_id=work.request_id, answer_text=respond.CRISIS_ANSWER,
                trace=tuple(work.trace)))
        if emergency.level is EmergencyLevel.STRONG:
            return _finish(work, IndicatorResult(
                status=IndicatorStatus.EMERGENCY_HANDLED,
                request_id=work.request_id, answer_text=respond.EMERGENCY_ANSWER,
                trace=tuple(work.trace)))

    # ---- IA02 指标标准化 ----
    degraded: list[str] = []
    if indicators is not None:
        payload = IndicatorPayload(tuple(indicators), question.strip())
    else:
        node = node_budget(work, BUDGET_EXTRACT_MS)
        if node is None:
            return _finish(work, IndicatorResult(
                status=IndicatorStatus.BUDGET_EXHAUSTED,
                request_id=work.request_id, error_code="budget_blocked",
                trace=tuple(work.trace)))
        payload_result = _extract_with_repair(work, raw_text, node)
        if payload_result.status is not NodeStatus.OK:
            return _finish(work, IndicatorResult(
                status=IndicatorStatus.NEEDS_INPUT,
                request_id=work.request_id, error_code="extract_failed",
                follow_up_questions=(
                    "我没看懂这段内容。请按「指标名 + 数值 + 单位（如有参考范围也写上）」"
                    "的格式再发一次，例如：空腹血糖 7.2 mmol/L（参考 3.9-6.1）。",),
                trace=tuple(work.trace)))
        payload = payload_result.value

    if not payload.indicators:
        return _finish(work, IndicatorResult(
            status=IndicatorStatus.NEEDS_INPUT,
            request_id=work.request_id, error_code="no_indicators",
            follow_up_questions=(
                "请提供至少一项指标的名称和数值，例如：空腹血糖 7.2 mmol/L。",),
            trace=tuple(work.trace)))
    work.add_trace("IA02", "decision", {
        "indicators": len(payload.indicators),
        "source": "structured" if indicators is not None else "extracted",
    })

    # ---- IA03 确定性判读（只对照给定参考范围） ----
    uncertainties: list[str] = []
    follow_ups: list[str] = []
    readings: list[IndicatorReading] = []
    for item in payload.indicators:
        flag, notes = _assess(item)
        uncertainties.extend(notes[0])
        follow_ups.extend(notes[1])
        readings.append(IndicatorReading(
            name=item.name, value_text=item.value_text,
            value_numeric=item.value_numeric, unit=item.unit,
            reference_low=item.reference_low, reference_high=item.reference_high,
            flag=flag, sample_time=item.sample_time, note=item.note,
        ))
    work.add_trace("IA03", "decision", {
        "above": sum(r.flag is RangeFlag.ABOVE for r in readings),
        "below": sum(r.flag is RangeFlag.BELOW for r in readings),
        "unknown": sum(r.flag is RangeFlag.UNKNOWN for r in readings),
    })

    # ---- IA04 趋势（同指标多条，同单位才比） ----
    trends = _build_trends(readings)

    # ---- IA05 知识需求判定（确定性）→ 委派知识子 Agent ----
    evidence: list[KnowledgeReference] = []
    questions = _knowledge_questions(readings, payload.question)
    for index, q in enumerate(questions[:MAX_KNOWLEDGE_QUESTIONS]):
        evidence.append(_ask_knowledge(work, q, index))

    # ---- IA06 证据缺口归入不确定性 ----
    for ref in evidence:
        if not ref.answer:
            uncertainties.append(
                f"关于「{ref.question}」的资料获取未完成，相关解释从略。")
            degraded.append("knowledge_evidence")

    # ---- IA07 渲染输出 ----
    status = IndicatorStatus.COMPLETED
    if degraded:
        status = IndicatorStatus.PARTIAL
    result = IndicatorResult(
        status=status, request_id=work.request_id,
        readings=tuple(readings), trends=tuple(trends),
        evidence=tuple(evidence),
        uncertainties=tuple(dict.fromkeys(uncertainties)),
        follow_up_questions=tuple(dict.fromkeys(follow_ups)),
        degraded=tuple(dict.fromkeys(degraded)),
        trace=tuple(work.trace),
    )
    result.answer_text = render_answer_text(result)
    return result


# ==================== 内部实现 ====================


def _assess(item: IndicatorInput) -> tuple[RangeFlag, tuple[list, list]]:
    """确定性判读。参考范围完整且数值可用才给 above/below/within。"""
    uncertainties: list[str] = []
    follow_ups: list[str] = []
    if item.value_numeric is None and not item.value_text:
        uncertainties.append(f"「{item.name}」没有可判读的数值。")
        follow_ups.append(f"请补充「{item.name}」的检测数值。")
        return RangeFlag.UNKNOWN, (uncertainties, follow_ups)
    if item.value_numeric is None:
        # 文本型结果（阴性/++/...）：不做数值比较，只原样呈现
        return RangeFlag.UNKNOWN, (uncertainties, follow_ups)
    low, high = item.reference_low, item.reference_high
    if low is None and high is None:
        uncertainties.append(
            f"「{item.name}」未提供参考范围：不同实验室标准不同，"
            "无法判断是否异常。")
        follow_ups.append(
            f"如有检验报告单，请补充「{item.name}」的参考范围。")
        return RangeFlag.UNKNOWN, (uncertainties, follow_ups)
    if low is None or high is None:
        uncertainties.append(f"「{item.name}」参考范围不完整，仅按已有界限判读。")
    if low is not None and item.value_numeric < low:
        return RangeFlag.BELOW, (uncertainties, follow_ups)
    if high is not None and item.value_numeric > high:
        return RangeFlag.ABOVE, (uncertainties, follow_ups)
    return RangeFlag.WITHIN, (uncertainties, follow_ups)


def _build_trends(readings: list[IndicatorReading]) -> tuple[TrendNote, ...]:
    """同名指标 >1 条时生成趋势；时间未知或单位不一致只列点不给 delta。"""
    groups: dict[str, list[IndicatorReading]] = {}
    for r in readings:
        groups.setdefault(r.name, []).append(r)
    notes: list[TrendNote] = []
    for name, items in groups.items():
        if len(items) < 2:
            continue
        points = tuple(
            f"{r.sample_time or '时间未知'}: {r.value_text}"
            + (f" {r.unit}" if r.unit else "")
            for r in items
        )
        units = {r.unit for r in items}
        nums = [r.value_numeric for r in items]
        if len(units) > 1:
            notes.append(TrendNote(name, items[0].unit, points, "",
                                   False, "unit_mismatch"))
        elif any(v is None for v in nums):
            notes.append(TrendNote(name, items[0].unit, points, "",
                                   False, "value_missing"))
        elif not all(r.sample_time for r in items):
            notes.append(TrendNote(name, items[0].unit, points, "",
                                   False, "sample_time_missing"))
        else:
            delta = round(nums[-1] - nums[0], 6)
            sign = "+" if delta > 0 else ""
            notes.append(TrendNote(
                name, items[0].unit, points,
                f"较首次 {sign}{delta} {items[0].unit}".replace("  ", " "),
                True))
    return tuple(notes)


def _knowledge_questions(readings: list[IndicatorReading],
                         question: str) -> list[str]:
    """确定性知识需求：异常指标优先解释；无异常时解释首项指标。"""
    questions: list[str] = []
    flagged = [r for r in readings if r.flag in (RangeFlag.ABOVE, RangeFlag.BELOW)]
    for r in flagged[:MAX_KNOWLEDGE_QUESTIONS]:
        questions.append(_KNOWLEDGE_TEMPLATES[r.flag].format(name=r.name))
    if not questions and readings:
        target = readings[0]
        questions.append(_KNOWLEDGE_TEMPLATES[target.flag].format(name=target.name))
    return questions


def _extract_with_repair(work, raw_text: str, node_budget_ms: int) -> NodeResult:
    try:
        llm = get_llm(timeout=max(1.0, node_budget_ms / 1000))
    except Exception as exc:
        return NodeResult(NodeStatus.UNAVAILABLE, detail=type(exc).__name__)
    result = extract_indicators(raw_text, llm=llm, repair=False)
    if result.status is not NodeStatus.INVALID_OUTPUT:
        return result
    work.add_trace("IA02", "recovery", {"action": "retry"})
    return extract_indicators(raw_text, llm=llm, repair=True)


def _ask_knowledge(work, question: str, index: int) -> KnowledgeReference:
    """委派知识子 Agent：复用 task_spec 授权链（与 case_analysis_agent 同构）。"""
    launch = node_budget(work, MIN_LAUNCH_BUDGET_MS)
    if launch is None:
        work.add_trace("IA05", "decision", {"delegate": False,
                                            "reason": "budget_blocked"})
        return KnowledgeReference(question=question, status="rejected",
                                  detail="budget_blocked")
    sub = AgentState(request_id=f"{work.request_id}-kq{index}",
                     raw_question=question, resolved_question=question)
    sub.budget.deadline = work.budget.deadline
    sub.category = QuestionCategory.KNOWLEDGE
    sub.route = Route.KNOWLEDGE
    spec = task_spec.build_task_spec(sub)
    authorization = task_spec.authorize_task_spec(spec, sub)
    if not authorization.ok:
        work.add_trace("IA05", "decision", {
            "delegate": False, "reason": authorization.detail})
        return KnowledgeReference(question=question, status="rejected",
                                  detail=authorization.detail)
    work.add_trace("IA05", "decision", {
        "delegate": "knowledge_subagent", "spec_id": spec.spec_id,
        "question_len": len(question)})
    try:
        result = knowledge.run(spec, authorization.permit, recovery=sub.recovery)
    except Exception as exc:
        work.add_trace("IA05", "error",
                       {"reason": f"worker_exception:{type(exc).__name__}"})
        return KnowledgeReference(question=question, status="exception",
                                  detail="worker_exception")
    for record in result.tool_calls:
        record.seq = len(work.tool_calls) + 1
        work.tool_calls.append(record)
    for trace in result.trace:
        trace.seq = len(work.trace) + 1
        work.trace.append(trace)
    if result.status is SubagentStatus.OK:
        return KnowledgeReference(question=question, status="ok",
                                  answer=result.answer,
                                  sources=tuple(result.sources))
    return KnowledgeReference(question=question, status=result.status.value,
                              detail=result.detail or result.status.value)


def _finish(work, result: IndicatorResult) -> IndicatorResult:
    if not result.answer_text and result.status in (
        IndicatorStatus.COMPLETED, IndicatorStatus.PARTIAL,
    ):
        result.answer_text = render_answer_text(result)
    result.trace = tuple(work.trace)
    return result


__all__ = [
    "INDICATOR_DEADLINE_S",
    "run_indicator_explanation",
]
