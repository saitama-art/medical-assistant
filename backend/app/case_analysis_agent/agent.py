"""纵向病例分析 Agent 主流程（任务书 §二）：CA00–CA12 的唯一控制流。

与主 Agent 编排器同构的职责边界：本模块唯一决定分支走向与终止状态，
负责预算分配、trace 写入、运行登记与降级标记；数据访问在 adapter.py
（薄封装 app.case_analysis.service），结构化视图在 context.py，确定性
分析由 service 完成，医学知识解释委派给主 Agent 既有的知识子 Agent，
LLM 叙述及其程序校验在 narrative.py。

主流程（对应任务书 §二.1–8）：

    CA00 入参校验（身份/患者 id/时间范围/关注点长度）
    CA01 安全门：关注点文本过主 Agent 急症规则；规则故障保守降级
    CA02 患者归属校验（service 层 SQL 强制，本层只消费结果）
    CA03 分析运行登记（失败不阻断分析，记降级）
    CA04 数据装载：事实/时间线/观察/用药/就诊（逐项容错）
    CA05 结构化上下文组装（保留发生/录入时间、否定、不确定性、审核、来源）
    CA06 确定性分析：指标两两比较 + 候选时间关联草稿
    CA07 知识需求判定（确定性规则，不由模型决定）
    CA08 按需委派知识子 Agent（task_spec 授权 + 启动预算检查 + 结果验收）
    CA09 证据汇总：支持/反证/缺失/不确定性 → 待医生核实问题
    CA10 可选 LLM 叙述（预算守卫 + 程序校验，失败仅降级）
    CA11 输出校验：引用 id 与患者归属、日期白名单、指令性表述
    CA12 持久化收口：候选关联落库 + 运行状态更新

本模块不抛异常：一切失败归一为 CaseAnalysisResult.status + 机器可读
error_code / degraded 标记。
"""

from __future__ import annotations

import math
import time
import uuid

from ..case_analysis.models import AssociationType
from ..medical_agent import respond, safety, task_spec
from ..medical_agent.state import (
    AgentState,
    EmergencyLevel,
    NodeStatus,
    QuestionCategory,
    Route,
    SubagentStatus,
)
from ..medical_agent.subagents import knowledge
from ..rag.chain import get_llm
from ..rag.service import MAX_QUESTION_LEN
from . import adapter, narrative
from . import config as _cfg
from .context import build_analysis_context
from .outputs import (
    CaseAnalysisResult,
    CaseAnalysisStatus,
    DoctorQuestion,
    KnowledgeReference,
    render_answer_text,
)

_MAX_DOCTOR_QUESTIONS = 20


# ==================== 入口 ====================


def run_case_analysis(
    owner_user_id: int,
    patient_id: str,
    *,
    focus_question: str = "",
    time_range_start: float | None = None,
    time_range_end: float | None = None,
) -> CaseAnalysisResult:
    """执行一次完整的病例纵向分析。不抛异常。"""
    work = AgentState(
        request_id=f"ca-{uuid.uuid4().hex[:12]}",
        raw_question=focus_question or "",
        resolved_question=focus_question.strip() if focus_question else "",
    )
    work.budget.deadline = time.monotonic() + _cfg.CASE_ANALYSIS_DEADLINE_S
    try:
        return _run(
            work, owner_user_id, patient_id,
            time_range_start=time_range_start, time_range_end=time_range_end,
        )
    except Exception as exc:  # 兜底：任何未预期异常都归一为结构化失败
        work.add_trace("CA00", "error", {"reason": f"internal:{type(exc).__name__}"})
        return _finish(work, CaseAnalysisStatus.PROCESSING_FAILED,
                       patient_id, error_code="internal_error")


def _run(
    work: AgentState,
    owner_user_id: int,
    patient_id: str,
    *,
    time_range_start: float | None,
    time_range_end: float | None,
) -> CaseAnalysisResult:
    degraded: list[str] = []

    # ---- CA00 入参校验 ----
    validation_error = _validate_inputs(
        owner_user_id, patient_id, time_range_start, time_range_end,
    )
    if validation_error:
        work.add_trace("CA00", "decision", {"valid": False, "reason": validation_error})
        return _finish(work, CaseAnalysisStatus.VALIDATION_FAILED, patient_id,
                       error_code=validation_error)
    work.add_trace("CA00", "decision", {"valid": True})

    # ---- CA01 安全门（复用主 Agent 急症规则，作用于医生输入的关注点文本） ----
    if work.resolved_question:
        try:
            hit = safety.check_emergency(work.resolved_question)
        except safety.EmergencyRuleError:
            work.add_trace("CA01", "error", {"reason": "emergency_rule_fault"})
            result = _finish(work, CaseAnalysisStatus.PROCESSING_FAILED,
                             patient_id, error_code="emergency_rule_fault")
            result.answer_text = respond.CONSERVATIVE_MEDICAL_NOTICE
            return result
        work.add_trace("CA01", "decision", {"level": hit.level.value})
        if hit.level is EmergencyLevel.STRONG:
            return _finish(work, CaseAnalysisStatus.EMERGENCY_HANDLED, patient_id)
    else:
        work.add_trace("CA01", "decision", {"level": EmergencyLevel.NONE.value,
                                            "skipped": "no_focus_question"})

    # ---- CA02 患者归属校验（service 层不区分不存在/不属于，防枚举） ----
    patient, err = adapter.get_patient(patient_id, owner_user_id)
    if err:
        work.add_trace("CA02", "decision", {"access": "denied"})
        status = (CaseAnalysisStatus.ACCESS_DENIED if err == "access_denied"
                  else CaseAnalysisStatus.PROCESSING_FAILED)
        return _finish(work, status, patient_id, error_code=err)
    work.add_trace("CA02", "decision", {"access": "granted"})

    # ---- CA03 分析运行登记（登记失败不阻断分析，记降级） ----
    run, err = adapter.save_analysis_run(
        patient_id, owner_user_id,
        time_range_start=time_range_start, time_range_end=time_range_end,
        input_summary=f"focus={'yes' if work.resolved_question else 'no'}",
    )
    run_id: str | None = None
    if err or run is None:
        degraded.append("run_persistence_failed")
        work.add_trace("CA03", "error", {"reason": err or "run_missing"})
    else:
        run_id = run.run_id
        work.add_trace("CA03", "decision", {"run_registered": True})

    # ---- CA04 数据装载（逐项容错；全部失败才判存储故障） ----
    sections: dict[str, tuple] = {}
    read_failures = 0
    for name, call in (
        ("encounters", lambda: adapter.list_encounters(
            patient_id, owner_user_id,
            start=time_range_start, end=time_range_end)),
        ("facts", lambda: adapter.get_patient_history(
            patient_id, owner_user_id,
            start=time_range_start, end=time_range_end)),
        ("medications", lambda: adapter.get_patient_medications(
            patient_id, owner_user_id,
            start=time_range_start, end=time_range_end)),
        ("observations", lambda: adapter.get_patient_observations(
            patient_id, owner_user_id,
            start=time_range_start, end=time_range_end)),
        ("timeline", lambda: adapter.get_patient_timeline(
            patient_id, owner_user_id)),
    ):
        value, call_err = call()
        if call_err:
            read_failures += 1
            degraded.append(f"{name}_read_failed")
            sections[name] = ()
        else:
            sections[name] = tuple(value)
    work.add_trace("CA04", "decision", {
        "read_failures": read_failures,
        "counts": {k: len(v) for k, v in sections.items()},
    })
    if read_failures == 5:
        _close_run(work, run_id, owner_user_id, ok=False, error_code="storage_error")
        return _finish(work, CaseAnalysisStatus.PROCESSING_FAILED, patient_id,
                       run_id=run_id, error_code="storage_error",
                       degraded=tuple(degraded))

    # ---- CA05 结构化上下文 ----
    context = build_analysis_context(
        patient_id,
        sections["encounters"], sections["facts"], sections["medications"],
        sections["observations"], sections["timeline"],
    )
    work.add_trace("CA05", "decision", {
        "empty": context.empty,
        "unknown_time_count": context.unknown_time_count,
        "pending_facts": context.pending_fact_count,
    })
    if context.empty:
        _close_run(work, run_id, owner_user_id, ok=True)
        return _finish(work, CaseAnalysisStatus.NO_HISTORY, patient_id,
                       run_id=run_id, context=context,
                       timeline=tuple(sections["timeline"]),
                       degraded=tuple(degraded))

    # ---- CA06 确定性分析（时间先后 ≠ 因果，由 service/analysis 保证） ----
    comparisons, err = adapter.compare_observations(
        patient_id, owner_user_id,
        start=time_range_start, end=time_range_end,
    )
    if err:
        degraded.append("comparison_failed")
        comparisons = []
    drafts, err = adapter.find_temporal_associations(
        patient_id, owner_user_id,
        start=time_range_start, end=time_range_end,
    )
    if err:
        degraded.append("association_failed")
        drafts = []
    comparisons = tuple(comparisons)
    drafts = tuple(drafts)
    work.add_trace("CA06", "decision", {
        "comparisons": len(comparisons),
        "comparable": sum(1 for c in comparisons if c.comparable),
        "drafts": len(drafts),
    })

    evidence_insufficient = (
        not drafts and not any(c.comparable for c in comparisons)
    )

    # ---- CA07 知识需求判定（确定性规则） + CA08 按需委派知识子 Agent ----
    knowledge_refs: list[KnowledgeReference] = []
    questions = _build_knowledge_questions(context, drafts, work.resolved_question)
    if evidence_insufficient and not work.resolved_question:
        questions = []  # 证据不足且无关注点：不做无谓委派
    work.add_trace("CA07", "decision", {"knowledge_questions": len(questions)})
    for index, (question, related_ids) in enumerate(questions):
        ref = _ask_knowledge(work, question, related_ids, index)
        knowledge_refs.append(ref)
        if ref.status != "ok":
            degraded.append(f"knowledge_{index}_failed")

    # ---- CA09 证据汇总：支持/反证/缺失/不确定性 → 待医生核实问题 ----
    doctor_questions = _build_doctor_questions(context, comparisons, drafts)
    work.add_trace("CA09", "decision", {
        "doctor_questions": len(doctor_questions),
    })

    # ---- CA10 可选 LLM 叙述（预算守卫 + 程序校验；失败仅降级） ----
    narrative_text = ""
    if not evidence_insufficient:
        node_budget = _cfg.node_budget(work, _cfg.BUDGET_NARRATIVE_MS)
        if node_budget is None:
            degraded.append("narrative_budget_blocked")
            work.add_trace("CA10", "decision", {"skipped": "budget"})
        else:
            try:
                llm = get_llm(timeout=max(1.0, node_budget / 1000))
                narrative_result = narrative.generate_narrative(
                    context, comparisons, drafts, knowledge_refs,
                    focus_question=work.resolved_question, llm=llm,
                )
            except Exception as exc:
                narrative_result = None
                work.add_trace("CA10", "error",
                               {"reason": f"llm:{type(exc).__name__}"})
            if narrative_result is not None and narrative_result.status is NodeStatus.OK:
                narrative_text = narrative_result.value or ""
                work.add_trace("CA10", "decision", {"narrative": "ok"},
                               node_budget_ms=node_budget)
            else:
                degraded.append("narrative_degraded")
                work.add_trace("CA10", "decision", {
                    "narrative": "degraded",
                    "reason": (narrative_result.detail
                               if narrative_result is not None else "llm_init_failed"),
                }, node_budget_ms=node_budget)

    # ---- CA11 输出校验：关联引用的 id 必须属于本患者上下文 ----
    drafts, dropped = _drop_foreign_drafts(context, drafts)
    if dropped:
        degraded.append("foreign_associations_dropped")
    work.add_trace("CA11", "decision", {
        "drafts_kept": len(drafts), "drafts_dropped": dropped,
    })

    # ---- CA12 持久化收口：候选关联落库 + 运行状态更新 ----
    associations: tuple = ()
    if run_id and drafts:
        saved, err = adapter.save_candidate_associations(
            run_id, owner_user_id, list(drafts),
        )
        if err:
            degraded.append("association_persistence_failed")
            work.add_trace("CA12", "error", {"reason": err})
        else:
            associations = tuple(saved)
    final_ok = not degraded
    _close_run(
        work, run_id, owner_user_id, ok=True,
        error_code="" if final_ok else ",".join(degraded)[:200],
    )

    if evidence_insufficient:
        status = CaseAnalysisStatus.INSUFFICIENT_EVIDENCE
    elif degraded:
        status = CaseAnalysisStatus.PARTIAL
    else:
        status = CaseAnalysisStatus.COMPLETED

    return _finish(
        work, status, patient_id,
        run_id=run_id, context=context,
        timeline=tuple(sections["timeline"]),
        comparisons=comparisons,
        associations=associations,
        association_drafts=() if associations else drafts,
        knowledge=tuple(knowledge_refs),
        doctor_questions=doctor_questions,
        narrative=narrative_text,
        degraded=tuple(degraded),
    )


# ==================== CA07/CA08 知识委派 ====================


def _build_knowledge_questions(context, drafts, focus_question: str):
    """确定性生成知识问题。只有确需医学知识解释时才产生问题（任务书 §二.5）：
    医生的显式关注点优先；其余仅来自需要补充证据的用药-症状关联草稿。
    """
    questions: list[tuple[str, tuple[str, ...]]] = []
    if focus_question and len(focus_question) <= MAX_QUESTION_LEN:
        questions.append((focus_question, ()))
    for draft in drafts:
        if len(questions) >= _cfg.MAX_KNOWLEDGE_QUESTIONS:
            break
        if draft.association_type is not AssociationType.MEDICATION_SYMPTOM:
            continue
        if not draft.needs_further_evidence:
            continue
        names = [
            context.display_names.get(fid, "") for fid in draft.fact_ids
        ]
        names = [n for n in names if n]
        if not names:
            continue
        question = (
            f"{'、'.join(names)}之间是否存在已知的医学关联或注意事项？"
        )
        if len(question) > MAX_QUESTION_LEN:
            continue
        questions.append((question, tuple(draft.fact_ids)))
    return questions


def _ask_knowledge(
    work: AgentState, question: str, related_ids: tuple[str, ...], index: int,
) -> KnowledgeReference:
    """委派一次知识子 Agent：复用 task_spec 授权链与启动预算检查。

    子 Agent 状态每问新建（不复用父状态），产出经 SubagentResult 契约
    交还，trace/tool_calls 重排 seq 后并入父流程，保证追踪连续。
    """
    sub = AgentState(
        request_id=f"{work.request_id}-kq{index}",
        raw_question=question,
        resolved_question=question,
    )
    sub.budget.deadline = work.budget.deadline
    sub.category = QuestionCategory.KNOWLEDGE
    sub.route = Route.KNOWLEDGE

    spec = task_spec.build_task_spec(sub)
    authorization = task_spec.authorize_task_spec(spec, sub)
    if not authorization.ok:
        work.add_trace("CA08", "decision", {
            "delegate": False, "reason": authorization.detail,
        })
        return KnowledgeReference(
            question=question, status="rejected",
            detail=authorization.detail, related_fact_ids=related_ids,
        )

    work.add_trace("CA08", "decision", {
        "delegate": "knowledge_subagent", "spec_id": spec.spec_id,
        "question_len": len(question),
    })
    try:
        result = knowledge.run(spec, authorization.permit, recovery=sub.recovery)
    except Exception as exc:
        work.add_trace("CA08", "error",
                       {"reason": f"worker_exception:{type(exc).__name__}"})
        return KnowledgeReference(
            question=question, status="exception",
            detail="worker_exception", related_fact_ids=related_ids,
        )

    for record in result.tool_calls:
        record.seq = len(work.tool_calls) + 1
        work.tool_calls.append(record)
    for trace in result.trace:
        trace.seq = len(work.trace) + 1
        work.trace.append(trace)

    validation = task_spec.validate_worker_result(spec, result)
    if not validation.ok:
        work.add_trace("CA08", "decision", {
            "worker_result": "invalid", "reason": validation.detail,
        })
        return KnowledgeReference(
            question=question, status="invalid_result",
            detail=validation.detail, related_fact_ids=related_ids,
        )
    if result.status is not SubagentStatus.OK:
        return KnowledgeReference(
            question=question, status="failed",
            detail=result.status.value, related_fact_ids=related_ids,
        )
    return KnowledgeReference(
        question=question, status="ok",
        answer=result.answer, sources=tuple(result.sources),
        related_fact_ids=related_ids,
    )


# ==================== CA09 待医生核实 ====================


def _build_doctor_questions(context, comparisons, drafts) -> tuple[DoctorQuestion, ...]:
    """全部由确定性规则推出：缺失信息、未知时间、待审核、不可比较。
    模型不参与生成，也就不存在模型编造问题的可能。
    """
    questions: list[DoctorQuestion] = []
    for draft in drafts:
        for item in draft.missing_information:
            questions.append(DoctorQuestion(
                topic="候选关联待补充证据", detail=item,
                related_ids=tuple(draft.fact_ids),
            ))
    if context.unknown_time_count:
        questions.append(DoctorQuestion(
            topic="时间信息缺失",
            detail=(
                f"有 {context.unknown_time_count} 条条目发生时间未知，"
                "已排在时间线末尾单独标记；补充时间后才能参与时间关联分析"
            ),
        ))
    if context.pending_fact_count:
        questions.append(DoctorQuestion(
            topic="事实待审核",
            detail=f"{context.pending_fact_count} 条结构化事实仍为待审核状态",
        ))
    for comp in comparisons:
        if not comp.comparable and comp.reason:
            questions.append(DoctorQuestion(
                topic="指标不可比较",
                detail=f"{comp.indicator}：{comp.reason}",
            ))
    if context.negated_fact_count:
        questions.append(DoctorQuestion(
            topic="否定语义提示",
            detail=(
                f"{context.negated_fact_count} 条事实为否定语义（如『无/未』），"
                "复核时请注意其与阳性记录的语义差异"
            ),
        ))
    return tuple(questions[:_MAX_DOCTOR_QUESTIONS])


# ==================== CA11 输出校验 ====================


def _drop_foreign_drafts(context, drafts):
    """关联草稿引用的每个 id 都必须属于本患者上下文（归属校验的纵深防御）。"""
    kept = tuple(d for d in drafts if set(d.fact_ids) <= context.known_ids)
    return kept, len(drafts) - len(kept)


# ==================== 内部工具 ====================


def _validate_inputs(owner_user_id, patient_id, start, end) -> str:
    """返回机器可读错误码；空串表示合法。"""
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) \
            or owner_user_id <= 0:
        return "owner_user_id_invalid"
    if not isinstance(patient_id, str) or not patient_id.strip():
        return "patient_id_invalid"
    for value in (start, end):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            return "time_range_invalid"
    if start is not None and end is not None and start > end:
        return "time_range_inverted"
    return ""


def _close_run(work, run_id, owner_user_id, *, ok: bool, error_code: str = "") -> None:
    """更新运行状态；登记已失败（run_id 为空）或更新失败都只记 trace。"""
    if not run_id:
        return
    _, err = adapter.update_analysis_run(
        run_id, owner_user_id,
        status="completed" if ok else "failed",
        error_code=error_code,
    )
    if err:
        work.add_trace("CA12", "error", {"reason": f"run_update:{err}"})


def _finish(
    work: AgentState,
    status: CaseAnalysisStatus,
    patient_id: str,
    *,
    run_id: str | None = None,
    context=None,
    timeline: tuple = (),
    comparisons: tuple = (),
    associations: tuple = (),
    association_drafts: tuple = (),
    knowledge: tuple = (),
    doctor_questions: tuple = (),
    narrative: str = "",
    degraded: tuple = (),
    error_code: str = "",
) -> CaseAnalysisResult:
    """终态收口：落状态、确定性渲染用户可读文本、并入 trace。"""
    if isinstance(patient_id, str):
        safe_patient_id = patient_id
    else:
        safe_patient_id = ""
    result = CaseAnalysisResult(
        status=status,
        request_id=work.request_id,
        patient_id=safe_patient_id,
        run_id=run_id,
        context=context,
        timeline=timeline,
        comparisons=comparisons,
        associations=associations,
        association_drafts=association_drafts,
        knowledge=knowledge,
        doctor_questions=doctor_questions,
        narrative=narrative,
        degraded=degraded,
        error_code=error_code,
        trace=tuple(work.trace),
    )
    result.answer_text = render_answer_text(result)
    work.add_trace("CA12", "decision", {
        "status": status.value,
        "degraded": list(degraded),
    })
    result.trace = tuple(work.trace)
    return result


__all__ = ["run_case_analysis"]
