"""确定性分析工具（任务书 §五）：时间线、指标比较、时间关系、候选关联。

铁律：
- 全部纯函数，不调用 LLM、不做 IO、不写库；
- 时间先后不等于因果；数值变化不解释成病情恶化/改善；
- 未知时间不猜、缺失值不补；
- 不含任何疾病诊断或治疗建议。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .models import (
    AssociationDraft,
    AssociationType,
    Certainty,
    ClinicalFact,
    ComparisonEntry,
    FactType,
    MedicationEvent,
    MedicationEventType,
    Observation,
    ObservationPoint,
    Polarity,
    TemporalRelation,
    TimelineEntry,
)


# ==================== A. 时间线排序 ====================


def build_timeline(
    encounters=(),
    facts=(),
    medications=(),
    observations=(),
) -> list[TimelineEntry]:
    """把多类事件排成一条时间线。

    规则：event_time 已知者按 (event_time, recorded_at, kind, ref_id) 升序；
    event_time 未知者排在**最后**并标记 unknown_time=True——未知时间不插到
    序列中间冒充已知位置。recorded_at 只作次序兜底，绝不改写成发生时间。
    """
    entries: list[TimelineEntry] = []

    for enc in encounters:
        entries.append(TimelineEntry(
            kind="encounter", ref_id=enc.encounter_id,
            event_time=enc.encounter_time, recorded_at=enc.recorded_at,
            unknown_time=enc.encounter_time is None,
            summary=f"就诊记录（{enc.source_type}）",
            source_ref=enc.encounter_id,
        ))
    for fact in facts:
        entries.append(TimelineEntry(
            kind="fact", ref_id=fact.fact_id,
            event_time=fact.onset_time, recorded_at=fact.recorded_at,
            unknown_time=fact.onset_time is None,
            summary=f"{fact.fact_type.value}:{fact.display_name}",
            source_ref=fact.encounter_id or "",
        ))
    for med in medications:
        entries.append(TimelineEntry(
            kind="medication", ref_id=med.medication_event_id,
            event_time=med.start_time, recorded_at=med.created_at,
            unknown_time=med.start_time is None,
            summary=f"{med.event_type.value}:{med.medication_name}",
            source_ref=med.encounter_id or "",
        ))
    for obs in observations:
        entries.append(TimelineEntry(
            kind="observation", ref_id=obs.observation_id,
            event_time=obs.sample_time,
            recorded_at=obs.reported_at or obs.created_at,
            unknown_time=obs.sample_time is None,
            summary=f"{obs.indicator_name}={_fmt_value(obs.value_numeric, obs.value_text)}{obs.unit}",
            source_ref=obs.encounter_id or "",
        ))

    known = sorted(
        (e for e in entries if not e.unknown_time),
        key=lambda e: (e.event_time, e.recorded_at, e.kind, e.ref_id),
    )
    unknown = sorted(
        (e for e in entries if e.unknown_time),
        key=lambda e: (e.recorded_at, e.kind, e.ref_id),
    )
    return known + unknown


# ==================== B. 检查指标趋势比较 ====================


def compare_observations(
    observations: list[Observation],
    *,
    indicator: str | None = None,
) -> list[ComparisonEntry]:
    """对同一指标的观察序列做确定性两两比较。

    可比条件（全部满足才给 delta）：
    1. normalized_name（或 indicator_name）一致；
    2. 单位逐字一致——未确认换算规则时**不做跨单位比较**；
    3. 两次都有 value_numeric 且都有 sample_time。
    任一不满足则 comparable=False 且 reason 给出机器可读原因。
    """
    groups: dict[str, list[Observation]] = {}
    for obs in observations:
        if indicator is not None and obs.indicator_name != indicator \
                and obs.normalized_name != indicator.strip().lower():
            continue
        key = obs.normalized_name or obs.indicator_name.strip().lower()
        groups.setdefault(key, []).append(obs)

    out: list[ComparisonEntry] = []
    for key, items in groups.items():
        timed = sorted(
            (o for o in items if o.sample_time is not None),
            key=lambda o: (o.sample_time, o.created_at),
        )
        timeless = [o for o in items if o.sample_time is None]
        for first, second in zip(timed, timed[1:]):
            out.append(_compare_pair(key, first, second))
        for obs in timeless:
            out.append(ComparisonEntry(
                indicator=key, unit=obs.unit,
                earlier=_point(obs), later=None, delta=None,
                comparable=False, reason="sample_time_missing",
            ))
    return out


def _point(obs: Observation) -> ObservationPoint:
    return ObservationPoint(
        observation_id=obs.observation_id, sample_time=obs.sample_time,
        value_numeric=obs.value_numeric, value_text=obs.value_text,
        unit=obs.unit, reference_low=obs.reference_low,
        reference_high=obs.reference_high, abnormal_flag=obs.abnormal_flag,
        source_text=obs.source_text,
    )


def _compare_pair(indicator: str, first: Observation,
                  second: Observation) -> ComparisonEntry:
    p1, p2 = _point(first), _point(second)
    if first.unit != second.unit:
        return ComparisonEntry(
            indicator=indicator, unit=first.unit, earlier=p1, later=p2,
            delta=None, comparable=False, reason="unit_mismatch",
        )
    if first.value_numeric is None or second.value_numeric is None:
        return ComparisonEntry(
            indicator=indicator, unit=first.unit, earlier=p1, later=p2,
            delta=None, comparable=False, reason="value_missing",
        )
    return ComparisonEntry(
        indicator=indicator, unit=first.unit, earlier=p1, later=p2,
        delta=round(second.value_numeric - first.value_numeric, 6),
        comparable=True,
    )


# ==================== C. 症状与用药时间关系 ====================


def symptom_medication_relation(
    symptom_onset: float | None,
    med_start: float | None,
    med_end: float | None,
) -> TemporalRelation:
    """症状发生时间与一次用药的时间关系。

    判定规则（任何关键时间缺失即 UNKNOWN，不猜）：
    - symptom_onset 或 med_start 缺失 → UNKNOWN；
    - onset < start → BEFORE；
    - med_end 已知且 onset > end → AFTER；
    - 其余（start <= onset 且未超出已知 end）→ DURING。
    时间关系只是线索，不代表不良反应成立。
    """
    if symptom_onset is None or med_start is None:
        return TemporalRelation.UNKNOWN
    if symptom_onset < med_start:
        return TemporalRelation.BEFORE
    if med_end is not None and symptom_onset > med_end:
        return TemporalRelation.AFTER
    return TemporalRelation.DURING


# ==================== D. 历史事实关联（纯函数侧） ====================


def relate_facts_by_time(
    facts: list[ClinicalFact],
    *,
    fact_types: list[FactType] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> list[ClinicalFact]:
    """按类型与时间窗过滤事实；不做语义相似度合并。

    repository.list_facts 已完成归属与 SQL 过滤，本函数供内存内二次过滤
    （如同一患者多类型交叉）使用。时间未知的事实不因窗口被丢弃。
    """
    out = []
    for fact in facts:
        if fact_types and fact.fact_type not in fact_types:
            continue
        if fact.onset_time is not None:
            if start is not None and fact.onset_time < start:
                continue
            if end is not None and fact.onset_time > end:
                continue
        out.append(fact)
    return out


# ==================== E. 候选关联生成 ====================

# 时间接近窗口：用药开始后 N 天内出现的症状算"时间接近"线索
_TEMPORAL_PROXIMITY_SECONDS = 30 * 24 * 3600


def build_association_candidates(
    patient_id: str,
    facts: list[ClinicalFact],
    medications: list[MedicationEvent],
    comparisons: list[ComparisonEntry] | None = None,
) -> list[AssociationDraft]:
    """规则化生成候选关联（仅为待核实线索，不是因果结论）。

    规则集（全部显式、可审计）：
    E1 用药-症状时间接近：症状 onset 在用药 [start, start+30d] 区间内
       （end 已知时不超出 end+30d）→ MEDICATION_SYMPTOM；
       时间缺失 → 不生成（UNKNOWN 不硬造关联）。
    E2 指标趋势：comparable=True 且 delta != 0 的指标对 → OBSERVATION_TREND，
       附带参考范围与单位，不解释方向含义。
    每条候选都带 supporting / contradicting / missing / certainty=UNCERTAIN /
    needs_further_evidence=True。
    """
    drafts: list[AssociationDraft] = []
    symptoms = [f for f in facts
                if f.fact_type is FactType.SYMPTOM
                and f.polarity is not Polarity.NEGATIVE]

    for med in medications:
        if med.event_type is MedicationEventType.MENTIONED:
            continue  # 仅提及不参与时间关联
        for symptom in symptoms:
            relation = symptom_medication_relation(
                symptom.onset_time, med.start_time, med.end_time,
            )
            if relation is not TemporalRelation.DURING:
                continue
            if med.start_time is not None and symptom.onset_time is not None:
                proximity_end = med.start_time + _TEMPORAL_PROXIMITY_SECONDS
                if med.end_time is not None:
                    proximity_end = min(proximity_end,
                                        med.end_time + _TEMPORAL_PROXIMITY_SECONDS)
                if symptom.onset_time > proximity_end:
                    continue
            missing = []
            if med.event_type is MedicationEventType.PRESCRIBED:
                missing.append("患者是否实际服药未知（处方不等于服药）")
            if med.dose is None:
                missing.append("剂量未知")
            if symptom.onset_time is None or med.start_time is None:
                missing.append("关键时间缺失")
            drafts.append(AssociationDraft(
                association_type=AssociationType.MEDICATION_SYMPTOM,
                fact_ids=(med.medication_event_id, symptom.fact_id),
                temporal_relationship=TemporalRelation.DURING,
                supporting_evidence=(
                    f"症状「{symptom.display_name}」时间落在「{med.medication_name}」用药时间范围内",
                ),
                contradicting_evidence=(),
                missing_information=tuple(missing),
                explanation="时间接近的用药-症状线索，需进一步医学证据核实，"
                            "不构成不良反应结论。",
                certainty=Certainty.UNCERTAIN,
                needs_further_evidence=True,
            ))

    for comp in comparisons or []:
        if not comp.comparable or comp.delta in (None, 0):
            continue
        drafts.append(AssociationDraft(
            association_type=AssociationType.OBSERVATION_TREND,
            fact_ids=tuple(
                p.observation_id for p in (comp.earlier, comp.later) if p),
            temporal_relationship=TemporalRelation.UNKNOWN,
            supporting_evidence=(
                f"指标「{comp.indicator}」两次测量差值 {comp.delta} {comp.unit}",
            ),
            contradicting_evidence=(),
            missing_information=("变化原因需结合临床背景",),
            explanation="同一指标同单位两次测量存在数值差异；方向含义需专业判断，"
                        "本模块不做恶化/改善解释。",
            certainty=Certainty.UNCERTAIN,
            needs_further_evidence=True,
        ))
    return drafts


def new_association_id() -> str:
    return uuid.uuid4().hex


def _fmt_value(numeric, text: str) -> str:
    if numeric is not None:
        return str(numeric)
    return text or "?"


__all__ = [
    "build_association_candidates",
    "build_timeline",
    "compare_observations",
    "new_association_id",
    "relate_facts_by_time",
    "symptom_medication_relation",
]
