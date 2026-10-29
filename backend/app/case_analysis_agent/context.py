"""结构化分析上下文：把 service 读出的记录整理成分析/校验共用的视图。

纯函数，不做 IO、不调用 LLM。语义保留规则（任务书 §二.3）：
- 事件发生时间（onset/start/sample/encounter_time）与录入时间
  （recorded_at/created_at）分别保留，绝不互相改写；
- 否定（polarity）、不确定性（certainty）、审核状态（review_status）逐条保留；
- 来源（source_text / encounter_id / source_ref）逐条保留；
- 未知时间保持 None/unknown_time=True，不用任何猜测值回填。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType

from ..case_analysis.models import (
    Certainty,
    ClinicalFact,
    Encounter,
    MedicationEvent,
    Observation,
    Polarity,
    ReviewStatus,
    TimelineEntry,
)


@dataclass(frozen=True)
class AnalysisContext:
    """一次分析所依据的全部患者数据视图。所有字段只读。

    known_ids / allowed_dates 同时是输出校验的白名单：任何引用不在
    known_ids 中的 id、任何不在 allowed_dates 中的日期，都视为模型编造。
    """

    patient_id: str
    encounters: tuple[Encounter, ...] = ()
    facts: tuple[ClinicalFact, ...] = ()
    medications: tuple[MedicationEvent, ...] = ()
    observations: tuple[Observation, ...] = ()
    timeline: tuple[TimelineEntry, ...] = ()

    fact_count: int = 0
    encounter_count: int = 0
    medication_count: int = 0
    observation_count: int = 0
    pending_fact_count: int = 0
    negated_fact_count: int = 0
    uncertain_fact_count: int = 0
    unknown_time_count: int = 0

    indicators: tuple[str, ...] = ()
    known_ids: frozenset[str] = frozenset()
    allowed_dates: frozenset[str] = frozenset()   # "%Y-%m-%d"，含全部发生/录入时间
    # id → 展示名，供确定性构造知识问题；不含原文全文。
    display_names: object = field(default_factory=lambda: MappingProxyType({}))

    @property
    def empty(self) -> bool:
        return not (
            self.encounters or self.facts or self.medications or self.observations
        )


def build_analysis_context(
    patient_id: str,
    encounters,
    facts,
    medications,
    observations,
    timeline,
) -> AnalysisContext:
    """把各类记录汇总为只读上下文；汇总值全部由输入推导，不引入外部数据。"""
    encounters = tuple(encounters)
    facts = tuple(facts)
    medications = tuple(medications)
    observations = tuple(observations)
    timeline = tuple(timeline)

    known_ids: set[str] = set()
    dates: set[str] = set()
    names: dict[str, str] = {}

    def _collect_time(ts: float | None) -> None:
        if ts is not None:
            dates.add(time.strftime("%Y-%m-%d", time.localtime(ts)))

    for enc in encounters:
        known_ids.add(enc.encounter_id)
        _collect_time(enc.encounter_time)
        _collect_time(enc.recorded_at)
    for fact in facts:
        known_ids.add(fact.fact_id)
        if fact.display_name:
            names[fact.fact_id] = fact.display_name
        _collect_time(fact.onset_time)
        _collect_time(fact.end_time)
        _collect_time(fact.recorded_at)
    for med in medications:
        known_ids.add(med.medication_event_id)
        names[med.medication_event_id] = med.medication_name
        _collect_time(med.start_time)
        _collect_time(med.end_time)
        _collect_time(med.created_at)
    for obs in observations:
        known_ids.add(obs.observation_id)
        if obs.indicator_name:
            names[obs.observation_id] = obs.indicator_name
        _collect_time(obs.sample_time)
        _collect_time(obs.reported_at)

    return AnalysisContext(
        patient_id=patient_id,
        encounters=encounters,
        facts=facts,
        medications=medications,
        observations=observations,
        timeline=timeline,
        fact_count=len(facts),
        encounter_count=len(encounters),
        medication_count=len(medications),
        observation_count=len(observations),
        pending_fact_count=sum(
            1 for f in facts if f.review_status is ReviewStatus.PENDING
        ),
        negated_fact_count=sum(
            1 for f in facts if f.polarity is Polarity.NEGATIVE
        ),
        uncertain_fact_count=sum(
            1 for f in facts if f.certainty is not Certainty.CONFIRMED
        ),
        unknown_time_count=sum(1 for e in timeline if e.unknown_time),
        indicators=tuple(
            sorted({o.indicator_name for o in observations if o.indicator_name})
        ),
        known_ids=frozenset(known_ids),
        allowed_dates=frozenset(dates),
        display_names=MappingProxyType(names),
    )


def filter_timeline_by_range(
    timeline,
    start: float | None,
    end: float | None,
) -> tuple[TimelineEntry, ...]:
    """把全量时间线裁剪到本次分析范围内（任务书：时间线与分析范围一致）。

    规则：
    - event_time 已知的条目，只有落在 [start, end] 内才保留；
    - event_time 未知（unknown_time=True）的条目一律保留——它们本就不属于
      任何时间区间，靠 unknown_time 标记与范围内事件区分，不得伪装成范围内
      事件，也不得被范围过滤静默丢弃。

    已知口径差异（service.get_patient_timeline 不接范围参数，过滤只能在本层
    做）：medication 的时间线 event_time 取 start_time，而 SQL 侧对用药的
    范围过滤是"区间重叠"语义；跨界用药会留在 medications 列表但其时间线条目
    可能被裁掉。结论性内容（drafts/comparisons）不依赖时间线，不受影响。
    """
    out = []
    for entry in timeline:
        if entry.event_time is None:
            out.append(entry)
        elif (start is None or entry.event_time >= start) and (
            end is None or entry.event_time <= end
        ):
            out.append(entry)
    return tuple(out)


__all__ = ["AnalysisContext", "build_analysis_context", "filter_timeline_by_range"]
