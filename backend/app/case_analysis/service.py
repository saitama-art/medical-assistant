"""纵向病例分析的对外服务接口：另一终端（病例分析 Agent）的唯一调用面。

纪律：
- Agent 只能调用本模块函数，不得接触 db.connect() 或 SQL；
- 每个患者级函数第一个参数都是 owner_user_id，归属校验在 repository 层
  强制，绕不过去；
- 返回值全部来自 models.py 的不可变 dataclass；异常为
  CaseNotFoundError / CaseAccessDeniedError / CaseValidationError；
- 本层不调用 LLM、不生成报告、不做诊断。
"""

from __future__ import annotations

from . import analysis, repository
from .models import (
    AnalysisRun,
    AssociationDraft,
    CandidateAssociation,
    ClinicalFact,
    ComparisonEntry,
    Encounter,
    MedicationEvent,
    Observation,
    Patient,
    TimelineEntry,
)


# ==================== 患者与就诊 ====================


def create_patient(owner_user_id: int, display_label: str = "") -> Patient:
    """创建患者记录（内部 ID 标识，不按姓名匹配）。"""
    return repository.create_patient(owner_user_id, display_label)


def get_patient(patient_id: str, owner_user_id: int) -> Patient:
    """查询患者；不存在或不属于调用者抛 CaseAccessDeniedError。"""
    return repository.get_patient_checked(patient_id, owner_user_id)


def list_patients(owner_user_id: int, *, limit: int = 200) -> list[Patient]:
    return repository.list_patients(owner_user_id, limit=limit)


def create_encounter(patient_id: str, owner_user_id: int, raw_note: str, *,
                     encounter_time: float | None = None,
                     source_type: str = "manual_note") -> Encounter:
    """创建就诊记录，保留原文；encounter_time 未知时传 None。"""
    return repository.create_encounter(
        patient_id, owner_user_id, raw_note,
        encounter_time=encounter_time, source_type=source_type,
    )


def list_encounters(patient_id: str, owner_user_id: int, *,
                    start: float | None = None, end: float | None = None,
                    limit: int = 500) -> list[Encounter]:
    return repository.list_encounters(
        patient_id, owner_user_id, start=start, end=end, limit=limit,
    )


# ==================== 结构化事实写入 ====================


def save_clinical_facts(patient_id: str, owner_user_id: int,
                        facts: list[ClinicalFact]) -> list[ClinicalFact]:
    """保存抽取的事实（默认待审核）；encounter_id 必须属于同一患者。"""
    return repository.save_clinical_facts(patient_id, owner_user_id, facts)


def save_medication_events(patient_id: str, owner_user_id: int,
                           events: list[MedicationEvent]) -> list[MedicationEvent]:
    return repository.save_medication_events(patient_id, owner_user_id, events)


def save_observations(patient_id: str, owner_user_id: int,
                      observations: list[Observation]) -> list[Observation]:
    return repository.save_observations(patient_id, owner_user_id, observations)


# ==================== 患者历史查询（Agent 主要读接口） ====================


def get_patient_history(patient_id: str, owner_user_id: int, *,
                        fact_types: list[str] | None = None,
                        start: float | None = None,
                        end: float | None = None,
                        review_status: str | None = None,
                        limit: int = 1000) -> list[ClinicalFact]:
    """按患者/类型/时间窗查询临床事实，含原文依据与来源就诊记录 id。"""
    return repository.list_facts(
        patient_id, owner_user_id, fact_types=fact_types,
        start=start, end=end, review_status=review_status, limit=limit,
    )


def get_patient_timeline(patient_id: str, owner_user_id: int) -> list[TimelineEntry]:
    """患者完整时间线：就诊+事实+用药+观察，未知时间排尾并单独标记。"""
    encounters = repository.list_encounters(patient_id, owner_user_id)
    facts = repository.list_facts(patient_id, owner_user_id)
    medications = repository.list_medication_events(patient_id, owner_user_id)
    observations = repository.list_observations(patient_id, owner_user_id)
    return analysis.build_timeline(encounters, facts, medications, observations)


def get_patient_observations(patient_id: str, owner_user_id: int, *,
                             indicator: str | None = None,
                             start: float | None = None,
                             end: float | None = None,
                             limit: int = 1000) -> list[Observation]:
    return repository.list_observations(
        patient_id, owner_user_id, indicator=indicator,
        start=start, end=end, limit=limit,
    )


def get_patient_medications(patient_id: str, owner_user_id: int, *,
                            start: float | None = None,
                            end: float | None = None,
                            limit: int = 1000) -> list[MedicationEvent]:
    return repository.list_medication_events(
        patient_id, owner_user_id, start=start, end=end, limit=limit,
    )


# ==================== 确定性分析 ====================


def compare_observations(patient_id: str, owner_user_id: int, *,
                         indicator: str | None = None,
                         start: float | None = None,
                         end: float | None = None) -> list[ComparisonEntry]:
    """同指标同单位的确定性两两比较；不可比较时给出机器可读原因。"""
    observations = repository.list_observations(
        patient_id, owner_user_id, indicator=indicator, start=start, end=end,
    )
    return analysis.compare_observations(observations, indicator=indicator)


def find_temporal_associations(patient_id: str, owner_user_id: int, *,
                               start: float | None = None,
                               end: float | None = None) -> list[AssociationDraft]:
    """生成候选关联草稿（不落库）：用药-症状时间接近 + 指标趋势。"""
    facts = repository.list_facts(patient_id, owner_user_id,
                                  start=start, end=end)
    medications = repository.list_medication_events(patient_id, owner_user_id,
                                                    start=start, end=end)
    comparisons = compare_observations(patient_id, owner_user_id,
                                       start=start, end=end)
    return analysis.build_association_candidates(
        patient_id, facts, medications, comparisons,
    )


# ==================== 分析运行与候选关联存储 ====================


def save_analysis_run(patient_id: str, owner_user_id: int, *,
                      time_range_start: float | None = None,
                      time_range_end: float | None = None,
                      input_summary: str = "") -> AnalysisRun:
    """创建分析运行记录（status=pending）；完成后用 update_analysis_run 更新。"""
    return repository.create_analysis_run(
        patient_id, owner_user_id,
        time_range_start=time_range_start, time_range_end=time_range_end,
        input_summary=input_summary,
    )


def update_analysis_run(run_id: str, owner_user_id: int, *,
                        status: str, error_code: str = "") -> AnalysisRun:
    return repository.update_analysis_run(
        run_id, owner_user_id, status=status, error_code=error_code,
    )


def save_candidate_associations(run_id: str, owner_user_id: int,
                                drafts: list,
                                ) -> list[CandidateAssociation]:
    """把 find_temporal_associations 或 Agent 自产的草稿落库（默认待审核）。"""
    return repository.save_candidate_associations(run_id, owner_user_id, drafts)


def list_candidate_associations(run_id: str, owner_user_id: int,
                                ) -> list[CandidateAssociation]:
    return repository.list_candidate_associations(run_id, owner_user_id)


__all__ = [
    "compare_observations",
    "create_encounter",
    "create_patient",
    "find_temporal_associations",
    "get_patient",
    "get_patient_history",
    "get_patient_medications",
    "get_patient_observations",
    "get_patient_timeline",
    "list_candidate_associations",
    "list_encounters",
    "list_patients",
    "save_analysis_run",
    "save_candidate_associations",
    "save_clinical_facts",
    "save_medication_events",
    "save_observations",
    "update_analysis_run",
]
