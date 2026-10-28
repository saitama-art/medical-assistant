"""纵向病例分析的数据模型：dataclass + 枚举，不含 SQL、不做 IO。

语义约束（任务书 §三）：
- 未知时间一律允许为 None，禁止用模型猜测值回填；
- polarity/certainty/review_status 区分原文语义，结构化结果默认 pending；
- 时间戳为 epoch 秒（float），与 db.now_ts() 一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ==================== 枚举 ====================


class PatientStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class FactType(str, Enum):
    SYMPTOM = "symptom"
    DISEASE = "disease"
    HISTORY = "history"
    LAB_RELATED = "lab_related"
    OTHER = "other"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class Certainty(str, Enum):
    CONFIRMED = "confirmed"
    POSSIBLE = "possible"
    UNCERTAIN = "uncertain"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class MedicationEventType(str, Enum):
    """用药事件语义：开处方 ≠ 实际服药，逐字对应任务书 §三.4。"""

    PRESCRIBED = "prescribed"            # 医生开具
    REPORTED_ACTIVE = "reported_active"  # 患者报告正在服用
    REPORTED_STOPPED = "reported_stopped"  # 患者报告已停药
    MENTIONED = "mentioned"              # 历史中仅提及
    UNKNOWN = "unknown"                  # 实际服药情况未知


class AbnormalFlag(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TemporalRelation(str, Enum):
    """症状/事件与用药的时间关系（任务书 §五.C）。"""

    BEFORE = "before"    # 症状发生在用药开始之前
    DURING = "during"    # 症状发生在用药期间
    AFTER = "after"      # 症状发生在用药结束之后
    UNKNOWN = "unknown"  # 时间信息不足，无法判定


class AssociationType(str, Enum):
    """候选关联类型：只表示待核实线索，不是因果结论。"""

    MEDICATION_SYMPTOM = "medication_symptom"          # 用药与症状的时间接近
    OBSERVATION_TREND = "observation_trend"            # 指标序列变化
    FACT_CO_OCCURRENCE = "fact_co_occurrence"          # 同时段事实共现


# ==================== 异常 ====================


class CaseAnalysisError(Exception):
    """本模块的领域异常基类。code 为机器可读短标识，不含病历内容。"""

    code = "case_analysis_error"


class CaseNotFoundError(CaseAnalysisError):
    code = "not_found"


class CaseAccessDeniedError(CaseAnalysisError):
    """归属校验失败：不区分"不存在"与"不属于你"，防枚举（同 conversations 风格）。"""

    code = "access_denied"


class CaseValidationError(CaseAnalysisError):
    code = "validation_failed"


# ==================== 记录结构 ====================


@dataclass(frozen=True)
class Patient:
    patient_id: str
    owner_user_id: int
    display_label: str = ""
    status: PatientStatus = PatientStatus.ACTIVE
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class Encounter:
    encounter_id: str
    patient_id: str
    encounter_time: float | None     # None = 发生时间未知
    recorded_at: float
    source_type: str
    raw_note: str
    created_by: int
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class ClinicalFact:
    fact_id: str
    patient_id: str
    encounter_id: str | None
    fact_type: FactType
    display_name: str
    code: str = ""
    value_text: str = ""
    value_numeric: float | None = None
    unit: str = ""
    polarity: Polarity = Polarity.UNKNOWN
    certainty: Certainty = Certainty.UNCERTAIN
    onset_time: float | None = None
    end_time: float | None = None
    recorded_at: float = 0.0
    source_text: str = ""
    source_start: int | None = None
    source_end: int | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class MedicationEvent:
    medication_event_id: str
    patient_id: str
    encounter_id: str | None
    medication_name: str
    normalized_name: str = ""
    dose: float | None = None
    dose_unit: str = ""
    frequency: str = ""
    route: str = ""
    start_time: float | None = None
    end_time: float | None = None
    event_type: MedicationEventType = MedicationEventType.UNKNOWN
    usage_status: str = "unknown"
    source_text: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class Observation:
    observation_id: str
    patient_id: str
    encounter_id: str | None
    indicator_name: str
    normalized_name: str = ""
    value_numeric: float | None = None
    value_text: str = ""
    unit: str = ""
    reference_low: float | None = None
    reference_high: float | None = None
    abnormal_flag: AbnormalFlag = AbnormalFlag.UNKNOWN
    sample_time: float | None = None
    reported_at: float | None = None
    source_text: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING
    created_at: float = 0.0


@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    patient_id: str
    owner_user_id: int
    triggered_at: float
    time_range_start: float | None = None
    time_range_end: float | None = None
    status: RunStatus = RunStatus.PENDING
    input_summary: str = ""
    error_code: str = ""
    created_at: float = 0.0


@dataclass(frozen=True)
class CandidateAssociation:
    """候选关联：仅供另一终端的 Agent 进一步核实，不构成因果或诊断结论。"""

    association_id: str
    run_id: str
    patient_id: str
    association_type: AssociationType
    fact_ids: tuple[str, ...] = ()
    temporal_relationship: TemporalRelation = TemporalRelation.UNKNOWN
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    explanation: str = ""
    certainty: Certainty = Certainty.UNCERTAIN
    review_status: ReviewStatus = ReviewStatus.PENDING
    created_at: float = 0.0


@dataclass(frozen=True)
class TimelineEntry:
    """时间线条目：event_time 为 None 时 unknown_time=True，单独标记。"""

    kind: str                    # encounter / fact / medication / observation
    ref_id: str
    event_time: float | None     # 发生时间（可为空）
    recorded_at: float           # 记录时间（必有）
    unknown_time: bool
    summary: str                 # 不含原文全文的短摘要
    source_ref: str = ""


@dataclass(frozen=True)
class ObservationPoint:
    observation_id: str
    sample_time: float | None
    value_numeric: float | None
    value_text: str
    unit: str
    reference_low: float | None
    reference_high: float | None
    abnormal_flag: AbnormalFlag
    source_text: str = ""


@dataclass(frozen=True)
class ComparisonEntry:
    """两次观察值之间的确定性比较结果。delta 为 None 时 reason 给出原因。"""

    indicator: str
    unit: str
    earlier: ObservationPoint | None
    later: ObservationPoint | None
    delta: float | None
    comparable: bool
    reason: str = ""               # 不可比较原因（单位不一致/数值缺失等）


@dataclass(frozen=True)
class AssociationDraft:
    """build_association_candidates 的输出（落库前），结构与 CandidateAssociation 对应。"""

    association_type: AssociationType
    fact_ids: tuple[str, ...]
    temporal_relationship: TemporalRelation
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    explanation: str = ""
    certainty: Certainty = Certainty.UNCERTAIN
    needs_further_evidence: bool = True


__all__ = [
    "AbnormalFlag", "AnalysisRun", "AssociationDraft", "AssociationType",
    "CandidateAssociation", "CaseAccessDeniedError", "CaseAnalysisError",
    "CaseNotFoundError", "CaseValidationError", "Certainty", "ClinicalFact",
    "ComparisonEntry", "Encounter", "FactType", "MedicationEvent",
    "MedicationEventType", "Observation", "ObservationPoint", "Patient",
    "PatientStatus", "Polarity", "ReviewStatus", "RunStatus",
    "TemporalRelation", "TimelineEntry",
]
