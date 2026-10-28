"""纵向病例分析的 SQL 持久化层：参数化查询 + 归属校验的唯一出口。

纪律（任务书 §四/§七）：
- 所有 SQL 参数化，不拼接用户输入；
- 每个患者级读/写先经 _assert_patient_owner 校验归属，不存在与不属于你
  抛同一个 CaseAccessDeniedError（防枚举，与 conversations 一致）；
- 日志只记机器标识，不输出病历原文；
- 只负责存取，不做任何分析判断。
"""

from __future__ import annotations

import json
import sqlite3
import uuid

from ..db import connect, init_db, now_ts
from .models import (
    AbnormalFlag,
    AnalysisRun,
    AssociationDraft,
    AssociationType,
    CandidateAssociation,
    CaseAccessDeniedError,
    CaseValidationError,
    Certainty,
    ClinicalFact,
    Encounter,
    FactType,
    MedicationEvent,
    MedicationEventType,
    Observation,
    Patient,
    PatientStatus,
    Polarity,
    ReviewStatus,
    RunStatus,
    TemporalRelation,
)


def _new_id() -> str:
    return uuid.uuid4().hex


def _require_str(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseValidationError(f"{name}_required")
    return value.strip()


def _require_owner(owner_user_id) -> int:
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) \
            or owner_user_id <= 0:
        raise CaseValidationError("owner_user_id_invalid")
    return owner_user_id


def _opt_time(value, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaseValidationError(f"{name}_invalid")
    return float(value)


# ==================== 行映射 ====================


def _patient_from_row(row: sqlite3.Row) -> Patient:
    return Patient(
        patient_id=row["patient_id"], owner_user_id=int(row["owner_user_id"]),
        display_label=row["display_label"], status=PatientStatus(row["status"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _encounter_from_row(row: sqlite3.Row) -> Encounter:
    return Encounter(
        encounter_id=row["encounter_id"], patient_id=row["patient_id"],
        encounter_time=row["encounter_time"], recorded_at=row["recorded_at"],
        source_type=row["source_type"], raw_note=row["raw_note"],
        created_by=int(row["created_by"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _fact_from_row(row: sqlite3.Row) -> ClinicalFact:
    return ClinicalFact(
        fact_id=row["fact_id"], patient_id=row["patient_id"],
        encounter_id=row["encounter_id"], fact_type=FactType(row["fact_type"]),
        display_name=row["display_name"], code=row["code"],
        value_text=row["value_text"], value_numeric=row["value_numeric"],
        unit=row["unit"], polarity=Polarity(row["polarity"]),
        certainty=Certainty(row["certainty"]), onset_time=row["onset_time"],
        end_time=row["end_time"], recorded_at=row["recorded_at"],
        source_text=row["source_text"], source_start=row["source_start"],
        source_end=row["source_end"], review_status=ReviewStatus(row["review_status"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _med_from_row(row: sqlite3.Row) -> MedicationEvent:
    return MedicationEvent(
        medication_event_id=row["medication_event_id"],
        patient_id=row["patient_id"], encounter_id=row["encounter_id"],
        medication_name=row["medication_name"],
        normalized_name=row["normalized_name"], dose=row["dose"],
        dose_unit=row["dose_unit"], frequency=row["frequency"], route=row["route"],
        start_time=row["start_time"], end_time=row["end_time"],
        event_type=MedicationEventType(row["event_type"]),
        usage_status=row["usage_status"], source_text=row["source_text"],
        review_status=ReviewStatus(row["review_status"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _obs_from_row(row: sqlite3.Row) -> Observation:
    return Observation(
        observation_id=row["observation_id"], patient_id=row["patient_id"],
        encounter_id=row["encounter_id"], indicator_name=row["indicator_name"],
        normalized_name=row["normalized_name"], value_numeric=row["value_numeric"],
        value_text=row["value_text"], unit=row["unit"],
        reference_low=row["reference_low"], reference_high=row["reference_high"],
        abnormal_flag=AbnormalFlag(row["abnormal_flag"]),
        sample_time=row["sample_time"], reported_at=row["reported_at"],
        source_text=row["source_text"], review_status=ReviewStatus(row["review_status"]),
        created_at=row["created_at"],
    )


def _run_from_row(row: sqlite3.Row) -> AnalysisRun:
    return AnalysisRun(
        run_id=row["run_id"], patient_id=row["patient_id"],
        owner_user_id=int(row["owner_user_id"]), triggered_at=row["triggered_at"],
        time_range_start=row["time_range_start"],
        time_range_end=row["time_range_end"], status=RunStatus(row["status"]),
        input_summary=row["input_summary"], error_code=row["error_code"],
        created_at=row["created_at"],
    )


def _assoc_from_row(row: sqlite3.Row) -> CandidateAssociation:
    return CandidateAssociation(
        association_id=row["association_id"], run_id=row["run_id"],
        patient_id=row["patient_id"],
        association_type=AssociationType(row["association_type"]),
        fact_ids=tuple(json.loads(row["fact_ids"])),
        temporal_relationship=TemporalRelation(row["temporal_relationship"]),
        supporting_evidence=tuple(json.loads(row["supporting_evidence"])),
        contradicting_evidence=tuple(json.loads(row["contradicting_evidence"])),
        missing_information=tuple(json.loads(row["missing_information"])),
        explanation=row["explanation"], certainty=Certainty(row["certainty"]),
        review_status=ReviewStatus(row["review_status"]), created_at=row["created_at"],
    )


# ==================== 归属校验 ====================


def get_patient(conn: sqlite3.Connection, patient_id: str,
                owner_user_id: int) -> Patient:
    """归属校验的唯一入口：不存在或不属于调用者抛同一 CaseAccessDeniedError。"""
    _require_str(patient_id, "patient_id")
    _require_owner(owner_user_id)
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_id=?", (patient_id,),
    ).fetchone()
    if row is None or int(row["owner_user_id"]) != int(owner_user_id):
        raise CaseAccessDeniedError("patient_access_denied")
    return _patient_from_row(row)


def _assert_encounter_in_patient(conn: sqlite3.Connection,
                                 encounter_id: str | None,
                                 patient_id: str) -> None:
    """事实/用药/观察关联的就诊记录必须属于同一患者。"""
    if encounter_id is None:
        return
    _require_str(encounter_id, "encounter_id")
    row = conn.execute(
        "SELECT patient_id FROM encounters WHERE encounter_id=?", (encounter_id,),
    ).fetchone()
    if row is None or row["patient_id"] != patient_id:
        raise CaseValidationError("encounter_patient_mismatch")


# ==================== Patient ====================


def create_patient(owner_user_id: int, display_label: str = "") -> Patient:
    _require_owner(owner_user_id)
    init_db()
    ts = now_ts()
    patient = Patient(
        patient_id=_new_id(), owner_user_id=owner_user_id,
        display_label=(display_label or "")[:100], created_at=ts, updated_at=ts,
    )
    with connect() as conn:
        conn.execute(
            "INSERT INTO patients(patient_id, owner_user_id, display_label, status,"
            " created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (patient.patient_id, owner_user_id, patient.display_label,
             patient.status.value, ts, ts),
        )
        conn.commit()
    return patient


def get_patient_checked(patient_id: str, owner_user_id: int) -> Patient:
    init_db()
    with connect() as conn:
        return get_patient(conn, patient_id, owner_user_id)


def list_patients(owner_user_id: int, *, limit: int = 200) -> list[Patient]:
    _require_owner(owner_user_id)
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE owner_user_id=? AND status='active'"
            " ORDER BY updated_at DESC LIMIT ?",
            (owner_user_id, limit),
        ).fetchall()
    return [_patient_from_row(r) for r in rows]


# ==================== Encounter ====================


def create_encounter(patient_id: str, owner_user_id: int, raw_note: str, *,
                     encounter_time: float | None = None,
                     source_type: str = "manual_note") -> Encounter:
    _require_str(raw_note, "raw_note")
    encounter_time = _opt_time(encounter_time, "encounter_time")
    _require_str(source_type, "source_type")
    init_db()
    ts = now_ts()
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        encounter = Encounter(
            encounter_id=_new_id(), patient_id=patient_id,
            encounter_time=encounter_time, recorded_at=ts,
            source_type=source_type, raw_note=raw_note,
            created_by=owner_user_id, created_at=ts, updated_at=ts,
        )
        conn.execute(
            "INSERT INTO encounters(encounter_id, patient_id, encounter_time,"
            " recorded_at, source_type, raw_note, created_by, created_at, updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (encounter.encounter_id, patient_id, encounter_time, ts, source_type,
             raw_note, owner_user_id, ts, ts),
        )
        conn.commit()
    return encounter


def list_encounters(patient_id: str, owner_user_id: int, *,
                    start: float | None = None, end: float | None = None,
                    limit: int = 500) -> list[Encounter]:
    init_db()
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        sql = "SELECT * FROM encounters WHERE patient_id=?"
        params: list = [patient_id]
        if start is not None:
            sql += " AND (encounter_time IS NULL OR encounter_time>=?)"
            params.append(_opt_time(start, "start"))
        if end is not None:
            sql += " AND (encounter_time IS NULL OR encounter_time<=?)"
            params.append(_opt_time(end, "end"))
        sql += " ORDER BY encounter_time IS NULL, encounter_time, recorded_at LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [_encounter_from_row(r) for r in rows]


# ==================== ClinicalFact ====================


def save_clinical_facts(patient_id: str, owner_user_id: int,
                        facts: list[ClinicalFact]) -> list[ClinicalFact]:
    """批量保存事实。encounter_id 必须属于同一患者；fact_id 缺失时生成。"""
    if not isinstance(facts, list) or not facts:
        raise CaseValidationError("facts_empty")
    init_db()
    ts = now_ts()
    out: list[ClinicalFact] = []
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        for fact in facts:
            if not isinstance(fact, ClinicalFact):
                raise CaseValidationError("fact_type_invalid")
            _require_str(fact.display_name, "display_name")
            if fact.patient_id and fact.patient_id != patient_id:
                raise CaseValidationError("fact_patient_mismatch")
            _assert_encounter_in_patient(conn, fact.encounter_id, patient_id)
            record = ClinicalFact(
                fact_id=fact.fact_id or _new_id(), patient_id=patient_id,
                encounter_id=fact.encounter_id, fact_type=FactType(fact.fact_type),
                display_name=fact.display_name.strip(), code=fact.code,
                value_text=fact.value_text, value_numeric=fact.value_numeric,
                unit=fact.unit, polarity=Polarity(fact.polarity),
                certainty=Certainty(fact.certainty),
                onset_time=_opt_time(fact.onset_time, "onset_time"),
                end_time=_opt_time(fact.end_time, "end_time"),
                recorded_at=fact.recorded_at or ts,
                source_text=fact.source_text,
                source_start=fact.source_start, source_end=fact.source_end,
                review_status=ReviewStatus(fact.review_status),
                created_at=ts, updated_at=ts,
            )
            conn.execute(
                "INSERT OR REPLACE INTO clinical_facts("
                "fact_id,patient_id,encounter_id,fact_type,code,display_name,"
                "value_text,value_numeric,unit,polarity,certainty,onset_time,"
                "end_time,recorded_at,source_text,source_start,source_end,"
                "review_status,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.fact_id, patient_id, record.encounter_id,
                 record.fact_type.value, record.code, record.display_name,
                 record.value_text, record.value_numeric, record.unit,
                 record.polarity.value, record.certainty.value, record.onset_time,
                 record.end_time, record.recorded_at, record.source_text,
                 record.source_start, record.source_end, record.review_status.value,
                 ts, ts),
            )
            out.append(record)
        conn.commit()
    return out


def list_facts(patient_id: str, owner_user_id: int, *,
               fact_types: list[str] | None = None,
               start: float | None = None, end: float | None = None,
               review_status: str | None = None,
               limit: int = 1000) -> list[ClinicalFact]:
    init_db()
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        sql = "SELECT * FROM clinical_facts WHERE patient_id=?"
        params: list = [patient_id]
        if fact_types:
            valid = [FactType(t).value for t in fact_types]
            sql += f" AND fact_type IN ({','.join('?' * len(valid))})"
            params.extend(valid)
        if start is not None:
            sql += " AND (onset_time IS NULL OR onset_time>=?)"
            params.append(_opt_time(start, "start"))
        if end is not None:
            sql += " AND (onset_time IS NULL OR onset_time<=?)"
            params.append(_opt_time(end, "end"))
        if review_status is not None:
            sql += " AND review_status=?"
            params.append(ReviewStatus(review_status).value)
        sql += " ORDER BY onset_time IS NULL, onset_time, recorded_at LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [_fact_from_row(r) for r in rows]


# ==================== MedicationEvent ====================


def save_medication_events(patient_id: str, owner_user_id: int,
                           events: list[MedicationEvent]) -> list[MedicationEvent]:
    if not isinstance(events, list) or not events:
        raise CaseValidationError("medication_events_empty")
    init_db()
    ts = now_ts()
    out: list[MedicationEvent] = []
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        for event in events:
            if not isinstance(event, MedicationEvent):
                raise CaseValidationError("medication_event_invalid")
            _require_str(event.medication_name, "medication_name")
            if event.patient_id and event.patient_id != patient_id:
                raise CaseValidationError("medication_patient_mismatch")
            _assert_encounter_in_patient(conn, event.encounter_id, patient_id)
            record = MedicationEvent(
                medication_event_id=event.medication_event_id or _new_id(),
                patient_id=patient_id, encounter_id=event.encounter_id,
                medication_name=event.medication_name.strip(),
                normalized_name=event.normalized_name
                or event.medication_name.strip().lower(),
                dose=event.dose, dose_unit=event.dose_unit,
                frequency=event.frequency, route=event.route,
                start_time=_opt_time(event.start_time, "start_time"),
                end_time=_opt_time(event.end_time, "end_time"),
                event_type=MedicationEventType(event.event_type),
                usage_status=event.usage_status, source_text=event.source_text,
                review_status=ReviewStatus(event.review_status),
                created_at=ts, updated_at=ts,
            )
            conn.execute(
                "INSERT OR REPLACE INTO medication_events("
                "medication_event_id,patient_id,encounter_id,medication_name,"
                "normalized_name,dose,dose_unit,frequency,route,start_time,"
                "end_time,event_type,usage_status,source_text,review_status,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.medication_event_id, patient_id, record.encounter_id,
                 record.medication_name, record.normalized_name, record.dose,
                 record.dose_unit, record.frequency, record.route,
                 record.start_time, record.end_time, record.event_type.value,
                 record.usage_status, record.source_text,
                 record.review_status.value, ts, ts),
            )
            out.append(record)
        conn.commit()
    return out


def list_medication_events(patient_id: str, owner_user_id: int, *,
                           start: float | None = None, end: float | None = None,
                           limit: int = 1000) -> list[MedicationEvent]:
    init_db()
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        sql = "SELECT * FROM medication_events WHERE patient_id=?"
        params: list = [patient_id]
        if start is not None:
            sql += " AND (start_time IS NULL OR end_time IS NULL OR end_time>=?)"
            params.append(_opt_time(start, "start"))
        if end is not None:
            sql += " AND (start_time IS NULL OR start_time<=?)"
            params.append(_opt_time(end, "end"))
        sql += " ORDER BY start_time IS NULL, start_time, created_at LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [_med_from_row(r) for r in rows]


# ==================== Observation ====================


def save_observations(patient_id: str, owner_user_id: int,
                      observations: list[Observation]) -> list[Observation]:
    if not isinstance(observations, list) or not observations:
        raise CaseValidationError("observations_empty")
    init_db()
    ts = now_ts()
    out: list[Observation] = []
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        for obs in observations:
            if not isinstance(obs, Observation):
                raise CaseValidationError("observation_invalid")
            _require_str(obs.indicator_name, "indicator_name")
            if obs.patient_id and obs.patient_id != patient_id:
                raise CaseValidationError("observation_patient_mismatch")
            _assert_encounter_in_patient(conn, obs.encounter_id, patient_id)
            record = Observation(
                observation_id=obs.observation_id or _new_id(),
                patient_id=patient_id, encounter_id=obs.encounter_id,
                indicator_name=obs.indicator_name.strip(),
                normalized_name=obs.normalized_name
                or obs.indicator_name.strip().lower(),
                value_numeric=obs.value_numeric, value_text=obs.value_text,
                unit=obs.unit, reference_low=obs.reference_low,
                reference_high=obs.reference_high,
                abnormal_flag=AbnormalFlag(obs.abnormal_flag),
                sample_time=_opt_time(obs.sample_time, "sample_time"),
                reported_at=_opt_time(obs.reported_at, "reported_at"),
                source_text=obs.source_text,
                review_status=ReviewStatus(obs.review_status), created_at=ts,
            )
            conn.execute(
                "INSERT OR REPLACE INTO observations("
                "observation_id,patient_id,encounter_id,indicator_name,"
                "normalized_name,value_numeric,value_text,unit,reference_low,"
                "reference_high,abnormal_flag,sample_time,reported_at,"
                "source_text,review_status,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.observation_id, patient_id, record.encounter_id,
                 record.indicator_name, record.normalized_name,
                 record.value_numeric, record.value_text, record.unit,
                 record.reference_low, record.reference_high,
                 record.abnormal_flag.value, record.sample_time,
                 record.reported_at, record.source_text,
                 record.review_status.value, ts),
            )
            out.append(record)
        conn.commit()
    return out


def list_observations(patient_id: str, owner_user_id: int, *,
                      indicator: str | None = None,
                      start: float | None = None, end: float | None = None,
                      limit: int = 1000) -> list[Observation]:
    init_db()
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        sql = "SELECT * FROM observations WHERE patient_id=?"
        params: list = [patient_id]
        if indicator is not None:
            sql += " AND (normalized_name=? OR indicator_name=?)"
            params.extend([indicator.strip().lower(), indicator.strip()])
        if start is not None:
            sql += " AND (sample_time IS NULL OR sample_time>=?)"
            params.append(_opt_time(start, "start"))
        if end is not None:
            sql += " AND (sample_time IS NULL OR sample_time<=?)"
            params.append(_opt_time(end, "end"))
        sql += " ORDER BY sample_time IS NULL, sample_time, created_at LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [_obs_from_row(r) for r in rows]


# ==================== AnalysisRun / CandidateAssociation ====================


def create_analysis_run(patient_id: str, owner_user_id: int, *,
                        time_range_start: float | None = None,
                        time_range_end: float | None = None,
                        input_summary: str = "") -> AnalysisRun:
    init_db()
    ts = now_ts()
    with connect() as conn:
        get_patient(conn, patient_id, owner_user_id)
        run = AnalysisRun(
            run_id=_new_id(), patient_id=patient_id, owner_user_id=owner_user_id,
            triggered_at=ts,
            time_range_start=_opt_time(time_range_start, "time_range_start"),
            time_range_end=_opt_time(time_range_end, "time_range_end"),
            status=RunStatus.PENDING, input_summary=input_summary[:500],
            created_at=ts,
        )
        conn.execute(
            "INSERT INTO analysis_runs(run_id,patient_id,owner_user_id,"
            "triggered_at,time_range_start,time_range_end,status,input_summary,"
            "error_code,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (run.run_id, patient_id, owner_user_id, ts, run.time_range_start,
             run.time_range_end, run.status.value, run.input_summary, "", ts),
        )
        conn.commit()
    return run


def update_analysis_run(run_id: str, owner_user_id: int, *,
                        status: str,
                        error_code: str = "") -> AnalysisRun:
    _require_str(run_id, "run_id")
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row is None or int(row["owner_user_id"]) != int(owner_user_id):
            raise CaseAccessDeniedError("run_access_denied")
        new_status = RunStatus(status)
        conn.execute(
            "UPDATE analysis_runs SET status=?, error_code=? WHERE run_id=?",
            (new_status.value, error_code, run_id),
        )
        conn.commit()
        return _run_from_row(conn.execute(
            "SELECT * FROM analysis_runs WHERE run_id=?", (run_id,),
        ).fetchone())


def save_candidate_associations(run_id: str, owner_user_id: int,
                                drafts: list[CandidateAssociation | AssociationDraft],
                                ) -> list[CandidateAssociation]:
    """保存一次分析的候选关联。接受 AssociationDraft（分析层产出）或
    CandidateAssociation；run_id/patient_id 一律以运行记录为准。"""
    if not isinstance(drafts, list):
        raise CaseValidationError("associations_invalid")
    init_db()
    ts = now_ts()
    out: list[CandidateAssociation] = []
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row is None or int(row["owner_user_id"]) != int(owner_user_id):
            raise CaseAccessDeniedError("run_access_denied")
        patient_id = row["patient_id"]
        for draft in drafts:
            record = CandidateAssociation(
                association_id=getattr(draft, "association_id", "") or _new_id(),
                run_id=run_id, patient_id=patient_id,
                association_type=AssociationType(draft.association_type),
                fact_ids=tuple(draft.fact_ids),
                temporal_relationship=TemporalRelation(draft.temporal_relationship),
                supporting_evidence=tuple(draft.supporting_evidence),
                contradicting_evidence=tuple(draft.contradicting_evidence),
                missing_information=tuple(draft.missing_information),
                explanation=draft.explanation,
                certainty=Certainty(draft.certainty),
                review_status=ReviewStatus(
                    getattr(draft, "review_status", ReviewStatus.PENDING)),
                created_at=ts,
            )
            conn.execute(
                "INSERT OR REPLACE INTO candidate_associations("
                "association_id,run_id,patient_id,association_type,fact_ids,"
                "temporal_relationship,supporting_evidence,contradicting_evidence,"
                "missing_information,explanation,certainty,review_status,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.association_id, run_id, patient_id,
                 record.association_type.value,
                 json.dumps(list(record.fact_ids), ensure_ascii=False),
                 record.temporal_relationship.value,
                 json.dumps(list(record.supporting_evidence), ensure_ascii=False),
                 json.dumps(list(record.contradicting_evidence), ensure_ascii=False),
                 json.dumps(list(record.missing_information), ensure_ascii=False),
                 record.explanation, record.certainty.value,
                 record.review_status.value, ts),
            )
            out.append(record)
        conn.commit()
    return out


def list_candidate_associations(run_id: str, owner_user_id: int,
                                ) -> list[CandidateAssociation]:
    _require_str(run_id, "run_id")
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if row is None or int(row["owner_user_id"]) != int(owner_user_id):
            raise CaseAccessDeniedError("run_access_denied")
        rows = conn.execute(
            "SELECT * FROM candidate_associations WHERE run_id=?"
            " ORDER BY created_at, association_id", (run_id,),
        ).fetchall()
    return [_assoc_from_row(r) for r in rows]
