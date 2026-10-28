"""B 类：数据持久化与查询（8 用例）+ 迁移幂等。"""

import unittest

from app.case_analysis import service
from app.case_analysis.models import CaseValidationError, FactType
from app.db import connect, init_db
from tests._case_fixtures import CaseTestBase, T0, T1, T2, T3


class TestCaseRepository(CaseTestBase):
    def test_b1_full_write_read_roundtrip(self):
        patient = service.create_patient(self.owner, "虚构患者")
        enc = service.create_encounter(patient.patient_id, self.owner,
                                       "虚构：随访记录", encounter_time=T1)
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, encounter_id=enc.encounter_id)])
        service.save_medication_events(patient.patient_id, self.owner, [
            self.make_med(patient.patient_id, encounter_id=enc.encounter_id)])
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, encounter_id=enc.encounter_id)])
        self.assertEqual(len(service.list_encounters(patient.patient_id, self.owner)), 1)
        self.assertEqual(len(service.get_patient_history(patient.patient_id, self.owner)), 1)
        self.assertEqual(len(service.get_patient_medications(patient.patient_id, self.owner)), 1)
        self.assertEqual(len(service.get_patient_observations(patient.patient_id, self.owner)), 1)

    def test_b2_time_range_filter(self):
        patient = service.create_patient(self.owner)
        for t in (T0, T1, T2):
            service.save_clinical_facts(patient.patient_id, self.owner, [
                self.make_fact(patient.patient_id, onset=t)])
        got = service.get_patient_history(patient.patient_id, self.owner,
                                          start=T1, end=T1)
        self.assertEqual(len(got), 1)          # 边界时间闭区间
        self.assertEqual(got[0].onset_time, T1)

    def test_b3_fact_type_and_review_status_filter(self):
        patient = service.create_patient(self.owner)
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, name="恶心"),
            self.make_fact(patient.patient_id, name="2型糖尿病",
                           fact_type=FactType.DISEASE),
        ])
        symptoms = service.get_patient_history(
            patient.patient_id, self.owner, fact_types=["symptom"])
        self.assertEqual([f.display_name for f in symptoms], ["恶心"])
        pending = service.get_patient_history(
            patient.patient_id, self.owner, review_status="pending")
        self.assertEqual(len(pending), 2)
        confirmed = service.get_patient_history(
            patient.patient_id, self.owner, review_status="confirmed")
        self.assertEqual(confirmed, [])

    def test_b4_empty_result_not_error(self):
        patient = service.create_patient(self.owner)
        self.assertEqual(service.get_patient_history(patient.patient_id, self.owner), [])
        self.assertEqual(service.get_patient_medications(patient.patient_id, self.owner), [])
        self.assertEqual(service.get_patient_observations(patient.patient_id, self.owner), [])
        self.assertEqual(service.list_encounters(patient.patient_id, self.owner), [])

    def test_b5_stable_ordering_by_event_time(self):
        patient = service.create_patient(self.owner)
        for t in (T3, T0, T2, T1):              # 乱序写入
            service.save_clinical_facts(patient.patient_id, self.owner, [
                self.make_fact(patient.patient_id, onset=t)])
        got = service.get_patient_history(patient.patient_id, self.owner)
        self.assertEqual([f.onset_time for f in got], [T0, T1, T2, T3])

    def test_b6_unknown_time_records_kept(self):
        patient = service.create_patient(self.owner)
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, name="未知时间", onset=None),
            self.make_fact(patient.patient_id, name="已知时间", onset=T1),
        ])
        got = service.get_patient_history(patient.patient_id, self.owner)
        self.assertEqual(len(got), 2)          # 未知时间不因查询丢失
        ranged = service.get_patient_history(patient.patient_id, self.owner,
                                             start=T0, end=T2)
        self.assertEqual(len(ranged), 2)       # 未知时间不被时间窗误杀

    def test_b7_foreign_encounter_rejected_and_batch_rollback(self):
        p1, p2 = service.create_patient(self.owner), service.create_patient(self.owner)
        enc2 = service.create_encounter(p2.patient_id, self.owner, "虚构")
        good = self.make_fact(p1.patient_id)
        bad = self.make_fact(p1.patient_id, name="坏记录",
                             encounter_id=enc2.encounter_id)
        with self.assertRaises(CaseValidationError):
            service.save_clinical_facts(p1.patient_id, self.owner, [good, bad])
        # 部分失败必须整批回滚，不得留下 good 记录
        self.assertEqual(service.get_patient_history(p1.patient_id, self.owner), [])

    def test_b8_migration_idempotent(self):
        service.create_patient(self.owner, "虚构患者")
        v1 = init_db()
        init_db()                                # 重复执行
        with connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            n = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        from app.db import _SCHEMA_VERSION
        self.assertEqual(version, _SCHEMA_VERSION)
        self.assertEqual(n, 1)                   # 数据不因重复迁移丢失


if __name__ == "__main__":
    unittest.main()
