"""E 类：指标比较与确定性分析（8 用例）；F 类：候选关联与证据约束（6 用例）。"""

import unittest

from app.case_analysis import analysis, service
from app.case_analysis.models import (
    AssociationType,
    Certainty,
    MedicationEventType,
    Polarity,
    ReviewStatus,
    TemporalRelation,
)
from tests._case_fixtures import CaseTestBase, T0, T1, T2, T3, T4


class TestCaseComparison(CaseTestBase):
    def test_e1_multi_point_delta_chain(self):
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, value=7.2, sample_time=T1),
            self.make_obs(patient.patient_id, value=6.8, sample_time=T2),
            self.make_obs(patient.patient_id, value=6.1, sample_time=T3),
        ])
        comps = service.compare_observations(patient.patient_id, self.owner)
        good = [c for c in comps if c.comparable]
        self.assertEqual(len(good), 2)                    # 相邻两两比较
        self.assertAlmostEqual(good[0].delta, -0.4, places=4)
        self.assertAlmostEqual(good[1].delta, -0.7, places=4)
        for c in good:                                    # 原始值/单位/参考范围保留
            self.assertEqual(c.unit, "mmol/L")
            self.assertEqual(c.earlier.reference_low, 3.9)

    def test_e2_unit_mismatch_not_comparable(self):
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, value=7.2, sample_time=T1),
            self.make_obs(patient.patient_id, value=130.0, unit="mg/dL",
                          sample_time=T2),
        ])
        comps = service.compare_observations(patient.patient_id, self.owner)
        self.assertEqual(len(comps), 1)
        self.assertFalse(comps[0].comparable)
        self.assertEqual(comps[0].reason, "unit_mismatch")
        self.assertIsNone(comps[0].delta)                 # 不输出误导性差值

    def test_e3_missing_value_not_comparable(self):
        patient = service.create_patient(self.owner)
        obs = self.make_obs(patient.patient_id, sample_time=T2)
        object.__setattr__(obs, "value_numeric", None)
        object.__setattr__(obs, "value_text", "未见报告数值")
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, sample_time=T1), obs,
        ])
        comps = service.compare_observations(patient.patient_id, self.owner)
        self.assertEqual(comps[0].reason, "value_missing")

    def test_e4_reference_ranges_preserved_per_point(self):
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, value=7.2, sample_time=T1,
                          ref=(3.9, 6.1)),
            self.make_obs(patient.patient_id, value=6.5, sample_time=T2,
                          ref=(4.0, 7.0)),                # 不同实验室参考范围
        ])
        comp = service.compare_observations(patient.patient_id, self.owner)[0]
        self.assertEqual(comp.earlier.reference_high, 6.1)
        self.assertEqual(comp.later.reference_high, 7.0)

    def test_e5_duplicate_same_time_records(self):
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, value=7.2, sample_time=T1),
            self.make_obs(patient.patient_id, value=7.3, sample_time=T1),
        ])
        comps = service.compare_observations(patient.patient_id, self.owner)
        self.assertEqual(len(comps), 1)                   # 同刻也可比，确定性处理
        self.assertTrue(comps[0].comparable)

    def test_e6_no_data_in_range_returns_empty(self):
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, sample_time=T1),
        ])
        self.assertEqual(service.compare_observations(
            patient.patient_id, self.owner, start=T2, end=T3), [])

    def test_e7_reason_codes_machine_readable(self):
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, sample_time=None),
        ])
        comp = service.compare_observations(patient.patient_id, self.owner)[0]
        self.assertEqual(comp.reason, "sample_time_missing")
        self.assertFalse(comp.comparable)

    def test_e8_no_trend_language_in_output(self):
        """输出只有数值与状态，不含恶化/改善等医学解释（确定性边界）。"""
        patient = service.create_patient(self.owner)
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, value=7.2, sample_time=T1),
            self.make_obs(patient.patient_id, value=9.5, sample_time=T2),
        ])
        comp = service.compare_observations(patient.patient_id, self.owner)[0]
        self.assertTrue(comp.comparable)
        for word in ("恶化", "改善", "好转", "加重"):
            self.assertNotIn(word, comp.reason)


class TestCaseAssociations(CaseTestBase):
    def _seed_med_symptom(self, *, med_type=MedicationEventType.REPORTED_ACTIVE,
                          symptom_onset=T1, med_start=T0, med_end=None,
                          polarity=Polarity.POSITIVE):
        patient = service.create_patient(self.owner)
        med = service.save_medication_events(patient.patient_id, self.owner, [
            self.make_med(patient.patient_id, event_type=med_type,
                          start=med_start, end=med_end),
        ])[0]
        fact = service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, onset=symptom_onset,
                           polarity=polarity),
        ])[0]
        return patient, med, fact

    def test_f1_proximity_generates_candidate(self):
        patient, med, fact = self._seed_med_symptom()
        drafts = service.find_temporal_associations(patient.patient_id, self.owner)
        self.assertEqual(len(drafts), 1)
        d = drafts[0]
        self.assertIs(d.association_type, AssociationType.MEDICATION_SYMPTOM)
        self.assertEqual(set(d.fact_ids), {med.medication_event_id, fact.fact_id})
        self.assertIs(d.temporal_relationship, TemporalRelation.DURING)
        self.assertTrue(d.supporting_evidence)

    def test_f2_symptom_before_med_no_candidate(self):
        patient, _, _ = self._seed_med_symptom(symptom_onset=T0 - 100.0,
                                               med_start=T1)
        self.assertEqual(
            service.find_temporal_associations(patient.patient_id, self.owner), [])

    def test_f3_incomplete_time_no_fabricated_candidate(self):
        patient, _, _ = self._seed_med_symptom(symptom_onset=None)
        self.assertEqual(
            service.find_temporal_associations(patient.patient_id, self.owner), [])

    def test_f4_negated_symptom_excluded(self):
        patient, _, _ = self._seed_med_symptom(polarity=Polarity.NEGATIVE)
        self.assertEqual(
            service.find_temporal_associations(patient.patient_id, self.owner), [])

    def test_f5_prescribed_flags_missing_actual_use(self):
        patient, _, _ = self._seed_med_symptom(
            med_type=MedicationEventType.PRESCRIBED)
        drafts = service.find_temporal_associations(patient.patient_id, self.owner)
        self.assertEqual(len(drafts), 1)
        self.assertTrue(any("处方不等于服药" in m
                            for m in drafts[0].missing_information))
        self.assertTrue(any("剂量未知" in m
                            for m in drafts[0].missing_information))

    def test_f6_candidates_never_auto_confirmed(self):
        patient, _, _ = self._seed_med_symptom()
        run = service.save_analysis_run(patient.patient_id, self.owner)
        drafts = service.find_temporal_associations(patient.patient_id, self.owner)
        service.save_candidate_associations(run.run_id, self.owner, drafts)
        for assoc in service.list_candidate_associations(run.run_id, self.owner):
            self.assertIs(assoc.certainty, Certainty.UNCERTAIN)
            self.assertIs(assoc.review_status, ReviewStatus.PENDING)


class TestSymptomMedRelationMatrix(unittest.TestCase):
    """E/C 规则矩阵：纯函数全分支（并入自旧测试文件）。"""

    def test_relation_matrix(self):
        f = analysis.symptom_medication_relation
        self.assertIs(f(100.0, 200.0, 300.0), TemporalRelation.BEFORE)
        self.assertIs(f(250.0, 200.0, 300.0), TemporalRelation.DURING)
        self.assertIs(f(400.0, 200.0, 300.0), TemporalRelation.AFTER)
        self.assertIs(f(None, 200.0, 300.0), TemporalRelation.UNKNOWN)
        self.assertIs(f(250.0, None, 300.0), TemporalRelation.UNKNOWN)
        self.assertIs(f(250.0, 200.0, None), TemporalRelation.DURING)


if __name__ == "__main__":
    unittest.main()
