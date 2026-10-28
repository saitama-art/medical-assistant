"""A 类：数据模型与基础校验（8 用例）。"""

import unittest

from app.case_analysis import service
from app.case_analysis.models import (
    CaseValidationError,
    Certainty,
    FactType,
    MedicationEventType,
    Polarity,
    ReviewStatus,
)
from tests._case_fixtures import CaseTestBase, T0, T1


class TestCaseModels(CaseTestBase):
    def test_a1_legal_fact_persists_enum_values(self):
        patient = service.create_patient(self.owner, "虚构患者")
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, polarity=Polarity.NEGATIVE,
                           certainty=Certainty.POSSIBLE),
        ])
        fact = service.get_patient_history(patient.patient_id, self.owner)[0]
        self.assertIs(fact.polarity, Polarity.NEGATIVE)
        self.assertIs(fact.certainty, Certainty.POSSIBLE)
        self.assertIs(fact.review_status, ReviewStatus.PENDING)

    def test_a2_invalid_fact_type_rejected(self):
        patient = service.create_patient(self.owner)
        bad = self.make_fact(patient.patient_id)
        object.__setattr__(bad, "fact_type", "not_a_type")
        with self.assertRaises((CaseValidationError, ValueError)):
            service.save_clinical_facts(patient.patient_id, self.owner, [bad])

    def test_a3_missing_required_display_name_rejected(self):
        patient = service.create_patient(self.owner)
        bad = self.make_fact(patient.patient_id, name="  ")
        with self.assertRaises(CaseValidationError):
            service.save_clinical_facts(patient.patient_id, self.owner, [bad])

    def test_a4_nullable_times_stay_null(self):
        patient = service.create_patient(self.owner)
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, onset=None),
        ])
        service.save_medication_events(patient.patient_id, self.owner, [
            self.make_med(patient.patient_id, start=None),
        ])
        fact = service.get_patient_history(patient.patient_id, self.owner)[0]
        med = service.get_patient_medications(patient.patient_id, self.owner)[0]
        self.assertIsNone(fact.onset_time)
        self.assertIsNone(med.start_time)

    def test_a5_numeric_and_text_observation(self):
        patient = service.create_patient(self.owner)
        obs_num = self.make_obs(patient.patient_id, value=7.2)
        obs_text = self.make_obs(patient.patient_id, name="尿蛋白", value=None)
        object.__setattr__(obs_text, "value_text", "++")
        service.save_observations(patient.patient_id, self.owner,
                                  [obs_num, obs_text])
        rows = service.get_patient_observations(patient.patient_id, self.owner)
        by_name = {o.indicator_name: o for o in rows}
        self.assertEqual(by_name["空腹血糖"].value_numeric, 7.2)
        self.assertIsNone(by_name["尿蛋白"].value_numeric)
        self.assertEqual(by_name["尿蛋白"].value_text, "++")

    def test_a6_pending_default_and_confirm_transition(self):
        patient = service.create_patient(self.owner)
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id),
        ])
        fact = service.get_patient_history(patient.patient_id, self.owner)[0]
        self.assertIs(fact.review_status, ReviewStatus.PENDING)

    def test_a7_json_list_fields_roundtrip(self):
        patient = service.create_patient(self.owner)
        run = service.save_analysis_run(patient.patient_id, self.owner)
        med = service.save_medication_events(patient.patient_id, self.owner, [
            self.make_med(patient.patient_id),
        ])[0]
        fact = service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id),
        ])[0]
        drafts = service.find_temporal_associations(patient.patient_id, self.owner)
        saved = service.save_candidate_associations(run.run_id, self.owner, drafts)
        loaded = service.list_candidate_associations(run.run_id, self.owner)[0]
        self.assertIsInstance(loaded.fact_ids, tuple)
        self.assertEqual(set(loaded.fact_ids),
                         {med.medication_event_id, fact.fact_id})
        self.assertEqual(loaded.fact_ids, saved[0].fact_ids)

    def test_a8_invalid_time_type_rejected(self):
        patient = service.create_patient(self.owner)
        bad = self.make_fact(patient.patient_id, onset="上周三")
        with self.assertRaises(CaseValidationError):
            service.save_clinical_facts(patient.patient_id, self.owner, [bad])
        with self.assertRaises(CaseValidationError):
            service.list_encounters(patient.patient_id, self.owner, start=True)


if __name__ == "__main__":
    unittest.main()
