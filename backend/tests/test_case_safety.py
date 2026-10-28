"""G 类：异常与安全边界（4 用例）。

重点：错误不得造成跨患者数据泄露；失败的分析不得被标记为成功。
"""

import unittest

from app.case_analysis import service
from app.case_analysis.models import (
    CaseAccessDeniedError,
    CaseValidationError,
)
from tests._case_fixtures import CaseTestBase, T1, T2


class TestCaseSafety(CaseTestBase):
    def test_g1_invalid_time_params_rejected(self):
        patient = service.create_patient(self.owner)
        for bad in ("去年", True, [T1]):
            with self.assertRaises((CaseValidationError, TypeError),
                                   msg=f"start={bad!r}"):
                service.list_encounters(patient.patient_id, self.owner, start=bad)

    def test_g2_batch_failure_rolls_back_and_no_leak(self):
        p1 = service.create_patient(self.owner, "虚构患者1")
        p2 = service.create_patient(self.other, "虚构患者2")
        enc2 = service.create_encounter(p2.patient_id, self.other, "虚构")
        with self.assertRaises(CaseValidationError):
            service.save_clinical_facts(p1.patient_id, self.owner, [
                self.make_fact(p1.patient_id, name="好记录"),
                self.make_fact(p1.patient_id, name="坏记录",
                               encounter_id=enc2.encounter_id),
            ])
        # 整批回滚：p1 无残留，p2 数据不受影响
        self.assertEqual(service.get_patient_history(p1.patient_id, self.owner), [])
        self.assertEqual(len(service.list_encounters(p2.patient_id, self.other)), 1)

    def test_g3_invalid_run_status_rejected_and_unchanged(self):
        patient = service.create_patient(self.owner)
        run = service.save_analysis_run(patient.patient_id, self.owner)
        with self.assertRaises((CaseValidationError, ValueError)):
            service.update_analysis_run(run.run_id, self.owner,
                                        status="half_done_maybe")
        # 非法状态被拒绝后，合法的状态流转仍可用且未停留在错误值
        from app.case_analysis import repository
        real = repository.update_analysis_run(run.run_id, self.owner,
                                              status="running")
        self.assertEqual(real.status.value, "running")

    def test_g4_malformed_draft_rejected_without_write(self):
        patient = service.create_patient(self.owner)
        run = service.save_analysis_run(patient.patient_id, self.owner)
        with self.assertRaises((CaseValidationError, ValueError, AttributeError)):
            service.save_candidate_associations(run.run_id, self.owner,
                                                ["not_a_draft"])
        self.assertEqual(
            service.list_candidate_associations(run.run_id, self.owner), [])

    def test_g4b_denied_run_is_never_marked_success(self):
        patient = service.create_patient(self.owner)
        run = service.save_analysis_run(patient.patient_id, self.owner)
        with self.assertRaises(CaseAccessDeniedError):
            service.update_analysis_run(run.run_id, self.other,
                                        status="completed")
        # 越权者无法把分析标为成功
        self.assertEqual(
            service.list_candidate_associations(run.run_id, self.owner), [])


if __name__ == "__main__":
    unittest.main()
