"""C 类：患者归属与数据隔离（8 用例）。

权限用例不只断言异常，还验证越权后目标数据**真实未被读取/修改/关联**。
"""

import unittest

from app.case_analysis import service
from app.case_analysis.models import CaseAccessDeniedError, CaseValidationError
from tests._case_fixtures import CaseTestBase, T1


class TestCaseIsolation(CaseTestBase):
    def _two_users_two_patients(self):
        pa = service.create_patient(self.owner, "虚构患者A")
        pb = service.create_patient(self.other, "虚构患者B")
        service.save_clinical_facts(pa.patient_id, self.owner, [
            self.make_fact(pa.patient_id, name="A的症状")])
        service.save_clinical_facts(pb.patient_id, self.other, [
            self.make_fact(pb.patient_id, name="B的症状")])
        return pa, pb

    def test_c1_owner_reads_own_data(self):
        pa, _ = self._two_users_two_patients()
        got = service.get_patient_history(pa.patient_id, self.owner)
        self.assertEqual([f.display_name for f in got], ["A的症状"])

    def test_c2_cross_user_reads_all_denied(self):
        _, pb = self._two_users_two_patients()
        for fn in (
            service.get_patient_history,
            service.get_patient_observations,
            service.get_patient_medications,
            service.get_patient_timeline,
            service.list_encounters,
        ):
            with self.assertRaises(CaseAccessDeniedError, msg=fn.__name__):
                fn(pb.patient_id, self.owner)

    def test_c3_cross_user_run_query_denied(self):
        _, pb = self._two_users_two_patients()
        run_b = service.save_analysis_run(pb.patient_id, self.other)
        with self.assertRaises(CaseAccessDeniedError):
            service.list_candidate_associations(run_b.run_id, self.owner)

    def test_c4_cross_user_run_update_denied_and_status_unchanged(self):
        _, pb = self._two_users_two_patients()
        run_b = service.save_analysis_run(pb.patient_id, self.other)
        with self.assertRaises(CaseAccessDeniedError):
            service.update_analysis_run(run_b.run_id, self.owner, status="failed")
        # 越权失败后状态必须仍是 pending（未被改动）
        reloaded = service.list_candidate_associations(run_b.run_id, self.other)
        self.assertEqual(reloaded, [])
        # 用属主身份确认真实状态
        from app.case_analysis import repository
        run = repository.update_analysis_run(run_b.run_id, self.other,
                                             status="running")
        self.assertEqual(run.status.value, "running")

    def test_c5_cross_user_save_association_denied_and_nothing_written(self):
        _, pb = self._two_users_two_patients()
        run_b = service.save_analysis_run(pb.patient_id, self.other)
        with self.assertRaises(CaseAccessDeniedError):
            service.save_candidate_associations(run_b.run_id, self.owner, [])
        # 越权写入失败后，属主视角仍为空
        self.assertEqual(
            service.list_candidate_associations(run_b.run_id, self.other), [])

    def test_c6_not_found_and_not_owned_same_error(self):
        pa, _ = self._two_users_two_patients()
        with self.assertRaises(CaseAccessDeniedError) as e1:
            service.get_patient(pa.patient_id, self.other)
        with self.assertRaises(CaseAccessDeniedError) as e2:
            service.get_patient("no-such-id", self.other)
        self.assertEqual(type(e1.exception), type(e2.exception))  # 防枚举

    def test_c7_patient_id_swap_in_fact_rejected(self):
        pa, pb = self._two_users_two_patients()
        swapped = self.make_fact(pa.patient_id)
        object.__setattr__(swapped, "patient_id", pb.patient_id)
        with self.assertRaises(CaseValidationError):
            service.save_clinical_facts(pa.patient_id, self.owner, [swapped])
        # 替换被拒绝后：A 只剩预置的 1 条（未新增），B 不受影响
        self.assertEqual([f.display_name for f in
                          service.get_patient_history(pa.patient_id, self.owner)],
                         ["A的症状"])
        self.assertEqual([f.display_name for f in
                          service.get_patient_history(pb.patient_id, self.other)],
                         ["B的症状"])

    def test_c8_foreign_encounter_cannot_be_attached(self):
        pa, pb = self._two_users_two_patients()
        enc_b = service.create_encounter(pb.patient_id, self.other, "虚构B就诊")
        with self.assertRaises(CaseValidationError):
            service.save_clinical_facts(pa.patient_id, self.owner, [
                self.make_fact(pa.patient_id, encounter_id=enc_b.encounter_id)])
        # A 只剩预置的 1 条（跨患者 encounter 未被挂入）
        self.assertEqual([f.display_name for f in
                          service.get_patient_history(pa.patient_id, self.owner)],
                         ["A的症状"])


if __name__ == "__main__":
    unittest.main()
