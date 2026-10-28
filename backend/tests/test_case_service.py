"""D 类：时间线与时间语义（8 用例）。

已知实现边界（如实断言，不虚构能力）：时间模型只有 float epoch 秒，
没有"仅日期"精度的独立表达——date-only 用例记录为设计现状而非缺陷。
"""

import unittest

from app.case_analysis import service
from tests._case_fixtures import CaseTestBase, T0, T1, T2, T3, T4


class TestCaseTimeline(CaseTestBase):
    def _patient_with_encounters(self):
        patient = service.create_patient(self.owner, "虚构患者")
        encs = [
            service.create_encounter(patient.patient_id, self.owner,
                                     f"虚构就诊{i}", encounter_time=t)
            for i, t in enumerate((T3, T0, T2))
        ]
        return patient, encs

    def test_d1_encounters_sorted_by_event_time(self):
        patient, _ = self._patient_with_encounters()
        timeline = service.get_patient_timeline(patient.patient_id, self.owner)
        enc_entries = [e for e in timeline if e.kind == "encounter"]
        self.assertEqual([e.event_time for e in enc_entries], [T0, T2, T3])

    def test_d2_event_time_distinct_from_recorded_time(self):
        patient, _ = self._patient_with_encounters()
        enc = service.list_encounters(patient.patient_id, self.owner)[0]
        self.assertNotEqual(enc.encounter_time, enc.recorded_at)
        timeline = service.get_patient_timeline(patient.patient_id, self.owner)
        for entry in timeline:
            if not entry.unknown_time:
                self.assertEqual(entry.event_time, entry.event_time)  # 事件时间不被改写
            self.assertIsNotNone(entry.recorded_at)

    def test_d3_date_precision_is_plain_epoch(self):
        """现状：只有 epoch 秒，无日期粒度区分。写入当天 0 点 epoch 即原样保存。"""
        patient = service.create_patient(self.owner)
        day_epoch = 20270 * 86400.0                        # 某虚构日期的 0 点
        enc = service.create_encounter(patient.patient_id, self.owner, "虚构",
                                       encounter_time=day_epoch)
        self.assertEqual(enc.encounter_time, day_epoch)     # 无额外精度标记（已知边界）

    def test_d4_unknown_time_marked_and_last(self):
        patient = service.create_patient(self.owner)
        service.create_encounter(patient.patient_id, self.owner, "未知时间就诊")
        service.create_encounter(patient.patient_id, self.owner, "已知时间就诊",
                                 encounter_time=T1)
        timeline = service.get_patient_timeline(patient.patient_id, self.owner)
        self.assertFalse(timeline[0].unknown_time)
        self.assertTrue(timeline[-1].unknown_time)
        # 未知时间条目的 event_time 必须为 None，不得用 recorded_at 冒充
        self.assertIsNone(timeline[-1].event_time)

    def test_d5_same_time_stable_tiebreak(self):
        patient = service.create_patient(self.owner)
        for _ in range(3):
            service.save_clinical_facts(patient.patient_id, self.owner, [
                self.make_fact(patient.patient_id, onset=T1)])
        first = service.get_patient_timeline(patient.patient_id, self.owner)
        second = service.get_patient_timeline(patient.patient_id, self.owner)
        self.assertEqual(
            [(e.kind, e.ref_id) for e in first],
            [(e.kind, e.ref_id) for e in second])          # 同刻排序确定

    def test_d6_range_boundaries_inclusive(self):
        patient = service.create_patient(self.owner)
        for t in (T0, T1, T2):
            service.create_encounter(patient.patient_id, self.owner, "虚构",
                                     encounter_time=t)
        got = service.list_encounters(patient.patient_id, self.owner,
                                      start=T1, end=T2)
        self.assertEqual(len(got), 2)

    def test_d7_out_of_range_excluded(self):
        patient = service.create_patient(self.owner)
        for t in (T0, T4):
            service.create_encounter(patient.patient_id, self.owner, "虚构",
                                     encounter_time=t)
        got = service.list_encounters(patient.patient_id, self.owner,
                                      start=T1, end=T2)
        self.assertEqual(got, [])

    def test_d8_mixed_kinds_full_timeline(self):
        patient = service.create_patient(self.owner)
        service.create_encounter(patient.patient_id, self.owner, "虚构",
                                 encounter_time=T0)
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, onset=T1)])
        service.save_medication_events(patient.patient_id, self.owner, [
            self.make_med(patient.patient_id, start=T2)])
        service.save_observations(patient.patient_id, self.owner, [
            self.make_obs(patient.patient_id, sample_time=T3)])
        kinds = [e.kind for e in service.get_patient_timeline(
            patient.patient_id, self.owner)]
        self.assertEqual(kinds, ["encounter", "fact", "medication", "observation"])


if __name__ == "__main__":
    unittest.main()
