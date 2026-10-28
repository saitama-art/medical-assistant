"""病例分析测试的共享 fixture：临时 SQLite + 虚构数据构造器。

不被 unittest discover 收集（下划线前缀）；各 test_case_*.py 从这里继承。
隔离方案：APP_DB_PATH 指向 TemporaryDirectory 内的独立库文件，
不触碰开发库；每条用例独立建库，互不依赖。
"""

import os
import tempfile
import unittest

from app.auth import users
from app.auth.passwords import hash_password
from app.case_analysis.models import (
    AbnormalFlag,
    Certainty,
    ClinicalFact,
    FactType,
    MedicationEvent,
    MedicationEventType,
    Observation,
    Polarity,
)

T0, T1, T2, T3, T4 = 1000.0, 2000.0, 3000.0, 4000.0, 5000.0


class CaseTestBase(unittest.TestCase):
    """每条用例一个全新临时库；预置两个用户，归属测试直接用。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["APP_DB_PATH"] = os.path.join(self.tmp.name, "case.db")
        self.addCleanup(os.environ.pop, "APP_DB_PATH", None)
        self.owner = users.create_user("case-owner", hash_password("pw-o-123456"))
        self.other = users.create_user("case-other", hash_password("pw-x-123456"))

    # ---- 构造器（全部虚构数据） ----

    def make_fact(self, patient_id, *, name="恶心", onset=T1,
                  fact_type=FactType.SYMPTOM, polarity=Polarity.POSITIVE,
                  certainty=Certainty.CONFIRMED, encounter_id=None,
                  recorded_at=T2, source_text="虚构原文片段"):
        return ClinicalFact(
            fact_id="", patient_id=patient_id, encounter_id=encounter_id,
            fact_type=fact_type, display_name=name, polarity=polarity,
            certainty=certainty, onset_time=onset, recorded_at=recorded_at,
            source_text=source_text,
        )

    def make_med(self, patient_id, *, name="二甲双胍",
                 event_type=MedicationEventType.REPORTED_ACTIVE,
                 start=T0, end=None, encounter_id=None, dose=None):
        return MedicationEvent(
            medication_event_id="", patient_id=patient_id,
            encounter_id=encounter_id, medication_name=name,
            dose=dose, start_time=start, end_time=end, event_type=event_type,
        )

    def make_obs(self, patient_id, *, name="空腹血糖", value=7.2, unit="mmol/L",
                 sample_time=T1, encounter_id=None, ref=(3.9, 6.1),
                 flag=AbnormalFlag.HIGH):
        return Observation(
            observation_id="", patient_id=patient_id, encounter_id=encounter_id,
            indicator_name=name, value_numeric=value, unit=unit,
            reference_low=ref[0], reference_high=ref[1],
            abnormal_flag=flag, sample_time=sample_time,
        )
