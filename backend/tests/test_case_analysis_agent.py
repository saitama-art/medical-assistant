"""纵向病例分析 Agent 集成验证（第一轮，任务书 §三 7 个用例）。

隔离与 mock 约定：
- 数据库：CaseTestBase 每条用例独立临时库（APP_DB_PATH），不碰开发库；
- LLM：一律 mock——不涉及叙述语义的用例直接 mock
  ``narrative.generate_narrative`` 返回固定合法结果；叙述校验用例（用例 4）
  mock ``agent.get_llm`` 返回预置文本的假模型；
- 知识委派：不需要验证知识路径的用例把
  ``config.MAX_KNOWLEDGE_QUESTIONS`` 置 0，保证零真实检索/LLM 调用；
- 适配层故障注入：patch ``app.case_analysis_agent.adapter.*``（Agent 经
  ``adapter.X`` 属性访问，调用时才取，patch 即生效）。
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from app import db
from app.case_analysis import service
from app.case_analysis.models import (
    AbnormalFlag,
    Certainty,
    ComparisonEntry,
    FactType,
    MedicationEventType,
    ObservationPoint,
    Polarity,
)
from app.case_analysis_agent import run_case_analysis
from app.case_analysis_agent.outputs import CaseAnalysisStatus
from app.medical_agent.state import NodeResult, NodeStatus
from tests._case_fixtures import CaseTestBase, T0, T1, T2, T3

_AGENT = "app.case_analysis_agent.agent"
_ADAPTER = "app.case_analysis_agent.adapter"
_CONFIG = "app.case_analysis_agent.config"
_NARRATIVE = "app.case_analysis_agent.narrative"

RANGE_START, RANGE_END = 2000.0, 4000.0


def _run_row(run_id):
    with db.connect() as conn:
        return conn.execute(
            "SELECT status, error_code FROM analysis_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()


def _assoc_rows(run_id):
    with db.connect() as conn:
        return conn.execute(
            "SELECT run_id, patient_id, fact_ids FROM candidate_associations"
            " WHERE run_id=?",
            (run_id,),
        ).fetchall()


def _trace(result, node, event):
    return [t for t in result.trace if t.node == node and t.event == event]


def _ok_narrative(*args, **kwargs):
    return NodeResult(NodeStatus.OK, "占位叙述文本")


class _FakeLLM:
    """narrative.generate_narrative 只调用 llm.invoke 并读 .content。"""

    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self._content)


class TestRunClosure(CaseTestBase):
    """用例 1/2：运行登记的异常收口与关闭失败。"""

    def test_1_unexpected_exception_closes_run_failed(self):
        patient = service.create_patient(self.owner, "虚构患者")
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, fact_type=FactType.HISTORY),
        ])
        with mock.patch(f"{_AGENT}.build_analysis_context",
                        side_effect=RuntimeError("boom")):
            result = run_case_analysis(self.owner, patient.patient_id)
        # 不抛异常即通过调用；以下为状态断言
        self.assertIs(result.status, CaseAnalysisStatus.PROCESSING_FAILED)
        self.assertEqual(result.error_code, "internal_error")
        self.assertIsNotNone(result.run_id)
        row = _run_row(result.run_id)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_code"], "internal_error")
        errors = _trace(result, "CA00", "error")
        self.assertTrue(any("internal:RuntimeError" == t.detail["reason"]
                            for t in errors))

    def test_2_run_close_failure_degrades_not_completed(self):
        patient = service.create_patient(self.owner, "虚构患者")
        service.save_clinical_facts(patient.patient_id, self.owner, [
            self.make_fact(patient.patient_id, onset=T1),
        ])
        service.save_medication_events(patient.patient_id, self.owner, [
            self.make_med(patient.patient_id, start=T0),
        ])
        with mock.patch(f"{_ADAPTER}.update_analysis_run",
                        return_value=(None, "storage_error")), \
             mock.patch(f"{_NARRATIVE}.generate_narrative", _ok_narrative), \
             mock.patch(f"{_CONFIG}.MAX_KNOWLEDGE_QUESTIONS", 0):
            result = run_case_analysis(self.owner, patient.patient_id)
        # 业务分析成功（候选关联已落库），但审计记录关闭失败 → 不得 COMPLETED
        self.assertIs(result.status, CaseAnalysisStatus.PARTIAL)
        self.assertIn("run_update_failed", result.degraded)
        self.assertTrue(_trace(result, "CA12", "error"))
        self.assertTrue(result.associations)          # 业务落库已成功
        row = _run_row(result.run_id)
        self.assertEqual(row["status"], "pending")    # 实际行为：关闭失败留下 pending


class TestTimelineRange(CaseTestBase):
    """用例 3：范围内/范围外/未知时间三类记录 + 跨界用药口径差异记录。"""

    def test_3_timeline_range_filtering(self):
        patient = service.create_patient(self.owner, "虚构患者")
        pid = patient.patient_id
        f_in, f_out, f_unknown = service.save_clinical_facts(pid, self.owner, [
            self.make_fact(pid, name="范围内事件", onset=3000.0,
                           fact_type=FactType.HISTORY),
            self.make_fact(pid, name="范围外事件", onset=1000.0,
                           fact_type=FactType.HISTORY),
            self.make_fact(pid, name="未知时间事件", onset=None,
                           fact_type=FactType.HISTORY),
        ])
        m_cross, = service.save_medication_events(pid, self.owner, [
            self.make_med(pid, start=1000.0, end=3000.0),  # 跨界：起点在范围外
        ])
        result = run_case_analysis(
            self.owner, pid,
            time_range_start=RANGE_START, time_range_end=RANGE_END,
        )

        timeline_ids = {e.ref_id for e in result.timeline}
        self.assertIn(f_in.fact_id, timeline_ids)          # 范围内保留
        self.assertNotIn(f_out.fact_id, timeline_ids)      # 范围外排除
        unknown = [e for e in result.timeline if e.ref_id == f_unknown.fact_id]
        self.assertEqual(len(unknown), 1)                  # 未知时间保留
        self.assertTrue(unknown[0].unknown_time)

        # 口径差异记录（不判定为缺陷）：用药按区间重叠保留在 medications，
        # 其时间线条目 event_time=start_time=1000.0 在范围外被裁掉
        self.assertIn(m_cross.medication_event_id,
                      {m.medication_event_id for m in result.context.medications})
        self.assertNotIn(m_cross.medication_event_id, timeline_ids)

        # 未知时间记录不参与时间先后/关联判断（无 symptom 事实，零草稿）
        self.assertEqual(result.association_drafts, ())
        self.assertEqual(result.comparisons, ())
        self.assertIs(result.status, CaseAnalysisStatus.INSUFFICIENT_EVIDENCE)


class TestNarrativeSafety(CaseTestBase):
    """用例 4：叙述安全校验。合法/陌生 id/陌生日期/指令性/因果/畸形/超长。"""

    def _patient_with_comparable_obs(self):
        patient = service.create_patient(self.owner, "虚构患者")
        pid = patient.patient_id
        fact, = service.save_clinical_facts(pid, self.owner, [
            self.make_fact(pid, onset=T1),
        ])
        service.save_observations(pid, self.owner, [
            self.make_obs(pid, value=7.2, sample_time=T1),
            self.make_obs(pid, value=6.8, sample_time=T2),
        ])
        return pid, fact.fact_id

    def _run_with_content(self, pid, content):
        with mock.patch(f"{_AGENT}.get_llm",
                        lambda **kw: _FakeLLM(content)):
            return run_case_analysis(self.owner, pid)

    def test_4_narrative_validation_matrix(self):
        pid, fact_id = self._patient_with_comparable_obs()
        legal = json.dumps({
            "summary": "两次空腹血糖测量存在差异，原因需结合临床背景。",
            "highlights": ["1970-01-01 有两次血糖记录"],
            "caveats": ["变化含义需专业判断"],
            "referenced_ids": [fact_id],
        }, ensure_ascii=False)
        marker = "非法叙述独有标记文本"
        bad_cases = {
            "foreign_id": {"summary": "复述。", "highlights": [], "caveats": [],
                           "referenced_ids": ["不存在的id"]},
            "foreign_date": {"summary": f"2025-05-05{marker}", "highlights": [],
                             "caveats": [], "referenced_ids": [fact_id]},
            "directive": {"summary": f"建议停药观察。{marker}", "highlights": [],
                          "caveats": [], "referenced_ids": [fact_id]},
            "causal": {"summary": f"用药导致指标变化。{marker}", "highlights": [],
                       "caveats": [], "referenced_ids": [fact_id]},
            "malformed": "这不是JSON" + marker,
            "overlong": {"summary": marker + "长" * 5000, "highlights": [],
                         "caveats": [], "referenced_ids": [fact_id]},
        }

        result = self._run_with_content(pid, legal)
        self.assertTrue(result.narrative)                       # 合法叙述保留在字段
        self.assertNotIn("narrative_degraded", result.degraded)
        self.assertNotIn("两次空腹血糖测量存在差异", result.answer_text)  # 非权威，不进正文
        self.assertIs(result.status, CaseAnalysisStatus.COMPLETED)

        for name, payload in bad_cases.items():
            with self.subTest(case=name):
                content = payload if isinstance(payload, str) else json.dumps(
                    payload, ensure_ascii=False)
                result = self._run_with_content(pid, content)
                self.assertEqual(result.narrative, "")
                self.assertIn("narrative_degraded", result.degraded)
                self.assertNotIn(marker, result.answer_text)    # 不污染确定性结果
                self.assertIn("### 指标变化", result.answer_text)
                self.assertIs(result.status, CaseAnalysisStatus.PARTIAL)


class TestPartialReadFailure(CaseTestBase):
    """用例 5：事实读取失败 + 关联查询成功 → 保守丢弃。"""

    def test_5_facts_read_failed_but_associations_found(self):
        patient = service.create_patient(self.owner, "虚构患者")
        pid = patient.patient_id
        service.save_clinical_facts(pid, self.owner, [
            self.make_fact(pid, onset=T1),          # symptom，供真实关联查询命中
        ])
        service.save_medication_events(pid, self.owner, [
            self.make_med(pid, start=T0),
        ])
        with mock.patch(f"{_ADAPTER}.get_patient_history",
                        return_value=(None, "storage_error")), \
             mock.patch(f"{_NARRATIVE}.generate_narrative", _ok_narrative), \
             mock.patch(f"{_CONFIG}.MAX_KNOWLEDGE_QUESTIONS", 0):
            result = run_case_analysis(self.owner, pid)

        self.assertIs(result.status, CaseAnalysisStatus.PARTIAL)
        self.assertIn("facts_read_failed", result.degraded)
        self.assertIn("foreign_associations_dropped", result.degraded)
        # 无法核验归属的候选关联被保守丢弃，不出现在任何输出段
        self.assertEqual(result.associations, ())
        self.assertEqual(result.association_drafts, ())
        self.assertNotIn("候选关联（", result.answer_text)
        # trace 区分：CA04 事实读取失败（error）与 CA06 关联查询成功（decision）
        ca04_err = _trace(result, "CA04", "error")
        self.assertTrue(any(t.detail.get("section") == "facts" for t in ca04_err))
        ca06 = _trace(result, "CA06", "decision")
        self.assertTrue(ca06 and ca06[0].detail["drafts"] >= 1)
        ca11 = _trace(result, "CA11", "decision")
        self.assertTrue(ca11 and ca11[0].detail["drafts_dropped"] >= 1)


class TestFullFlow(CaseTestBase):
    """用例 6：完整流程 + 权限约束 + 防枚举 + 语义保留。"""

    def test_6_full_flow_consistency(self):
        patient = service.create_patient(self.owner, "虚构患者")
        pid = patient.patient_id
        service.create_encounter(pid, self.owner, "虚构就诊原文",
                                 encounter_time=T0)
        service.save_clinical_facts(pid, self.owner, [
            self.make_fact(pid, name="恶心", onset=T1),
            self.make_fact(pid, name="皮疹", onset=T2,
                           polarity=Polarity.NEGATIVE),
            self.make_fact(pid, name="头晕", onset=T3,
                           certainty=Certainty.POSSIBLE),
        ])
        service.save_medication_events(pid, self.owner, [
            self.make_med(pid, start=T0),
        ])
        service.save_observations(pid, self.owner, [
            self.make_obs(pid, value=7.2, sample_time=T1),
            self.make_obs(pid, value=6.8, sample_time=T2),
            self.make_obs(pid, value=7.0, sample_time=None),   # 未知采样时间
        ])
        with mock.patch(f"{_NARRATIVE}.generate_narrative", _ok_narrative), \
             mock.patch(f"{_CONFIG}.MAX_KNOWLEDGE_QUESTIONS", 0):
            result = run_case_analysis(self.owner, pid)

        self.assertIs(result.status, CaseAnalysisStatus.COMPLETED)
        # run 一致性与落库状态
        row = _run_row(result.run_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["error_code"], "")
        # 候选关联只关联到本次运行与本患者
        rows = _assoc_rows(result.run_id)
        self.assertEqual(len(rows), len(result.associations))
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["run_id"], result.run_id)
            self.assertEqual(r["patient_id"], pid)
        # 语义保留：否定/不确定/待审核/未知时间
        ctx = result.context
        self.assertEqual(ctx.negated_fact_count, 1)
        self.assertEqual(ctx.uncertain_fact_count, 1)
        self.assertEqual(ctx.pending_fact_count, 3)
        unknown = [e for e in result.timeline if e.unknown_time]
        self.assertTrue(unknown)
        # 不可比较条目给出机器可读原因而不是差值
        bad = [c for c in result.comparisons if not c.comparable]
        self.assertTrue(any(c.reason == "sample_time_missing" for c in bad))
        # 候选关联表述不得越界为因果结论
        self.assertIn("候选关联（时间接近 ≠ 因果", result.answer_text)
        self.assertNotIn("占位叙述文本", result.answer_text)

    def test_6b_access_denied_indistinguishable(self):
        patient = service.create_patient(self.owner, "虚构患者")
        denied = run_case_analysis(self.other, patient.patient_id)
        missing = run_case_analysis(self.owner, "不存在ed患者id")
        # 不区分"不存在"与"不属于你"，两处状态与错误码完全一致
        self.assertIs(denied.status, CaseAnalysisStatus.ACCESS_DENIED)
        self.assertIs(missing.status, CaseAnalysisStatus.ACCESS_DENIED)
        self.assertEqual(denied.error_code, missing.error_code)
        self.assertEqual(denied.answer_text, missing.answer_text)


class TestRenderDirtyData(CaseTestBase):
    """用例 7（观察项）：脏数值导致渲染异常时的兜底行为，只记录不修复。"""

    def test_7_render_failure_observation(self):
        patient = service.create_patient(self.owner, "虚构患者")
        pid = patient.patient_id
        service.save_clinical_facts(pid, self.owner, [
            self.make_fact(pid, fact_type=FactType.HISTORY),
        ])
        dirty = ComparisonEntry(
            indicator="空腹血糖", unit="",
            earlier=ObservationPoint(
                observation_id="x", sample_time=T1,
                value_numeric="非数值",  # 类型损坏：render 的 :g 格式化将抛错
                value_text="", unit="", reference_low=None,
                reference_high=None, abnormal_flag=AbnormalFlag.HIGH,
            ),
            # earlier/later 同时存在才会进入 _fmt_point 渲染分支
            later=ObservationPoint(
                observation_id="y", sample_time=T2,
                value_numeric=7.0, value_text="", unit="",
                reference_low=None, reference_high=None,
                abnormal_flag=AbnormalFlag.NORMAL,
            ),
            delta=None, comparable=True, reason="",
        )
        escaped = None
        with mock.patch(f"{_ADAPTER}.compare_observations",
                        return_value=([dirty], "")), \
             mock.patch(f"{_NARRATIVE}.generate_narrative", _ok_narrative), \
             mock.patch(f"{_CONFIG}.MAX_KNOWLEDGE_QUESTIONS", 0):
            try:
                result = run_case_analysis(self.owner, pid)
            except Exception as exc:                      # 记录是否逃逸
                escaped = exc
                result = None

        with db.connect() as conn:
            rows = conn.execute(
                "SELECT status, error_code FROM analysis_runs").fetchall()
        # 实际行为记录（断言即文档）：
        self.assertIsNone(escaped)          # 兜底 _finish 用空数据渲染，异常未逃逸
        self.assertIs(result.status, CaseAnalysisStatus.PROCESSING_FAILED)
        self.assertEqual(result.error_code, "internal_error")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")     # 运行记录未悬挂
        self.assertEqual(rows[0]["error_code"], "internal_error")


if __name__ == "__main__":
    unittest.main()
