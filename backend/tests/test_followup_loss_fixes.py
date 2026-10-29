"""P1：进展式追问触发、历史缺失文案分流、resolver no-op 三态（2026-10-29）。

M1：「已经持续三天了」等进展式陈述触发 N02；
M2：历史在但改写失败 → RESOLUTION_FAILED_REPLY，不再谎称历史丢失；
M3：resolver no-op → repair 一次 → 软触发接受自含、硬指代继续澄清。
"""

import json
import unittest

from app.medical_agent import respond, session
from app.medical_agent.complex_handler import assess_resolution
from app.medical_agent.resolver import resolve_reference
from app.medical_agent.router import has_followup_lead
from app.medical_agent.state import (
    AgentState,
    NodeResult,
    NodeStatus,
    ResolutionResult,
    ResolutionStatus,
    TerminationReason,
)
from tests.test_medical_agent_step4 import FakeLLM, _fresh_session, _run


class TestProgressTrigger(unittest.TestCase):
    """M1：进展式陈述触发（不能仅凭关键词，需长度+标记约束）。"""

    def test_progress_statements_trigger(self):
        for text in ("已经持续三天了", "还是不舒服", "现在还在疼", "第三天了"):
            with self.subTest(text=text):
                self.assertTrue(has_followup_lead(text))

    def test_complete_statements_not_triggered(self):
        for text in ("我已经吃了二甲双胍三天了", "每天都在跑步", "持续服药两年了。"):
            with self.subTest(text=text):
                self.assertFalse(has_followup_lead(text))

    def test_long_progress_not_triggered(self):
        self.assertFalse(has_followup_lead("已经持续三天了而且越来越严重该怎么办呢"))


class TestResolverNoopThreeState(unittest.TestCase):
    """M3：no-op 首次→可修复失败；repair 后再 no-op → SELF_CONTAINED。"""

    def _state(self, question="已经持续三天了"):
        state = AgentState(request_id="t", session_id="s", raw_question=question)
        state.resolved_question = question
        state.history = [("user", "最近经常头疼")]
        return state

    def test_noop_first_pass_is_invalid(self):
        llm = FakeLLM(resolve=json.dumps({"question": "已经持续三天了"},
                                         ensure_ascii=False))
        out = resolve_reference(self._state(), 6000, repair=False, llm_factory=lambda timeout: llm)
        self.assertIs(out.status, NodeStatus.INVALID_OUTPUT)
        self.assertEqual(out.detail, "resolution_noop")

    def test_noop_on_repair_becomes_self_contained(self):
        llm = FakeLLM(resolve=json.dumps({"question": "已经持续三天了"},
                                         ensure_ascii=False))
        out = resolve_reference(self._state(), 6000, repair=True, llm_factory=lambda timeout: llm)
        self.assertIs(out.status, NodeStatus.OK)
        self.assertIs(out.value.status, ResolutionStatus.SELF_CONTAINED)
        self.assertEqual(out.value.question, "已经持续三天了")

    def test_self_contained_soft_trigger_continues(self):
        """软触发（无硬指代词）的 no-op：接受原句继续，不澄清。"""
        state = AgentState()
        result = NodeResult(
            NodeStatus.OK,
            ResolutionResult(ResolutionStatus.SELF_CONTAINED, "已经持续三天了"),
        )
        decision = assess_resolution(state, result)
        from app.medical_agent.complex_handler import RecoveryAction
        self.assertIs(decision.action, RecoveryAction.CONTINUE)

    def test_self_contained_hard_reference_clarifies(self):
        """硬指代（它/这个）的 no-op：不得当自含，继续澄清。"""
        state = AgentState()
        result = NodeResult(
            NodeStatus.OK,
            ResolutionResult(ResolutionStatus.SELF_CONTAINED, "它的副作用呢？"),
        )
        decision = assess_resolution(state, result)
        self.assertIs(decision.termination, TerminationReason.CONTEXT_MISSING)


class TestClarifyTextSplit(unittest.TestCase):
    """M2：历史在但改写失败 → 新文案；历史截断 → 旧文案。"""

    def test_resolution_failure_with_history_uses_new_text(self):
        sid = _fresh_session()
        session.complete(sid, "最近经常头疼", "答：注意休息。", got_lock=True)
        llm = FakeLLM(resolve=json.dumps({"question": ""}, ensure_ascii=False))
        state, _ = _run(sid, "还是不舒服", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.CONTEXT_MISSING)
        self.assertIn("你刚才聊的内容我都还在", state.answer_text)
        self.assertNotIn("不再保留", state.answer_text)

    def test_truncated_history_keeps_old_text(self):
        """历史被截断（超过窗口）时仍用 CONTEXT_MISSING_REPLY。"""
        sid = _fresh_session()
        for i in range(session.MAX_CONTEXT_TURNS + 1):
            session.complete(sid, f"问题{i}", f"回答{i}", got_lock=True)
        llm = FakeLLM(resolve=json.dumps({"question": ""}, ensure_ascii=False))
        state, _ = _run(sid, "那什么原因", llm, None)
        self.assertIs(state.termination_reason, TerminationReason.CONTEXT_MISSING)
        self.assertIn("不再保留", state.answer_text)


if __name__ == "__main__":
    unittest.main()


class TestThreeTurnConsultation(unittest.TestCase):
    """任务书场景：「最近经常头疼」→「已经持续三天了」必须走 N02 补全而非澄清。"""

    def test_progress_statement_resolves_with_history(self):
        sid = _fresh_session()
        session.complete(sid, "最近经常头疼", "答：注意休息。", got_lock=True)
        llm = FakeLLM(
            resolve=json.dumps({"question": "头疼已经持续三天了，应该怎么办？"},
                               ensure_ascii=False),
            classify={"category": "knowledge", "confidence": 0.9},
            judge={"answerable": True, "supporting_quote": "依据片段"},
        )
        from app.rag import service
        ok = service.RetrievalResult(
            outcome=None,
            chunks=[service.RetrievedChunk(content="依据片段", document="d.pdf",
                                           page=1, distance=0.3)])
        state, _ = _run(sid, "已经持续三天了", llm, ok)
        self.assertIn("头疼", state.resolved_question)
        self.assertIs(state.termination_reason, TerminationReason.COMPLETED)
