"""Task Spec 构造、独立授权、单次 permit 与 Worker 结果验收测试。"""

import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.medical_agent import task_spec
from app.medical_agent.state import (
    AgentError,
    AgentErrorKind,
    AgentState,
    EmergencyHit,
    EmergencyLevel,
    EvidenceJudgement,
    QuestionCategory,
    RetrievedChunk,
    RetrievalResult,
    Route,
    Source,
    SubagentResult,
    SubagentStatus,
    ToolCallRecord,
    ToolStatus,
)
from app.medical_agent.subagents import knowledge


class TaskSpecTestBase(unittest.TestCase):
    def setUp(self):
        self.state = AgentState(
            request_id="req-task-spec",
            session_id="s-task-spec",
            raw_question="二甲双胍的适应症是什么？",
            resolved_question="二甲双胍的适应症是什么？",
            category=QuestionCategory.KNOWLEDGE,
            route=Route.KNOWLEDGE,
        )
        self.state.budget.deadline = time.monotonic() + 30
        self.spec = task_spec.build_task_spec(self.state)

    def valid_result(self, **overrides):
        chunk = RetrievedChunk(content="直接依据原文", document="d.pdf", page=1)
        values = dict(
            status=SubagentStatus.OK,
            spec_id=self.spec.spec_id,
            resolved_question=self.spec.inputs["question"],
            answer="依据资料作出的回答。",
            sources=[Source(document="d.pdf", page=1)],
            evidence=EvidenceJudgement(
                sufficient=True,
                stage2_quote_verified=True,
                stage3_answerable=True,
                supporting_quote="直接依据原文",
            ),
            retrieval=RetrievalResult(chunks=[chunk]),
            tool_rounds=1,
        )
        values.update(overrides)
        return SubagentResult(**values)


class TestTaskSpecAuthorization(TaskSpecTestBase):
    def test_build_is_pure_and_has_required_fields(self):
        with patch("app.medical_agent.tools.call_tool") as call_tool:
            spec = task_spec.build_task_spec(self.state)
        call_tool.assert_not_called()
        self.assertFalse(self.state.recovery.worker_launched)
        for field in (
            "intent", "inputs", "artifacts", "side_effects",
            "constraints", "acceptance_criteria",
        ):
            self.assertTrue(hasattr(spec, field))
        self.assertTrue(task_spec.validate_task_spec(spec).ok)

    def test_authorization_is_independent_and_single_use(self):
        authorization = task_spec.authorize_task_spec(self.spec, self.state)
        self.assertTrue(authorization.ok)
        self.assertFalse(self.state.recovery.worker_launched)
        self.assertTrue(task_spec.consume_permit(self.spec, authorization.permit))
        self.assertFalse(task_spec.consume_permit(self.spec, authorization.permit))

    def test_invalid_intent_and_route_cannot_get_permit(self):
        invalid = replace(self.spec, intent=QuestionCategory.TRIAGE)
        self.assertFalse(task_spec.authorize_task_spec(invalid, self.state).ok)
        self.state.route = Route.TRIAGE
        self.assertFalse(task_spec.authorize_task_spec(self.spec, self.state).ok)

    def test_changed_input_fingerprint_cannot_get_permit(self):
        invalid = replace(self.spec, inputs={"question": "另一个问题"})
        result = task_spec.validate_task_spec(invalid)
        self.assertFalse(result.ok)
        self.assertEqual(result.detail, "input_fingerprint_mismatch")

    def test_unknown_acceptance_criterion_is_not_executable(self):
        invalid = replace(
            self.spec,
            acceptance_criteria=self.spec.acceptance_criteria + ("invented_check",),
        )
        self.assertFalse(task_spec.validate_task_spec(invalid).ok)

    def test_emergency_budget_and_duplicate_launch_are_blocked(self):
        self.state.emergency = EmergencyHit(level=EmergencyLevel.STRONG)
        self.assertFalse(task_spec.authorize_task_spec(self.spec, self.state).ok)
        self.state.emergency = EmergencyHit()
        self.state.budget.deadline = time.monotonic() + 1
        self.assertFalse(task_spec.authorize_task_spec(self.spec, self.state).ok)

        self.state = AgentState(
            raw_question=self.state.raw_question,
            resolved_question=self.state.resolved_question,
            category=QuestionCategory.KNOWLEDGE,
            route=Route.KNOWLEDGE,
        )
        self.state.budget.deadline = time.monotonic() + 30
        self.state.recovery.worker_launched = True
        spec = task_spec.build_task_spec(self.state)
        self.assertFalse(task_spec.authorize_task_spec(spec, self.state).ok)

    def test_worker_without_matching_permit_executes_no_tool(self):
        permit = task_spec.WorkerPermit(
            spec_id="wrong",
            allowed_tools=self.spec.constraints.allowed_tools,
            deadline=self.spec.constraints.deadline,
            allowed_side_effects=(),
        )
        with patch("app.medical_agent.tools.call_tool") as call_tool:
            result = knowledge.run(self.spec, permit)
        call_tool.assert_not_called()
        self.assertIs(result.status, SubagentStatus.SAFETY_REJECTED)
        self.assertIs(result.error.kind, AgentErrorKind.SAFETY_REJECTED)


class TestWorkerResultValidation(TaskSpecTestBase):
    def test_valid_ok_result_passes_contract(self):
        self.assertTrue(task_spec.validate_worker_result(self.spec, self.valid_result()).ok)

    def test_ok_with_empty_answer_or_missing_evidence_is_rejected(self):
        for overrides in (
            {"answer": ""},
            {"evidence": None},
            {"evidence": EvidenceJudgement(sufficient=True)},
        ):
            with self.subTest(overrides=overrides):
                result = self.valid_result(**overrides)
                self.assertFalse(task_spec.validate_worker_result(self.spec, result).ok)

    def test_fabricated_quote_and_source_are_rejected(self):
        result = self.valid_result()
        result.evidence.supporting_quote = "检索片段中不存在的话"
        self.assertFalse(task_spec.validate_worker_result(self.spec, result).ok)
        result = self.valid_result(sources=[Source(document="fake.pdf", page=99)])
        validation = task_spec.validate_worker_result(self.spec, result)
        self.assertFalse(validation.ok)
        self.assertEqual(validation.detail, "sources_not_from_retrieval")

    def test_changed_constraints_side_effects_and_tools_are_rejected(self):
        cases = (
            self.valid_result(resolved_question="被改写的问题"),
            self.valid_result(side_effects=("write_record",)),
            self.valid_result(tool_calls=[ToolCallRecord(
                seq=1, tool="unregistered", args_digest="d", status=ToolStatus.OK,
            )]),
        )
        for result in cases:
            with self.subTest(detail=result.resolved_question):
                self.assertFalse(task_spec.validate_worker_result(self.spec, result).ok)

    def test_business_failure_does_not_require_answer(self):
        result = self.valid_result(
            status=SubagentStatus.INSUFFICIENT,
            answer="",
            evidence=EvidenceJudgement(reason="stage3_not_answerable"),
        )
        self.assertTrue(task_spec.validate_worker_result(self.spec, result).ok)

    def test_processing_failure_requires_structured_error(self):
        result = self.valid_result(status=SubagentStatus.PROCESSING_FAILED, answer="")
        self.assertFalse(task_spec.validate_worker_result(self.spec, result).ok)
        result.error = AgentError("N08", AgentErrorKind.MODEL_TIMEOUT)
        self.assertTrue(task_spec.validate_worker_result(self.spec, result).ok)


if __name__ == "__main__":
    unittest.main()
