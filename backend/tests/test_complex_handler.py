"""复杂处理层纯策略测试：无网络、无模型、无持久化写入。"""

import time
import unittest

from app.medical_agent import complex_handler
from app.medical_agent.state import (
    AgentErrorKind,
    AgentState,
    IntentAssessment,
    NodeResult,
    NodeStatus,
    QuestionCategory,
    RecoveryState,
    ResolutionResult,
    ResolutionStatus,
    TerminationReason,
    UnresolvedKind,
)


class TestComplexHandler(unittest.TestCase):
    def setUp(self):
        self.state = AgentState()
        self.state.budget.deadline = time.monotonic() + 30

    def test_valid_resolution_continues(self):
        result = NodeResult(
            NodeStatus.OK,
            ResolutionResult(ResolutionStatus.RESOLVED, "二甲双胍的用法用量"),
        )
        decision = complex_handler.assess_resolution(self.state, result)
        self.assertIs(decision.action, complex_handler.RecoveryAction.CONTINUE)

    def test_context_required_is_business_result(self):
        result = NodeResult(
            NodeStatus.OK,
            ResolutionResult(ResolutionStatus.CONTEXT_REQUIRED),
        )
        decision = complex_handler.assess_resolution(self.state, result)
        self.assertIs(decision.action, complex_handler.RecoveryAction.CLARIFY)
        self.assertIs(decision.termination, TerminationReason.CONTEXT_MISSING)
        self.assertIsNone(decision.error)

    def test_explicit_unknown_is_not_technical_failure(self):
        result = NodeResult(
            NodeStatus.OK,
            IntentAssessment(QuestionCategory.UNCLASSIFIABLE, 0.2),
        )
        decision = complex_handler.assess_intent(self.state, result)
        self.assertIs(decision.termination, TerminationReason.UNCLASSIFIED)
        self.assertIsNone(decision.error)

    def test_low_confidence_does_not_block_answerable_categories(self):
        """2026-10-28 置信度策略：低置信度只是辅助信号，不单独阻断可回答类别。"""
        for category in (QuestionCategory.KNOWLEDGE, QuestionCategory.TRIAGE,
                         QuestionCategory.DAILY):
            with self.subTest(category=category):
                assessment = IntentAssessment(category, 0.35)
                decision = complex_handler.assess_intent(
                    self.state, NodeResult(NodeStatus.OK, assessment),
                )
                self.assertIs(decision.action, complex_handler.RecoveryAction.CONTINUE)
                self.assertEqual(assessment.confidence, 0.35)

    def test_low_confidence_clarifies_only_non_answerable(self):
        assessment = IntentAssessment(QuestionCategory.EMERGENCY, 0.35)
        decision = complex_handler.assess_intent(
            self.state, NodeResult(NodeStatus.OK, assessment),
        )
        self.assertIs(decision.termination, TerminationReason.CLARIFICATION_REQUIRED)

    def test_missing_input_does_not_block_answerable_categories(self):
        """2026-10-28 策略：可回答类别的 missing_input 不阻断回答。"""
        for category in (QuestionCategory.KNOWLEDGE, QuestionCategory.TRIAGE,
                         QuestionCategory.DAILY):
            with self.subTest(category=category):
                value = IntentAssessment(
                    category, 0.85, UnresolvedKind.MISSING_INPUT, ("wound_severity",),
                )
                decision = complex_handler.assess_intent(
                    self.state, NodeResult(NodeStatus.OK, value),
                )
                self.assertIs(decision.action, complex_handler.RecoveryAction.CONTINUE)

    def test_missing_input_still_clarifies_when_category_not_answerable(self):
        value = IntentAssessment(
            QuestionCategory.EMERGENCY, 0.85,
            UnresolvedKind.MISSING_INPUT, ("location",),
        )
        decision = complex_handler.assess_intent(
            self.state, NodeResult(NodeStatus.OK, value),
        )
        self.assertIs(decision.termination, TerminationReason.CLARIFICATION_REQUIRED)

    def test_intent_conflict_requires_clarification(self):
        value = IntentAssessment(
            QuestionCategory.KNOWLEDGE, 0.95, UnresolvedKind.INTENT_CONFLICT, (),
        )
        decision = complex_handler.assess_intent(
            self.state, NodeResult(NodeStatus.OK, value),
        )
        self.assertIs(decision.termination, TerminationReason.CLARIFICATION_REQUIRED)

    def test_timeout_and_unavailable_have_distinct_errors(self):
        cases = (
            (NodeStatus.TIMEOUT, AgentErrorKind.MODEL_TIMEOUT),
            (NodeStatus.UNAVAILABLE, AgentErrorKind.MODEL_UNAVAILABLE),
        )
        for status, kind in cases:
            with self.subTest(status=status):
                decision = complex_handler.assess_node_result(
                    self.state, "N04", NodeResult(status),
                )
                self.assertIs(decision.action, complex_handler.RecoveryAction.TERMINATE)
                self.assertIs(decision.termination, TerminationReason.PROCESSING_FAILED)
                self.assertEqual(decision.error.node, "N04")
                self.assertIs(decision.error.kind, kind)

    def test_budget_never_retries(self):
        decision = complex_handler.assess_node_result(
            self.state, "N08", NodeResult(NodeStatus.BUDGET_BLOCKED),
        )
        self.assertIs(decision.termination, TerminationReason.BUDGET_EXHAUSTED)
        self.assertIs(decision.error.kind, AgentErrorKind.BUDGET_BLOCKED)
        self.state.budget.deadline = time.monotonic() - 1
        self.assertFalse(
            complex_handler.authorize_retry(self.state, "N08", "repair", "input")
        )

    def test_invalid_output_repairs_at_most_once(self):
        result = NodeResult(NodeStatus.INVALID_OUTPUT)
        first = complex_handler.assess_node_result(self.state, "N04", result)
        self.assertIs(first.action, complex_handler.RecoveryAction.RETRY_NODE)
        self.assertTrue(
            complex_handler.record_retry(self.state, "N04", "repair", "input")
        )
        second = complex_handler.assess_node_result(self.state, "N04", result)
        self.assertIs(second.action, complex_handler.RecoveryAction.TERMINATE)
        self.assertIs(second.error.kind, AgentErrorKind.MODEL_OUTPUT_INVALID)
        self.assertFalse(
            complex_handler.record_retry(self.state, "N04", "repair", "different")
        )
        self.assertEqual(self.state.recovery.repair_total, 1)

    def test_same_call_digest_cannot_be_registered_twice(self):
        self.assertTrue(
            complex_handler.record_retry(self.state, "N02", "repair", "same")
        )
        self.assertFalse(
            complex_handler.record_retry(self.state, "N02", "repair", "same")
        )

    def test_insufficient_time_blocks_retry_without_counting(self):
        self.state.budget.deadline = time.monotonic() - 1
        self.assertFalse(
            complex_handler.record_retry(self.state, "N04", "repair", "input")
        )
        self.assertEqual(self.state.recovery.repair_total, 0)
        self.assertEqual(self.state.recovery.attempts, {})

    def test_constraint_violation_survives_failed_repair(self):
        result = NodeResult(
            NodeStatus.INVALID_OUTPUT,
            detail=AgentErrorKind.MODEL_CONSTRAINT_VIOLATION.value,
        )
        complex_handler.record_retry(self.state, "N02", "repair", "input")
        decision = complex_handler.assess_resolution(self.state, result)
        self.assertIs(
            decision.error.kind, AgentErrorKind.MODEL_CONSTRAINT_VIOLATION,
        )

    def test_malformed_ok_payload_is_invalid_output(self):
        decision = complex_handler.assess_intent(
            self.state, NodeResult(NodeStatus.OK, None),
        )
        self.assertIs(decision.action, complex_handler.RecoveryAction.RETRY_NODE)

    def test_safety_rejection_is_terminal(self):
        decision = complex_handler.safety_rejection()
        self.assertIs(decision.action, complex_handler.RecoveryAction.TERMINATE)
        self.assertIs(decision.termination, TerminationReason.SAFETY_REJECTED)
        self.assertIs(decision.error.kind, AgentErrorKind.SAFETY_REJECTED)

    def test_recovery_state_is_not_shared_between_requests(self):
        other = AgentState()
        self.assertIsNot(self.state.recovery, other.recovery)
        self.state.recovery.attempts["N04:repair"] = 1
        self.assertEqual(other.recovery, RecoveryState())


if __name__ == "__main__":
    unittest.main()
