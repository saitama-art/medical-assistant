"""复杂处理层的纯决策策略，不调用模型、工具或 Worker。

编排器仍是唯一控制流：本模块只把节点结果映射为继续、一次有界修复、
澄清或终止。重试计数由 ``record_retry`` 显式登记，避免“判断可重试”本身
产生副作用。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum

from . import config as _cfg
from . import tools
from .state import (
    AgentError,
    AgentErrorKind,
    AgentState,
    IntentAssessment,
    NodeResult,
    NodeStatus,
    QuestionCategory,
    ResolutionResult,
    ResolutionStatus,
    TerminationReason,
    UnresolvedKind,
    exception_result,
)


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    RETRY_NODE = "retry_node"
    CLARIFY = "clarify"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    termination: TerminationReason | None = None
    error: AgentError | None = None


_CONTINUE = RecoveryDecision(RecoveryAction.CONTINUE)


def assess_node_result(
    state: AgentState, node: str, result: NodeResult,
) -> RecoveryDecision:
    """统一解释节点执行状态，不解释任何业务 Payload。"""
    if result.status is NodeStatus.OK:
        return _CONTINUE
    if result.status is NodeStatus.BUDGET_BLOCKED:
        return RecoveryDecision(
            RecoveryAction.TERMINATE,
            TerminationReason.BUDGET_EXHAUSTED,
            AgentError(node, AgentErrorKind.BUDGET_BLOCKED),
        )
    if result.status is NodeStatus.TIMEOUT:
        return _processing(node, AgentErrorKind.MODEL_TIMEOUT)
    if result.status is NodeStatus.UNAVAILABLE:
        return _processing(node, AgentErrorKind.MODEL_UNAVAILABLE)
    if result.status is NodeStatus.INVALID_OUTPUT:
        attempt_key = f"{node}:repair"
        if (
            state.recovery.repair_total < _cfg.MAX_MODEL_REPAIRS
            and state.recovery.attempts.get(attempt_key, 0) < 1
        ):
            return RecoveryDecision(RecoveryAction.RETRY_NODE)
        kind = AgentErrorKind.MODEL_OUTPUT_INVALID
        if result.detail == AgentErrorKind.MODEL_CONSTRAINT_VIOLATION.value:
            kind = AgentErrorKind.MODEL_CONSTRAINT_VIOLATION
        return _processing(node, kind, result.detail)
    return _processing(node, AgentErrorKind.MODEL_OUTPUT_INVALID, "unknown_node_status")


def assess_resolution(
    state: AgentState, result: NodeResult[ResolutionResult], node: str = "N02",
) -> RecoveryDecision:
    """N02：合法缺上下文走重述，技术失败不得伪装成上下文缺失。"""
    decision = assess_node_result(state, node, result)
    if decision.action is not RecoveryAction.CONTINUE:
        return decision
    value = result.value
    if not isinstance(value, ResolutionResult) or not isinstance(value.question, str):
        return _invalid_payload(state, node)
    if value.status is ResolutionStatus.CONTEXT_REQUIRED:
        if value.question:
            return _invalid_payload(state, node)
        return RecoveryDecision(
            RecoveryAction.CLARIFY, TerminationReason.CONTEXT_MISSING,
        )
    if value.status is ResolutionStatus.RESOLVED and value.question.strip():
        return _CONTINUE
    return _invalid_payload(state, node)


def assess_intent(
    state: AgentState, result: NodeResult[IntentAssessment],
) -> RecoveryDecision:
    """N04：业务未知、低置信度和未解决问题分别决策。"""
    decision = assess_node_result(state, "N04", result)
    if decision.action is not RecoveryAction.CONTINUE:
        return decision
    value = result.value
    if not _valid_intent(value):
        return _invalid_payload(state, "N04")

    if value.unresolved is UnresolvedKind.CONTEXT_REQUIRED:
        return RecoveryDecision(
            RecoveryAction.CLARIFY, TerminationReason.CONTEXT_MISSING,
        )
    if value.unresolved in (UnresolvedKind.MISSING_INPUT, UnresolvedKind.INTENT_CONFLICT):
        return RecoveryDecision(
            RecoveryAction.CLARIFY, TerminationReason.CLARIFICATION_REQUIRED,
        )
    if value.category is QuestionCategory.UNCLASSIFIABLE:
        return RecoveryDecision(
            RecoveryAction.TERMINATE, TerminationReason.UNCLASSIFIED,
        )
    if value.confidence < _cfg.CATEGORY_CONFIDENCE_MIN:
        return RecoveryDecision(
            RecoveryAction.CLARIFY, TerminationReason.CLARIFICATION_REQUIRED,
        )
    return _CONTINUE


def authorize_retry(
    state: AgentState, node: str, purpose: str, input_digest: str,
) -> bool:
    """检查次数、总修复预算、调用摘要和剩余时间；不修改状态。"""
    if not node or not purpose or not input_digest:
        return False
    if state.recovery.repair_total >= _cfg.MAX_MODEL_REPAIRS:
        return False
    attempt_key = f"{node}:{purpose}"
    if state.recovery.attempts.get(attempt_key, 0) >= 1:
        return False
    if _call_digest(node, purpose, input_digest) in state.recovery.call_digests:
        return False
    return state.budget.remaining_ms() >= tools.MIN_USABLE_BUDGET_MS


def record_retry(
    state: AgentState, node: str, purpose: str, input_digest: str,
) -> bool:
    """授权后原子登记一次修复；未获授权时不产生任何计数变化。"""
    if not authorize_retry(state, node, purpose, input_digest):
        return False
    attempt_key = f"{node}:{purpose}"
    state.recovery.attempts[attempt_key] = (
        state.recovery.attempts.get(attempt_key, 0) + 1
    )
    state.recovery.call_digests.add(_call_digest(node, purpose, input_digest))
    state.recovery.repair_total += 1
    return True


def safety_rejection(node: str = "WORKER") -> RecoveryDecision:
    return RecoveryDecision(
        RecoveryAction.TERMINATE,
        TerminationReason.SAFETY_REJECTED,
        AgentError(node, AgentErrorKind.SAFETY_REJECTED),
    )


def _valid_intent(value) -> bool:
    if not isinstance(value, IntentAssessment):
        return False
    if not isinstance(value.category, QuestionCategory):
        return False
    if not isinstance(value.unresolved, UnresolvedKind):
        return False
    if (
        isinstance(value.confidence, bool)
        or not isinstance(value.confidence, (int, float))
        or not math.isfinite(value.confidence)
        or not 0.0 <= value.confidence <= 1.0
    ):
        return False
    if not isinstance(value.missing_inputs, tuple) or not all(
        isinstance(item, str) and item for item in value.missing_inputs
    ):
        return False
    if value.unresolved is UnresolvedKind.MISSING_INPUT:
        return bool(value.missing_inputs)
    return not value.missing_inputs


def _invalid_payload(state: AgentState, node: str) -> RecoveryDecision:
    return assess_node_result(
        state, node, NodeResult(NodeStatus.INVALID_OUTPUT, detail="invalid_payload"),
    )


def _processing(
    node: str, kind: AgentErrorKind, detail: str = "",
) -> RecoveryDecision:
    return RecoveryDecision(
        RecoveryAction.TERMINATE,
        TerminationReason.PROCESSING_FAILED,
        AgentError(node, kind, detail),
    )


def _call_digest(node: str, purpose: str, input_digest: str) -> str:
    raw = f"{node}\0{purpose}\0{input_digest}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "RecoveryAction",
    "RecoveryDecision",
    "assess_intent",
    "assess_node_result",
    "assess_resolution",
    "authorize_retry",
    "exception_result",
    "record_retry",
    "safety_rejection",
]
