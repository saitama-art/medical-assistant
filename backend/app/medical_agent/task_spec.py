"""知识 Worker 的最小 Task Spec、独立授权与结果验收。

Task Spec 的构造是纯函数，不执行工具。只有后端 ``authorize_task_spec`` 能签发
单请求 WorkerPermit；模型输出、Task Spec 字段本身均不构成执行权限。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from . import config as _cfg
from . import tools
from ..rag import service
from .state import (
    AgentState,
    EmergencyLevel,
    EvidenceJudgement,
    QuestionCategory,
    RetrievalResult,
    Route,
    Source,
    SubagentResult,
    SubagentStatus,
    ToolCallRecord,
)


EXPECTED_ARTIFACTS = (
    "answer", "sources", "evidence", "retrieval", "tool_calls", "trace",
)
ACCEPTANCE_CRITERIA = frozenset({
    "evidence_sufficient",
    "evidence_quote_verified",
    "answer_nonempty",
    "sources_from_retrieval",
    "allowed_tools_only",
    "no_side_effects",
    "input_constraints_retained",
})


@dataclass(frozen=True)
class TaskConstraints:
    raw_question: str
    allowed_tools: tuple[str, ...]
    deadline: float
    max_rounds: int
    input_fingerprint: str


@dataclass(frozen=True)
class TaskSpec:
    spec_id: str
    intent: QuestionCategory | None
    inputs: Mapping[str, object]
    artifacts: tuple[str, ...]
    side_effects: tuple[str, ...]
    constraints: TaskConstraints
    acceptance_criteria: tuple[str, ...]


@dataclass
class WorkerPermit:
    """进程内单次授权记录，不是对外安全令牌。"""

    spec_id: str
    allowed_tools: tuple[str, ...]
    deadline: float
    allowed_side_effects: tuple[str, ...]
    consumed: bool = False


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class AuthorizationResult:
    permit: WorkerPermit | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.permit is not None


def build_task_spec(state: AgentState) -> TaskSpec:
    """从已路由状态构造知识任务；不做 IO，也不表示已获执行许可。"""
    question = state.resolved_question
    raw_question = state.raw_question
    return TaskSpec(
        spec_id=f"{state.request_id or 'request'}-knowledge-0",
        intent=state.category,
        inputs=MappingProxyType({"question": question}),
        artifacts=EXPECTED_ARTIFACTS,
        side_effects=(),
        constraints=TaskConstraints(
            raw_question=raw_question,
            allowed_tools=(tools.RETRIEVAL_TOOL,),
            deadline=state.budget.deadline,
            max_rounds=tools.MAX_TOOL_ROUNDS,
            input_fingerprint=_fingerprint(state.category, raw_question, question),
        ),
        acceptance_criteria=tuple(sorted(ACCEPTANCE_CRITERIA)),
    )


def validate_task_spec(spec: TaskSpec) -> ValidationResult:
    """校验结构、输入、工具能力、副作用和可执行验收标准。"""
    if not isinstance(spec, TaskSpec):
        return ValidationResult(False, "spec_identity_invalid")
    if not isinstance(spec.spec_id, str) or not spec.spec_id.strip():
        return ValidationResult(False, "spec_identity_invalid")
    if spec.intent is not QuestionCategory.KNOWLEDGE:
        return ValidationResult(False, "intent_not_allowed")
    if not isinstance(spec.inputs, Mapping) or set(spec.inputs) != {"question"}:
        return ValidationResult(False, "inputs_invalid")
    question = spec.inputs.get("question")
    if (
        not isinstance(question, str)
        or not question.strip()
        or question != question.strip()
        or len(question) > service.MAX_QUESTION_LEN
    ):
        return ValidationResult(False, "question_invalid")
    if not isinstance(spec.constraints, TaskConstraints):
        return ValidationResult(False, "constraints_invalid")
    if not isinstance(spec.constraints.raw_question, str) or not spec.constraints.raw_question.strip():
        return ValidationResult(False, "raw_question_missing")
    if (
        not isinstance(spec.artifacts, tuple)
        or len(spec.artifacts) != len(EXPECTED_ARTIFACTS)
        or set(spec.artifacts) != set(EXPECTED_ARTIFACTS)
    ):
        return ValidationResult(False, "artifacts_invalid")
    if not isinstance(spec.side_effects, tuple) or spec.side_effects:
        return ValidationResult(False, "side_effects_not_allowed")
    if spec.constraints.allowed_tools != (tools.RETRIEVAL_TOOL,):
        return ValidationResult(False, "tool_whitelist_invalid")
    tool = tools.get_tool(tools.RETRIEVAL_TOOL)
    if tool is None or tool.side_effects:
        return ValidationResult(False, "tool_capability_mismatch")
    if (
        isinstance(spec.constraints.deadline, bool)
        or not isinstance(spec.constraints.deadline, (int, float))
        or not math.isfinite(spec.constraints.deadline)
        or spec.constraints.deadline <= 0
    ):
        return ValidationResult(False, "deadline_invalid")
    if (
        isinstance(spec.constraints.max_rounds, bool)
        or not isinstance(spec.constraints.max_rounds, int)
        or not 1 <= spec.constraints.max_rounds <= tools.MAX_TOOL_ROUNDS
    ):
        return ValidationResult(False, "round_limit_invalid")
    expected_fingerprint = _fingerprint(
        spec.intent, spec.constraints.raw_question, question,
    )
    if spec.constraints.input_fingerprint != expected_fingerprint:
        return ValidationResult(False, "input_fingerprint_mismatch")
    if (
        not isinstance(spec.acceptance_criteria, tuple)
        or len(spec.acceptance_criteria) != len(ACCEPTANCE_CRITERIA)
        or set(spec.acceptance_criteria) != ACCEPTANCE_CRITERIA
    ):
        return ValidationResult(False, "acceptance_criteria_unsupported")
    return ValidationResult(True)


def authorize_task_spec(spec: TaskSpec, state: AgentState) -> AuthorizationResult:
    """独立执行授权：结构合格仍须满足当前路由、安全和单次启动条件。"""
    valid = validate_task_spec(spec)
    if not valid.ok:
        return AuthorizationResult(detail=valid.detail)
    if state.route is not Route.KNOWLEDGE or state.category is not QuestionCategory.KNOWLEDGE:
        return AuthorizationResult(detail="route_not_allowed")
    if state.emergency.level is EmergencyLevel.STRONG:
        return AuthorizationResult(detail="emergency_blocked")
    if (
        state.resolved_question != spec.inputs["question"]
        or state.raw_question != spec.constraints.raw_question
        or state.budget.deadline != spec.constraints.deadline
    ):
        return AuthorizationResult(detail="state_spec_mismatch")
    if state.recovery.worker_launched:
        return AuthorizationResult(detail="worker_already_launched")
    if state.budget.remaining_ms() < _cfg.MIN_LAUNCH_BUDGET_MS:
        return AuthorizationResult(detail="budget_blocked")
    return AuthorizationResult(permit=WorkerPermit(
        spec_id=spec.spec_id,
        allowed_tools=spec.constraints.allowed_tools,
        deadline=spec.constraints.deadline,
        allowed_side_effects=spec.side_effects,
    ))


def consume_permit(spec: TaskSpec, permit: WorkerPermit) -> bool:
    """Worker 入口的原子单次消费；失败时不得执行任何工具。"""
    if not validate_task_spec(spec).ok or not isinstance(permit, WorkerPermit):
        return False
    if permit.consumed:
        return False
    if (
        permit.spec_id != spec.spec_id
        or permit.allowed_tools != spec.constraints.allowed_tools
        or permit.deadline != spec.constraints.deadline
        or permit.allowed_side_effects != spec.side_effects
    ):
        return False
    permit.consumed = True
    return True


def validate_worker_result(spec: TaskSpec, result: SubagentResult) -> ValidationResult:
    """验收 Worker 契约；结构通过不代表医学内容已被证明正确。"""
    if not isinstance(result, SubagentResult) or result.spec_id != spec.spec_id:
        return ValidationResult(False, "result_spec_mismatch")
    if not isinstance(result.status, SubagentStatus):
        return ValidationResult(False, "result_status_invalid")
    if result.resolved_question != spec.inputs["question"]:
        return ValidationResult(False, "input_constraints_changed")
    if not isinstance(result.answer, str):
        return ValidationResult(False, "answer_type_invalid")
    if (
        not isinstance(result.side_effects, tuple)
        or not all(isinstance(item, str) for item in result.side_effects)
        or set(result.side_effects) - set(spec.side_effects)
    ):
        return ValidationResult(False, "undeclared_side_effects")
    if not isinstance(result.tool_calls, list) or not all(
        isinstance(record, ToolCallRecord) for record in result.tool_calls
    ):
        return ValidationResult(False, "tool_records_invalid")
    if any(record.tool not in spec.constraints.allowed_tools for record in result.tool_calls):
        return ValidationResult(False, "tool_whitelist_violation")
    if (
        isinstance(result.tool_rounds, bool)
        or not isinstance(result.tool_rounds, int)
        or not 0 <= result.tool_rounds <= spec.constraints.max_rounds
    ):
        return ValidationResult(False, "round_limit_violation")
    if not isinstance(result.sources, list) or not all(
        isinstance(source, Source) for source in result.sources
    ):
        return ValidationResult(False, "sources_invalid")

    if result.sources:
        if not isinstance(result.retrieval, RetrievalResult):
            return ValidationResult(False, "sources_without_retrieval")
        retrieved = {(chunk.document, chunk.page) for chunk in result.retrieval.chunks}
        if any((source.document, source.page) not in retrieved for source in result.sources):
            return ValidationResult(False, "sources_not_from_retrieval")

    if result.status is not SubagentStatus.OK:
        if result.answer:
            return ValidationResult(False, "partial_answer_on_failure")
        if result.status in (
            SubagentStatus.PROCESSING_FAILED, SubagentStatus.SAFETY_REJECTED,
        ) and result.error is None:
            return ValidationResult(False, "structured_error_missing")
        return ValidationResult(True)

    if not isinstance(result.answer, str) or not result.answer.strip():
        return ValidationResult(False, "answer_empty")
    if not isinstance(result.retrieval, RetrievalResult) or not result.retrieval.chunks:
        return ValidationResult(False, "retrieval_missing")
    evidence = result.evidence
    if (
        not isinstance(evidence, EvidenceJudgement)
        or not evidence.sufficient
        or not evidence.stage2_quote_verified
        or evidence.stage3_answerable is not True
        or not service.verify_quote(evidence.supporting_quote, result.retrieval.chunks)
    ):
        return ValidationResult(False, "evidence_contract_invalid")
    if not result.sources:
        return ValidationResult(False, "sources_missing")
    return ValidationResult(True)


def _fingerprint(intent, raw_question: str, resolved_question: str) -> str:
    payload = json.dumps(
        [
            intent.value if isinstance(intent, QuestionCategory) else None,
            raw_question,
            resolved_question,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACCEPTANCE_CRITERIA",
    "EXPECTED_ARTIFACTS",
    "AuthorizationResult",
    "TaskConstraints",
    "TaskSpec",
    "ValidationResult",
    "WorkerPermit",
    "authorize_task_spec",
    "build_task_spec",
    "consume_permit",
    "validate_task_spec",
    "validate_worker_result",
]
