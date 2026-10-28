"""主 Agent 的统一状态定义。

本模块只含状态/枚举定义与由其衍生的最小辅助方法，不含业务逻辑，不做 IO。

状态对象 AgentState 的生命周期 = 单个请求：由编排器在入口新建，各节点读写，
请求结束后丢弃。禁止把它挂到模块级变量上，否则不同请求之间会串数据。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .context_engineering import (
        ContextPackage, MessageContext, StructuredFact, TaskRecord,
    )


# ==================== 枚举 ====================


class QuestionCategory(str, Enum):
    """意图分类结果。分类失败/超时/非法输出时使用 UNCLASSIFIABLE，不得猜最接近的一类。"""

    KNOWLEDGE = "knowledge"
    TRIAGE = "triage"
    EMERGENCY = "emergency"
    CHITCHAT = "chitchat"
    UNCLASSIFIABLE = "unclassifiable"


class Route(str, Enum):
    """条件路由选定的执行路径。"""

    EMERGENCY = "emergency"
    CONTEXT_MISSING = "context_missing"
    CHITCHAT = "chitchat"
    UNCLASSIFIABLE = "unclassifiable"
    TRIAGE = "triage"
    KNOWLEDGE = "knowledge"


class EmergencyLevel(str, Enum):
    """紧急风险等级。STRONG 拦截并直接返回紧急应答；WEAK 只追加风险提示。"""

    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class ContextStatus(str, Enum):
    """会话上下文可用性。"""

    NOT_NEEDED = "not_needed"
    OK = "ok"
    MISSING = "missing"
    DEGRADED_NO_HISTORY = "degraded_no_history"


class NodeStatus(str, Enum):
    """模型节点的执行状态；OK 不代表业务上可回答或可执行。"""

    OK = "ok"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_OUTPUT = "invalid_output"
    BUDGET_BLOCKED = "budget_blocked"


class ResolutionStatus(str, Enum):
    """N02 的业务解析结果，与模型调用是否成功分开表达。"""

    RESOLVED = "resolved"
    CONTEXT_REQUIRED = "context_required"


class UnresolvedKind(str, Enum):
    """N04 识别出的未解决业务问题，不授予任何执行权限。"""

    NONE = "none"
    MISSING_INPUT = "missing_input"
    CONTEXT_REQUIRED = "context_required"
    INTENT_CONFLICT = "intent_conflict"


class AgentErrorKind(str, Enum):
    """可安全写入 trace/API 的机器错误类型，不包含异常原文。"""

    MODEL_TIMEOUT = "model_timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    MODEL_CONSTRAINT_VIOLATION = "model_constraint_violation"
    WORKER_RESULT_INVALID = "worker_result_invalid"
    SAFETY_REJECTED = "safety_rejected"
    BUDGET_BLOCKED = "budget_blocked"


class RetrievalOutcome(str, Enum):
    """检索/生成结果状态。五个值互斥，不可混用。

    EMPTY_RESULT（检索成功但没有内容）与 RETRIEVAL_ERROR（检索执行失败）
    必须区分开，二者对用户给出不同提示。
    """

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    EMPTY_RESULT = "empty_result"
    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"


class ToolStatus(str, Enum):
    """单次工具调用的结果状态。任何非 OK 状态都不得被当作成功结果使用。"""

    OK = "ok"
    INVALID_ARGS = "invalid_args"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    BLOCKED_REPEAT = "blocked_repeat"
    BLOCKED_MAX_ROUNDS = "blocked_max_rounds"
    BLOCKED_BUDGET = "blocked_budget"


class TerminationReason(str, Enum):
    """流程终止原因。所有终止路径都必须落到其中一个值。"""

    COMPLETED = "completed"
    EMERGENCY_HANDLED = "emergency_handled"
    EMERGENCY_RULE_FAULT = "emergency_rule_fault"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EMPTY_KNOWLEDGE_BASE = "empty_knowledge_base"
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"
    UNCLASSIFIED = "unclassified"
    CONTEXT_MISSING = "context_missing"
    ARGUMENT_INVALID = "argument_invalid"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    REPEATED_CALL_BLOCKED = "repeated_call_blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CLARIFICATION_REQUIRED = "clarification_required"
    PROCESSING_FAILED = "processing_failed"
    SAFETY_REJECTED = "safety_rejected"


class SubagentStatus(str, Enum):
    """子 Agent 结构化返回的状态。父 Agent 据此映射终止原因与用户文案，

    不解释子 Agent 内部细节——status + detail（机器可读短标识）就是全部契约。
    """

    OK = "ok"                                # 证据充分且生成成功
    INSUFFICIENT = "insufficient"            # 有片段但证据判定不足
    EMPTY_RESULT = "empty_result"            # 检索成功返回 0 条
    RETRIEVAL_FAILED = "retrieval_failed"    # 检索超时/异常且不可重试
    GENERATION_FAILED = "generation_failed"  # 证据充分但生成失败
    BUDGET_EXHAUSTED = "budget_exhausted"    # 环节预算不足未启动
    ARGUMENT_INVALID = "argument_invalid"    # 参数校验不通过（未执行）
    REPEATED_CALL_BLOCKED = "repeated_call_blocked"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    PROCESSING_FAILED = "processing_failed"
    SAFETY_REJECTED = "safety_rejected"


# ==================== 记录结构 ====================


T = TypeVar("T")


@dataclass(frozen=True)
class NodeResult(Generic[T]):
    """N02/N04/N08 的统一节点出口。detail 只能是安全机器标识。"""

    status: NodeStatus
    value: T | None = None
    detail: str = ""


def exception_result(exc: Exception) -> NodeResult:
    """把调用边界异常映射为安全状态，不保留异常消息或堆栈。"""
    name = type(exc).__name__
    normalized = name.lower().replace("_", "")
    if isinstance(exc, TimeoutError) or "timeout" in normalized or "timedout" in normalized:
        return NodeResult(NodeStatus.TIMEOUT, detail=name)
    return NodeResult(NodeStatus.UNAVAILABLE, detail=name)


@dataclass(frozen=True)
class AgentError:
    """请求级结构化错误；最终出口仍由 TerminationReason 表达。"""

    node: str
    kind: AgentErrorKind
    detail: str = ""


@dataclass(frozen=True)
class ResolutionResult:
    status: ResolutionStatus
    question: str = ""


@dataclass(frozen=True)
class IntentAssessment:
    """N04 的原始业务评估；低置信度不得篡改 category。"""

    category: QuestionCategory
    confidence: float
    unresolved: UnresolvedKind = UnresolvedKind.NONE
    missing_inputs: tuple[str, ...] = ()


@dataclass
class RecoveryState:
    """单请求恢复预算。摘要和计数均不得包含用户原文。"""

    attempts: dict[str, int] = field(default_factory=dict)
    call_digests: set[str] = field(default_factory=set)
    repair_total: int = 0
    clarified: bool = False
    worker_launched: bool = False


@dataclass
class EmergencyHit:
    """紧急风险检查结果。

    同时保留 rule_id 与命中词，便于事后审计"当时为什么判定为紧急/不紧急"。
    """

    level: EmergencyLevel = EmergencyLevel.NONE
    rule_id: str = ""
    terms: tuple[str, ...] = ()
    hit_negated: bool = False


@dataclass
class EvidenceJudgement:
    """证据充分性判定结果，分段记录以便定位是哪一段拦下的。

    stage1_gate1 为 None 表示距离阈值尚未校准、该段被禁用（不是"通过"）。
    stage3_answerable 为 None 表示该段未启用。
    """

    sufficient: bool = False
    stage1_gate1: bool | None = None
    stage2_quote_verified: bool = False
    stage3_answerable: bool | None = None
    supporting_quote: str | None = None
    top1_distance: float | None = None
    distance_metric: str | None = None
    reason: str = ""


@dataclass
class RetrievedChunk:
    """检索返回的一个片段。只保留回答所需字段，不原样持有 langchain Document。"""

    content: str
    document: str = ""
    page: int | None = None
    distance: float | None = None


@dataclass
class RetrievalResult:
    """一次检索调用的结果。outcome 为 None 表示尚未执行检索。"""

    outcome: RetrievalOutcome | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    distance_metric: str | None = None
    latency_ms: int = 0
    error: str = ""


@dataclass
class Source:
    """对用户展示的来源。document 取自检索结果的 metadata，不由模型生成。"""

    document: str
    page: int | None = None
    quote: str | None = None


@dataclass
class ToolCallRecord:
    """单次工具调用记录。args_digest 只存摘要，不存用户原文。"""

    seq: int
    tool: str
    args_digest: str
    status: ToolStatus
    latency_ms: int = 0
    budget_ms: int = 0
    round: int = 0


@dataclass
class TraceRecord:
    """执行追踪记录。detail 中不得出现用户问题原文、密钥、堆栈或内部路径。"""

    seq: int
    node: str
    event: str
    detail: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    budget_left_ms: int | None = None
    node_budget_ms: int | None = None


@dataclass
class SubagentResult:
    """子 Agent 返回给父 Agent 的结构化结果（委派的唯一出口契约）。

    父 Agent 合并 tool_calls / trace 时负责重排 seq（见 orchestrator）；
    evidence / retrieval 为 None 表示该环节未产出（如参数被拦截）。
    """

    status: SubagentStatus
    spec_id: str = ""
    resolved_question: str = ""
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    evidence: EvidenceJudgement | None = None
    retrieval: RetrievalResult | None = None
    tool_rounds: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    trace: list[TraceRecord] = field(default_factory=list)
    detail: str = ""
    error: AgentError | None = None
    side_effects: tuple[str, ...] = ()


@dataclass
class BudgetState:
    """时间预算。deadline 是 time.monotonic() 的绝对时刻，不是剩余量。"""

    deadline: float = 0.0
    node_budget_ms: int = 0

    def remaining_ms(self) -> int:
        if not self.deadline:
            return 0
        return max(0, int((self.deadline - time.monotonic()) * 1000))


# ==================== 主状态对象 ====================


@dataclass
class AgentState:
    """单次请求的全部状态。

    每请求新建一次；不要复用、不要缓存、不要放进模块级容器。
    """

    # —— 请求标识 ——
    request_id: str = ""
    session_id: str = ""
    turn_id: int = 0

    # —— 输入与上下文 ——
    raw_question: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)
    context_status: ContextStatus = ContextStatus.NOT_NEEDED
    resolved_question: str = ""
    # 上下文工程运行态；可靠持久化由 context_engineering + SQLite 负责。
    context_package: ContextPackage | None = None
    context_history: list[MessageContext] = field(default_factory=list)
    context_tasks: list[TaskRecord] = field(default_factory=list)
    context_facts: list[StructuredFact] = field(default_factory=list)
    task_record: TaskRecord | None = None
    pending_facts: list[StructuredFact] = field(default_factory=list)
    context_owner_user_id: int | None = None
    context_persistent: bool = False

    # —— 判定结果 ——
    emergency: EmergencyHit = field(default_factory=EmergencyHit)
    category: QuestionCategory | None = None
    category_confidence: float = 0.0
    intent_assessment: IntentAssessment | None = None
    route: Route | None = None

    # —— 工具与证据 ——
    tool_rounds: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    consecutive_failures: int = 0
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    evidence: EvidenceJudgement = field(default_factory=EvidenceJudgement)

    # —— 输出 ——
    sources: list[Source] = field(default_factory=list)
    answer_text: str = ""
    risk_notice: str = ""

    # —— 控制与追踪 ——
    budget: BudgetState = field(default_factory=BudgetState)
    recovery: RecoveryState = field(default_factory=RecoveryState)
    task_spec_id: str = ""
    termination_reason: TerminationReason | None = None
    error: AgentError | None = None
    trace: list[TraceRecord] = field(default_factory=list)

    def add_trace(
        self,
        node: str,
        event: str,
        detail: dict[str, Any] | None = None,
        latency_ms: int | None = None,
        node_budget_ms: int | None = None,
    ) -> TraceRecord:
        """追加一条追踪记录，序号自增。"""
        record = TraceRecord(
            seq=len(self.trace) + 1,
            node=node,
            event=event,
            detail=detail or {},
            latency_ms=latency_ms,
            budget_left_ms=self.budget.remaining_ms() if self.budget.deadline else None,
            node_budget_ms=node_budget_ms,
        )
        self.trace.append(record)
        return record
