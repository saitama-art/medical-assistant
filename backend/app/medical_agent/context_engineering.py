"""上下文工程 V1：待续任务、可追溯事实与单次模型调用的上下文包。

原始消息仍由 conversations.messages 保存；本模块不复制消息、不写 Chroma，
也不生成摘要。持久化只覆盖确有多轮延续需要的任务和用户明确提供的事实。
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping

from .. import conversations
from ..db import connect, init_db, now_ts
from . import calculator
from .state import AgentState, ContextStatus, TerminationReason

DEFAULT_TOKEN_BUDGET = 3000


class TaskStatus(str, Enum):
    WAITING_FOR_INPUT = "waiting_for_input"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContextAssemblyStatus(str, Enum):
    OK = "ok"
    BUDGET_EXCEEDED = "budget_exceeded"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    owner_user_id: int
    conversation_id: str
    task_type: str
    status: TaskStatus
    required_inputs: tuple[str, ...]
    # 计算任务的参数值是数值；followup 的 anchor_question 是用户原文（字符串）
    provided_inputs: Mapping[str, object] = field(default_factory=dict)
    missing_inputs: tuple[str, ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0
    attempt: int = 0
    result_ref: str = ""
    error_code: str = ""

    def with_inputs(self, values: Mapping[str, float], *, now: float) -> "TaskRecord":
        if self.status not in (TaskStatus.WAITING_FOR_INPUT, TaskStatus.READY):
            raise ValueError("task_not_open_for_input")
        provided = {**self.provided_inputs, **values}
        allowed = {k: float(provided[k]) for k in self.required_inputs if k in provided}
        missing = tuple(k for k in self.required_inputs if k not in allowed)
        status = TaskStatus.READY if not missing else TaskStatus.WAITING_FOR_INPUT
        return replace(self, provided_inputs=allowed, missing_inputs=missing,
                       status=status, updated_at=now, error_code="")

    def mark_running(self, *, now: float) -> "TaskRecord":
        if self.status is not TaskStatus.READY or self.missing_inputs:
            raise ValueError("task_not_ready")
        return replace(self, status=TaskStatus.RUNNING, attempt=self.attempt + 1,
                       updated_at=now)


@dataclass(frozen=True)
class StructuredFact:
    fact_id: str
    owner_user_id: int
    conversation_id: str
    source_message_id: int | None
    fact_type: str
    value: float
    unit: str
    event_time: float
    recorded_at: float
    status: str = "active"
    freshness: str = "historical_unconfirmed"


@dataclass(frozen=True)
class MessageContext:
    message_id: int | None
    role: str
    content: str
    created_at: float = 0.0
    turn: int = 0


@dataclass(frozen=True)
class EvidenceContext:
    content: str
    document: str = ""
    page: int | None = None


@dataclass(frozen=True)
class ContextConflict:
    kind: str
    values: tuple[str, ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ContextPackage:
    current_question: str
    task: TaskRecord | None
    recent_history: tuple[MessageContext, ...]
    relevant_facts: tuple[StructuredFact, ...]
    retrieval_evidence: tuple[EvidenceContext, ...]
    missing_information: tuple[str, ...]
    conflicts: tuple[ContextConflict, ...]
    source_refs: tuple[str, ...]
    token_estimate: int
    token_budget: int
    context_status: ContextStatus
    assembly_status: ContextAssemblyStatus
    omitted_history_turns: int = 0
    candidate_task_count: int = 0


@dataclass(frozen=True)
class TaskTypeSpec:
    """任务类型注册项（M5，2026-10-29）。dynamic=True 表示缺失字段由
    运行期校验决定（followup），否则必须等于 required_inputs 才能 READY。"""

    required_inputs: tuple[str, ...]
    dynamic: bool = False


_TASK_REGISTRY: dict[str, TaskTypeSpec] = {
    "bmi": TaskTypeSpec(("height_cm", "weight_kg")),
    "standard_weight": TaskTypeSpec(("height_cm",)),
    "followup": TaskTypeSpec((), dynamic=True),
}

FOLLOWUP_TASK_TYPE = "followup"
FOLLOWUP_TTL_SECONDS = 7 * 24 * 3600


def task_type_spec(task_type: str) -> TaskTypeSpec:
    """注册表查询：未注册的 task_type 明确失败，不静默绕过（M5）。"""
    from ..case_analysis.models import CaseValidationError
    try:
        return _TASK_REGISTRY[task_type]
    except KeyError:
        raise CaseValidationError(f"unknown_task_type:{task_type}")


# 兼容别名：原 _REQUIRED_INPUTS 的读取方（事实选择）只看带必填参数的类型。
_REQUIRED_INPUTS = {
    name: spec.required_inputs
    for name, spec in _TASK_REGISTRY.items() if spec.required_inputs
}
_INPUT_LABELS = {"height_cm": "身高", "weight_kg": "体重"}
_UNITS = {"height_cm": "cm", "weight_kg": "kg"}
_EXPLICIT_MEASUREMENT = re.compile(
    r"身高|体重|厘米|公分|kg|公斤|千克|斤|lb|磅", re.I,
)
_PARAMETER_FILLER = re.compile(
    r"[\s，,。．.!！?？~～;；:：、我的是有现在目前身高体重和与还有再]+"
)


def _new_task(owner: int, conversation_id: str, kind: str, *, now: float) -> TaskRecord:
    required = task_type_spec(kind).required_inputs
    return TaskRecord(
        task_id=uuid.uuid4().hex, owner_user_id=owner,
        conversation_id=conversation_id, task_type=kind,
        status=TaskStatus.WAITING_FOR_INPUT, required_inputs=required,
        missing_inputs=required, created_at=now, updated_at=now,
    )


def _extract_parameter_update(text: str) -> dict[str, float]:
    """只接受带参数语义的补充句；裸数字和含其他诉求的句子不关联任务。"""
    if not isinstance(text, str) or not _EXPLICIT_MEASUREMENT.search(text):
        return {}
    height = calculator._parse_height(text)
    weight = calculator._parse_weight(text)
    remainder = text
    for pattern in (*calculator._HEIGHT_PATTERNS, *calculator._WEIGHT_PATTERNS):
        remainder = pattern.sub("", remainder)
    if _PARAMETER_FILLER.sub("", remainder):
        return {}
    values = {}
    if height is not None:
        values["height_cm"] = float(height)
    if weight is not None:
        values["weight_kg"] = float(weight)
    return values


def _bare_number_args(text: str, task: TaskRecord) -> dict[str, float]:
    """唯一待续任务场景：把裸数字补充句按任务必需参数解析。

    系统刚发出「请提供身高和体重」的澄清时，用户回「170 65」「身高170 70」
    就是该澄清的直接回答，不是新问题——此时数字与任务的关联是唯一确定的。
    仅当会话里只有一个待续任务时由调用方使用；多候选必须消歧，不走这里。
    """
    if not isinstance(text, str) or not re.search(r"\d", text):
        return {}
    stripped = text.strip()
    # 必须是「参数回答」形态：短句且无问句/诉求词——
    # 「我身高170，二甲双胍怎么吃？」是知识问题，不是澄清的回答。
    if len(stripped) > 20 or re.search(r"[？?]|什么|怎么|怎么办|吗|呢|哪|为什么|如何", stripped):
        return {}
    height = calculator._parse_height(text)
    weight = calculator._parse_weight(text)
    height, weight = calculator._parse_bare_numbers(text, height, weight)
    values = {}
    for name, value in (("height_cm", height), ("weight_kg", weight)):
        if value is not None and name in task.required_inputs:
            values[name] = float(value)
    return values


def _current_facts(state: AgentState, values: Mapping[str, float], *, now: float):
    owner = state.context_owner_user_id or 0
    return [
        StructuredFact(
            fact_id=f"{state.request_id}-{name}", owner_user_id=owner,
            conversation_id=state.session_id, source_message_id=None,
            fact_type=name, value=float(value), unit=_UNITS[name],
            event_time=now, recorded_at=now, freshness="current_user_provided",
        )
        for name, value in values.items() if name in _UNITS
    ]


def _merge_facts(state: AgentState, facts: Iterable[StructuredFact]) -> None:
    by_type = {fact.fact_type: fact for fact in state.pending_facts}
    by_type.update({fact.fact_type: fact for fact in facts})
    state.pending_facts = list(by_type.values())


# ==================== M4：咨询续问任务（2026-10-29） ====================

# 缺失字段白名单：中英文、数字、常见计量/标点；拒绝引号、花括号、尖括号、
# 分号等注入形态——分类器的 missing_inputs 永远不当 SQL/代码碎片落库。
_MISSING_FIELD_RE = re.compile(r"^[A-Za-z0-9_一-鿿（）()\-—/，,。 .]{1,20}$")
_MAX_MISSING_FIELDS = 8


def _validate_missing_fields(values) -> tuple[str, ...] | None:
    """missing_inputs 白名单校验；任何一项非法即整体不创建（返回 None）。"""
    if not isinstance(values, (list, tuple)) or len(values) > _MAX_MISSING_FIELDS:
        return None
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or not _MISSING_FIELD_RE.fullmatch(value):
            return None
        out.append(value)
    return tuple(dict.fromkeys(out))


def _last_user_message(history) -> str:
    for role, content in reversed(list(history or [])):
        if role == "user" and isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def register_followup(state: AgentState, missing_inputs) -> TaskRecord | None:
    """创建或复用一条咨询续问待续任务（M4）。

    只在「历史可用」时创建；锚点是上一轮用户原始问题（不是模型改写产物）。
    missing_inputs 经白名单校验，失败则返回 None（调用方按无任务继续）。
    """
    anchor = _last_user_message(state.history)
    if not anchor:
        return None
    missing = _validate_missing_fields(missing_inputs)
    if missing is None:
        return None
    for task in state.context_tasks:
        if (task.task_type == FOLLOWUP_TASK_TYPE
                and task.status is TaskStatus.WAITING_FOR_INPUT
                and task.provided_inputs.get("anchor_question") == anchor):
            state.task_record = task
            return task
    now = now_ts()
    task = TaskRecord(
        task_id=uuid.uuid4().hex,
        owner_user_id=state.context_owner_user_id or 0,
        conversation_id=state.session_id,
        task_type=FOLLOWUP_TASK_TYPE,
        status=TaskStatus.WAITING_FOR_INPUT,
        required_inputs=(),
        provided_inputs={"anchor_question": anchor},
        missing_inputs=missing,
        created_at=now,
        updated_at=now,
    )
    state.task_record = task
    state.context_tasks = [*state.context_tasks, task]
    return task


def followup_anchor(state: AgentState) -> str:
    """唯一待续咨询任务的锚点；多个候选不自动选择，返回空串。"""
    waiting = [t for t in state.context_tasks
               if t.task_type == FOLLOWUP_TASK_TYPE
               and t.status is TaskStatus.WAITING_FOR_INPUT]
    if len(waiting) != 1:
        return ""
    anchor = waiting[0].provided_inputs.get("anchor_question")
    return anchor if isinstance(anchor, str) else ""


def associate_followup_on_resolution(state: AgentState) -> None:
    """N02 成功改写后：唯一待续咨询任务转 RUNNING。

    是否最终完成由 finalize_turn 按本轮真实业务结果决定——
    改写成功本身不等于任务完成（任务书 M4 规则 2/6）。
    """
    waiting = [t for t in state.context_tasks
               if t.task_type == FOLLOWUP_TASK_TYPE
               and t.status is TaskStatus.WAITING_FOR_INPUT]
    if len(waiting) == 1:
        # followup 是动态任务：WAITING→RUNNING 是合法转换（关联已确定）；
        # mark_running 的 READY 门槛只服务静态必填参数任务，这里显式转换。
        now = now_ts()
        state.task_record = replace(
            waiting[0], status=TaskStatus.RUNNING,
            attempt=waiting[0].attempt + 1, updated_at=now,
        )




def prepare_calculation(state: AgentState, plan) -> calculator.CalcPlan | None:
    """把 N04a 计划关联到唯一待续任务；不做 IO，不执行计算。"""
    now = now_ts()
    explicit = _extract_parameter_update(state.raw_question)
    if plan is not None:
        # 新计算任务允许从完整问句提取带标签参数；续接任务才要求参数-only 句。
        height = calculator._parse_height(state.raw_question)
        weight = calculator._parse_weight(state.raw_question)
        if height is not None:
            explicit["height_cm"] = float(height)
        if weight is not None:
            explicit["weight_kg"] = float(weight)
        explicit.update({
            key: float(value) for key, value in plan.args.items()
            if key in _UNITS and isinstance(value, (int, float))
        })
    _merge_facts(state, _current_facts(state, explicit, now=now))
    waiting = [task for task in state.context_tasks
               if task.status is TaskStatus.WAITING_FOR_INPUT]

    if plan is not None:
        candidates = [task for task in waiting if task.task_type == plan.kind]
        if plan.status == "ready":
            if len(candidates) == 1:
                state.task_record = candidates[0].with_inputs(
                    plan.args, now=now,
                ).mark_running(now=now)
            return plan
        if plan.status != "missing":
            return plan
        if len(candidates) > 1:
            return calculator.CalcPlan("ambiguous", plan.kind)
        task = candidates[0] if candidates else _new_task(
            state.context_owner_user_id or 0, state.session_id, plan.kind, now=now,
        )
        task = task.with_inputs(explicit, now=now)
        state.task_record = task
        if task.status is TaskStatus.READY:
            state.task_record = task = task.mark_running(now=now)
            return calculator.CalcPlan("ready", plan.kind, dict(task.provided_inputs),
                                       remaining=plan.remaining)
        return calculator.CalcPlan(
            "missing", plan.kind, dict(task.provided_inputs),
            missing=tuple(_INPUT_LABELS[name] for name in task.missing_inputs),
            remaining=plan.remaining,
        )

    if not explicit and len(waiting) == 1:
        # 只有一个待续任务时，裸数字就是上一轮澄清的直接回答，按任务参数解析。
        explicit = _bare_number_args(state.raw_question, waiting[0])
    if not explicit:
        return None
    compatible = [task for task in waiting
                  if any(name in task.required_inputs for name in explicit)]
    if not compatible:
        return None
    if len(compatible) != 1:
        return calculator.CalcPlan("ambiguous", compatible[0].task_type)
    task = compatible[0]
    for name, value in explicit.items():
        previous = task.provided_inputs.get(name)
        if previous is not None and not math.isclose(float(previous), value):
            return calculator.CalcPlan("conflict", task.task_type)
    task = task.with_inputs(explicit, now=now)
    state.task_record = task
    if task.status is TaskStatus.READY:
        state.task_record = task = task.mark_running(now=now)
        return calculator.CalcPlan("ready", task.task_type, dict(task.provided_inputs))
    return calculator.CalcPlan(
        "missing", task.task_type, dict(task.provided_inputs),
        missing=tuple(_INPUT_LABELS[name] for name in task.missing_inputs),
    )


# ---- 上下文处理：筛选、冲突、来源和调用前预算 ----


def estimate_tokens(text: str) -> int:
    """保守、无依赖的估算：非 ASCII 每字 1，ASCII 每 4 字 1。"""
    units = sum(1.0 if ord(char) > 255 else 0.25 for char in text)
    return int(math.ceil(units))


def build_context(
    *,
    current_question: str,
    history: Iterable[MessageContext | Mapping[str, Any]] = (),
    tasks: Iterable[TaskRecord] = (),
    facts: Iterable[StructuredFact] = (),
    retrieval_evidence: Iterable[EvidenceContext] = (),
    task: TaskRecord | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    context_status: ContextStatus = ContextStatus.NOT_NEEDED,
) -> ContextPackage:
    if not isinstance(current_question, str) or not current_question.strip():
        status = ContextAssemblyStatus.INVALID_INPUT
    elif isinstance(token_budget, bool) or token_budget <= 0:
        status = ContextAssemblyStatus.INVALID_INPUT
    else:
        status = ContextAssemblyStatus.OK

    messages = tuple(_message(item) for item in history)
    task_items = tuple(tasks)
    if task is None and len(task_items) == 1:
        task = task_items[0]
    evidence = tuple(retrieval_evidence)
    relevant_facts = _select_facts(current_question, task, tuple(facts))
    conflicts = _fact_conflicts(relevant_facts)
    missing = task.missing_inputs if task is not None else ()

    mandatory = "\n".join([
        current_question,
        _task_text(task),
        *(_fact_text(fact) for fact in relevant_facts),
        *(item.content for item in evidence),
        *(f"{item.kind}:{','.join(item.values)}" for item in conflicts),
    ])
    mandatory_tokens = estimate_tokens(mandatory)
    if status is ContextAssemblyStatus.OK and mandatory_tokens > token_budget:
        status = ContextAssemblyStatus.BUDGET_EXCEEDED

    selected: list[MessageContext] = []
    omitted = 0
    used = mandatory_tokens
    if status is ContextAssemblyStatus.OK:
        turns = _history_turns(messages)
        prioritized = _prioritize_turns(current_question, task, turns)
        omitted = len(turns)
        for turn_messages in prioritized:
            cost = estimate_tokens("\n".join(item.content for item in turn_messages)) + 8
            if used + cost > token_budget:
                continue
            selected.extend(turn_messages)
            used += cost
            omitted -= 1
        selected.sort(key=lambda item: (item.turn, item.message_id or 0))

    source_refs = _source_refs(selected, relevant_facts, evidence)
    return ContextPackage(
        current_question=current_question,
        task=task,
        recent_history=tuple(selected),
        relevant_facts=relevant_facts,
        retrieval_evidence=evidence,
        missing_information=tuple(missing),
        conflicts=conflicts,
        source_refs=source_refs,
        token_estimate=used,
        token_budget=token_budget,
        context_status=context_status,
        assembly_status=status,
        omitted_history_turns=omitted,
        candidate_task_count=len(task_items),
    )


def format_history(package: ContextPackage | None, fallback=()) -> str:
    """为既有 N02 用户模板提供历史文本；当前问题仍由独立变量传入。"""
    if package is None:
        items = [_message(item) for item in fallback]
    else:
        if package.assembly_status is not ContextAssemblyStatus.OK:
            return ""
        items = list(package.recent_history)
    return "\n".join(
        f"{'用户' if item.role == 'user' else '助手'}：{item.content}"
        for item in items
    )


def _message(item) -> MessageContext:
    if isinstance(item, MessageContext):
        return item
    if isinstance(item, tuple):
        return MessageContext(None, item[0], item[1])
    return MessageContext(
        item.get("id"), str(item.get("role", "")), str(item.get("content", "")),
        float(item.get("created_at") or 0), int(item.get("turn") or 0),
    )


def _select_facts(question, task, facts):
    needed = set(task.required_inputs) if task is not None else set()
    if re.search(r"bmi|体质指数", question, re.I):
        needed.update(_REQUIRED_INPUTS["bmi"])
    if "标准体重" in question or "理想体重" in question:
        needed.update(_REQUIRED_INPUTS["standard_weight"])
    return tuple(fact for fact in facts
                 if fact.status == "active" and fact.fact_type in needed)


def _fact_conflicts(facts):
    grouped = {}
    for fact in facts:
        grouped.setdefault(fact.fact_type, []).append(fact)
    conflicts = []
    for name, items in grouped.items():
        values = {(round(fact.value, 6), fact.unit) for fact in items}
        if len(values) > 1:
            conflicts.append(ContextConflict(
                kind=name,
                values=tuple(sorted(f"{value}:{unit}" for value, unit in values)),
                source_refs=tuple(f"fact:{fact.fact_id}" for fact in items),
            ))
    return tuple(conflicts)


def _history_turns(messages):
    grouped = {}
    for index, message in enumerate(messages):
        turn = message.turn or index // 2 + 1
        grouped.setdefault(turn, []).append(message)
    return [tuple(grouped[turn]) for turn in sorted(grouped)]


def _prioritize_turns(question, task, turns):
    if not turns:
        return []
    terms = _terms(question + " " + (task.task_type if task else ""))
    latest = turns[-1]
    relevant = [turn for turn in reversed(turns[:-1]) if _turn_relevant(turn, terms)]
    return [latest, *relevant]


def _turn_relevant(turn, terms) -> bool:
    text = " ".join(message.content for message in turn)
    return bool(terms & _terms(text))


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for run in re.findall(r"[一-鿿]+", text):
        words.update(run[index:index + 2] for index in range(max(0, len(run) - 1)))
    return {item for item in words if item}


def _task_text(task) -> str:
    if task is None:
        return ""
    return json.dumps({
        "task_type": task.task_type,
        "required_inputs": task.required_inputs,
        "provided_inputs": dict(task.provided_inputs),
        "missing_inputs": task.missing_inputs,
        "status": task.status.value,
    }, ensure_ascii=False)


def _fact_text(fact) -> str:
    return json.dumps({
        "fact_type": fact.fact_type, "value": fact.value, "unit": fact.unit,
        "event_time": fact.event_time, "recorded_at": fact.recorded_at,
        "source_message_id": fact.source_message_id, "status": fact.status,
        "freshness": fact.freshness,
    }, ensure_ascii=False)


def _source_refs(messages, facts, evidence):
    refs = []
    for message in messages:
        if message.message_id is not None:
            refs.append(f"message:{message.message_id}")
    for fact in facts:
        refs.append(f"fact:{fact.fact_id}")
        if fact.source_message_id is not None:
            refs.append(f"message:{fact.source_message_id}")
    for item in evidence:
        if item.document:
            refs.append(f"{item.document}#page={item.page}")
    return tuple(dict.fromkeys(refs))


# ---- 请求集成与 SQLite 持久化 ----


def _current_owner() -> int | None:
    try:
        from ..auth import deps
        return deps.current_owner.get()
    except Exception:
        return None


def begin_turn(state: AgentState, *, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
    """装载当前用户允许访问的历史、待续任务与事实；失败不越权回退。"""
    owner = _current_owner()
    history = list(state.history)
    tasks = []
    facts = []
    state.context_owner_user_id = owner
    state.context_persistent = False
    if owner is not None:
        try:
            history = conversations.list_messages(state.session_id, owner)
            tasks = load_pending_tasks(owner, state.session_id)
            facts = load_facts(owner, state.session_id)
            state.context_persistent = True
        except Exception:
            # 直接 run_agent 的旧调用没有数据库会话；只能使用 session 已授权的窗口。
            state.context_persistent = False
    state.context_history = [_message(item) for item in history]
    state.context_tasks = tasks
    state.context_facts = facts
    refresh_context(state, token_budget=token_budget)


def refresh_context(state: AgentState, *, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
    facts = [*state.context_facts, *state.pending_facts]
    state.context_package = build_context(
        current_question=state.resolved_question or state.raw_question,
        history=state.context_history,
        tasks=state.context_tasks,
        facts=facts,
        task=state.task_record,
        token_budget=token_budget,
        context_status=state.context_status,
    )


def finalize_turn(state: AgentState, source_message_id: int) -> bool:
    """原始消息已保存后，原子保存任务与事实；未完成不得标记 completed。"""
    if not getattr(state, "context_persistent", False):
        return False
    owner = state.context_owner_user_id
    if owner is None:
        return False
    task = state.task_record
    if task is not None and task.status is TaskStatus.RUNNING:
        if state.termination_reason is TerminationReason.COMPLETED:
            task = replace(task, status=TaskStatus.COMPLETED, result_ref=state.request_id,
                           updated_at=now_ts())
        elif state.termination_reason in (
            TerminationReason.CLARIFICATION_REQUIRED, TerminationReason.CONTEXT_MISSING,
        ):
            task = replace(task, status=TaskStatus.WAITING_FOR_INPUT, updated_at=now_ts())
        else:
            task = replace(
                task, status=TaskStatus.FAILED,
                error_code=state.termination_reason.value if state.termination_reason else "failed",
                updated_at=now_ts(),
            )
    facts = [replace(fact, source_message_id=source_message_id)
             for fact in state.pending_facts]
    persist_turn_context(owner, state.session_id, source_message_id, task, facts)
    state.task_record = task
    return True


def load_pending_tasks(owner: int, conversation_id: str) -> list[TaskRecord]:
    """加载待续任务；followup 超过 TTL 的任务标记 cancelled 并跳过（M4）。"""
    init_db()
    now = now_ts()
    with connect() as conn:
        rows = conn.execute(
            "SELECT t.* FROM agent_tasks t JOIN conversations c ON c.id=t.conversation_id"
            " WHERE t.owner_user_id=? AND t.conversation_id=? AND c.user_id=?"
            " AND c.deleted=0 AND t.status IN ('waiting_for_input','ready')"
            " ORDER BY t.updated_at, t.task_id",
            (owner, conversation_id, owner),
        ).fetchall()
        out: list[TaskRecord] = []
        for row in rows:
            task = _task_from_row(row)
            if (task.task_type == FOLLOWUP_TASK_TYPE
                    and now - task.updated_at > FOLLOWUP_TTL_SECONDS):
                conn.execute(
                    "UPDATE agent_tasks SET status='cancelled', updated_at=?"
                    " WHERE task_id=?",
                    (now, task.task_id),
                )
                continue
            out.append(task)
        conn.commit()
    return out


def load_facts(owner: int, conversation_id: str) -> list[StructuredFact]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT f.* FROM user_facts f JOIN conversations c ON c.id=f.conversation_id"
            " WHERE f.owner_user_id=? AND f.conversation_id=? AND c.user_id=?"
            " AND c.deleted=0 ORDER BY f.recorded_at, f.fact_id",
            (owner, conversation_id, owner),
        ).fetchall()
    return [_fact_from_row(row) for row in rows]


def persist_turn_context(owner, conversation_id, source_message_id, task, facts) -> None:
    """同事务校验消息归属并写任务/事实；跨用户来源一律拒绝。"""
    init_db()
    fact_items = tuple(facts)
    with connect() as conn:
        source = conn.execute(
            "SELECT m.id FROM messages m JOIN conversations c ON c.id=m.conversation_id"
            " WHERE m.id=? AND m.conversation_id=? AND c.user_id=? AND c.deleted=0"
            " AND m.role='user'",
            (source_message_id, conversation_id, owner),
        ).fetchone()
        if source is None:
            raise PermissionError("context_source_not_owned")
        if task is not None:
            if task.owner_user_id != owner or task.conversation_id != conversation_id:
                raise PermissionError("task_owner_mismatch")
            _save_task(conn, task)
        for fact in fact_items:
            if (fact.owner_user_id != owner
                    or fact.conversation_id != conversation_id
                    or fact.source_message_id != source_message_id):
                raise PermissionError("fact_owner_or_source_mismatch")
            conn.execute(
                "INSERT OR IGNORE INTO user_facts("
                "fact_id,owner_user_id,conversation_id,source_message_id,fact_type,"
                "value,unit,event_time,recorded_at,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (fact.fact_id, owner, conversation_id, source_message_id,
                 fact.fact_type, fact.value, fact.unit, fact.event_time,
                 fact.recorded_at, fact.status),
            )


def _save_task(conn, task: TaskRecord) -> None:
    values = (
        json.dumps(tuple(task.required_inputs)), json.dumps(dict(task.provided_inputs)),
        json.dumps(tuple(task.missing_inputs)), task.status.value, task.updated_at,
        task.attempt, task.result_ref, task.error_code,
    )
    existing = conn.execute(
        "SELECT owner_user_id, conversation_id FROM agent_tasks WHERE task_id=?",
        (task.task_id,),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO agent_tasks("
            "task_id,owner_user_id,conversation_id,task_type,status,required_inputs,"
            "provided_inputs,missing_inputs,created_at,updated_at,attempt,result_ref,error_code)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task.task_id, task.owner_user_id, task.conversation_id, task.task_type,
             task.status.value, values[0], values[1], values[2], task.created_at,
             task.updated_at, task.attempt, task.result_ref, task.error_code),
        )
        return
    if (int(existing["owner_user_id"]) != task.owner_user_id
            or existing["conversation_id"] != task.conversation_id):
        raise PermissionError("task_owner_mismatch")
    conn.execute(
        "UPDATE agent_tasks SET required_inputs=?,provided_inputs=?,missing_inputs=?,"
        "status=?,updated_at=?,attempt=?,result_ref=?,error_code=? WHERE task_id=?",
        (*values, task.task_id),
    )


def _task_from_row(row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"], owner_user_id=int(row["owner_user_id"]),
        conversation_id=row["conversation_id"], task_type=row["task_type"],
        status=TaskStatus(row["status"]),
        required_inputs=tuple(json.loads(row["required_inputs"])),
        provided_inputs=json.loads(row["provided_inputs"]),
        missing_inputs=tuple(json.loads(row["missing_inputs"])),
        created_at=row["created_at"], updated_at=row["updated_at"],
        attempt=int(row["attempt"]), result_ref=row["result_ref"],
        error_code=row["error_code"],
    )


def _fact_from_row(row) -> StructuredFact:
    return StructuredFact(
        fact_id=row["fact_id"], owner_user_id=int(row["owner_user_id"]),
        conversation_id=row["conversation_id"],
        source_message_id=int(row["source_message_id"]),
        fact_type=row["fact_type"], value=float(row["value"]), unit=row["unit"],
        event_time=row["event_time"], recorded_at=row["recorded_at"], status=row["status"],
    )


__all__ = [
    "ContextAssemblyStatus", "ContextConflict", "ContextPackage", "EvidenceContext",
    "MessageContext", "StructuredFact", "TaskRecord", "TaskStatus", "begin_turn",
    "build_context", "estimate_tokens", "finalize_turn", "format_history", "load_facts",
    "load_pending_tasks", "persist_turn_context", "prepare_calculation", "refresh_context",
]
