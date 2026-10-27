"""主 Agent 工具层：注册表、参数校验、调用拦截、结果封装、异常归一化。

职责边界：本模块**决定"能不能调、参数对不对、结果怎么装"**，不决定分支走向
（那是 orchestrator 的事）。调用方向 `tools.py → rag/service.py → chain.py`。

两类失败必须分开，任何非 OK 状态都不得被当作成功结果使用（需求 AG-04）：

    参数不合法   → INVALID_ARGS，**底层检索根本没被调用**
    调用被拦截   → BLOCKED_REPEAT / BLOCKED_MAX_ROUNDS / BLOCKED_BUDGET，同样不发出调用
    执行期失败   → TIMEOUT / RUNTIME_ERROR，调用发出但没拿到结果

配置项暂放在本模块顶部。设计文档 5.8 要求它们最终收敛到 `backend/app/config.py`；
本阶段不新增配置文件，因此先就地定义。

**本阶段只注册检索工具这一个。** 就医指引（triage）工具未注册：设计文档 5.7 提到
"两个工具用注册表 dict 即可"，但全文没有给出 triage 工具的参数 Schema 与返回内容，
而 triage 路径按 2.2 表不进入证据判定。为避免替它编一套未被评审过的契约，
留到 Step 4 随 router 一起定义——注册表本身加一个工具只是再加一次 register()。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..rag import service
from ..rag.service import (
    DEFAULT_TOP_K,
    MAX_QUESTION_LEN,
    MAX_TOP_K,
    MIN_TOP_K,
)
from .state import RetrievalOutcome, RetrievalResult, ToolCallRecord, ToolStatus


# ==================== 配置（设计文档 5.8） ====================

# 工具执行轮数上限。本方案实际最多 2 轮（N09 回边只允许一次）。
MAX_TOOL_ROUNDS = 3

# 低于此值不启动该环节，避免"启动了但必然超时"的假动作（设计文档 5.6）。
MIN_USABLE_BUDGET_MS = 3000


class InvalidArgsError(ValueError):
    """工具参数校验不通过。只携带机器可读标识，不携带用户原文。"""


# ==================== 工具定义 ====================


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的定义。刻意不引入工具基类体系——两个工具用 dict 注册表足够。"""

    name: str
    description: str
    validator: Callable[[dict], dict]
    handler: Callable[..., Any]
    side_effects: tuple[str, ...] = ()


@dataclass
class ToolInvocation:
    """一次工具调用的完整结果。

    record 是给 AgentState.tool_calls 用的追踪记录；retrieval 仅检索类工具产出。
    detail 是机器可读的短标识（如 invalid_args:question_too_long），供 trace 使用，
    **不得直接展示给用户**——用户可见文案一律取自 respond.py。
    """

    record: ToolCallRecord
    retrieval: RetrievalResult | None = None
    detail: str = ""

    @property
    def status(self) -> ToolStatus:
        return self.record.status

    @property
    def ok(self) -> bool:
        return self.record.status is ToolStatus.OK


# ==================== 参数校验 ====================


def compute_args_digest(args: dict) -> str:
    """参数摘要：长度 + 短 hash，**不记原文**（设计文档 3.2 / 待确认 7）。

    用于 N07 的"同工具 + 同参数"重复调用判定，以及 trace 留痕。
    """
    canonical = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"len={len(canonical)};sha1={digest}"


def _validate_search_args(raw: dict) -> dict:
    """search_medical_kb 的参数 Schema 校验，返回规范化后的参数。

    Raises:
        InvalidArgsError: 任一条件不满足。调用方据此产出 INVALID_ARGS 且不发出调用。
    """
    if not isinstance(raw, dict):
        raise InvalidArgsError("args_not_a_mapping")

    unknown = sorted(set(raw) - {"question", "k"})
    if unknown:
        # 多传参数说明调用方与 Schema 不一致，宁可拒绝也不要猜测其意图
        raise InvalidArgsError(f"unknown_param:{','.join(unknown)}")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise InvalidArgsError("question_empty")
    question = question.strip()
    if len(question) > MAX_QUESTION_LEN:
        raise InvalidArgsError("question_too_long")

    k = raw.get("k", DEFAULT_TOP_K)
    if isinstance(k, bool) or not isinstance(k, int):
        raise InvalidArgsError("k_not_int")
    if not (MIN_TOP_K <= k <= MAX_TOP_K):
        raise InvalidArgsError("k_out_of_range")

    return {"question": question, "k": k}


def _search_medical_kb(question: str, k: int) -> RetrievalResult:
    """检索工具的实现：直接交给 rag/service.py，本层不做任何检索逻辑。"""
    return service.retrieve_with_scores(question, k=k)


RETRIEVAL_TOOL = "search_medical_kb"
TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in TOOL_REGISTRY:
        raise ValueError(f"工具名重复注册：{spec.name}")
    TOOL_REGISTRY[spec.name] = spec
    return spec


register(ToolSpec(
    name=RETRIEVAL_TOOL,
    description="在已收录的医学资料中检索与问题相关的片段，返回片段及其距离。",
    validator=_validate_search_args,
    handler=_search_medical_kb,
))


def get_tool(name: str) -> ToolSpec | None:
    return TOOL_REGISTRY.get(name)


# ==================== 调用前拦截（N07 在调用之前判定） ====================


def check_call_allowed(
    tool_name: str,
    args_digest: str,
    *,
    state=None,
    budget_ms: int | None = None,
) -> tuple[ToolStatus | None, str]:
    """调用前拦截。返回 (拦截状态, 原因)；未被拦截时状态为 None。

    三类拦截都在**发出调用之前**发生，因此被拦截时底层一次都没执行：

        BLOCKED_REPEAT      同工具 + 同参数摘要此前已调用过
        BLOCKED_MAX_ROUNDS  tool_rounds 已达上限
        BLOCKED_BUDGET      分配到的环节预算低于 MIN_USABLE_BUDGET_MS

    "连续失败达阈值"不在这里判：AgentState 没有对应的 ToolStatus 取值，
    由 orchestrator 在 N09 守卫里处理（设计文档 4.1 第 9 条）。
    """
    if budget_ms is not None and budget_ms < MIN_USABLE_BUDGET_MS:
        return ToolStatus.BLOCKED_BUDGET, f"budget_too_small:{budget_ms}"

    if state is None:
        return None, ""

    rounds = getattr(state, "tool_rounds", 0) or 0
    if rounds >= MAX_TOOL_ROUNDS:
        return ToolStatus.BLOCKED_MAX_ROUNDS, f"rounds_exceeded:{rounds}"

    for record in getattr(state, "tool_calls", None) or []:
        if record.tool == tool_name and record.args_digest == args_digest:
            # 只拦"成功过或正在重试的同参数调用"吗？不——同参数重复本身即无意义，
            # 无论上次成功与否，重试意义相同，一律拦截。
            return ToolStatus.BLOCKED_REPEAT, f"repeat:{tool_name}"

    return None, ""


# ==================== 统一调用入口 ====================


def call_tool(
    tool_name: str,
    raw_args: dict,
    *,
    state=None,
    budget_ms: int | None = None,
) -> ToolInvocation:
    """主 Agent 调用工具的唯一入口。

    Args:
        tool_name: 注册表中的工具名。
        raw_args: 未校验的原始参数。
        state: 可选。提供时用于轮数/重复拦截，并填充记录的 seq 与 round。
        budget_ms: 本环节分配到的预算（毫秒）。低于 MIN_USABLE_BUDGET_MS 时不启动。

    Returns:
        ToolInvocation。**本函数不抛异常**：所有失败都归一为 ToolStatus，
        由 orchestrator 依状态分流，避免异常穿透到编排层之外。
    """
    started = time.monotonic()
    spec = get_tool(tool_name)
    seq, round_no = _position(state)
    budget = budget_ms if budget_ms is not None else 0

    if spec is None:
        return _finish(
            started, tool_name, "", ToolStatus.INVALID_ARGS, seq, round_no, budget,
            detail=f"unknown_tool:{tool_name}",
        )

    # 先校验，再拦截。顺序不能反：拦截依赖参数摘要，而摘要必须建立在**规范化后**
    # 的参数上，否则「同一个问题多打一个空格」会被算成两次不同的调用，重复拦截形同虚设。
    # 校验不发出任何底层调用，因此"被拦截时底层未被调用"这一要求不受影响。
    try:
        args = spec.validator(raw_args)
    except InvalidArgsError as exc:
        # 底层未发出调用 —— 测试 T-07 会断言这一点
        return _finish(started, tool_name, _probe_digest(raw_args), ToolStatus.INVALID_ARGS,
                       seq, round_no, budget, detail=f"invalid_args:{exc}")
    except Exception as exc:
        return _finish(started, tool_name, _probe_digest(raw_args), ToolStatus.INVALID_ARGS,
                       seq, round_no, budget, detail=f"invalid_args:{type(exc).__name__}")

    digest = compute_args_digest(args)

    blocked, reason = check_call_allowed(tool_name, digest, state=state, budget_ms=budget_ms)
    if blocked is not None:
        return _finish(started, tool_name, digest, blocked, seq, round_no, budget, detail=reason)

    try:
        retrieval = spec.handler(**args)
    except Exception as exc:
        status = ToolStatus.TIMEOUT if _is_timeout(exc) else ToolStatus.RUNTIME_ERROR
        return _finish(started, tool_name, digest, status, seq, round_no, budget,
                       detail=f"{status.value}:{type(exc).__name__}")

    return _finish(started, tool_name, digest, _status_from_retrieval(retrieval), seq, round_no, budget,
                   retrieval=retrieval, detail=_detail_from_retrieval(retrieval))


def _status_from_retrieval(retrieval: RetrievalResult) -> ToolStatus:
    """把检索结果状态翻译成工具状态。

    service.retrieve_with_scores 不抛异常而是把失败归一成 outcome，因此这里必须
    把 RETRIEVAL_ERROR 重新映射回 RUNTIME_ERROR / TIMEOUT，否则一次失败的检索会
    留下 ToolStatus.OK 的记录，违反"任何非 OK 状态都不得被当作成功"。

    EMPTY_RESULT 是**成功**：检索正常执行、正常返回 0 条（设计文档 4.1 第 3b 条）。
    """
    if retrieval.outcome is RetrievalOutcome.RETRIEVAL_ERROR:
        return ToolStatus.TIMEOUT if _is_timeout_name(retrieval.error) else ToolStatus.RUNTIME_ERROR
    return ToolStatus.OK


def _detail_from_retrieval(retrieval: RetrievalResult) -> str:
    if retrieval.outcome is None:
        return f"ok:chunks={len(retrieval.chunks)}"
    return f"{retrieval.outcome.value}" + (f":{retrieval.error}" if retrieval.error else "")


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or _is_timeout_name(type(exc).__name__)


def _is_timeout_name(name: str | None) -> bool:
    """仅凭异常类名判断超时。

    service 层按设计只保留异常类型名（不保留异常原文，避免路径/密钥外泄），
    因此这里只能用类名匹配。必须用子串而不是后缀：内置的 TimeoutError 以
    "Error" 结尾，用 endswith("Timeout") 会漏掉它，导致超时被误记成 RUNTIME_ERROR
    （测试 T-09② 要求超时被明确区分为 timeout）。
    覆盖 TimeoutError、httpx 的 ReadTimeout / ConnectTimeout / TimeoutException。
    """
    if not name:
        return False
    return "timeout" in name.lower()


def _probe_digest(raw_args) -> str:
    """参数校验前用的摘要，仅用于重复调用判定，容忍任意输入类型。"""
    try:
        return compute_args_digest(raw_args if isinstance(raw_args, dict) else {"__raw__": repr(raw_args)})
    except Exception:
        return "len=0;sha1=unknown"


def _position(state) -> tuple[int, int]:
    if state is None:
        return 1, 0
    return len(getattr(state, "tool_calls", None) or []) + 1, getattr(state, "tool_rounds", 0) or 0


def _finish(
    started: float,
    tool_name: str,
    args_digest: str,
    status: ToolStatus,
    seq: int,
    round_no: int,
    budget_ms: int,
    *,
    retrieval: RetrievalResult | None = None,
    detail: str = "",
) -> ToolInvocation:
    return ToolInvocation(
        record=ToolCallRecord(
            seq=seq,
            tool=tool_name,
            args_digest=args_digest,
            status=status,
            latency_ms=int((time.monotonic() - started) * 1000),
            budget_ms=budget_ms,
            round=round_no,
        ),
        retrieval=retrieval,
        detail=detail,
    )


__all__ = [
    "InvalidArgsError",
    "ToolSpec",
    "ToolInvocation",
    "TOOL_REGISTRY",
    "RETRIEVAL_TOOL",
    "MAX_TOOL_ROUNDS",
    "MIN_USABLE_BUDGET_MS",
    "register",
    "get_tool",
    "compute_args_digest",
    "check_call_allowed",
    "call_tool",
]
