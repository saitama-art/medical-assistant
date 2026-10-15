"""主 Agent 编排器：N00–N12 的唯一控制流（设计文档 1.1 / 2.1 / 2.2）。

职责边界：本模块**唯一决定分支走向**，负责预算分配、轮数更新、trace 写入、
终止原因落值；不实现检索/判定/生成本身（那些在 rag/service.py），不实现
参数校验与拦截（tools.py），不提供文案常量（respond.py）。

本阶段与设计的两处已知偏差（记录在案，非疏漏）：

    1. triage 工具未注册（tools.py 注释已说明：设计全文未给出其参数 Schema），
       triage 路径直接返回 respond.TRIAGE_GUIDANCE 固定文案，0 次工具调用。
    2. N09 重试的"改写后 query"落地为：问题不变、检索条数 k 增大（k 不同即
       通过 tools 层的重复调用拦截，语义上是"扩大候选再判一次"）。
"""

from __future__ import annotations

import os
import time
import uuid

from . import respond, router, safety, session, tools
from .state import (
    AgentState,
    ContextStatus,
    EmergencyHit,
    EmergencyLevel,
    EvidenceJudgement,
    QuestionCategory,
    RetrievalOutcome,
    Route,
    TerminationReason,
    ToolStatus,
)
from ..rag import service
from ..rag.chain import get_llm

# ==================== 配置（设计文档 5.8，最终收敛 config.py） ====================

REQUEST_DEADLINE = 45.0        # 整体硬上限（5.6 选项 A）
BUDGET_FOLLOWUP_MS = 6000      # N02 上下文补全
BUDGET_CLASSIFY_MS = 6000      # N04 意图识别
BUDGET_JUDGE_MS = 8000         # N08 第 3 段模型判定
BUDGET_GENERATE_MS = 12000     # N10 回答生成
TOOL_BUDGET_MS = 5000          # N07 单次工具执行
MAX_CONSECUTIVE_FAILURES = 2   # 工具连续失败达此值即终止（设计文档 4.1 第 9 条）
_MAX_RETRIES = 1               # N09 回边最多触发 1 次（2.2）
_K_STEP = 2                    # 重试时 k 的增量

RETRIEVAL_TOOL = "search_medical_kb"

_EMERGENCY_ROUTES = (Route.EMERGENCY,)


class AgentRequestError(ValueError):
    """N00 参数缺失。由接口层转 400，不进入编排（设计文档 2.1 N00）。消息只含
    机器可读标识，不含用户原文。"""


# ==================== 入口 ====================


def run_agent(session_id: str, question: str) -> AgentState:
    """执行一次完整的 N00–N12 流程。不抛异常（AgentRequestError 除外）。"""
    state = AgentState(
        request_id=uuid.uuid4().hex[:12],
        session_id=session_id,
        raw_question=question,
    )

    # ---- N00 请求受理与会话装载 ----
    if not session_id or not isinstance(question, str) or not question.strip():
        raise AgentRequestError("missing_session_id_or_question")
    state.budget.deadline = time.monotonic() + REQUEST_DEADLINE
    got_lock, history, ctx_status = session.acquire(session_id)
    state.history = history
    state.context_status = ctx_status
    state.turn_id = sum(1 for role, _ in history if role == "user") + 1
    state.add_trace("N00", "decision", {
        "context_status": ctx_status.value,
        "history_msgs": len(history),
        "turn_id": state.turn_id,
    })

    # ---- N01 紧急风险检查（原始输入） ----
    try:
        state.emergency = safety.check_emergency(question)
    except safety.EmergencyRuleError:
        state.add_trace("N01", "error", {"reason": "emergency_rule_fault"})
        return _finish(state, got_lock, TerminationReason.EMERGENCY_RULE_FAULT,
                       respond.CONSERVATIVE_MEDICAL_NOTICE)
    state.add_trace("N01", "decision", {
        "level": state.emergency.level.value, "rule_id": state.emergency.rule_id,
    })
    if state.emergency.level is EmergencyLevel.STRONG:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)

    # ---- N02 上下文补全（条件触发：有历史、未截断、含指代词） ----
    state.resolved_question = question.strip()
    if (
        state.history
        and state.context_status is not ContextStatus.MISSING
        and router.has_reference(state.resolved_question)
    ):
        node_budget = _node_budget(state, BUDGET_FOLLOWUP_MS)
        if node_budget is None:
            # 预算不足不启动该环节：不补全，交 N05 走"要求重述"
            state.context_status = ContextStatus.MISSING
            state.add_trace("N02", "budget", {"skipped": True})
        else:
            resolved = _resolve_reference(state, node_budget)
            if resolved:
                state.resolved_question = resolved
                state.context_status = ContextStatus.OK
            else:
                # 补全失败：不猜指代对象，交 N05 走"要求重述"
                state.context_status = ContextStatus.MISSING
            state.add_trace("N02", "decision", {"resolved": bool(resolved)},
                            node_budget_ms=node_budget)

    # ---- N03 紧急风险检查（补全后），取两趟中更强者 ----
    try:
        second = safety.check_emergency(state.resolved_question)
    except safety.EmergencyRuleError:
        state.add_trace("N03", "error", {"reason": "emergency_rule_fault"})
        return _finish(state, got_lock, TerminationReason.EMERGENCY_RULE_FAULT,
                       respond.CONSERVATIVE_MEDICAL_NOTICE)
    state.emergency = _merge_hits(state.emergency, second)
    state.add_trace("N03", "decision", {"level": state.emergency.level.value})
    if state.emergency.level is EmergencyLevel.STRONG:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)

    # ---- N04 意图识别 ----
    node_budget = _node_budget(state, BUDGET_CLASSIFY_MS)
    if node_budget is None:
        # 环节预算不足未启动：归入 unclassifiable（T-05⑤）
        state.category = QuestionCategory.UNCLASSIFIABLE
        state.category_confidence = 0.0
        state.add_trace("N04", "budget", {"skipped": True})
    else:
        llm = get_llm(timeout=max(1.0, node_budget / 1000))
        state.category, state.category_confidence = router.classify(
            state.resolved_question, llm=llm,
        )
        state.add_trace("N04", "decision", {
            "category": state.category.value,
            "confidence": round(state.category_confidence, 3),
        }, node_budget_ms=node_budget)

    # 弱紧急信号：回答顶部附加风险提示（不与知识库结论混写）
    if state.emergency.level is EmergencyLevel.WEAK:
        state.risk_notice = respond.RISK_NOTICE

    # ---- N05 条件路由 ----
    state.route = router.decide_route(
        state.category, state.emergency.level,
        state.context_status, state.resolved_question,
    )
    state.add_trace("N05", "decision", {"route": state.route.value})

    if state.route is Route.EMERGENCY:
        return _finish(state, got_lock, TerminationReason.EMERGENCY_HANDLED,
                       respond.EMERGENCY_ANSWER)
    if state.route is Route.CONTEXT_MISSING:
        return _finish(state, got_lock, TerminationReason.CONTEXT_MISSING,
                       respond.CONTEXT_MISSING_REPLY)
    if state.route is Route.CHITCHAT:
        return _finish(state, got_lock, TerminationReason.COMPLETED,
                       respond.CHITCHAT_REPLY)
    if state.route is Route.UNCLASSIFIABLE:
        return _finish(state, got_lock, TerminationReason.UNCLASSIFIED,
                       respond.UNCLASSIFIABLE_REPLY)
    if state.route is Route.TRIAGE:
        # 就医指引工具本阶段未注册（tools.py 注释）：直接给通用指引固定文案
        return _finish(state, got_lock, TerminationReason.COMPLETED,
                       respond.TRIAGE_GUIDANCE)
    # Route.KNOWLEDGE → N06→N07→N08→N09（下方检索—判定—生成闭环）
    return _run_knowledge(state, got_lock)


# ==================== 知识问答路径（N06–N10） ====================


def _run_knowledge(state: AgentState, got_lock: bool) -> AgentState:
    retries = 0
    k = service.DEFAULT_TOP_K

    while True:
        # ---- N06 参数校验 + N07 工具执行（校验在 tools.call_tool 内完成，
        #      校验不通过时底层检索不会被调用） ----
        node_budget = _node_budget(state, TOOL_BUDGET_MS)
        invocation = tools.call_tool(
            RETRIEVAL_TOOL,
            {"question": state.resolved_question, "k": k},
            state=state,
            # node_budget 为 None 时传剩余量，由 tools 层归一成 BLOCKED_BUDGET
            budget_ms=node_budget if node_budget is not None else state.budget.remaining_ms(),
        )
        # 全部调用记录（含被拦截/未执行）都写回，但只有实际执行才计轮数
        state.tool_calls.append(invocation.record)
        state.add_trace("N07", "tool_call", {
            "tool": invocation.record.tool,
            "status": invocation.record.status.value,
            "round": invocation.record.round,
            "detail": invocation.detail,
        }, latency_ms=invocation.record.latency_ms,
           node_budget_ms=invocation.record.budget_ms)

        status = invocation.status
        if status is ToolStatus.INVALID_ARGS:
            return _finish(state, got_lock, TerminationReason.ARGUMENT_INVALID,
                           respond.INVALID_ARGS_REPLY)
        if status is ToolStatus.BLOCKED_REPEAT:
            return _finish(state, got_lock, TerminationReason.REPEATED_CALL_BLOCKED,
                           respond.REPEATED_CALL_REPLY)
        if status is ToolStatus.BLOCKED_MAX_ROUNDS:
            return _finish(state, got_lock, TerminationReason.MAX_ROUNDS_REACHED,
                           respond.ROUNDS_EXCEEDED_REPLY)
        if status is ToolStatus.BLOCKED_BUDGET:
            return _finish(state, got_lock, TerminationReason.BUDGET_EXHAUSTED,
                           respond.BUDGET_EXHAUSTED_REPLY)
        if status in (ToolStatus.TIMEOUT, ToolStatus.RUNTIME_ERROR):
            state.consecutive_failures += 1
            if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                return _finish(state, got_lock, TerminationReason.RETRIEVAL_FAILED,
                               respond.RETRIEVAL_ERROR)
            retries += 1
            k = min(service.MAX_TOP_K, k + _K_STEP)
            continue

        # ToolStatus.OK：检索实际执行完成，计一轮
        state.tool_rounds += 1
        state.consecutive_failures = 0
        state.retrieval = invocation.retrieval  # type: ignore[assignment]

        # EMPTY_RESULT 是成功执行的"返回 0 条"，与检索失败措辞不同（FR-08）
        if state.retrieval.outcome is RetrievalOutcome.EMPTY_RESULT:
            return _finish(state, got_lock, TerminationReason.EMPTY_KNOWLEDGE_BASE,
                           respond.EMPTY_RESULT)

        # ---- N08 结果处理与证据判定（第 1+2 段纯计算，第 3 段条件调 LLM） ----
        state.sources = service.extract_sources(state.retrieval.chunks)
        node_budget = _node_budget(state, BUDGET_JUDGE_MS)
        if node_budget is None:
            # 预算不足不启动第 3 段：无判定能力，保守判不足（5.4 fail-safe 精神）
            state.evidence = EvidenceJudgement(reason="judge_skipped_budget")
            state.add_trace("N08", "budget", {"skipped": True})
        else:
            llm = get_llm(timeout=max(1.0, node_budget / 1000))
            state.evidence = service.judge_evidence(
                state.resolved_question, state.retrieval, llm=llm,
            )
            state.add_trace("N08", "judge", {
                "sufficient": state.evidence.sufficient,
                "stage1": state.evidence.stage1_gate1,
                "stage2": state.evidence.stage2_quote_verified,
                "stage3": state.evidence.stage3_answerable,
                "reason": state.evidence.reason,
            }, node_budget_ms=node_budget)

        # ---- N09 继续或结束（唯一回边，守卫见设计文档 2.2） ----
        if state.evidence.sufficient:
            break
        can_retry = (
            state.evidence.stage1_gate1 is not False  # 第 1 段没过，重试无意义
            and state.tool_rounds < tools.MAX_TOOL_ROUNDS
            and state.budget.remaining_ms() >= tools.MIN_USABLE_BUDGET_MS + TOOL_BUDGET_MS
            and retries < _MAX_RETRIES
        )
        state.add_trace("N09", "decision", {
            "sufficient": False, "can_retry": can_retry, "tool_rounds": state.tool_rounds,
        })
        if not can_retry:
            return _finish(state, got_lock, TerminationReason.INSUFFICIENT_EVIDENCE,
                           respond.EVIDENCE_INSUFFICIENT)
        retries += 1
        k = min(service.MAX_TOP_K, k + _K_STEP)  # 改写后的 query：扩大候选集

    # ---- N10 回答生成（只有证据充分才会到达这里） ----
    node_budget = _node_budget(state, BUDGET_GENERATE_MS)
    if node_budget is None:
        return _finish(state, got_lock, TerminationReason.BUDGET_EXHAUSTED,
                       respond.BUDGET_EXHAUSTED_REPLY)
    try:
        llm = get_llm(timeout=max(1.0, node_budget / 1000))
        answer = service.generate_answer(
            state.resolved_question, state.retrieval.chunks, llm=llm,
        )
    except service.RagServiceError as exc:
        # 消息是机器可读短标识（如 generate_failed:TimeoutError），可入 trace，不给用户
        state.add_trace("N10", "error", {"detail": str(exc)}, node_budget_ms=node_budget)
        return _finish(state, got_lock, TerminationReason.GENERATION_FAILED,
                       respond.GENERATION_ERROR)
    state.answer_text = answer
    return _finish(state, got_lock, TerminationReason.COMPLETED, answer)


# ==================== 内部工具 ====================


def _node_budget(state: AgentState, config_ms: int) -> int | None:
    """按 5.6 分配规则算本环节预算：min(配置预算, 剩余全局预算)；
    低于 MIN_USABLE_BUDGET 时返回 None（该环节不得启动）。"""
    remaining = state.budget.remaining_ms()
    node = min(config_ms, remaining)
    if node < tools.MIN_USABLE_BUDGET_MS:
        return None
    state.budget.node_budget_ms = node
    return node


def _merge_hits(first: EmergencyHit, second: EmergencyHit) -> EmergencyHit:
    """取两趟检查中更强的一档（N01 原文 / N03 补全后）。"""
    order = {EmergencyLevel.NONE: 0, EmergencyLevel.WEAK: 1, EmergencyLevel.STRONG: 2}
    if order[second.level] > order[first.level]:
        return second
    return first


RESOLVE_SYSTEM_PROMPT = """你是对话上下文整理器。用户本轮的问题引用了此前的对话，
请把它改写成一个不依赖上下文、独立完整的问题。

要求：
1. 只输出 JSON 对象，不要输出解释、前后缀或 markdown 代码块；
2. 改写后的问题必须保留用户本轮的原始意图，补全被指代的药名、症状或事物；
3. 确实无法确定指代对象时，输出 {{"question": ""}}，不要猜测；
4. 输出格式：{{"question": "改写后的完整问题"}}"""

RESOLVE_USER_PROMPT = """【此前的对话】
{history}

【用户本轮的问题】
{question}

请输出改写后的完整问题。"""


def _resolve_reference(state: AgentState, node_budget_ms: int) -> str | None:
    """N02 上下文补全。失败（LLM 异常/输出非法/指代无法确定）返回 None。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate

        history_text = "\n".join(
            f"{'用户' if role == 'user' else '助手'}：{content}"
            for role, content in state.history
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESOLVE_SYSTEM_PROMPT),
            ("human", RESOLVE_USER_PROMPT),
        ])
        messages = prompt.format_messages(history=history_text, question=state.resolved_question)
        llm = get_llm(timeout=max(1.0, node_budget_ms / 1000))
        response = llm.invoke(messages)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return None
        import json
        import re

        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return None
        payload = json.loads(match.group(0))
        question = payload.get("question")
        if not isinstance(question, str):
            return None
        question = question.strip()
        return question or None
    except Exception:
        return None


def _format_source(source) -> str:
    """来源展示格式：文档名（剥掉盘符与目录，只留 basename）+ 可选页码。

    extract_sources 已保证 document 非空；basename 后若仍为空则跳过该条，
    宁可不显示也不编造（设计文档 4.1「不伪造来源」）。
    """
    name = os.path.basename(source.document) if source.document else ""
    if not name:
        return ""
    if source.page:
        return f"{name} - 第{source.page}页"
    return name


def display_sources(state: AgentState) -> list[dict]:
    """接口层序列化用：只含展示所需的文档名与页码，不含原始路径。"""
    result = []
    for source in state.sources:
        text = _format_source(source)
        if text:
            result.append({"document": text, "page": source.page})
    return result


def _assemble(state: AgentState, body: str) -> str:
    """N11 回答组装：风险提示 / 无历史说明 → 正文 → 来源 → 免责声明，固定顺序。

    组装本身不抛异常；缺件按分支降级（来源为空则不拼来源段）。
    """
    parts: list[str] = []
    if state.risk_notice:
        parts.append(state.risk_notice)
    if state.context_status is ContextStatus.DEGRADED_NO_HISTORY:
        parts.append(respond.NO_HISTORY_NOTICE)
    parts.append(body)
    if state.termination_reason is TerminationReason.COMPLETED and state.sources:
        sources_text = "；".join(t for t in (_format_source(s) for s in state.sources) if t)
        if sources_text:
            parts.append(respond.SOURCE_HEADER.lstrip("\n") + sources_text)
    # 免责声明只跟在"含医学内容"的路径后：知识问答完成、紧急应答；
    # 闲聊/无法处理/重述等固定文案自身已完整，不再追加。
    show_disclaimer = (
        state.termination_reason is TerminationReason.EMERGENCY_HANDLED
        or (state.termination_reason is TerminationReason.COMPLETED
            and state.route is Route.KNOWLEDGE)
    )
    if show_disclaimer:
        parts.append(respond.DISCLAIMER)
    return "\n\n".join(parts)


def _finish(state: AgentState, got_lock: bool, reason: TerminationReason, body: str) -> AgentState:
    """终态收口：落终止原因、组装用户可见文本、写 trace、更新会话（N12）。

    会话更新失败不影响响应（session.complete 内部已兜底）。
    """
    state.termination_reason = reason
    state.answer_text = _assemble(state, body)
    state.add_trace("N12", "decision", {
        "termination": reason.value,
        "tool_rounds": state.tool_rounds,
        "tool_calls": len(state.tool_calls),
    })
    session.complete(
        state.session_id, state.raw_question, state.answer_text, got_lock=got_lock,
    )
    return state


__all__ = [
    "AgentRequestError",
    "run_agent",
    "display_sources",
    "REQUEST_DEADLINE",
    "RETRIEVAL_TOOL",
]
