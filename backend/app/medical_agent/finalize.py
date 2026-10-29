"""终态收口与回答组装（N11/N12）：从 orchestrator 拆出（2026-10-29 解耦）。

职责：来源聚合展示、N11 固定顺序组装、``finish``（唯一终态收口）与
``finish_recovery``（复杂层决策 → 终态）。不做任何节点业务逻辑。
"""

from __future__ import annotations

import os

from . import complex_handler, respond, session
from .state import (
    AgentState,
    ContextStatus,
    Route,
    TerminationReason,
)

# 来源展示上限：同一文档的页码最多列 3 个、文档最多列 3 个（其余以「等」示意）。
# 背景：检索 k=10 的每个 chunk 都产出来源，同一文档常连续出现 7~9 条页码，
# 用户可读性差（2026-10-20 实测反馈）。
_SOURCE_MAX_PAGES_PER_DOC = 3
_SOURCE_MAX_DOCS = 3


def format_sources(sources) -> list[str]:
    """把来源列表聚合成用户可读的展示串：按文档分组、页码升序去重、超限截断。

    返回每个文档一条（"文档 - 第2、7、12页"），保持检索相关度带来的首次出现顺序。
    """
    by_doc: dict[str, list[int]] = {}
    no_page: list[str] = []
    for s in sources:
        # 历史 Chroma 元数据可能是 Windows 路径；Linux 也需去掉盘符和目录。
        name = os.path.basename(s.document.replace("\\", "/")) if s.document else ""
        if not name:
            continue
        if s.page:
            pages = by_doc.setdefault(name, [])
            if s.page not in pages:
                pages.append(s.page)
        elif name not in no_page:
            no_page.append(name)

    out: list[str] = []
    for name, pages in by_doc.items():
        pages.sort()
        if len(pages) > _SOURCE_MAX_PAGES_PER_DOC:
            page_txt = "、".join(str(p) for p in pages[:_SOURCE_MAX_PAGES_PER_DOC]) + "页等"
        else:
            page_txt = "、".join(str(p) for p in pages) + "页"
        out.append(f"{name} - 第{page_txt}")
    for name in no_page:
        out.append(name)

    if len(out) > _SOURCE_MAX_DOCS:
        out = out[:_SOURCE_MAX_DOCS]
        out.append("等")
    return out


def display_sources(state: AgentState) -> list[dict]:
    """接口层序列化用：聚合后的展示串，不含原始路径。"""
    return [{"document": t, "page": None} for t in format_sources(state.sources)]


def assemble(state: AgentState, body: str) -> str:
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
        sources_text = "；".join(format_sources(state.sources))
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


def finish_recovery(
    state: AgentState,
    got_lock: bool,
    decision: complex_handler.RecoveryDecision,
    node: str,
) -> AgentState:
    """把纯策略决策映射到唯一终态收口，不在此重试或调用 Worker。"""
    reason = decision.termination
    if reason is None:
        raise ValueError("recovery_decision_missing_termination")
    if decision.error is not None:
        state.error = decision.error
    if decision.action is complex_handler.RecoveryAction.CLARIFY:
        state.recovery.clarified = True
    if reason is TerminationReason.CONTEXT_MISSING:
        state.route = Route.CONTEXT_MISSING
    elif reason is TerminationReason.UNCLASSIFIED:
        state.route = Route.UNCLASSIFIABLE
    state.add_trace(node, "recovery", {
        "action": decision.action.value,
        "termination": reason.value,
        "error_kind": decision.error.kind.value if decision.error else None,
    })
    return finish(state, got_lock, reason, respond.TERMINATION_TEXTS[reason])


def finish(state: AgentState, got_lock: bool, reason: TerminationReason, body: str) -> AgentState:
    """终态收口：落终止原因、组装用户可见文本、写 trace、更新会话（N12）。

    会话更新失败不影响响应（session.complete 内部已兜底）。
    """
    state.termination_reason = reason
    state.answer_text = assemble(state, body)
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
    "assemble",
    "display_sources",
    "finish",
    "finish_recovery",
    "format_sources",
]
