"""daily 路由的直接回应节点（N04b）：从 orchestrator 拆出（2026-10-29 解耦）。

0 检索、1 次轻量 LLM 调用；LLM 由调用方注入（llm_factory），
终态收口走 finalize.finish（唯一收口点）。
"""

from __future__ import annotations

from . import config as _cfg
from . import respond
from .finalize import finish
from .state import AgentState, TerminationReason

DAILY_SYSTEM_PROMPT = """你是医疗健康知识助手的日常对话模块。用户这句话不是医学知识查询，
请结合对话记录直接给出自然、简短、有用的回应（2~4 句话）：

1. 可以提供安全的一般性生活建议；不得给出诊断、处方或具体用药推荐；
2. 如有必要，最多再提一个开放式的追问，不要连环发问；
3. 若用户问的是「我之前说了什么/我们聊了什么」，只能依据给定对话记录回答，
   记录里没有的内容一律不说，不得编造；
4. 内容若涉及可能的危险信号，提醒尽快就医；
5. 不要输出 JSON，不要引用或编造任何资料来源。"""

DAILY_USER_PROMPT = """{history_block}用户：{question}"""


def answer_daily(state: AgentState, got_lock: bool, *, llm_factory) -> AgentState:
    """daily 路由：一次轻量 LLM 调用给出自然回应；失败回退固定文案。

    与知识路径的边界：不检索、不引用来源、不追加医学免责声明——回应性质
    是日常对话而非医学结论。会话历史仅作为回应语境，不改变安全约束。
    """
    node_budget = _cfg.node_budget(state, _cfg.BUDGET_DAILY_MS)
    if node_budget is None:
        return finish(state, got_lock, TerminationReason.BUDGET_EXHAUSTED,
                      respond.BUDGET_EXHAUSTED_REPLY)
    history_block = ""
    if state.history:
        recent = "\n".join(
            f"{'用户' if role == 'user' else '助手'}：{content}"
            for role, content in state.history[-6:]
        )
        history_block = f"【对话记录】\n{recent}\n\n"
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", DAILY_SYSTEM_PROMPT),
            ("human", DAILY_USER_PROMPT),
        ])
        messages = prompt.format_messages(
            history_block=history_block, question=state.resolved_question,
        )
        llm = llm_factory(timeout=max(1.0, node_budget / 1000))
        response = llm.invoke(messages)
    except Exception as exc:
        state.add_trace("N04b", "error", {
            "detail": type(exc).__name__,
        }, node_budget_ms=node_budget)
        return finish(state, got_lock, TerminationReason.COMPLETED,
                      respond.DAILY_FALLBACK_REPLY)
    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        return finish(state, got_lock, TerminationReason.COMPLETED,
                      respond.DAILY_FALLBACK_REPLY)
    state.add_trace("N04b", "decision", {
        "answer_len": len(content.strip()),
    }, node_budget_ms=node_budget)
    return finish(state, got_lock, TerminationReason.COMPLETED, content.strip())


__all__ = [
    "DAILY_SYSTEM_PROMPT",
    "DAILY_USER_PROMPT",
    "answer_daily",
]
