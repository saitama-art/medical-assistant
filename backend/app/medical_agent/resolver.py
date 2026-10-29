"""N02 上下文补全（指代消解）：从 orchestrator 拆出（2026-10-29 解耦）。

LLM 由调用方注入（llm_factory 参数），本模块不直接 import get_llm——
编排器仍是模型出口的唯一接线点，测试 patch 面不变。
"""

from __future__ import annotations

import json
import re

from . import complex_handler, context_engineering
from .state import (
    AgentState,
    NodeResult,
    NodeStatus,
    ResolutionResult,
    ResolutionStatus,
)

RESOLVE_SYSTEM_PROMPT = """你是对话上下文整理器。用户本轮的问题引用了此前的对话，
请把它改写成一个不依赖上下文、独立完整的问题。

要求：
1. 只输出 JSON 对象，不要输出解释、前后缀或 markdown 代码块；
2. 改写后的问题必须保留用户本轮的原始意图，补全被指代的药名、症状或事物；
3. **保留实体**：本轮问题中出现的或历史对话中已明确的具体实体（疾病名、症状、
   药品名、检查名）必须原样保留在改写后的问题里，只补全指代词所指的内容，
   不得把具体问题泛化成通用问题（如「腹泻要去医院的情况」不得泛化为
   「什么情况要去医院」）；
4. **保留否定词与限定语**：不、没有、别、未、无以及时间/人群限定
   （如「最近一周」「空腹」）必须原样保留，不得省略、反转或改写为反义；
5. 确实无法确定指代对象时，输出 {{"question": ""}}，不要猜测；
6. 输出格式：{{"question": "改写后的完整问题"}}"""

RESOLVE_USER_PROMPT = """【此前的对话】
{history}

【用户本轮的问题】
{question}

请输出改写后的完整问题。"""

RESOLVE_REPAIR_PROMPT = """

上一次输出未通过校验（含「原样返回未补全」的情况）。请重新输出合法 JSON：
question 必须把指代词替换为对话记录中的具体实体、保留否定词和限定语，
不得原样返回用户的问题；确实无法确定指代对象时只能输出空字符串。"""

_NEGATION_MARKERS = ("没有", "别", "不要", "未", "无")


def dropped_negation(raw: str, resolved: str) -> bool:
    """改写产物是否丢弃了原句的否定词（S1-D2，程序校验先于 LLM 信任）。

    「没有胃肠道反应吗」被改写成「有胃肠道反应吗」是语义反转，医疗场景
    绝不能放行。校验不通过→调用方降级 missing 走"请重述"，不猜。
    保守取向：宁可误判漏改（多一次重述）也不放过反转（错误医疗语义）。
    """
    return any(m in raw and m not in resolved for m in _NEGATION_MARKERS)


def resolve_reference(
    state: AgentState,
    node_budget_ms: int,
    *,
    repair: bool = False,
    llm_factory,
) -> NodeResult[ResolutionResult]:
    """执行一次 N02 补全；合法无法补全与调用/输出失败分别返回。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate

        history_text = context_engineering.format_history(
            state.context_package, state.history,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESOLVE_SYSTEM_PROMPT + (RESOLVE_REPAIR_PROMPT if repair else "")),
            ("human", RESOLVE_USER_PROMPT),
        ])
        messages = prompt.format_messages(
            history=history_text, question=state.resolved_question,
        )
        llm = llm_factory(timeout=max(1.0, node_budget_ms / 1000))
        response = llm.invoke(messages)
    except Exception as exc:
        return complex_handler.exception_result(exc)

    content = getattr(response, "content", None)
    if not isinstance(content, str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="content_not_text")

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="json_invalid")
    if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="question_invalid")

    resolved = payload["question"].strip()
    if dropped_negation(state.resolved_question, resolved):
        return NodeResult(
            NodeStatus.INVALID_OUTPUT, detail="model_constraint_violation",
        )
    if resolved and resolved == state.resolved_question.strip():
        # no-op 改写（2026-10-28 实测）：N02 只在检测到指代/追问时触发，
        # 原样返回意味着模型没有补全任何上下文——按可修复失败处理，
        # 走既有的一次 repair 路径，避免裸问句直接送分类器被判 unknown。
        return NodeResult(NodeStatus.INVALID_OUTPUT, detail="resolution_noop")
    if not resolved:
        return NodeResult(
            NodeStatus.OK, ResolutionResult(ResolutionStatus.CONTEXT_REQUIRED),
        )
    return NodeResult(
        NodeStatus.OK, ResolutionResult(ResolutionStatus.RESOLVED, resolved),
    )


__all__ = [
    "RESOLVE_REPAIR_PROMPT",
    "RESOLVE_SYSTEM_PROMPT",
    "RESOLVE_USER_PROMPT",
    "dropped_negation",
    "resolve_reference",
]
