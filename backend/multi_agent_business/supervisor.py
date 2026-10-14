import json
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from config import get_llm

AGENTS = ("revenue_agent", "customer_agent", "product_agent")

def supervisor_plan(state: dict) -> dict:
    """主 Agent 根据用户问题选择要调用的专业 Agent。"""
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "你是业务分析系统的 Supervisor 主 Agent。"
            "你不直接做数据分析，只负责为专业 Agent 分配任务。"
            "可选 Agent 仅有 revenue_agent、customer_agent、product_agent。"
            "根据问题选择一个或多个 Agent；若问题是综合经营分析，选择全部。"
            "只输出 JSON，例如 {\"agents\":[\"revenue_agent\",\"product_agent\"],"
            "\"reason\":\"...\"}。"
        )),
        HumanMessage(content=f"用户问题：{state.get('question', '')}"),
    ])
    raw = response.content.strip()
    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        selected = [name for name in parsed.get("agents", []) if name in AGENTS]
    except Exception:
        selected = list(AGENTS)

    if not selected:
        selected = list(AGENTS)

    # 保存主 Agent 的计划，方便追踪“为什么调用这些子 Agent”
    return {"plan": selected}

def dispatch_to_specialists(state: dict):
    """Send 将同一个问题分发给计划中的专业 Agent；多个 Send 可并行执行。"""
    plan = state.get("plan") or list(AGENTS)
    return [
        Send(agent_name, {
            "question": state.get("question", ""),
            "task": agent_name,
        })
        for agent_name in plan
    ]
