
import os

from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# 1. 定义工具：只负责具体动作
@tool
def calculate_total(sales: list[float]) -> float:
    """计算销售数据的总销售额。"""
    return sum(sales)


@tool
def calculate_average(sales: list[float]) -> float:
    """计算销售数据的平均销售额。"""
    return sum(sales) / len(sales) if sales else 0.0


@tool
def check_performance(total: float) -> str:
    """判断总销售额是否达到1000。"""
    return "销售表现良好" if total >= 1000 else "销售表现有待提升"


tools = [calculate_total, calculate_average, check_performance]


# 2. 配置模型，并将工具描述提供给模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

llm_with_tools = llm.bind_tools(tools)


# 3. 定义共享状态
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 4. Agent 节点：让模型决定下一步
def agent_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 5. 构建工作流
builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "agent")

app = builder.compile()


# 6. 执行任务
result = app.invoke({
    "messages": [
        (
            "user",
            "销售数据是 [200, 300, 600]。"
            "请计算总额、平均值，并判断销售表现，最后总结。"
        )
    ]
})

print(result["messages"][-1].content)
