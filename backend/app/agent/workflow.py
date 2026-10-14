
import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


# 1. 定义共享状态
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 2. 定义工具：只负责具体计算
@tool
def calculate_total(sales: list[float]) -> float:
    """计算一组销售数据的总销售额。"""
    return sum(sales)


@tool
def calculate_average(sales: list[float]) -> float:
    """计算一组销售数据的平均销售额。"""
    if not sales:
        raise ValueError("销售数据不能为空")
    return sum(sales) / len(sales)


tools = [calculate_total, calculate_average]


# 3. 配置 LLM，并把工具定义提供给模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

llm_with_tools = llm.bind_tools(tools)


# 4. Agent 节点：由模型决定下一步
def agent_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 5. 构建工作流
builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# 模型提出工具调用 -> tools
# 模型不再请求工具 -> END
builder.add_conditional_edges(
    "agent",
    tools_condition,
)

# 工具执行完成后，回到模型继续决策
builder.add_edge("tools", "agent")

app = builder.compile()


# 6. 执行任务
result = app.invoke({
    "messages": [
        HumanMessage(
            content=(
                "请分析这组销售数据 [200, 300, 600]。"
                "计算总销售额和平均销售额，并根据结果给出简短分析。"
                "需要时调用合适的工具，不要自行编造计算结果。"
            )
        )
    ]
})

# 7. 查看完整消息流程
for message in result["messages"]:
    print(f"\n角色：{message.type}")
    print(f"内容：{message.content}")

    if getattr(message, "tool_calls", None):
        print("工具调用：", message.tool_calls)
