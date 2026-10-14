
import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。城市可以是北京、上海或深圳。"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "深圳": "小雨，30°C",
    }
    return weather_data.get(city, f"{city}：暂无数据")


tools = [get_weather]

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
).bind_tools(tools)


def agent(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


#  构建工作流
builder = StateGraph(State)

builder.add_node("agent", agent)

builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition,
)

builder.add_edge("tools", "agent")

app = builder.compile()


if __name__ == "__main__":
    result = app.invoke({
        "messages": [
            {"role": "user", "content": "北京天气怎么样？"}
        ]
    })

    print(result["messages"][-1].content)
