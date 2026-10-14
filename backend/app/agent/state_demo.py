
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    step: int


def first_node(state: State) -> dict:
    print("\n--- first_node ---")
    print("进入节点时消息数：", len(state["messages"]))
    print("当前 step：", state["step"])

    return {
        "messages": [AIMessage(content="第一步执行完成")],
        "step": state["step"] + 1,
    }


def second_node(state: State) -> dict:
    print("\n--- second_node ---")
    print("进入节点时消息数：", len(state["messages"]))
    print("当前 step：", state["step"])

    return {
        "messages": [AIMessage(content="第二步执行完成")],
        "step": state["step"] + 1,
    }


builder = StateGraph(State)

builder.add_node("first", first_node)
builder.add_node("second", second_node)

builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

app = builder.compile()

result = app.invoke({
    "messages": [HumanMessage(content="开始执行")],
    "step": 0,
})

print("\n--- 最终 State ---")
print("step：", result["step"])

for message in result["messages"]:
    print(f"{message.type}: {message.content}")
