from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage, AIMessage

def chat(state: MessagesState):
    # Demo：用固定逻辑代替真实大模型
    user_msg = state["messages"][-1].content
    return {
        "messages": [
            AIMessage(content=f"我记住了当前消息：{user_msg}")
        ]
    }

builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")

checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-001"}}

app.invoke(
    {"messages": [HumanMessage(content="我喜欢柱状图")]},
    config=config
)

result = app.invoke(
    {"messages": [HumanMessage(content="我喜欢什么图表？")]},
    config=config
)

print(result["messages"][-2:])