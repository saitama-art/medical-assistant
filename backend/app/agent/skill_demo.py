import os
import json

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ---------- 1. 内部能力：普通 Python 函数 ----------

def clean_sales(sales: list[float]) -> list[float]:
    """过滤负数销售记录。"""
    return [x for x in sales if x >= 0]


def calculate_statistics(sales: list[float]) -> dict:
    """计算总额和平均值。"""
    return {
        "total": sum(sales),
        "average": sum(sales) / len(sales) if sales else 0,
    }


def evaluate_performance(total: float) -> str:
    """判断销售表现。"""
    return "销售表现良好" if total >= 1000 else "销售表现有待提升"


# ---------- 2. Skill：组合多个内部能力 ----------

@tool
def sales_analysis_skill(sales: list[float]) -> str:
    """对一组销售数据完成清洗、统计和表现判断。
    当用户要求完整分析销售数据时调用。
    """
    cleaned = clean_sales(sales)

    if not cleaned:
        return "没有可用于分析的有效销售数据。"

    stats = calculate_statistics(cleaned)
    conclusion = evaluate_performance(stats["total"])

    return json.dumps({
        "cleaned_sales": cleaned,
        **stats,
        "conclusion": conclusion,
    }, ensure_ascii=False)


# ---------- 3. Agent：只暴露 Skill 入口 ----------

tools = [sales_analysis_skill]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---------- 4. Workflow：组织执行流程 ----------

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


# ---------- 5. 运行 ----------

result = app.invoke({
    "messages": [
        (
            "user",
            "请分析销售数据 [200, 300, 600]，"
            "计算总额和平均值，并判断销售表现。"
        )
    ]
})

print(result["messages"][-1].content)