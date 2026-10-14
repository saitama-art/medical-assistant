from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

# 1. 定义工具
@tool
def query_sales() -> str:
    """查询销售数据"""
    return "产品A: 1000元, 产品B: 2000元"

@tool
def create_report(data: str) -> str:
    """根据数据生成分析报告"""
    return f"销售分析报告：{data}"

tools = [query_sales, create_report]
tools_by_name = {t.name: t for t in tools}


# 2. 定义 Agent 状态
class AgentState(TypedDict):
    goal: str
    plan: list[str]
    current_step: int
    result: str


# 3. Planning：制定计划
def planner(state: AgentState):
    response = llm.invoke(
        f"""请为以下任务制定执行计划，按顺序输出步骤。
        任务：{state['goal']}
        只输出步骤，每步一行。"""
    )

    plan = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return {"plan": plan, "current_step": 0}


# 4. Executor：执行当前步骤
def executor(state: AgentState):
    step = state["plan"][state["current_step"]]

    # 由 LLM 根据当前步骤选择工具
    response = llm.bind_tools(tools).invoke(
        f"当前任务：{step}\n原始目标：{state['goal']}"
    )

    if response.tool_calls:
        outputs = []
        for call in response.tool_calls:
            tool_result = tools_by_name[call["name"]].invoke(
                call["args"]
            )
            outputs.append(str(tool_result))

        return {"result": "\n".join(outputs)}

    return {"result": response.content}


# 5. 控制流程
def next_step(state: AgentState) -> Literal["executor", "finish"]:
    if state["current_step"] + 1 < len(state["plan"]):
        return "executor"
    return "finish"


def advance(state: AgentState):
    return {"current_step": state["current_step"] + 1}


def finish(state: AgentState):
    return {"result": state["result"]}


# 6. 构建图
graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("advance", advance)
graph.add_node("finish", finish)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "advance")
graph.add_conditional_edges(
    "advance",
    next_step,
    {
        "executor": "executor",
        "finish": "finish",
    },
)
graph.add_edge("finish", END)

app = graph.compile()

# 7. 执行
result = app.invoke({
    "goal": "查询销售数据并生成分析报告",
    "plan": [],
    "current_step": 0,
    "result": "",
})

print(result["result"])