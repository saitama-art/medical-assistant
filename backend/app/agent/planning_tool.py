
import os
import json
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END


# 1. 模型配置：沿用现有 DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepsee. k.com",
)


# 2. 接入已有的 RAG 向量库
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


@tool
def retrieve_knowledge(query: str) -> str:
    """检索知识库，查询与业务问题相关的文档依据。"""
    docs = vector_store.similarity_search(query, k=3)

    if not docs:
        return "知识库没有检索到相关内容。"

    return "\n\n".join(
        doc.page_content for doc in docs
    )


@tool
def query_sales(product: str = "") -> str:
    """查询示例销售数据，可按产品名称筛选。"""
    data = {
        "产品A": 1000,
        "产品B": 2000,
        "产品C": 1500,
    }

    if product:
        return str({
            k: v for k, v in data.items()
            if product in k
        })

    return str(data)


@tool
def analyze_sales(data: str) -> str:
    """分析销售数据，计算销售总额和最高销售产品。"""
    sales = json.loads(data)

    if not sales:
        return "没有可分析的销售数据。"

    total = sum(sales.values())
    top_product = max(sales, key=sales.get)

    return json.dumps({
        "total_sales": total,
        "top_product": top_product,
        "top_sales": sales[top_product],
    }, ensure_ascii=False)


tools = [
    retrieve_knowledge,
    query_sales,
    analyze_sales,
]
tools_by_name = {t.name: t for t in tools}


# 3. 定义整个 Agent 的状态
class AgentState(TypedDict):
    goal: str
    plan: list[dict]
    current_step: int
    step_results: list[dict]
    retry_count: int
    replan_count: int
    decision: str
    final_answer: str


# 4. Planner：制定结构化计划
def planner(state: AgentState):
    history = json.dumps(
        state["step_results"],
        ensure_ascii=False,
    )

    response = llm.invoke([
        (
            "system",
            """你是一个任务规划器。
            将用户目标拆解为可执行的步骤。
            可用工具：retrieve_knowledge、query_sales、
            analyze_sales。
            根据任务需要安排工具，不要无意义地调用所有工具。
            如果已有结果可复用，不要重复安排已完成工作。
            只输出 JSON 数组，例如：
            [{"task":"查询销售数据"},{"task":"分析销售数据"}]"""
        ),
        (
            "human",
            f"用户目标：{state['goal']}\n"
            f"已有执行结果：{history}"
        ),
    ])

    content = response.content.strip()
    content = content.removeprefix("```json").removesuffix("```").strip()

    plan = json.loads(content)

    if not isinstance(plan, list) or not plan:
        raise ValueError("Planner 没有生成有效计划")

    return {
        "plan": plan,
        "current_step": 0,
        "decision": "execute",
    }


# 5. Executor：根据当前步骤选择并执行工具
def executor(state: AgentState):
    step = state["plan"][state["current_step"]]
    history = json.dumps(
        state["step_results"],
        ensure_ascii=False,
    )

    response = llm.bind_tools(tools).invoke([
        (
            "system",
            """你是一个任务执行器。
            根据当前任务选择合适的工具。
            如果前面步骤已有结果，应优先利用这些结果。
            只能调用提供的工具；不要编造工具执行结果。"""
        ),
        (
            "human",
            f"用户目标：{state['goal']}\n"
            f"当前任务：{step['task']}\n"
            f"之前的执行结果：{history}"
        ),
    ])

    outputs = []

    for call in response.tool_calls:
        try:
            result = tools_by_name[call["name"]].invoke(
                call["args"]
            )
            outputs.append({
                "tool": call["name"],
                "result": str(result),
                "error": False,
            })
        except Exception as exc:
            outputs.append({
                "tool": call["name"],
                "result": str(exc),
                "error": True,
            })

    if not response.tool_calls:
        outputs.append({
            "tool": "llm_response",
            "result": response.content,
            "error": False,
        })

    return {
        "step_results": [
            *state["step_results"],
            {
                "step": step["task"],
                "outputs": outputs,
            },
        ],
        "decision": "evaluate",
    }


# 6. Evaluator：评估当前步骤结果
def evaluator(state: AgentState):
    latest = state["step_results"][-1]
    outputs = latest["outputs"]

    has_error = any(item["error"] for item in outputs)

    if has_error:
        if state["retry_count"] < 1:
            return {"decision": "retry"}
        if state["replan_count"] < 1:
            return {"decision": "replan"}
        return {"decision": "finish"}

    evaluation = llm.invoke([
        (
            "system",
            """判断当前任务结果是否足以完成当前步骤。
            如果足够，只输出 PASS；如果不够，只输出 FAIL。
            不要输出其他内容。"""
        ),
        (
            "human",
            f"当前任务：{latest['step']}\n"
            f"执行结果：{json.dumps(outputs, ensure_ascii=False)}"
        ),
    ])

    verdict = evaluation.content.strip().upper()

    if verdict.startswith("PASS"):
        return {"decision": "advance"}

    if state["retry_count"] < 1:
        return {"decision": "retry"}

    if state["replan_count"] < 1:
        return {"decision": "replan"}

    return {"decision": "finish"}


# 7. 控制进度和重新规划次数
def advance(state: AgentState):
    return {
        "current_step": state["current_step"] + 1,
        "retry_count": 0,
    }


def retry_step(state: AgentState):
    return {
        "retry_count": state["retry_count"] + 1,
    }


def replan(state: AgentState):
    return {
        "replan_count": state["replan_count"] + 1,
        "retry_count": 0,
    }


def route_after_evaluation(
    state: AgentState,
) -> Literal["advance", "retry", "replan", "finish"]:
    return state["decision"]


def route_after_advance(
    state: AgentState,
) -> Literal["executor", "finish"]:
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "finish"


def finish(state: AgentState):
    response = llm.invoke([
        (
            "system",
            "根据实际执行结果回答用户。"
            "区分知识库依据和数据分析结果，不要编造数据。"
        ),
        (
            "human",
            f"用户目标：{state['goal']}\n"
            f"执行记录：{json.dumps(state['step_results'], ensure_ascii=False)}"
        ),
    ])

    return {"final_answer": response.content}


# 8. 构建 LangGraph
graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("evaluator", evaluator)
graph.add_node("advance", advance)
graph.add_node("retry_step", retry_step)
graph.add_node("replan", replan)
graph.add_node("finish", finish)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "evaluator")

graph.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "advance": "advance",
        "retry": "retry_step",
        "replan": "replan",
        "finish": "finish",
    },
)

graph.add_edge("retry_step", "executor")
graph.add_edge("replan", "planner")

graph.add_conditional_edges(
    "advance",
    route_after_advance,
    {
        "executor": "executor",
        "finish": "finish",
    },
)

graph.add_edge("finish", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "goal": "查询销售数据，分析最高销售产品，并结合知识库生成分析报告",
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "retry_count": 0,
        "replan_count": 0,
        "decision": "",
        "final_answer": "",
    })

    print(result["final_answer"])
