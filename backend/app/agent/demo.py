from typing import TypedDict
from unittest import result

from  langgraph.graph import  StateGraph ,START,END

class State(TypedDict):
    question: str
    answer: str
    question_type:str

def classify_question(state: State) -> dict:
    question = state["question"]

    medical_keywords = ["高血压","糖尿病","症状","药物","治疗"]


    if "多少" in question or "计算"  in question:
            return {"question_type": "calculate"}

    if any(keyword in question for keyword in medical_keywords):
          return  {"question_type":  "rag"}


    return {"question_type" :"chat"}


def router_question(state: State) -> str:
    return  state["question_type"]

def rag_node(state: State) -> dict:
    # 模拟rag接口
    return {"answer": f"进入知识库查询{state['question']}"}

def calculate_answer(state: State) -> dict:
    return    {"answer": "进入了计算节点"}


def chat_node(state:State) -> dict:
    return   {"answer":f"进入了聊天节点 :{state['question']}"}


def process_question(state: State) -> dict:
    question = state["question"]

    return {
        "question": question.strip()
    }


def generate_answer(state: State) -> dict:
    question = state["question"]

    return {
        "answer": f"你问的问题是：{question}"
    }

builder = StateGraph(State)

state = {
    "question": "高血压有哪些常见症状？  ",
    "answer": ""
}

# 注册工作流
builder.add_node("classify_question", classify_question)
builder.add_node("calculate_answer", calculate_answer)
builder.add_node("chat_node", chat_node)
builder.add_node("rag_node", rag_node)

# 开启节点
builder.add_edge(START,"classify_question" )

# 注册路由
builder.add_conditional_edges(
    "classify_question",
    router_question,
    {
        "calculate_answer": "calculate_answer",
        "chat": "chat_node",
        "rag": "rag_node"
    }
)
builder.add_edge("chat_node", END)
builder.add_edge("rag_node", END)


app = builder.compile()

result = app.invoke({
    "question": "你好？",
    "answer": "",
    "question_type": ""
})

print(result)

