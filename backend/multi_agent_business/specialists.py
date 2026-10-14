import json
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm
from data_source import load_retail_data
from analytics import analyze_revenue, analyze_customers, analyze_products

SPECIALIST_CONFIG = {
    "revenue_agent": {
        "title": "收入与订单分析专家",
        "task": "分析净收入、订单量、月度收入变化，并指出指标定义和限制。",
        "fn": analyze_revenue,
    },
    "customer_agent": {
        "title": "客户分析专家",
        "task": "分析客户贡献、客户集中度，并明确缺失客户 ID 的处理方式。",
        "fn": analyze_customers,
    },
    "product_agent": {
        "title": "商品分析专家",
        "task": "分析商品收入、销量排名，并避免把相关性写成因果关系。",
        "fn": analyze_products,
    },
}

def run_specialist(state: dict, agent_name: str) -> dict:
    """专业 Agent：先用真实数据计算，再由 LLM 解释，不让模型凭空编造指标。"""
    config = SPECIALIST_CONFIG[agent_name]
    df = load_retail_data()
    evidence = config["fn"](df)

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            f"你是{config['title']}。只能依据提供的真实计算结果进行分析。"
            "区分事实与推测；没有证据就明确说无法判断。"
            "使用中文，输出：主要发现、业务含义、限制。不要伪造数字。"
        )),
        HumanMessage(content=(
            f"用户问题：{state.get('question', '')}\n"
            f"本次任务：{config['task']}\n"
            f"真实计算结果 JSON：{json.dumps(evidence, ensure_ascii=False, default=str)}"
        )),
    ])
    return {
        "findings": [{
            "agent": agent_name,
            "task": config["task"],
            "evidence": evidence,
            "analysis": response.content,
        }]
    }

def revenue_agent(state: dict) -> dict:
    return run_specialist(state, "revenue_agent")

def customer_agent(state: dict) -> dict:
    return run_specialist(state, "customer_agent")

def product_agent(state: dict) -> dict:
    return run_specialist(state, "product_agent")
