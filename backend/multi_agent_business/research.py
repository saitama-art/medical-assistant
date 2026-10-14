import json
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm
from data_source import load_retail_data
from analytics import analyze_returns_by_month

def targeted_research(state: dict) -> dict:
    """按验证 Agent 指出的缺口做一次定向计算与解释。"""
    df = load_retail_data()
    evidence = analyze_returns_by_month(df)
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "你是定向补充研究 Agent。只针对验证指出的证据缺口分析。"
            "基于提供的真实计算结果，不要编造数据。中文输出简洁结论和局限。"
        )),
        HumanMessage(content=(
            f"用户问题：{state.get('question', '')}\n"
            f"待补充证据：{state.get('research_note', '')}\n"
            f"验证原因：{state.get('validation_reason', '')}\n"
            f"真实计算结果：{json.dumps(evidence, ensure_ascii=False, default=str)}"
        )),
    ])
    current_round = state.get("research_round", 0) + 1
    return {
        "findings": [{
            "agent": "targeted_research",
            "task": "根据验证缺口补充退货与取消订单证据",
            "evidence": evidence,
            "analysis": response.content,
        }],
        "research_round": current_round,
        "research_note": response.content,
    }
