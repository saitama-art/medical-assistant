import json
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm

MAX_RESEARCH_ROUNDS = 2

def validation_agent(state: dict) -> dict:
    findings = state.get("findings", [])
    research_round = state.get("research_round", 0)
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "你是证据审查 Agent。审查分析是否回答了用户问题，关键数字是否有计算证据，"
            "是否存在无依据因果判断或遗漏重要限制。"
            "只输出 JSON：{\"status\":\"SUPPORTED\"或\"INSUFFICIENT\","
            "\"reason\":\"具体原因\",\"missing_evidence\":\"需要补充什么\"}。"
        )),
        HumanMessage(content=(
            f"用户问题：{state.get('question', '')}\n"
            f"现有发现：{json.dumps(findings, ensure_ascii=False, default=str)}\n"
            f"补充研究轮数：{research_round}\n"
            f"最近补充研究备注：{state.get('research_note', '')}"
        )),
    ])
    raw = response.content.strip()
    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        status = parsed.get("status", "INSUFFICIENT")
        if status not in ("SUPPORTED", "INSUFFICIENT"):
            status = "INSUFFICIENT"
        reason = str(parsed.get("reason", "未提供原因"))
        missing = str(parsed.get("missing_evidence", "未明确"))
    except Exception:
        status, reason, missing = "INSUFFICIENT", "验证输出不是有效 JSON，按不充分处理。", "请重试验证"

    # 达到上限后允许继续生成报告，但报告会明确披露证据不足，而非伪装成通过
    if status != "SUPPORTED" and research_round >= MAX_RESEARCH_ROUNDS:
        status = "STOP_WITH_LIMITATION"

    return {
        "validation_result": status,
        "validation_reason": reason,
        "research_note": missing,
    }

def route_after_validation(state: dict) -> str:
    status = state.get("validation_result", "INSUFFICIENT")
    if status == "SUPPORTED":
        return "report"
    if status == "STOP_WITH_LIMITATION":
        return "report_with_limitations"
    return "targeted_research"
