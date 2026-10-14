import json
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm, OUTPUT_DIR

def generate_report(state: dict) -> dict:
    llm = get_llm()
    status = state.get("validation_result", "UNKNOWN")
    findings = state.get("findings", [])
    response = llm.invoke([
        SystemMessage(content=(
            "你是企业经营分析报告 Agent。使用中文输出 Markdown 报告，"
            "包含执行摘要、主要发现、建议、数据口径与局限。"
            "只能使用提供的发现，不得编造数字。若验证状态不是 SUPPORTED，"
            "必须在开头显著说明证据尚不充分及其原因。历史数据不能描述为当前市场趋势。"
        )),
        HumanMessage(content=(
            f"用户问题：{state.get('question', '')}\n"
            f"验证状态：{status}\n"
            f"验证原因：{state.get('validation_reason', '')}\n"
            f"发现：{json.dumps(findings, ensure_ascii=False, default=str)}"
        )),
    ])
    report = response.content
    if status != "SUPPORTED":
        report = (
            f"> **证据限制提示：验证状态为 {status}。** "
            f"{state.get('validation_reason', '部分结论仍需要进一步验证。')}\n\n"
            + report
        )
    path = Path(OUTPUT_DIR) / "business_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{path}")
    return {"final_report": report}
