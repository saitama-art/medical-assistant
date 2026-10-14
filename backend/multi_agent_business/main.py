from graph import build_graph

QUESTION = (
    "请分析这份零售数据中的收入表现、客户贡献和商品表现，"
    "指出需要进一步核查的退货或取消订单问题，并给出有证据支持的经营建议。"
)

def main():
    app = build_graph()
    initial_state = {
        "question": QUESTION,
        "findings": [],
        "research_round": 0,
        "research_note": "",
    }

    print("启动 Supervisor 多 Agent 业务分析工作流...")
    result = app.invoke(initial_state, config={"recursion_limit": 20})

    print("\n===== Supervisor 调度计划 =====")
    print(result.get("plan", []))
    print("\n===== 验证结果 =====")
    print(result.get("validation_result"))
    print(result.get("validation_reason"))
    print("\n===== 最终报告预览 =====")
    print((result.get("final_report") or "")[:3000])

if __name__ == "__main__":
    main()
