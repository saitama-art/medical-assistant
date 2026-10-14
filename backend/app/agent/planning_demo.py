
def create_plan(need_chart: bool, need_report: bool) -> list[str]:
    """根据用户需求生成执行计划。"""
    plan = [
        "read_excel",
        "clean_data",
        "calculate_statistics",
    ]

    if need_chart:
        plan.append("generate_chart")

    if need_report:
        plan.append("generate_report")

    return plan


def execute_step(step: str):
    """模拟执行一个步骤。"""
    print(f"正在执行：{step}")


# 用户要求：分析数据，并生成图表和报告
plan = create_plan(
    need_chart=True,
    need_report=True,
)

print("执行计划：", plan)

for step in plan:
    execute_step(step)

print("任务完成")
