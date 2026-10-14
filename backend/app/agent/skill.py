
from dataclasses import dataclass


# 两个基础 Tool
def calculate_total(sales: list[float]) -> float:
    return sum(sales)


def calculate_average(sales: list[float]) -> float:
    if not sales:
        raise ValueError("销售数据不能为空")
    return sum(sales) / len(sales)


# Skill：组织一类业务任务的执行步骤
@dataclass
class SalesAnalysisSkill:
    name: str = "sales_analysis"

    def run(self, sales: list[float]) -> dict:
        if not sales:
            raise ValueError("销售数据不能为空")

        total = calculate_total(sales)
        average = calculate_average(sales)

        return {
            "total": total,
            "average": round(average, 2),
            "count": len(sales),
        }


skill = SalesAnalysisSkill()
result = skill.run([200, 300, 600])

print(result)
