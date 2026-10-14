
from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Production Operations Server")

# 模拟生产数据
LINES = {
    "L01": {
        "status": "运行中",
        "output": 820,
        "target": 1000,
        "temperature": 72,
    },
    "L02": {
        "status": "停机",
        "output": 420,
        "target": 900,
        "temperature": 95,
    },
    "L03": {
        "status": "运行中",
        "output": 960,
        "target": 1000,
        "temperature": 68,
    },
}

ALARMS = [
    {
        "line_id": "L02",
        "level": "严重",
        "message": "设备温度过高",
    },
    {
        "line_id": "L01",
        "level": "警告",
        "message": "产量低于目标",
    },
]


@mcp.tool()
def get_all_line_status() -> list[dict]:
    """查询所有生产线的运行状态、产量、目标产量和设备温度。"""
    return [
        {"line_id": line_id, **data}
        for line_id, data in LINES.items()
    ]


@mcp.tool()
def get_recent_alarms() -> list[dict]:
    """查询近期生产线告警，包含产线编号、告警级别和告警描述。"""
    return ALARMS


@mcp.tool()
def calculate_output_gap(
    line_id: str,
    actual_output: int,
    target_output: int,
) -> dict:
    """计算指定生产线的产量差距和目标完成率。"""
    if target_output <= 0:
        return {"error": "目标产量必须大于 0"}

    gap = target_output - actual_output

    return {
        "line_id": line_id,
        "actual_output": actual_output,
        "target_output": target_output,
        "output_gap": gap,
        "completion_rate": round(
            actual_output / target_output * 100, 2
        ),
    }


@mcp.tool()
def create_maintenance_report(
    analysis: str,
    recommendations: str,
) -> dict:
    """根据分析结论和处理建议生成运维报告。"""
    return {
        "report_id": "OPS-" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis": analysis,
        "recommendations": recommendations,
        "status": "generated",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
