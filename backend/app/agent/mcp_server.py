from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Production Server")

PRODUCTION_DATA = {
    "L01": {"status": "运行中", "output": 1200},
    "L02": {"status": "停机检修", "output": 0},
    "L03": {"status": "运行中", "output": 980},
}


@mcp.tool()
def get_production_status(line_id: str) -> dict:
    """查询指定生产线的运行状态和产量。line_id 例如 L01、L02、L03。"""
    data = PRODUCTION_DATA.get(line_id)

    if data is None:
        return {"error": f"未找到生产线 {line_id}"}

    return {
        "line_id": line_id,
        **data,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")