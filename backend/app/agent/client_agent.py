import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0,
)


async def main():
    server_path = Path(__file__).with_name("server.py")

    if not server_path.exists():
        raise FileNotFoundError(
            f"MCP Server 文件不存在：{server_path}"
        )

    client = MultiServerMCPClient({
        "production": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_path)],
        }
    })

    # 发现 MCP Server 提供的工具
    tools = await client.get_tools()

    print("发现的工具：")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
你是一名生产运维分析助手。

任务要求：
1. 必须先查询所有生产线的状态和产量。
2. 查询近期告警，并结合生产线数据分析异常。
3. 对每条生产线调用产量差距计算工具，不能自行编造计算结果。
4. 根据数据确定问题优先级，区分严重故障和一般产能不足。
5. 最后调用运维报告工具，保存分析结论和处理建议。
6. 最终向用户汇总报告编号、异常情况、产量差距和处理建议。

规则：
- 只能使用工具返回的数据，不得编造事实。
- 如果工具调用失败，要说明失败，不得假装成功。
- 产量差距 = 目标产量 - 实际产量。
- 严重设备告警优先处理。
"""
    )

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "请检查今天所有生产线的运行情况，"
                    "找出异常，计算产量差距，"
                    "按优先级提出处理建议，并生成运维报告。"
                ),
            }
        ]
    })

    print("\n========== Agent 最终回答 ==========")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
