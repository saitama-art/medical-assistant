import json
import os
import httpx
from openai import OpenAI

# ==================== 1. 定义工具（普通函数） ====================
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    # 真实场景：调天气 API
    # 演示：返回模拟数据
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "深圳": "小雨，30°C",
    }
    return weather_data.get(city, f"{city}：暂无数据")


def calculate(expression: str) -> str:
    """计算数学表达式，如 '1+2*3'"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


# ==================== 2. 定义 Tool Schema ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '1+2*3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ==================== 3. 工具名 → 函数 的映射 ====================
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ==================== 4. 调用 LLM ====================
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    http_client=httpx.Client(trust_env=False, timeout=35),
)


def chat(user_message: str):
    print(f"\n用户：{user_message}")

    # 第一轮：把问题 + 工具列表发给 LLM
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
    )

    message = response.choices[0].message

    # 判断：LLM 有没有要求调用工具
    if message.tool_calls:
        print(f"LLM 决定调用工具：")

        # 逐个执行工具
        tool_messages = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"  → {tool_name}({tool_args})")

            # 执行工具
            result = TOOL_MAP[tool_name](**tool_args)
            print(f"  ← 结果：{result}")

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 第二轮：把工具结果返回给 LLM，生成最终回答
        second_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": user_message},
                message,
                *tool_messages,
            ],
        )
        final_answer = second_response.choices[0].message.content
    else:
        # LLM 没调工具，直接回答
        final_answer = message.content

    print(f"AI：{final_answer}")
    return final_answer


# ==================== 5. 测试 ====================
if __name__ == "__main__":
    chat("北京天气怎么样？")
    chat("上海天气怎么样？")
    chat("帮我算一下 123 * 456")
    chat("深圳天气怎么样？再帮我算一下 100 / 4")