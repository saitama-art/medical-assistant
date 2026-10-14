import json
import os
import httpx
from numpy import char
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


def get_bmi(height: float  ,weight:float|int) -> str| float:
    """计算身体bmi，如  86 /  (1.5 * 1.5)'"""
    try:
        result = weight / (height * height)
        return f"result = {result:.2f}"
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
            "name": "get_bmi",
            "description": "计算bmi，就是身高除以体重的平方",
            "parameters": {
                "type": "object",
                "properties": {
                    "height": {
                        "type": "number",
                        "description": "身高单位米 如身高1.76米",
                    },
                    "weight": {
                        "type": "number",
                        "description": "体重 如体重 86公斤",
                    }
                },
                "required": ["height","weight"],
            },
        },
    },
]


# ==================== 3. 工具名 → 函数 的映射 ====================
TOOL_MAP = {
    "get_weather": get_weather,
    "get_bmi": get_bmi,
}


# ==================== 4. 调用 LLM ====================
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    http_client=httpx.Client(trust_env=False, timeout=35),
)



def agent_loop(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    print(messages,'::messages')
    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        print(response.choices[0].message,':::::::response.choices[0].message')

        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"  → {tool_name}({tool_args})")

            result = TOOL_MAP[tool_name](**tool_args)
            print(f"  ← 结果：{result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })


if __name__ == "__main__":
    print(agent_loop("北京天气怎么样？"))
    print(agent_loop("我的身高1.78，体重86，帮我算BMI"))