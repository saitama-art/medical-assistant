import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有读取到 DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
)
print("当前工作目录：", os.getcwd())
print("是否读取到 Key：", bool(api_key))
print("Key 长度：", len(api_key) if api_key else 0)