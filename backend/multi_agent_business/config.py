from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# config.py 位于项目根目录
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR.parent / "data" / "Online Retail.xlsx"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR.parent.parent / ".env")
def get_llm(temperature: float = 0):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "没有读取到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 并配置密钥。"
        )
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=temperature,
    )
