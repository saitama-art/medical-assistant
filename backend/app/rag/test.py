"""当前 LLM 配置的冒烟检查：模型、温度与一次真实 ping。

配置来源与生产完全一致（chain.get_llm → .env 的 LLM_* / DEEPSEEK_*），
所以换模型后无需改本文件。用法（backend/ 目录下）：

    python -m app.rag.test
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .chain import get_llm  # noqa: E402


def main() -> int:
    model = os.environ.get("LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.environ.get(
        "LLM_BASE_URL",
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    print(f"model={model}")
    print(f"base_url={base_url}")
    try:
        response = get_llm(timeout=20).invoke("ping")
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    print(f"OK: {str(response.content)[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
