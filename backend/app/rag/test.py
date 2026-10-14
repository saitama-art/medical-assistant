import httpx
import os

api_key = os.environ["DEEPSEEK_API_KEY"]
url = "https://api.deepseek.com/v1/chat/completions"

client = httpx.Client(trust_env=False, timeout=20)

resp = client.post(
    url,
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model":"deepseek-v4-flash",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":100
    }
)
print(resp.status_code)
print(resp.text)