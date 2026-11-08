import requests
import time

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3:4b",
    "messages": [
        {
            "role": "user",
            "content": "你好，请用一句话介绍自己。"
        }
    ],
    "stream": False,
    "think": False,
    "options": {
        "num_predict": 128
    }
}

print("正在请求 Ollama...")
start = time.time()

try:
    response = requests.post(
        url,
        json=payload,
        timeout=(10, 600)
    )

    response.raise_for_status()

    result = response.json()

    print("\nQwen3 回答：")
    print(result["message"]["content"])

    print(f"\n耗时：{time.time() - start:.1f} 秒")

except requests.exceptions.Timeout:
    print("请求超时，请检查 Ollama 和模型运行状态。")

except requests.exceptions.RequestException as e:
    print("API 请求失败：", e)