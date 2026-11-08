import requests
import time

url = "http://localhost:11434/api/embed"

payload = {
    "model": "bge-m3",
    "input": ["这是一个本地 PDF RAG 测试"]
}

print("正在测试 BGE-M3...")

start = time.time()

try:
    response = requests.post(
        url,
        json=payload,
        timeout=(10, 600)
    )

    response.raise_for_status()

    result = response.json()
    embedding = result["embeddings"][0]

    print("Embedding 成功！")
    print("向量维度：", len(embedding))
    print("耗时：", round(time.time() - start, 2), "秒")

except Exception as e:
    print("测试失败：", e)
