import requests
from app.config import OLLAMA_BASE_URL, EMBED_MODEL


def get_embeddings(texts):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"]


if __name__ == "__main__":
    result = get_embeddings(["这是一个 PDF RAG 测试"])
    print("Embedding 成功！")
    print("向量维度：", len(result[0]))
