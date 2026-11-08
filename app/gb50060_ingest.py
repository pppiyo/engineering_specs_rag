
import json
from pathlib import Path

from app.config import (
    CHROMA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GB50060_DATA_DIR,
    GB50060_COLLECTION,
    GB50060_PDF_NAME,
)
from app.embeddings import get_embeddings


def load_page_index():
    """读取 GB50060 页面索引。"""
    index_path = GB50060_DATA_DIR / "page_index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"找不到页面索引：{index_path}"
        )

    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_text(text):
    """按配置切分文本，保留重叠内容。"""
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def prepare_documents():
    """读取页面 OCR，生成文本块及 metadata。"""
    index = load_page_index()

    document_id = index["document_id"]
    pages = index["pages"]

    documents = []
    metadatas = []
    ids = []
    page_stats = []

    for page_info in pages:
        page_num = int(page_info["pdf_page"])
        ocr_path = Path(page_info["ocr_source"])

        if not ocr_path.exists():
            raise FileNotFoundError(
                f"第 {page_num} 页 OCR 文件不存在：{ocr_path}"
            )

        with ocr_path.open("r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        raw_text = ocr_data.get("raw_text", "")

        if not isinstance(raw_text, str):
            raise TypeError(
                f"第 {page_num} 页 raw_text 不是字符串"
            )

        chunks = split_text(raw_text)

        for chunk_num, chunk in enumerate(chunks):
            documents.append(chunk)

            metadatas.append({
                "document_id": document_id,
                "source": GB50060_PDF_NAME,
                "page": page_num,
                "chunk": chunk_num,
                "page_type": page_info["page_type"],
            })

            ids.append(
                f"{document_id}_p{page_num:04d}_c{chunk_num:04d}"
            )

        page_stats.append({
            "page": page_num,
            "page_type": page_info["page_type"],
            "text_length": len(raw_text),
            "chunk_count": len(chunks),
            "ocr_source": str(ocr_path),
        })

    return documents, metadatas, ids, page_stats


def preview_ingest():
    """只读预览，不连接或写入 Chroma。"""
    documents, metadatas, ids, page_stats = (
        prepare_documents()
    )

    print("\n========== GB50060 入库预览 ==========")
    print("文档 ID：GB50060_2008")
    print("目标 collection：", GB50060_COLLECTION)
    print("页面数量：", len(page_stats))
    print("文本块总数：", len(documents))

    print("\n页面切块统计：")

    for item in page_stats:
        print(
            f"第 {item['page']:02d} 页 | "
            f"{item['page_type']:<20} | "
            f"文本 {item['text_length']:4d} 字 | "
            f"文本块 {item['chunk_count']}"
        )

    print("\n前 3 个文本块示例：")

    for i in range(min(3, len(documents))):
        print(f"\n--- 文本块 {i + 1} ---")
        print("ID：", ids[i])
        print("Metadata：")
        print(json.dumps(
            metadatas[i],
            ensure_ascii=False,
            indent=2
        ))
        print("文本：")
        print(documents[i][:300])

    print("\n预览完成：没有写入 Chroma。")


def ingest():
    """正式写入独立 GB50060 collection。"""
    import chromadb

    documents, metadatas, ids, _ = prepare_documents()

    if not documents:
        print("没有可入库的文本。")
        return

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=GB50060_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 32

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        embeddings = get_embeddings(batch)

        collection.upsert(
            ids=batch_ids,
            documents=batch,
            metadatas=batch_metadatas,
            embeddings=embeddings
        )

        print(
            f"已入库："
            f"{min(i + batch_size, len(documents))}"
            f"/{len(documents)}"
        )

    print("\nGB50060 入库完成！")
    print("Collection：", GB50060_COLLECTION)
    print("记录数：", collection.count())


if __name__ == "__main__":
    ingest()
