import json
from app.corrections import apply_corrections
import requests
import pymupdf
import chromadb
import pytesseract

from PIL import Image
from pathlib import Path

from app.config import (
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    CHROMA_PATH,
    PDF_FOLDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# image process module
def save_pdf_pages(pdf_file):
    """
    将 PDF 每一页按 2 倍分辨率完整渲染并保存为 PNG。
    不裁剪、不旋转、不修改原 PDF。
    """
    output_dir = (
        Path("data/pages") / pdf_file.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []

    with pymupdf.open(pdf_file) as doc:
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )

            image_path = (
                output_dir / f"page_{page_num:04d}.png"
            )

            pix.save(str(image_path))
            saved_images.append(image_path)

            print(
                f"已保存第 {page_num} 页："
                f"{image_path} "
                f"（{pix.width} × {pix.height}）"
            )

    return saved_images


def ocr_page(page):
    """将 PDF 页面转换为图片，并识别中英文文字。"""
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    text = pytesseract.image_to_string(
        image,
        lang="chi_sim+eng",
        config="--psm 3"
    )

    return text



# image process module
def ocr_page_with_boxes(image_path):
    """
    识别页面图片中的文字，并记录每个文字框的坐标。
    坐标单位为图片像素。
    """
    image = Image.open(image_path)

    data = pytesseract.image_to_data(
        image,
        lang="chi_sim+eng",
        config="--psm 3",
        output_type=pytesseract.Output.DICT
    )

    results = []

    for i, text in enumerate(data["text"]):
        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        results.append({
            "text": text,
            "confidence": confidence,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
            "line_num": data["line_num"][i]
        })

    return results

# image process module
def save_ocr_boxes(results, image_path, output_path):
    """保存 OCR 文字、坐标及页面关联信息。"""
    data = {
        "image": str(image_path),
        "image_width": Image.open(image_path).width,
        "image_height": Image.open(image_path).height,
        "coordinate_unit": "pixel",
        "coordinate_origin": "top_left",
        "coordinate_scale": 2,
        "items": results
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"OCR 坐标数据已保存：{output_path}")


# image process module
def draw_ocr_boxes(image_path, results, output_path):
    """在页面图片上绘制 OCR 文字框。"""
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for item in results:
        left = item["left"]
        top = item["top"]
        right = left + item["width"]
        bottom = top + item["height"]

        draw.rectangle(
            [left, top, right, bottom],
            outline="red",
            width=2
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(output_path)
    print(f"坐标检查图已保存：{output_path}")


# image process record
def build_page_ocr_record(pdf_name, page_num, image_path, results):
    """建立页面、OCR 文字和坐标之间的关联。"""
    image = Image.open(image_path)

    return {
        "source": pdf_name,
        "page": page_num,
        "page_image": str(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "coordinate_unit": "pixel",
        "coordinate_origin": "top_left",
        "coordinate_scale": 2,
        "items": results
    }

# image process module
def map_chunk_to_ocr_boxes(chunk, ocr_items):
    """
    根据文本块内容，匹配对应的 OCR 文字框。
    返回匹配文字框的坐标列表。
    """
    def normalize(text):
        return "".join(text.split())

    normalized_chunk = normalize(chunk)
    matched_boxes = []

    for item in ocr_items:
        word = normalize(item["text"])

        if not word:
            continue

        if word in normalized_chunk:
            matched_boxes.append({
                "text": item["text"],
                "bbox": [
                    item["left"],
                    item["top"],
                    item["left"] + item["width"],
                    item["top"] + item["height"]
                ],
                "confidence": item["confidence"]
            })

    return matched_boxes


# image process moodule
def group_ocr_items_by_line(items):
    """按 OCR 行组织文字，并计算每行的坐标范围。"""
    lines = {}

    for item in items:
        key = (
            item["block_num"],
            item["par_num"],
            item["line_num"]
        )

        lines.setdefault(key, []).append(item)

    results = []

    for key, words in lines.items():
        words = sorted(words, key=lambda x: x["left"])

        x0 = min(w["left"] for w in words)
        y0 = min(w["top"] for w in words)
        x1 = max(
            w["left"] + w["width"] for w in words
        )
        y1 = max(
            w["top"] + w["height"] for w in words
        )

        results.append({
            "text": "".join(w["text"] for w in words),
            "bbox": [x0, y0, x1, y1],
            "items": words
        })

    results.sort(
        key=lambda x: (x["bbox"][1], x["bbox"][0])
    )

    return results


# image process module
def build_ocr_chunks(lines, max_chars=800):
    """
    将 OCR 行组合成文本块，同时保留每行坐标。
    """
    chunks = []
    current_lines = []
    current_length = 0

    def save_chunk():
        if not current_lines:
            return

        x0 = min(line["bbox"][0] for line in current_lines)
        y0 = min(line["bbox"][1] for line in current_lines)
        x1 = max(line["bbox"][2] for line in current_lines)
        y1 = max(line["bbox"][3] for line in current_lines)

        chunks.append({
            "text": "\n".join(
                line["text"] for line in current_lines
            ),
            "bbox": [x0, y0, x1, y1],
            "line_boxes": [
                line["bbox"] for line in current_lines
            ]
        })

    for line in lines:
        text = line["text"]

        if current_lines and current_length + len(text) > max_chars:
            save_chunk()
            current_lines = []
            current_length = 0

        current_lines.append(line)
        current_length += len(text)

    save_chunk()

    return chunks


def split_text(text):
    text = "\n".join(
        line.strip() for line in text.splitlines()
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


def get_embeddings(texts):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=(10, 600)
    )

    response.raise_for_status()
    return response.json()["embeddings"]


def ingest_pdfs():
    pdf_dir = Path(PDF_FOLDER)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("没有找到 PDF，请先放入 data/pdfs/")
        return

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="design_documents",
        metadata={"hnsw:space": "cosine"}
    )

    for pdf_file in pdf_files:
        print(f"\n正在解析：{pdf_file.name}")

        documents = []
        metadatas = []
        ids = []
	    # 保存未经校对的原始文本
        raw_pages = {}

        with pymupdf.open(pdf_file) as doc:
            for page_num, page in enumerate(doc):
                # 先尝试直接提取文字
                text = page.get_text().strip()

                # 如果没有文字，则使用 OCR
                if len(text) < 20:
                    print(
                        f"第 {page_num + 1} 页："
                        "启动中文 OCR..."
                    )
                    text = ocr_page(page)

                # 保存未经校对的原始文本
                raw_pages[str(page_num + 1)] = text

                # 应用人工校对
                corrected_text = apply_corrections(
                    text,
                    pdf_file.name,
                    page_num + 1
                )

                chunks = split_text(corrected_text)


                for chunk_num, chunk in enumerate(chunks):
                    documents.append(chunk)

                    metadatas.append({
                        "source": pdf_file.name,
                        "page": page_num + 1,
                        "chunk": chunk_num
                    })

                    ids.append(
                        f"{pdf_file.name}_p{page_num+1}_c{chunk_num}"
                    )

        # 保存未经校对的原始文本
        raw_dir = Path("data/ocr_original")
        raw_dir.mkdir(parents=True, exist_ok=True)

        raw_file = raw_dir / f"{pdf_file.stem}.json"

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(
                raw_pages,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"原始文本已保存：{raw_file}")

        print(f"提取文本块：{len(documents)}")

        if not documents:
            print("没有提取到文本，跳过该 PDF")
            continue

        batch_size = 32

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = get_embeddings(batch)

            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=batch,
                metadatas=metadatas[i:i + batch_size],
                embeddings=embeddings
            )

            print(
                f"已入库："
                f"{min(i + batch_size, len(documents))}"
                f"/{len(documents)}"
            )

    print("\nPDF 向量入库完成！")
    print("ChromaDB 总记录数：", collection.count())



# image process module

def ingest_diagram_pdf():
    """
    将已经完成 OCR 和坐标整理的图像型 PDF 页面，
    单独写入正式知识库 design_documents。
    不重新 OCR，不影响其他 PDF。
    """
    pdf_name = "test_diagram.pdf"

    record_file = (
        Path("data/ocr_original")
        / "test_diagram_page_1.json"
    )

    if not record_file.exists():
        raise FileNotFoundError(
            f"找不到 OCR 坐标文件：{record_file}"
        )

    with open(record_file, "r", encoding="utf-8") as f:
        page_record = json.load(f)

    # 读取页面 OCR 数据
    ocr_items = page_record.get("items", [])

    if not ocr_items:
        print("没有 OCR 文字，停止入库。")
        return

    # 按行组织文字和坐标
    lines = group_ocr_items_by_line(ocr_items)

    # 组合成带坐标的文本块
    chunks = build_ocr_chunks(lines)

    if not chunks:
        print("没有生成文本块，停止入库。")
        return

    # 连接正式知识库
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="design_documents",
        metadata={"hnsw:space": "cosine"}
    )

    # 准备文本和 metadata
    documents = []
    metadatas = []
    ids = []

    for chunk_num, chunk in enumerate(chunks):
        documents.append(chunk["text"])

        metadatas.append({
            "source": pdf_name,
            "page": int(page_record["page"]),
            "chunk": chunk_num,

            "page_image": page_record["page_image"],
            "image_width": int(page_record["image_width"]),
            "image_height": int(page_record["image_height"]),

            "bbox": json.dumps(
                chunk["bbox"],
                ensure_ascii=False
            ),
            "line_boxes": json.dumps(
                chunk["line_boxes"],
                ensure_ascii=False
            ),

            "coordinate_unit": page_record["coordinate_unit"],
            "coordinate_origin": page_record["coordinate_origin"],
            "coordinate_scale": int(
                page_record["coordinate_scale"]
            )
        })

        ids.append(
            f"{pdf_name}_p{page_record['page']}_c{chunk_num}"
        )

    # 生成向量并写入正式知识库
    embeddings = get_embeddings(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\n图像 PDF 正式入库完成！")
    print("文件：", pdf_name)
    print("页面数：1")
    print("OCR 行数：", len(lines))
    print("文本块数：", len(documents))
    print("正式知识库总记录数：", collection.count())



if __name__ == "__main__":
    ingest_pdfs()
