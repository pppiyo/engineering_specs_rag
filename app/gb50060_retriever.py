import json
import re
from collections import defaultdict
from pathlib import Path

import chromadb

from app.config import (
    CHROMA_PATH,
    GB50060_COLLECTION,
    GB50060_DATA_DIR,
)
from app.embeddings import get_embeddings


# =========================
# 基础路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGE_INDEX_PATH = (
    GB50060_DATA_DIR / "page_index.json"
)


# =========================
# 获取独立 Collection
# =========================

def get_gb50060_collection():
    """
    只获取 GB50060 独立知识库。
    不访问旧版 design_documents。
    """

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    return client.get_collection(
        name=GB50060_COLLECTION
    )


# =========================
# 加载页面索引
# =========================

def load_page_index():
    if not PAGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"找不到页面索引：{PAGE_INDEX_PATH}"
        )

    with open(
        PAGE_INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# =========================
# 路径转换
# =========================

def resolve_project_path(path):
    """
    将 page_index 中相对于项目根目录的路径
    转换为绝对路径。
    """

    if not path:
        return None

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# =========================
# OCR 坐标转换
# =========================

def get_page_line_boxes(ocr_path):
    """
    将 OCR 文字块按 block_num、par_num、
    line_num 合并成行级 bbox。

    返回格式：
    [
        [x1, y1, x2, y2],
        ...
    ]
    """

    if not ocr_path or not ocr_path.exists():
        return []

    with open(
        ocr_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    lines = defaultdict(list)

    for item in data.get("items", []):
        key = (
            item.get("block_num"),
            item.get("par_num"),
            item.get("line_num"),
        )

        try:
            left = int(item["left"])
            top = int(item["top"])
            width = int(item["width"])
            height = int(item["height"])
        except (KeyError, TypeError, ValueError):
            continue

        if width <= 0 or height <= 0:
            continue

        lines[key].append({
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
        })

    line_boxes = []

    for items in lines.values():
        x1 = min(item["left"] for item in items)
        y1 = min(item["top"] for item in items)
        x2 = max(item["right"] for item in items)
        y2 = max(item["bottom"] for item in items)

        line_boxes.append([
            x1,
            y1,
            x2,
            y2,
        ])

    return line_boxes



def get_article_line_boxes(ocr_data, document, question=None):
    """
    优先根据用户问题中的条文编号定位。
    如果问题没有编号，则选择命中文本中第一条规范。
    每次只高亮一条规范。
    """

    number_pattern = re.compile(
        r"(?<!\d)\d+\.\d+\.\d+(?!\d)"
    )

    # 优先从用户问题中提取条文编号
    question_numbers = number_pattern.findall(question or "")

    # 没有明确编号时，取命中文本中的第一条
    document_numbers = number_pattern.findall(document or "")

    if question_numbers:
        target_number = question_numbers[0]
    elif document_numbers:
        target_number = document_numbers[0]
    else:
        return None

    # 按 OCR 的 block、paragraph、line 分组
    lines = defaultdict(list)

    for item in ocr_data.get("items", []):
        key = (
            item.get("block_num"),
            item.get("par_num"),
            item.get("line_num"),
        )

        try:
            left = int(item["left"])
            top = int(item["top"])
            width = int(item["width"])
            height = int(item["height"])
        except (KeyError, TypeError, ValueError):
            continue

        if width <= 0 or height <= 0:
            continue

        lines[key].append(item)

    page_lines = []

    for key, items in lines.items():
        items.sort(key=lambda x: int(x["left"]))

        text = "".join(
            item.get("text", "")
            for item in items
        )

        bbox = [
            min(int(item["left"]) for item in items),
            min(int(item["top"]) for item in items),
            max(
                int(item["left"]) + int(item["width"])
                for item in items
            ),
            max(
                int(item["top"]) + int(item["height"])
                for item in items
            ),
        ]

        page_lines.append({
            "text": text,
            "top": bbox[1],
            "left": bbox[0],
            "bbox": bbox,
        })

    page_lines.sort(
        key=lambda x: (x["top"], x["left"])
    )

    # 查找页面中所有条文编号
    anchor_pattern = re.compile(
        r"^\s*(\d+\.\d+\.\d+)(?!\d)"
    )

    anchors = []

    for i, line in enumerate(page_lines):
        match = anchor_pattern.search(line["text"])

        if match:
            anchors.append({
                "index": i,
                "number": match.group(1),
            })

    # 定位目标条文
    target_anchor = next(
        (
            anchor
            for anchor in anchors
            if anchor["number"] == target_number
        ),
        None
    )

    if target_anchor is None:
        return None

    start = target_anchor["index"]
    end = len(page_lines)

    # 遇到下一条规范时停止
    for anchor in anchors:
        if anchor["index"] > start:
            end = anchor["index"]
            break

    return [
        line["bbox"]
        for line in page_lines[start:end]
    ] or None


# =========================
# 页面来源信息
# =========================

def build_page_source(
    metadata,
    page_index,
    document=None,
    question=None
):
    """
    根据 Chroma metadata 获取页面图片，
    并尝试生成条文级高亮坐标。
    """

    page_number = metadata.get("page")

    page_record = next(
        (
            item
            for item in page_index.get("pages", [])
            if item.get("pdf_page") == page_number
        ),
        None
    )

    if page_record is None:
        return {
            "source": metadata.get("source"),
            "page": page_number,
            "page_image": None,
            "bbox": None,
            "line_boxes": [],
            "coordinate_unit": None,
            "coordinate_scale": None,
        }

    image_path = resolve_project_path(
        page_record.get("display_image")
    )

    ocr_path = resolve_project_path(
        page_record.get("ocr_source")
    )

    ocr_data = {}

    if ocr_path and ocr_path.exists():
        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

    # 优先使用条文级高亮
    line_boxes = get_article_line_boxes(
        ocr_data,
        document,
        question
    )    

    # 无法可靠定位时，回退到整页高亮
    if line_boxes is None:
        line_boxes = get_page_line_boxes(ocr_path)

    return {
        "source": metadata.get("source"),
        "page": page_number,
        "page_image": str(image_path) if image_path else None,
        "bbox": None,
        "line_boxes": line_boxes,
        "coordinate_unit": ocr_data.get("coordinate_unit"),
        "coordinate_scale": ocr_data.get("coordinate_scale"),
    }


# =========================
# 语义检索
# =========================


def search_gb50060_documents(question, top_k=5):
    """
    GB50060 检索：
    1. 先执行 bge-m3 语义检索。
    2. 如果问题包含三级条文编号，优先查找正文中
       以该编号开头的条文。
    3. 语义检索结果作为补充。
    只访问 GB50060 独立 Collection。
    """

    question = question.strip()

    if not question:
        return []

    collection = get_gb50060_collection()
    total_count = collection.count()

    if total_count == 0:
        return []

    # 提取用户问题中的三级条文编号，例如 3.0.1
    clause_match = re.search(
        r"(?<![\d.])(\d+\.\d+\.\d+)(?![\d.])",
        question
    )
    clause_number = (
        clause_match.group(1)
        if clause_match
        else None
    )

    # 生成查询向量
    query_embedding = get_embeddings([question])[0]

    if len(query_embedding) != 1024:
        raise ValueError(
            "查询向量维度错误："
            f"当前为 {len(query_embedding)}，预期为1024"
        )

    # 原有语义检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    semantic_results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        semantic_results.append({
            "document": document,
            "metadata": metadata,
            "distance": distance,
        })

    # 如果问题包含明确条文编号，扫描整个 Collection
    exact_results = []

    if clause_number:
        all_records = collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        for document, metadata in zip(
            all_records.get("documents", []),
            all_records.get("metadatas", []),
        ):
            if not document:
                continue

            # 匹配独立条文编号，并检查其后是否有正文
            pattern = (
                r"(?<![\d.])"
                + re.escape(clause_number)
                + r"(?![\d.])"
            )

            match = re.search(pattern, document)

            if not match:
                continue

            # 避免仅因正文引用了该编号就认定为目标条文
            following_text = document[match.end():].lstrip()

            if not following_text:
                continue

            # 只优先考虑主文页面
            if metadata.get("page_type") != "main_text":
                continue

            exact_results.append({
                "document": document,
                "metadata": metadata,
                "distance": None,
            })


    # 有精确条文命中时，只返回精确结果；
    # 没有精确命中时，才使用语义检索结果。
    candidate_results = (
        exact_results if exact_results else semantic_results
    )

    combined_results = []
    seen = set()

    for item in candidate_results:
        metadata = item["metadata"]

        key = (
            metadata.get("page"),
            metadata.get("chunk"),
        )

        if key in seen:
            continue

        seen.add(key)
        combined_results.append(item)

        if len(combined_results) >= top_k:
            break

    # # 精确条文优先，语义结果补充，并按页面和 chunk 去重
    # combined_results = []
    # seen = set()

    # for item in exact_results + semantic_results:
        # metadata = item["metadata"]

        # key = (
        #     metadata.get("page"),
        #     metadata.get("chunk"),
        # )

        # if key in seen:
        #     continue

        # seen.add(key)
        # combined_results.append(item)

        # if len(combined_results) >= top_k:
        #     break

    # 统一生成页面图片和高亮来源
    page_index = load_page_index()

    output = []

    for item in combined_results:
        document = item["document"]
        metadata = item["metadata"]

        source = build_page_source(
            metadata,
            page_index,
            document,
            question,
        )

        output.append({
            "document": document,
            "metadata": metadata,
            "distance": item["distance"],
            "source": source,
        })

    return output


# =========================
# 检索结果来源整理
# =========================

def build_gb50060_source_list(results):
    """
    按 PDF 页码合并来源，避免同一页
    多个 chunk 重复展示。
    """

    sources = []
    seen = set()

    for result in results:
        source = result.get("source", {})

        key = (
            source.get("source"),
            source.get("page"),
        )

        if key in seen:
            continue

        seen.add(key)
        sources.append(source)

    return sources




