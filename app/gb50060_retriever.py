import json
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

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

PAGE_INDEX_PATH = GB50060_DATA_DIR / "page_index.json"


# =========================
# 获取独立 Collection
# =========================

def get_gb50060_collection():
    """只获取 GB50060 独立知识库。"""

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    return client.get_collection(name=GB50060_COLLECTION)


# =========================
# 加载页面索引
# =========================

def load_page_index():
    if not PAGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"找不到页面索引：{PAGE_INDEX_PATH}"
        )

    with open(PAGE_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 路径转换
# =========================

def resolve_project_path(path):
    """将相对路径转换为项目绝对路径。"""

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
    """将 OCR 文字块合并为行级 bbox。"""

    if not ocr_path or not ocr_path.exists():
        return []

    with open(ocr_path, "r", encoding="utf-8") as f:
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
        line_boxes.append([
            min(item["left"] for item in items),
            min(item["top"] for item in items),
            max(item["right"] for item in items),
            max(item["bottom"] for item in items),
        ])

    return line_boxes


# =========================
# 条文编号辅助函数
# =========================

CLAUSE_PATTERN = re.compile(
    r"(?<![\d.])"
    r"(\d+)\s*\.\s*(\d+)\s*\.\s*(\d+)"
    r"(?![\d.])"
)

REFERENCE_SUFFIX_PATTERN = re.compile(
    r"(的规定|规定|的要求|要求|所列|见|参见|按|参照)$"
)


def normalize_clause_number(match):
    """将 3 . 0 . 1 统一为 3.0.1。"""

    return ".".join(match.groups())


def is_reference_number(document, match):
    """
    判断编号是否属于引用，而不是条文起点。

    例如：
    应符合本规范附录 B 的规定。
    """

    prefix = document[:match.start()].rstrip()

    prefix = re.sub(r"\s+", "", prefix)

    return bool(REFERENCE_SUFFIX_PATTERN.search(prefix))


def extract_clause_text(document, clause_number):
    """
    从原始 chunk 中提取指定条文。

    规则：
    1. 支持 3.0.1 和 3 . 0 . 1。
    2. 排除引用中的条文编号。
    3. 从目标编号开始提取。
    4. 遇到后续有效条文编号时停止。
    5. 不修改原始 document。
    """

    if not document or not clause_number:
        return document

    matches = list(CLAUSE_PATTERN.finditer(document))

    target_match = None
    target_index = None

    for index, match in enumerate(matches):

        if normalize_clause_number(match) != clause_number:
            continue

        if is_reference_number(document, match):
            continue

        target_match = match
        target_index = index
        break

    if target_match is None:
        return document

    start = target_match.start()
    end = len(document)

    for match in matches[target_index + 1:]:

        if is_reference_number(document, match):
            continue

        # 编号前必须存在空白，避免误把普通数字内容当作条文边界
        if match.start() > 0:
            previous_char = document[match.start() - 1]

            if not previous_char.isspace():
                continue

        end = match.start()
        break

    result = document[start:end].strip()

    return result if result else document


# =========================
# OCR 条文高亮
# =========================

def get_article_line_boxes(ocr_data, document, question=None):
    """
    根据问题中的条文编号定位 OCR 行。

    如果没有明确编号，则选择检索文本中的第一条规范。
    """

    question_numbers = CLAUSE_PATTERN.findall(question or "")
    document_numbers = CLAUSE_PATTERN.findall(document or "")

    if question_numbers:
        target_number = ".".join(question_numbers[0])
    elif document_numbers:
        target_number = ".".join(document_numbers[0])
    else:
        return None

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

    for items in lines.values():
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

    anchor_pattern = re.compile(
        r"^\s*(\d+\.\d+\.\d+)(?!\d)"
    )

    anchors = []

    for index, line in enumerate(page_lines):
        match = anchor_pattern.search(line["text"])

        if match:
            anchors.append({
                "index": index,
                "number": match.group(1),
            })

    target_anchor = next(
        (
            anchor
            for anchor in anchors
            if anchor["number"] == target_number
        ),
        None,
    )

    if target_anchor is None:
        return None

    start = target_anchor["index"]
    end = len(page_lines)

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
    question=None,
):
    """根据 metadata 获取页面图片和 OCR 高亮坐标。"""

    page_number = metadata.get("page")

    page_record = next(
        (
            item
            for item in page_index.get("pages", [])
            if item.get("pdf_page") == page_number
        ),
        None,
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

    line_boxes = get_article_line_boxes(
        ocr_data,
        document,
        question,
    )

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
# 中文分词
# =========================

def tokenize_chinese(text):
    """中文使用字符二元组，同时保留英文和数字词。"""

    text = (text or "").lower()
    tokens = []

    tokens.extend(
        re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text)
    )

    chinese_parts = re.findall(
        r"[\u4e00-\u9fff]+",
        text,
    )

    for part in chinese_parts:
        if len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(
                part[i:i + 2]
                for i in range(len(part) - 1)
            )

    return tokens


# =========================
# BM25 检索
# =========================

def lexical_search(question, documents, metadatas, limit=15):
    """使用 BM25 和关键词覆盖率进行检索。"""

    valid_records = [
        (doc, meta)
        for doc, meta in zip(documents, metadatas)
        if doc and meta
    ]

    if not valid_records:
        return []

    corpus = [
        tokenize_chinese(doc)
        for doc, _ in valid_records
    ]

    query_tokens = tokenize_chinese(question)
    unique_query_tokens = set(query_tokens)

    if not unique_query_tokens:
        return []

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    max_bm25 = max(bm25_scores) if len(bm25_scores) else 0

    ranked = []

    for index, (document, metadata) in enumerate(valid_records):

        doc_tokens = set(corpus[index])

        matched_tokens = unique_query_tokens & doc_tokens

        coverage = (
            len(matched_tokens) / len(unique_query_tokens)
            if unique_query_tokens else 0
        )

        bm25_score = (
            bm25_scores[index] / max_bm25
            if max_bm25 > 0 else 0
        )

        final_score = (
            coverage * 0.60
            + bm25_score * 0.40
        )

        ranked.append({
            "document": document,
            "metadata": metadata,
            "distance": None,
            "lexical_score": final_score,
            "coverage": coverage,
        })

    ranked.sort(
        key=lambda item: item["lexical_score"],
        reverse=True,
    )

    return ranked[:limit]


# =========================
# 主检索函数
# =========================

def search_gb50060_documents(question, top_k=3):
    """
    GB50060 混合检索。

    明确条文编号：
        精确提取对应条文，不调用 embedding。

    自然语言：
        BM25 + BGE-M3 + RRF。

    最终返回检索文本和 PDF 来源。
    """

    question = (question or "").strip()

    if not question:
        return []

    collection = get_gb50060_collection()

    total_count = collection.count()

    if total_count == 0:
        return []

    page_index = load_page_index()

    all_records = collection.get(
        include=["documents", "metadatas"]
    )

    all_documents = all_records.get("documents") or []
    all_metadatas = all_records.get("metadatas") or []

    clause_match = re.search(
        r"(?<![\d.])(\d+\.\d+\.\d+)(?![\d.])",
        question,
    )

    clause_number = (
        clause_match.group(1)
        if clause_match
        else None
    )

    candidate_results = []

    # =====================================
    # 路径一：明确条文编号
    # =====================================

    if clause_number:

        pattern = re.compile(
            r"(?<![\d.])"
            + re.escape(clause_number)
            + r"(?![\d.])"
        )

        for document, metadata in zip(
            all_documents,
            all_metadatas,
        ):

            if not document or not metadata:
                continue

            if metadata.get("page_type") != "main_text":
                continue

            match = pattern.search(document)

            if not match:
                continue

            prefix = re.sub(
                r"\s+",
                "",
                document[:match.start()].rstrip(),
            )

            if REFERENCE_SUFFIX_PATTERN.search(prefix):
                continue

            if not document[match.end():].strip():
                continue

            # 只提取目标条文用于前端显示和模型上下文
            clause_text = extract_clause_text(
                document,
                clause_number,
            )

            candidate_results.append({
                "document": clause_text,
                "original_document": document,
                "metadata": metadata,
                "distance": None,
            })

        if not candidate_results:
            print("未找到对应条文：", clause_number)
            return []

        print("检索模式：条文编号精确检索")
        print("条文编号：", clause_number)

    # =====================================
    # 路径二：BM25 + BGE-M3 + RRF
    # =====================================

    else:

        main_documents = []
        main_metadatas = []

        for document, metadata in zip(
            all_documents,
            all_metadatas,
        ):

            if not document or not metadata:
                continue

            if metadata.get("page_type") != "main_text":
                continue

            main_documents.append(document)
            main_metadatas.append(metadata)

        if not main_documents:
            print("没有找到规范正文")
            return []

        # BM25
        lexical_results = lexical_search(
            question,
            main_documents,
            main_metadatas,
            limit=15,
        )

        print("BM25检索数量：", len(lexical_results))

        print("\n========== BM25 前10名 ==========")

        for rank, item in enumerate(lexical_results[:10], start=1):
            print(
                rank,
                "页码：", item["metadata"].get("page"),
                "分数：", round(item["lexical_score"], 4),
                "覆盖率：", round(item["coverage"], 4),
                "内容：", item["document"][:100],
            )



        # BGE-M3
        query_embedding = get_embeddings([question])[0]

        if len(query_embedding) != 1024:
            raise ValueError(
                f"查询向量维度错误：{len(query_embedding)}，预期1024"
            )

        semantic_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(15, total_count),
            where={"page_type": "main_text"},
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        semantic_documents = (
            semantic_results.get("documents") or [[]]
        )[0]

        semantic_metadatas = (
            semantic_results.get("metadatas") or [[]]
        )[0]

        semantic_distances = (
            semantic_results.get("distances") or [[]]
        )[0]

        semantic_items = []

        for document, metadata, distance in zip(
            semantic_documents,
            semantic_metadatas,
            semantic_distances,
        ):

            if not document or not metadata:
                continue

            semantic_items.append({
                "document": document,
                "metadata": metadata,
                "distance": distance,
            })

        print("BGE-M3检索数量：", len(semantic_items))


        print("\n========== BGE-M3 前10名 ==========")

        for rank, item in enumerate(semantic_items[:10], start=1):
            print(
                rank,
                "页码：", item["metadata"].get("page"),
                "距离：", round(item["distance"], 4)
                if item.get("distance") is not None else None,
                "内容：", item["document"][:100],
            )


        # RRF 融合
        rrf_k = 60
        fused = {}

        def get_result_key(item):
            metadata = item["metadata"]

            page = metadata.get("page")
            chunk = metadata.get("chunk")

            if page is not None and chunk is not None:
                return page, chunk

            return page, item["document"]

        for rank, item in enumerate(lexical_results, start=1):

            key = get_result_key(item)

            if key not in fused:
                fused[key] = {
                    **item,
                    "rrf_score": 0.0,
                }

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)

        for rank, item in enumerate(semantic_items, start=1):

            key = get_result_key(item)

            if key not in fused:
                fused[key] = {
                    **item,
                    "rrf_score": 0.0,
                }

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)

            if item.get("distance") is not None:
                fused[key]["distance"] = item["distance"]

        ranked_results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        candidate_results = ranked_results[:top_k]

        print("检索模式：BM25 + BGE-M3 + RRF")

        for rank, item in enumerate(
            ranked_results[:5],
            start=1,
        ):
            print(
                "融合排名：", rank,
                "页码：", item["metadata"].get("page"),
                "条文内容：", item["document"][:150],
                "RRF分数：", round(item["rrf_score"], 6),
            )

    # =====================================
    # 构建最终结果及 PDF 来源
    # =====================================

    output = []
    seen = set()

    for item in candidate_results:

        document = item["document"]
        metadata = item["metadata"]

        key = (
            metadata.get("page"),
            metadata.get("chunk"),
        )

        if key in seen:
            continue

        seen.add(key)

        # 高亮使用原始 chunk，避免提取后的文字影响定位
        source_document = item.get(
            "original_document",
            document,
        )

        source = build_page_source(
            metadata=metadata,
            page_index=page_index,
            document=source_document,
            question=question,
        )

        output.append({
            "document": document,
            "metadata": metadata,
            "distance": item.get("distance"),
            "source": source,
        })

        if len(output) >= top_k:
            break

    print("最终检索结果数量：", len(output))

    for item in output:
        print(
            "来源页码：",
            item["source"].get("page"),
            "图片：",
            item["source"].get("page_image"),
        )

    return output


# =========================
# 检索结果来源整理
# =========================

def build_gb50060_source_list(results):
    """按 PDF 页码合并来源，避免同一页重复展示。"""

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