import re
import json
import requests
import chromadb

from app.config import (
    OLLAMA_BASE_URL,
    EMBED_MODEL,
    CHROMA_PATH,
    TOP_K,
)


def get_collection():
    """连接 ChromaDB 知识库。"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="design_documents",
        metadata={"hnsw:space": "cosine"}
    )


def get_query_embedding(question):
    """将问题转换为 BGE-M3 向量。"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": [question]
        },
        timeout=(10, 600)
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search_documents(question, top_k=TOP_K):
    """语义检索文本块，并返回关联的页面图片和坐标信息。"""
    collection = get_collection()

    count = collection.count()
    if count == 0:
        return []

    query_embedding = get_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        metadata = metadata or {}

        matches.append({
            "text": document or "",
            "source": metadata.get("source", "未知文件"),
            "page": metadata.get("page", "未知页码"),
            "chunk": metadata.get(
                "chunk",
                metadata.get("chunk_id", -1)
            ),
            "distance": distance,

            # 页面图片与 OCR 坐标
            "page_image": metadata.get("page_image"),
            "bbox": parse_json_metadata(
                metadata.get("bbox")
            ),
            "line_boxes": parse_json_metadata(
                metadata.get("line_boxes"),
                default=[]
            ),
            "coordinate_unit": metadata.get(
                "coordinate_unit"
            ),
            "coordinate_scale": metadata.get(
                "coordinate_scale"
            )
        })

    return matches


# image process module
def parse_json_metadata(value, default=None):
    """安全解析 ChromaDB 中以 JSON 字符串保存的 metadata。"""
    if value is None:
        return default

    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    return default



def safe_number(value):
    """将页码、chunk 编号安全转换为数字。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# added image process module
def get_all_records(source=None):
    """读取知识库文本块及页面定位信息，并按文件、页码、chunk 排序。"""
    collection = get_collection()

    data = collection.get(
        include=["documents", "metadatas"]
    )

    records = []

    for document, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):
        metadata = metadata or {}

        current_source = metadata.get(
            "source", "未知文件"
        )

        if source and current_source != source:
            continue

        records.append({
            "text": document or "",
            "source": current_source,
            "page": metadata.get("page", 0),
            "chunk": metadata.get(
                "chunk",
                metadata.get("chunk_id", -1)
            ),

            # 页面图片与 OCR 坐标
            "page_image": metadata.get("page_image"),
            "bbox": parse_json_metadata(
                metadata.get("bbox")
            ),
            "line_boxes": parse_json_metadata(
                metadata.get("line_boxes"),
                default=[]
            ),
            "coordinate_unit": metadata.get(
                "coordinate_unit"
            ),
            "coordinate_scale": metadata.get(
                "coordinate_scale"
            )
        })

    records.sort(
        key=lambda item: (
            item["source"],
            safe_number(item["page"]),
            safe_number(item["chunk"])
        )
    )

    return records


def merge_chunk_text(blocks):
    """合并连续文本块，去除相邻文本块的重叠部分。"""
    merged = ""

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        if not merged:
            merged = block
            continue

        # 只检查当前文本块开头与已合并文本末尾的重叠
        max_overlap = min(len(merged), len(block), 300)
        overlap = 0

        for size in range(max_overlap, 0, -1):
            if merged[-size:] == block[:size]:
                overlap = size
                break

        if overlap:
            merged += block[overlap:]
        else:
            merged += "\n" + block

    return merged.strip()


def search_clause(clause, source=None):
    """
    按三级条文编号检索完整条文。
    支持跨页、跨 chunk 合并。
    每个页面分别保存图片和坐标定位信息。
    例如：4.1.7
    """

    if not re.fullmatch(r"\d+\.\d+\.\d+", clause):
        return []

    records = get_all_records(source=source)

    target_pattern = re.compile(
        rf"(?<![\d.]){re.escape(clause)}(?![\d.])"
    )

    parts = clause.split(".")
    prefix = ".".join(parts[:-1])

    sibling_pattern = re.compile(
        rf"(?<![\d.]){re.escape(prefix)}\.\d+(?![\d.])"
    )

    sources = list(dict.fromkeys(
        item["source"] for item in records
    ))

    matches = []

    for current_source in sources:
        source_records = [
            item for item in records
            if item["source"] == current_source
        ]

        collecting = False
        blocks = []
        pages = []
        locations = []

        for item in source_records:
            text = item["text"]

            if not collecting:
                target_match = target_pattern.search(text)

                if not target_match:
                    continue

                collecting = True
                text = text[target_match.start():]

            # 找到下一个同级条文编号时停止
            next_clause = None

            for sibling in sibling_pattern.finditer(text):
                if sibling.group() != clause:
                    next_clause = sibling
                    break

            if next_clause:
                text = text[:next_clause.start()]

            if text.strip():
                blocks.append(text.strip())
                pages.append(item["page"])

                # 按页保存定位信息，避免混合不同页面的坐标
                location = {
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "page_image": item.get("page_image"),
                    "bbox": item.get("bbox"),
                    "line_boxes": item.get("line_boxes") or [],
                    "coordinate_unit": item.get("coordinate_unit"),
                    "coordinate_scale": item.get("coordinate_scale"),
                }

                # 同一页面只保留一份定位记录
                location_key = (
                    location["source"],
                    location["page"],
                    location["page_image"],
                )

                existing_keys = {
                    (
                        loc["source"],
                        loc["page"],
                        loc["page_image"],
                    )
                    for loc in locations
                }

                if location_key not in existing_keys:
                    locations.append(location)

            if next_clause:
                break

        if blocks:
            result = {
                "text": merge_chunk_text(blocks),
                "source": current_source,
                "page": (
                    f"{min(pages, key=safe_number)}-"
                    f"{max(pages, key=safe_number)}"
                    if pages else "未知页码"
                ),
                "chunk": -1,
                "clause": clause,
                "locations": locations,
            }

            # 单页条文保留顶层定位字段，兼容现有 build_source_list()
            if len(locations) == 1:
                result.update({
                    "page_image": locations[0]["page_image"],
                    "bbox": locations[0]["bbox"],
                    "line_boxes": locations[0]["line_boxes"],
                    "coordinate_unit": locations[0]["coordinate_unit"],
                    "coordinate_scale": locations[0]["coordinate_scale"],
                })

            matches.append(result)

    return matches

def expand_clause_results(results):
    """
    优先保留语义检索结果。
    只扩展最相关的前几个文本块中的条文编号，
    避免将大量无关条文加入上下文。
    """

    if not results:
        return []

    clause_pattern = re.compile(
        r"(?<![\d.])(\d+\.\d+\.\d+)(?![\d.])"
    )

    expanded = []
    seen = set()

    # 先保留原始语义检索结果，避免丢失相关上下文
    for item in results:
        key = (
            item["source"],
            item["page"],
            item["text"]
        )

        if key not in seen:
            seen.add(key)
            expanded.append(item)

    # 只从最相关的前 2 个结果中识别条文编号
    clauses = []
    clause_seen = set()

    for item in results[:2]:
        for clause in clause_pattern.findall(item["text"]):
            key = (item["source"], clause)

            if key not in clause_seen:
                clause_seen.add(key)
                clauses.append(key)

    # 补齐识别到的条文
    for source, clause in clauses:
        clause_results = search_clause(
            clause,
            source=source
        )

        for item in clause_results:
            key = (
                item["source"],
                item.get("clause", ""),
                item["text"]
            )

            if key not in seen:
                seen.add(key)
                expanded.append(item)

    return expanded



def search_section(section, source=None):
    """
    按两级章节编号检索完整章节内容。
    例如：4.2

    先根据完整原始 chunk 去除切块重叠，
    再截取目标章节，避免章节起点截断后无法识别重叠。
    """

    if not re.fullmatch(r"\d+\.\d+", section):
        return []

    records = get_all_records(source=source)

    major_prefix, minor = section.split(".")
    current_number = int(minor)

    # 匹配真正的两级章节标题，允许编号后没有空格
    # (?![\d.]) 防止将 5.1.1 误识别为 5.1
    section_pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(section)}(?![\d.])[ \t]*[^\n]*"
    )

    # 匹配同一大章节下的其他两级标题
    end_pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(major_prefix)}\.(\d+)(?![\d.])[ \t]*[^\n]*"
    )

    sources = list(dict.fromkeys(
        item["source"] for item in records
    ))

    matches = []

    for current_source in sources:
        source_records = [
            item for item in records
            if item["source"] == current_source
        ]

        collecting = False
        blocks = []
        pages = []

        # 保存完整的上一个原始 chunk
        previous_raw_text = None

        for item in source_records:
            raw_text = item["text"].strip()
            text = raw_text

            if not text:
                previous_raw_text = raw_text
                continue

            # 先根据完整原始 chunk 去除重叠
            if previous_raw_text is not None:
                max_overlap = min(
                    len(previous_raw_text),
                    len(raw_text),
                    300
                )

                overlap = 0

                for size in range(max_overlap, 0, -1):
                    if previous_raw_text.endswith(raw_text[:size]):
                        overlap = size
                        break

                text = raw_text[overlap:]

            # 保存完整原始 chunk，不能保存截取后的 text
            previous_raw_text = raw_text

            if not collecting:
                match = section_pattern.search(text)

                if not match:
                    continue

                collecting = True
                text = text[match.start():]

            # 检查是否到达下一个同级章节
            end_match = end_pattern.search(text)
            stop = False

            if end_match:
                candidate_number = int(end_match.group(1))

                if candidate_number > current_number:
                    text = text[:end_match.start()]
                    stop = True

            if text.strip():
                blocks.append(text.strip())
                pages.append(item["page"])

            if stop:
                break

        if blocks:
            matches.append({
                "text": "\n".join(blocks),
                "source": current_source,
                "page": (
                    f"{min(pages, key=safe_number)}-"
                    f"{max(pages, key=safe_number)}"
                    if pages else "未知页码"
                ),
                "chunk": -1
            })

    return matches


if __name__ == "__main__":
    section = input(
        "请输入章节编号（例如 4.1），或条文编号（例如 4.1.7）："
    ).strip()

    if re.fullmatch(r"\d+\.\d+\.\d+", section):
        results = search_clause(section)
    else:
        results = search_section(section)

    print(f"\n找到 {len(results)} 条结果：")

    for i, item in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print("来源：", item["source"])
        print("页码：", item["page"])
        print(item["text"])
