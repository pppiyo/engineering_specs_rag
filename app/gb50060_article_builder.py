
import json
import re
from collections import defaultdict
from pathlib import Path

from app.gb50060_retriever import get_gb50060_collection
from app.corrections import apply_corrections


ARTICLE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(\d{1,2})\s*[.,，。]\s*"
    r"(\d{1,2})\s*[.,，。]\s*"
    r"(\d{1,2})"
    r"(?![\d])"
    r"(?!\s*[-－—]\s*\d)"
)

REFERENCE_SUFFIX_PATTERN = re.compile(
    r"^(的规定|规定|的要求|要求|所列|见|参见|按|参照)"
)

CHUNK_ID_PATTERN = re.compile(
    r"_p(\d+)_c(\d+)$"
)

OUTPUT_PATH = Path(
    "data/documents/GB50060_2008/article_records.json"
)

MIN_OVERLAP = 20


def normalize_article_number(match):
    return ".".join(match.groups())


def normalize_for_comparison(text):
    return re.sub(r"\s+", "", text)


def normalize_with_positions(text):
    """
    去除空白，同时记录每个字符在原文中的位置。
    用于重叠检测，不修改原始 OCR 文本。
    """
    chars = []
    positions = []

    for index, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            positions.append(index)

    return "".join(chars), positions


def find_overlap(previous_text, next_text):
    """
    检查前一个文本的结尾与下一个文本的开头是否重叠。

    返回：
    - 重叠字符数
    - 下一个文本在原始字符串中的截断位置
    """
    previous_normalized, _ = normalize_with_positions(
        previous_text
    )

    next_normalized, next_positions = normalize_with_positions(
        next_text
    )

    max_length = min(
        len(previous_normalized),
        len(next_normalized),
    )

    for length in range(max_length, MIN_OVERLAP - 1, -1):

        if (
            previous_normalized[-length:]
            == next_normalized[:length]
        ):
            if length >= len(next_positions):
                cut_position = len(next_text)
            else:
                cut_position = next_positions[length]

            return length, cut_position

    return 0, 0


def get_chunk_sort_key(item):
    """
    根据 Chunk ID 提取页码和 Chunk 序号。
    无法识别时使用 metadata 中的页码和原始索引。
    """
    match = CHUNK_ID_PATTERN.search(item["chunk_id"] or "")

    if match:
        return (
            int(match.group(1)),
            int(match.group(2)),
        )

    return (
        int(item["page"] or 0),
        item["original_index"],
    )


def merge_page_chunks(chunks):
    """
    将同一 PDF 页的多个 Chunk 按顺序重建为完整页面文本。
    """
    if not chunks:
        return "", {
            "source_chunk_ids": [],
            "merge_details": [],
        }

    chunks = sorted(
        chunks,
        key=get_chunk_sort_key,
    )

    merged_text = ""
    merge_details = []

    for index, chunk in enumerate(chunks):

        document = chunk["document"] or ""

        if index == 0:
            merged_text = document

            merge_details.append({
                "chunk_id": chunk["chunk_id"],
                "overlap_chars": 0,
                "action": "first_chunk",
            })

            continue

        overlap_length, cut_position = find_overlap(
            merged_text,
            document,
        )

        if overlap_length >= MIN_OVERLAP:

            remaining_text = document[cut_position:]
            merged_text += remaining_text
            action = "overlap_removed"

        else:
            merged_text += "\n" + document
            action = "appended_without_overlap"

        merge_details.append({
            "chunk_id": chunk["chunk_id"],
            "overlap_chars": overlap_length,
            "action": action,
        })

    source_chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    return merged_text, {
        "source_chunk_ids": source_chunk_ids,
        "merge_details": merge_details,
    }


def is_likely_article_start(document, match):
    """
    判断编号是否是真正的条文起点。
    """
    suffix = document[match.end():]
    normalized_suffix = re.sub(r"\s+", "", suffix)

    if REFERENCE_SUFFIX_PATTERN.match(normalized_suffix):
        return False

    return True


def get_article_segments(document):
    """
    识别条文起点并切分。

    返回：
    - segments：识别出的条文
    - prefix_text：第一个有效条文编号之前的文字

    prefix_text 用于处理上一页条文延续到当前页的情况。
    """
    all_matches = list(
        ARTICLE_PATTERN.finditer(document)
    )

    valid_matches = []
    seen_article_numbers = set()

    for match in all_matches:

        if not is_likely_article_start(document, match):
            continue

        article_no = normalize_article_number(match)

        if article_no in seen_article_numbers:
            continue

        seen_article_numbers.add(article_no)
        valid_matches.append(match)

    if not valid_matches:
        return [], document.strip()

    # 第一个条文编号之前的文字可能是上一页条文的续文。
    prefix_text = document[:valid_matches[0].start()].strip()

    segments = []

    for index, match in enumerate(valid_matches):

        start = match.start()

        if index + 1 < len(valid_matches):
            end = valid_matches[index + 1].start()
        else:
            end = len(document)

        text = document[start:end].strip()

        if not text:
            continue

        segments.append({
            "article_no": normalize_article_number(match),
            "original_number": match.group(0),
            "text": text,
            "normalized_text": normalize_for_comparison(text),
        })

    return segments, prefix_text


def classify_segment(text):
    length = len(text.strip())

    if length < 30:
        return "short_segment"

    if length < 60:
        return "needs_review"

    return "unreviewed"


def mark_duplicate_candidates(records):
    """
    对同页、同编号、完全相同的记录进行标记。
    不删除记录。
    """
    groups = defaultdict(list)

    for record in records:
        key = (
            str(record["page"]),
            record["article_no"],
        )

        groups[key].append(record)

    for group in groups.values():

        if len(group) < 2:
            continue

        normalized_groups = defaultdict(list)

        for record in group:
            normalized_groups[
                record["normalized_text"]
            ].append(record)

        for same_records in normalized_groups.values():

            if len(same_records) > 1:
                for record in same_records:
                    record["duplicate_candidate"] = True

    return records


def append_cross_page_prefix(
    previous_record,
    prefix_text,
    current_page,
    current_chunk_ids,
):
    """
    将当前页第一个条文编号之前的文字，
    追加到上一页最后一条条文记录。

    同时更新来源页码和来源 Chunk ID。
    """
    if not previous_record or not prefix_text:
        return False

    previous_record["text"] += "\n" + prefix_text

    previous_record["normalized_text"] = (
        normalize_for_comparison(
            previous_record["text"]
        )
    )

    previous_record["text_length"] = len(
        previous_record["text"]
    )

    previous_record["status"] = classify_segment(
        previous_record["text"]
    )

    # 保留原始条文起始页，另行记录实际涉及的所有页面。
    source_pages = previous_record.setdefault(
        "source_pages",
        [previous_record["page"]],
    )

    if current_page not in source_pages:
        source_pages.append(current_page)

    source_chunk_ids = previous_record.setdefault(
        "source_chunk_ids",
        [],
    )

    for chunk_id in current_chunk_ids:
        if chunk_id not in source_chunk_ids:
            source_chunk_ids.append(chunk_id)

    continuation_details = previous_record.setdefault(
        "cross_page_continuations",
        [],
    )

    continuation_details.append({
        "continued_from_page": current_page,
        "text": prefix_text,
        "source_chunk_ids": current_chunk_ids,
    })

    return True


def build_article_records():

    print("正在读取 gb50060_2008 collection...")

    collection = get_gb50060_collection()

    result = collection.get(
        include=["documents", "metadatas"]
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []

    page_groups = defaultdict(list)

    main_text_count = 0

    # 第一步：按照 PDF 页码收集 Chunk。
    for index, (document, metadata) in enumerate(
        zip(documents, metadatas)
    ):

        metadata = metadata or {}

        if metadata.get("page_type") != "main_text":
            continue

        main_text_count += 1

        chunk_id = (
            ids[index]
            if index < len(ids)
            else None
        )

        page = metadata.get("page")

        if page is None:
            match = CHUNK_ID_PATTERN.search(chunk_id or "")
            if match:
                page = int(match.group(1))

        if page is None:
            print(f"警告：无法识别 Chunk 页码：{chunk_id}")
            continue

        page_groups[str(page)].append({
            "page": page,
            "chunk_id": chunk_id,
            "document": document,
            "metadata": metadata,
            "original_index": index,
        })

    records = []
    page_merge_reports = []
    cross_page_count = 0

    # 第二步：逐页重建、校正、识别条文和处理跨页续文。
    for page_key in sorted(
        page_groups,
        key=lambda value: int(value),
    ):

        chunks = page_groups[page_key]

        page_text, merge_info = merge_page_chunks(chunks)

        source = chunks[0]["metadata"].get("source", "")

        # 应用人工核对过的 OCR 修正规则。
        page_text = apply_corrections(
            page_text,
            source,
            int(page_key),
        )

        page_merge_reports.append({
            "page": int(page_key),
            **merge_info,
        })

        segments, prefix_text = get_article_segments(
            page_text
        )

        current_page = int(page_key)
        current_chunk_ids = merge_info["source_chunk_ids"]

        # 当前页第一个条文之前的文字，归入上一页最后一条条文。
        if prefix_text and records:

            previous_record = records[-1]

            attached = append_cross_page_prefix(
                previous_record,
                prefix_text,
                current_page,
                current_chunk_ids,
            )

            if attached:
                cross_page_count += 1

                print(
                    f"跨页续文：第{current_page}页前缀"
                    f"追加至条文 {previous_record['article_no']}"
                    f"（起始页：第{previous_record['page']}页）"
                )

        for segment_index, segment in enumerate(segments):

            text = segment["text"]

            record = {
                "record_id": (
                    f"page_{page_key}_"
                    f"article_{segment['article_no']}_"
                    f"segment_{segment_index}"
                ),

                "article_no": segment["article_no"],
                "original_number": segment["original_number"],

                # 条文起始页。
                "page": current_page,

                # 实际涉及的页面，跨页后会追加。
                "source_pages": [current_page],

                # 兼容原有字段。
                "source_chunk_id": (
                    current_chunk_ids[0]
                    if current_chunk_ids
                    else None
                ),

                "source_chunk_ids": list(current_chunk_ids),

                "segment_index": segment_index,

                "text": text,
                "normalized_text": segment["normalized_text"],
                "text_length": len(text),

                "duplicate_candidate": False,
                "status": classify_segment(text),

                "merge_details": merge_info["merge_details"],

                # 没有跨页时为空列表。
                "cross_page_continuations": [],
            }

            records.append(record)

    # 第三步：标记完全重复记录。
    mark_duplicate_candidates(records)

    output = {
        "collection": "gb50060_2008",
        "source_document_count": len(documents),
        "main_text_chunk_count": main_text_count,
        "page_count": len(page_groups),
        "article_segment_count": len(records),
        "cross_page_continuation_count": cross_page_count,
        "page_merge_reports": page_merge_reports,
        "records": records,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    duplicate_count = sum(
        record["duplicate_candidate"]
        for record in records
    )

    short_count = sum(
        record["status"] == "short_segment"
        for record in records
    )

    review_count = sum(
        record["status"] == "needs_review"
        for record in records
    )

    merged_count = sum(
        detail["action"] == "overlap_removed"
        for report in page_merge_reports
        for detail in report["merge_details"]
    )

    print("\n页面重建与条文整理完成")
    print("-" * 45)
    print(f"Collection 原始记录数：{len(documents)}")
    print(f"正文 Chunk 数：{main_text_count}")
    print(f"重建页面数：{len(page_groups)}")
    print(f"检测到重叠并去重的 Chunk 数：{merged_count}")
    print(f"跨页续文合并次数：{cross_page_count}")
    print(f"识别出的条文片段数：{len(records)}")
    print(f"重复候选片段数：{duplicate_count}")
    print(f"短片段数：{short_count}")
    print(f"需要复核片段数：{review_count}")
    print(f"\n输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    build_article_records()
