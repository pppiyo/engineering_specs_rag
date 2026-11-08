import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from app.config import (
    BASE_DIR,
    GB50060_DATA_DIR,
    GB50060_PAGE_IMAGE_DIR,
    GB50060_OCR_ORIGINAL_DIR,
    GB50060_COORDINATE_DIR,
    GB50060_PAGE_RECORD_DIR,
    GB50060_STATUS_FILE,
    GB50060_PDF_NAME,
    GB50060_OCR_SCALE,
    GB50060_OCR_LANG,
    GB50060_OCR_CONFIG,
)


VERSION = "1.0"


def sha256_file(path):
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def save_json(path, data):
    """原子写入 JSON，避免中途退出造成文件损坏。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def load_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def group_items_by_line(items):
    """按 Tesseract 的 block、paragraph、line 信息分组。"""
    groups = {}

    for item in items:
        text = item["text"].strip()

        if not text:
            continue

        key = (
            item["block_num"],
            item["par_num"],
            item["line_num"],
        )

        groups.setdefault(key, []).append(item)

    lines = []

    for line_items in groups.values():
        line_items.sort(key=lambda item: item["left"])

        x1 = min(item["left"] for item in line_items)
        y1 = min(item["top"] for item in line_items)
        x2 = max(
            item["left"] + item["width"]
            for item in line_items
        )
        y2 = max(
            item["top"] + item["height"]
            for item in line_items
        )

        lines.append({
            "text": " ".join(
                item["text"] for item in line_items
            ),
            "bbox": [x1, y1, x2, y2],
            "items": line_items,
        })

    lines.sort(key=lambda line: (
        line["bbox"][1],
        line["bbox"][0],
    ))

    return lines


def build_raw_text(lines):
    """由 OCR 行数据生成原始文本。"""
    return "\n".join(line["text"] for line in lines)


def load_status():
    return load_json(GB50060_STATUS_FILE, default={})


def save_status(status):
    save_json(GB50060_STATUS_FILE, status)


def get_page_paths(page_number):
    stem = f"page_{page_number:04d}"

    return {
        "image": GB50060_PAGE_IMAGE_DIR / f"{stem}.png",
        "original": GB50060_OCR_ORIGINAL_DIR / f"{stem}.json",
        "coordinates": GB50060_COORDINATE_DIR / f"{stem}.json",
        "record": GB50060_PAGE_RECORD_DIR / f"{stem}.json",
    }


def is_page_completed(status, page_number, pdf_hash):
    """只有文件完整且 PDF 和 OCR 参数一致时才跳过。"""
    page_status = status.get("pages", {}).get(
        str(page_number), {}
    )

    if page_status.get("status") != "success":
        return False

    if page_status.get("pdf_sha256") != pdf_hash:
        return False

    if page_status.get("ocr_scale") != GB50060_OCR_SCALE:
        return False

    if page_status.get("ocr_lang") != GB50060_OCR_LANG:
        return False

    if page_status.get("ocr_config") != GB50060_OCR_CONFIG:
        return False

    paths = get_page_paths(page_number)

    return all(path.exists() for path in paths.values())


def process_page(document, page_number, pdf_hash):
    """处理单页 PDF，保存图片、原始 OCR 和坐标。"""
    page = document.load_page(page_number - 1)

    matrix = pymupdf.Matrix(
        GB50060_OCR_SCALE,
        GB50060_OCR_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    paths = get_page_paths(page_number)
    paths["image"].parent.mkdir(parents=True, exist_ok=True)

    pixmap.save(str(paths["image"]))

    with Image.open(paths["image"]) as image:
        image_width, image_height = image.size

        data = pytesseract.image_to_data(
            image,
            lang=GB50060_OCR_LANG,
            config=GB50060_OCR_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

    items = []

    for index, text in enumerate(data["text"]):
        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence = -1.0

        items.append({
            "text": text,
            "confidence": confidence,
            "left": int(data["left"][index]),
            "top": int(data["top"][index]),
            "width": int(data["width"][index]),
            "height": int(data["height"][index]),
            "block_num": int(data["block_num"][index]),
            "par_num": int(data["par_num"][index]),
            "line_num": int(data["line_num"][index]),
        })

    lines = group_items_by_line(items)
    raw_text = build_raw_text(lines)

    relative_image = str(
        paths["image"].relative_to(BASE_DIR)
    )

    metadata = {
        "source": GB50060_PDF_NAME,
        "page": page_number,
        "pdf_sha256": pdf_hash,
        "page_image": relative_image,
        "image_width": image_width,
        "image_height": image_height,
        "coordinate_unit": "pixel",
        "coordinate_origin": "top_left",
        "coordinate_scale": GB50060_OCR_SCALE,
        "ocr_lang": GB50060_OCR_LANG,
        "ocr_config": GB50060_OCR_CONFIG,
        "version": VERSION,
    }

    original_data = {
        **metadata,
        "raw_text": raw_text,
        "items": items,
    }

    coordinate_data = {
        **metadata,
        "items": items,
        "lines": lines,
    }

    page_record = {
        **metadata,
        "page_type": "unclassified",
        "raw_text": raw_text,
        "items": items,
        "lines": lines,
        "status": "success",
        "processed_at": datetime.now().isoformat(),
    }

    save_json(paths["original"], original_data)
    save_json(paths["coordinates"], coordinate_data)
    save_json(paths["record"], page_record)

    return {
        "page": page_number,
        "image": relative_image,
        "image_width": image_width,
        "image_height": image_height,
        "item_count": len(items),
        "line_count": len(lines),
        "text_length": len(raw_text),
    }


def run_ocr(page_number=None, force=False):
    """运行 OCR。指定页码时只处理该页。"""
    pdf_path = BASE_DIR / "data" / "pdfs" / GB50060_PDF_NAME

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"找不到 PDF 文件：{pdf_path}"
        )

    GB50060_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_hash = sha256_file(pdf_path)
    status = load_status()

    status["source"] = GB50060_PDF_NAME
    status["pdf_sha256"] = pdf_hash
    status["ocr_scale"] = GB50060_OCR_SCALE
    status["ocr_lang"] = GB50060_OCR_LANG
    status["ocr_config"] = GB50060_OCR_CONFIG
    status["version"] = VERSION

    with pymupdf.open(pdf_path) as document:
        total_pages = len(document)

        if page_number is not None:
            if page_number < 1 or page_number > total_pages:
                raise ValueError(
                    f"页码必须在 1 到 {total_pages} 之间"
                )

            pages = [page_number]
        else:
            pages = list(range(1, total_pages + 1))

        status.setdefault("pages", {})

        print(f"PDF：{GB50060_PDF_NAME}")
        print(f"总页数：{total_pages}")
        print(f"本次处理：{pages}")
        print(f"OCR 参数：{GB50060_OCR_SCALE}x / "
              f"{GB50060_OCR_LANG} / {GB50060_OCR_CONFIG}")

        for number in pages:
            if not force and is_page_completed(
                status, number, pdf_hash
            ):
                print(f"第 {number} 页：已完成，跳过")
                continue

            print(f"\n正在处理第 {number} 页……")

            try:
                result = process_page(
                    document,
                    number,
                    pdf_hash,
                )

                status["pages"][str(number)] = {
                    "status": "success",
                    "pdf_sha256": pdf_hash,
                    "ocr_scale": GB50060_OCR_SCALE,
                    "ocr_lang": GB50060_OCR_LANG,
                    "ocr_config": GB50060_OCR_CONFIG,
                    "record": str(
                        get_page_paths(number)["record"].relative_to(
                            BASE_DIR
                        )
                    ),
                    "processed_at": datetime.now().isoformat(),
                }

                save_status(status)

                print("处理成功：")
                print(json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ))

            except Exception as error:
                status["pages"][str(number)] = {
                    "status": "error",
                    "error": str(error),
                    "pdf_sha256": pdf_hash,
                    "processed_at": datetime.now().isoformat(),
                }

                save_status(status)
                raise

    print("\nOCR 任务结束。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="只处理指定页码",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理",
    )

    args = parser.parse_args()

    run_ocr(
        page_number=args.page,
        force=args.force,
    )
