import json
from datetime import datetime
from pathlib import Path

import pytesseract
from PIL import Image

from app.config import (
    BASE_DIR,
    GB50060_DATA_DIR,
    GB50060_OCR_CORRECTED_DIR,
    GB50060_OCR_LANG,
    GB50060_OCR_CONFIG,
    GB50060_PDF_NAME,
)

VERSION = "1.0"

SOURCE_IMAGE_DIR = (
    GB50060_DATA_DIR / "page_images_corrected"
)

COORDINATE_DIR = (
    GB50060_OCR_CORRECTED_DIR / "coordinates"
)

PAGE_RECORD_DIR = (
    GB50060_OCR_CORRECTED_DIR / "page_records"
)


def save_json(path, data):
    """原子写入 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def load_json(path):
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


def process_page(page_number):
    """对旋转校正后的图片执行 OCR。"""

    stem = f"page_{page_number:04d}"

    image_path = SOURCE_IMAGE_DIR / f"{stem}.png"

    original_record_path = (
        GB50060_DATA_DIR
        / "page_records"
        / f"{stem}.json"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"找不到校正图片：{image_path}"
        )

    if not original_record_path.exists():
        raise FileNotFoundError(
            f"找不到原始页面记录：{original_record_path}"
        )

    original_record = load_json(original_record_path)

    with Image.open(image_path) as image:
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

    raw_text = "\n".join(
        line["text"] for line in lines
    )

    metadata = {
        "source": GB50060_PDF_NAME,
        "page": page_number,
        "page_type": original_record.get(
            "page_type", "unclassified"
        ),
        "page_image": str(
            image_path.relative_to(BASE_DIR)
        ),
        "image_width": image_width,
        "image_height": image_height,
        "rotation_correction": "clockwise_90_degrees",
        "coordinate_unit": "pixel",
        "coordinate_origin": "top_left",
        "coordinate_scale": 2,
        "coordinate_reference": "corrected_image",
        "ocr_lang": GB50060_OCR_LANG,
        "ocr_config": GB50060_OCR_CONFIG,
        "version": VERSION,
        "processed_at": datetime.now().isoformat(),
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
        "raw_text": raw_text,
        "items": items,
        "lines": lines,
        "status": "success",
    }

    save_json(
        GB50060_OCR_CORRECTED_DIR / f"{stem}.json",
        original_data,
    )

    save_json(
        COORDINATE_DIR / f"{stem}.json",
        coordinate_data,
    )

    save_json(
        PAGE_RECORD_DIR / f"{stem}.json",
        page_record,
    )

    return {
        "page": page_number,
        "page_type": metadata["page_type"],
        "image_size": f"{image_width}x{image_height}",
        "item_count": len(items),
        "line_count": len(lines),
        "text_length": len(raw_text),
        "output": str(
            (GB50060_OCR_CORRECTED_DIR / f"{stem}.json")
            .relative_to(BASE_DIR)
        ),
    }


def main():
    GB50060_OCR_CORRECTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for page_number in [15, 17]:
        print(f"\n正在处理第 {page_number} 页……")

        result = process_page(page_number)

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ))

    print("\n校正页面 OCR 全部完成。")


if __name__ == "__main__":
    main()
