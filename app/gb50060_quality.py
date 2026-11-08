import json
import re
from pathlib import Path
from datetime import datetime

from app.config import (
    GB50060_PAGE_RECORD_DIR,
    GB50060_OCR_ORIGINAL_DIR,
    GB50060_DATA_DIR,
)

TOTAL_PAGES = 62
REPORT_PATH = GB50060_DATA_DIR / "quality_report.json"

# 常见 OCR 异常字符，仅用于提示，不自动修改
SUSPICIOUS_PATTERNS = [
    r"[�]",
    r"[�]",
    r"[|\\]{2,}",
    r"[A-Za-z]{8,}",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_page(page_number):
    stem = f"page_{page_number:04d}"

    record_path = GB50060_PAGE_RECORD_DIR / f"{stem}.json"
    original_path = GB50060_OCR_ORIGINAL_DIR / f"{stem}.json"

    result = {
        "page": page_number,
        "record_exists": record_path.exists(),
        "original_exists": original_path.exists(),
        "issues": [],
    }

    if not record_path.exists() or not original_path.exists():
        result["issues"].append("文件缺失")
        return result

    record = load_json(record_path)
    original = load_json(original_path)

    text = original.get("raw_text", "")
    items = original.get("items", [])

    result["page_type"] = record.get("page_type", "unknown")
    result["text_length"] = len(text)
    result["line_count"] = len(record.get("lines", []))
    result["item_count"] = len(items)

    if not text.strip():
        result["issues"].append("空白页")

    elif len(text.strip()) < 20:
        result["issues"].append("文本过少")

    if not items:
        result["issues"].append("OCR没有识别到文字")

    confidences = [
        item.get("confidence", -1)
        for item in items
        if item.get("confidence", -1) >= 0
    ]

    if confidences:
        low_count = sum(c < 50 for c in confidences)
        result["low_confidence_count"] = low_count
        result["low_confidence_ratio"] = round(
            low_count / len(confidences), 4
        )
        result["average_confidence"] = round(
            sum(confidences) / len(confidences), 2
        )

        if low_count / len(confidences) >= 0.2:
            result["issues"].append("低置信度文字比例较高")
    else:
        result["issues"].append("缺少有效置信度数据")

    matched = []
    for pattern in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, text)
        matched.extend(matches)

    if matched:
        result["suspicious_characters"] = matched[:30]
        result["issues"].append("存在疑似异常字符")

    return result


def main():
    results = [
        check_page(page)
        for page in range(1, TOTAL_PAGES + 1)
    ]

    issue_pages = [
        item for item in results
        if item["issues"]
    ]

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_pages": TOTAL_PAGES,
        "checked_pages": len(results),
        "pages_with_issues": len(issue_pages),
        "pages": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("========== OCR 质量检测 ==========")
    print(f"检查页数：{len(results)}")
    print(f"发现问题页面：{len(issue_pages)}")

    for item in issue_pages:
        print(
            f"第{item['page']}页："
            + "；".join(item["issues"])
        )

    print(f"\n完整报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
