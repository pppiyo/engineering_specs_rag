import json
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RECORD_DIR = BASE_DIR / "data" / "documents" / "GB50060_2008" / "page_records"
BACKUP_DIR = BASE_DIR / "data" / "documents" / "GB50060_2008" / "page_records_backup"

TOTAL_PAGES = 62

def classify_page(page):
    if page == 1:
        return "cover"
    if page == 2:
        return "publication_info"
    if page == 3:
        return "preface"
    if page == 4:
        return "table_of_contents"
    if 5 <= page <= 27:
        return "main_text"
    if 28 <= page <= 30:
        return "appendix"
    if page == 31:
        return "terminology"
    if page == 32:
        return "explanatory_cover"
    if page == 33:
        return "explanatory_contents"
    if 34 <= page <= 62:
        return "explanatory_text"
    raise ValueError(f"无效页码：{page}")

def main():
    if not RECORD_DIR.exists():
        raise FileNotFoundError(f"找不到页面记录目录：{RECORD_DIR}")

    records = [
        RECORD_DIR / f"page_{page:04d}.json"
        for page in range(1, TOTAL_PAGES + 1)
    ]

    missing = [path.name for path in records if not path.exists()]
    if missing:
        raise FileNotFoundError(f"页面记录不完整：{missing}")

    loaded = []
    for path in records:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "success":
            raise ValueError(f"页面状态异常：{path.name}")
        if data.get("page") != int(path.stem.split("_")[1]):
            raise ValueError(f"页码与文件名不一致：{path.name}")
        loaded.append((path, data))

    if BACKUP_DIR.exists():
        raise FileExistsError(
            f"备份目录已存在，为避免覆盖而停止：{BACKUP_DIR}"
        )

    shutil.copytree(RECORD_DIR, BACKUP_DIR)
    print(f"已备份原始页面记录：{BACKUP_DIR}")

    counts = {}

    for path, data in loaded:
        page = data["page"]
        page_type = classify_page(page)

        data["page_type"] = page_type
        data["classification"] = {
            "method": "manual_page_range_v1",
            "classified_at": datetime.now().isoformat(),
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        counts[page_type] = counts.get(page_type, 0) + 1

    print("\n========== 页面分类完成 ==========")
    for page_type, count in counts.items():
        print(f"{page_type}: {count}")

    print(f"\n总计：{sum(counts.values())} 页")
    print("原始 OCR 和坐标数据未修改。")

if __name__ == "__main__":
    main()
