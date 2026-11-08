
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"

# 现有知识库配置（保持不变）
CHROMA_PATH = str(BASE_DIR / "data" / "chroma")
PDF_FOLDER = str(BASE_DIR / "data" / "pdfs")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 15


# =====================================
# GB 50060-2008 独立 OCR 配置
# =====================================

GB50060_DATA_DIR = (
    BASE_DIR / "data" / "documents" / "GB50060_2008"
)

# 页面图片
GB50060_PAGE_IMAGE_DIR = (
    GB50060_DATA_DIR / "page_images"
)

# 原始 OCR 数据
GB50060_OCR_ORIGINAL_DIR = (
    GB50060_DATA_DIR / "ocr_original"
)

# 人工校正文本
GB50060_OCR_CORRECTED_DIR = (
    GB50060_DATA_DIR / "ocr_corrected"
)

# OCR 坐标
GB50060_COORDINATE_DIR = (
    GB50060_DATA_DIR / "coordinates"
)

# 逐页完整记录
GB50060_PAGE_RECORD_DIR = (
    GB50060_DATA_DIR / "page_records"
)

# 逐页处理状态
GB50060_STATUS_FILE = (
    GB50060_DATA_DIR / "processing_status.json"
)

# 正式 PDF
GB50060_PDF_NAME = (
    "GB+50060-2008+3~110KV高压配电装置设计规范.pdf"
)

# 独立 Chroma collection
GB50060_COLLECTION = "gb50060_2008"

# OCR 参数
GB50060_OCR_SCALE = 2
GB50060_OCR_LANG = "chi_sim+eng"
GB50060_OCR_CONFIG = "--psm 3"
