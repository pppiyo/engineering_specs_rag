from pathlib import Path

import streamlit as st
import requests
from PIL import Image, ImageDraw


# =========================
# 基础配置
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

API_URL = "http://127.0.0.1:8000/ask"


# =========================
# 页面设置
# =========================

st.set_page_config(
    page_title="电气设计规范知识库",
    page_icon="📚",
    layout="wide"
)


# =========================
# 页面样式
# =========================

st.markdown(
    """
    <style>
    /* 页面背景 */
    .stApp {
        background-color: #F8F9FB;
    }

    /* 查询按钮 */
    div.stButton > button {
        background-color: #526579;
        color: #FFFFFF;
        border: 1px solid #526579;
        border-radius: 6px;
        padding: 8px 24px;
        font-size: 15px;
        font-weight: 500;
    }

    /* 鼠标悬停 */
    div.stButton > button:hover {
        background-color: #405266;
        color: #FFFFFF;
        border-color: #405266;
    }

    /* 输入框 */
    .stTextArea textarea {
        border: 1px solid #D5DAE1;
        border-radius: 6px;
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 页面标题
# =========================

st.markdown(
    """
    <h2 style="
        font-size: 24px;
        font-weight: 600;
        color: #344256;
        margin-bottom: 4px;
    ">
        电气设计规范知识库
    </h2>
    """,
    unsafe_allow_html=True
)

st.caption("基于本地 PDF、OCR、Chroma 和 Ollama 的 RAG 知识库")

st.markdown(
    """
    <hr style="
        margin-top: 8px;
        margin-bottom: 12px;
        border: none;
        border-top: 1px solid #D5DAE1;
    ">
    """,
    unsafe_allow_html=True
)


# =========================
# 图片高亮函数
# =========================

def draw_highlights(image_path, line_boxes):
    """
    在 PDF 页面图片上逐行绘制高亮框。

    line_boxes 坐标采用原始图片像素坐标，
    不需要除以 coordinate_scale。
    """

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGBA")

    # 创建透明图层
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for box in line_boxes or []:

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(value) for value in box]
        except (TypeError, ValueError):
            continue

        # 防止坐标超出图片范围
        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))
        x2 = max(0, min(x2, image.width))
        y2 = max(0, min(y2, image.height))

        if x2 <= x1 or y2 <= y1:
            continue

        # 半透明黄色填充
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(255, 230, 0, 65),
            outline=(255, 140, 0, 230),
            width=3
        )

    # 合并透明图层
    result = Image.alpha_composite(image, overlay)

    return result.convert("RGB")


# =========================
# 显示来源和页面定位
# =========================

def display_sources(sources):

    if not sources:
        st.info("本次检索没有返回来源信息。")
        return

    st.subheader("检索来源与原文定位")

    for index, source in enumerate(sources, start=1):

        filename = source.get("source") or "未知文件"
        page = source.get("page")
        page_image = source.get("page_image")
        line_boxes = source.get("line_boxes") or []

        page_text = f"第 {page} 页" if page is not None else "页码未知"

        with st.expander(
            f"{index}. {filename} · {page_text}",
            expanded=bool(page_image)
        ):

            st.write(f"**文件名称：** {filename}")
            st.write(f"**页码：** {page_text}")

            if not page_image:
                st.caption("该来源暂无页面图片定位信息。")
                continue

            highlighted_image = draw_highlights(
                page_image,
                line_boxes
            )

            if highlighted_image is None:
                st.warning(
                    f"找不到页面图片：{page_image}"
                )
                continue

            st.image(
                highlighted_image,
                caption=f"{filename} · {page_text} · OCR 行级定位",
                use_container_width=True
            )

            st.caption(
                f"已加载 {len(line_boxes)} 个行级定位框"
            )


# =========================
# 问题输入
# =========================

question = st.text_area(
    "请输入你的问题",
    placeholder="例如：电气设备的绝缘耐受水平应符合什么要求？",
    height=120
)


# =========================
# 查询
# =========================

if st.button("查询", type="primary"):

    if not question.strip():

        st.warning("请输入问题后再查询。")

    else:

        with st.spinner("正在检索知识库并生成回答..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=600
                )

                response.raise_for_status()

                result = response.json()

                # 显示回答
                st.subheader("回答结果")

                st.text(result.get("answer", ""))

                # 显示来源和图片定位
                display_sources(
                    result.get("sources", [])
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "无法连接 FastAPI，请先启动后端服务。"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "请求超时，请检查 Ollama 是否正常运行。"
                )

            except requests.exceptions.HTTPError as e:

                st.error(f"接口请求失败：{e}")

                try:
                    st.json(response.json())
                except Exception:
                    pass

            except Exception as e:

                st.error(f"发生错误：{e}")