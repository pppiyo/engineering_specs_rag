
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="电气设计规范知识库",
    page_icon="📚",
    layout="wide"
)

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

# st.title("电气设计规范智能问答")
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

# st.divider()
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

question = st.text_area(
    "请输入你的问题",
    placeholder="例如：电气设备的绝缘耐受水平应符合什么要求？",
    height=120
)

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

                st.subheader("回答结果")
                st.text(result["answer"])

            except requests.exceptions.ConnectionError:
                st.error("无法连接 FastAPI，请先启动后端服务。")

            except requests.exceptions.Timeout:
                st.error("请求超时，请检查 Ollama 是否正常运行。")

            except requests.exceptions.HTTPError as e:
                st.error(f"接口请求失败：{e}")
                try:
                    st.json(response.json())
                except Exception:
                    pass

            except Exception as e:
                st.error(f"发生错误：{e}")
