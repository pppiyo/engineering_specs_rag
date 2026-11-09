import sys
from pathlib import Path

# 将项目根目录加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.gb50060_retriever import search_gb50060_documents
from app.gb50060_rag import (
    extract_question_core,
    select_relevant_results,
    extract_relevant_clause,
)


TEST_QUESTIONS = [
    ("T01", "屋外配电装置中的电气设备和绝缘子需要采取哪些措施？"),
    ("T02", "3.0.1 条规定了什么？"),
    ("T03", "严重积雪覆冰地区的配电装置应采取哪些措施？"),
    ("T04", "隔离开关的破冰厚度有什么要求？"),
    ("T05", "配电装置中电气设备的网状遮栏高度有什么要求？"),
    ("T06", "屋外配电装置的安全净距有什么要求？"),
    ("T07", "配电装置室长度较大时，出口设置有什么要求？"),
    ("T08", "配电装置室是否需要设置事故通风装置？"),
    ("T09", "配电装置屋内通道有什么要求？"),
    ("T10", "屋外配电装置裸露带电部分上方和下方对线路有什么要求？"),
    ("T11", "特种耐热导体的最高工作温度如何选择？"),
    ("T12", "配电装置中的导体和电器对环境相对湿度有什么要求？"),
]


def get_page(metadata):
    return (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("pdf_page")
        or "未知"
    )


def run_test(test_id, question):
    print("\n" + "=" * 70)
    print(f"{test_id}：{question}")
    print("=" * 70)

    # 1. 原始检索
    results = search_gb50060_documents(question, top_k=3)

    print(f"\n原始检索结果数量：{len(results)}")

    if not results:
        print("未检索到结果")
        return

    for index, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        document = item.get("document") or ""

        print(f"\n原始结果 {index}")
        print(f"PDF页码：{get_page(metadata)}")
        print(f"文本：{document[:300]}")

    # 2. 提取问题核心对象
    question_core = extract_question_core(question)
    print(f"\n问题核心对象：{question_core}")

    # 3. 生成前相关性筛选
    selected = select_relevant_results(question, results)

    print(f"筛选结果数量：{len(selected)}")

    if not selected:
        print("筛选后没有结果")
        return

    # 4. 条文截取
    for index, item in enumerate(selected, start=1):
        metadata = item.get("metadata") or {}
        document = item.get("document") or ""

        clause_text = extract_relevant_clause(
            document,
            question_core,
        )

        print(f"\n筛选结果 {index}")
        print(f"PDF页码：{get_page(metadata)}")
        print(f"条文截取字符数：{len(clause_text)}")
        print("截取文本：")
        print(clause_text)


def main():
    print("GB50060-2008 检索回归测试")
    print(f"测试题数量：{len(TEST_QUESTIONS)}")

    for test_id, question in TEST_QUESTIONS:
        try:
            run_test(test_id, question)
        except Exception as exc:
            print(f"\n{test_id} 测试发生异常：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("全部测试题执行完毕")
    print("=" * 70)


if __name__ == "__main__":
    main()
