"""
GB50060-2008 独立 RAG 问答模块。

只使用 gb50060_2008 Collection，
不调用旧版知识库检索流程。
"""

from app.gb50060_retriever import (
    search_gb50060_documents,
    build_gb50060_source_list,
)

from app.rag import generate_answer


def ask_gb50060_question(question, return_details=False):
    """
    基于 GB50060-2008 知识库回答问题。

    Args:
        question: 用户问题
        return_details: 是否返回结构化回答和来源

    Returns:
        默认返回回答字符串。
        return_details=True 时返回 answer 和 sources。
    """

    question = question.strip()

    if not question:
        raise ValueError("问题不能为空")

    # 1. 只检索 GB50060 独立知识库
    results = search_gb50060_documents(
        question,
        top_k=5,
    )

    # 2. 没有检索到内容时，不让模型凭空回答
    if not results:
        answer = "在 GB50060-2008 知识库中没有检索到相关内容。"
        sources = []

    else:
        # 3. 整理检索文本
        context_parts = []

        for index, result in enumerate(results, start=1):
            document = result.get("document", "")
            metadata = result.get("metadata", {})

            page = metadata.get("page", "未知")

            context_parts.append(
                f"【检索片段 {index}｜PDF第 {page} 页】\n"
                f"{document}"
            )

        context = "\n\n".join(context_parts)

        # 4. 复用现有 Ollama 模型生成函数
        answer = generate_answer(
            question=question,
            context=context,
        )

        # 5. 整理页面来源，保留图片和高亮坐标
        sources = build_gb50060_source_list(results)

        if not answer:
            answer = "模型未能生成有效回答，请检查检索内容。"

    if return_details:
        return {
            "answer": answer,
            "sources": sources,
        }

    return answer
