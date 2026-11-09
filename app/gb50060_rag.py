
"""
GB50060-2008 独立 RAG 问答模块。

只使用 gb50060_2008 Collection，
不调用旧版知识库检索流程。
"""

import re
import unicodedata

from app.gb50060_retriever import (
    search_gb50060_documents,
    build_gb50060_source_list,
)

from app.rag import generate_answer


def normalize_text(text):
    """
    统一字符格式并删除空白和标点。
    只用于匹配，不修改原始 OCR 文本。
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.lower()

    return re.sub(
        r"""[\s，。！？、；：,.!?;:（）()《》【】\[\]"'“”‘’]""",
        "",
        text,
    )


def extract_question_core(question):
    """从问题中去掉常见提问后缀，提取核心对象描述。"""

    core = normalize_text(question)

    suffixes = [
        "需要采取哪些措施",
        "应采取哪些措施",
        "需要采取什么措施",
        "应采取什么措施",
        "有哪些措施",
        "采取哪些措施",
        "采取什么措施",
        "有什么要求",
        "有哪些要求",
        "有什么规定",
        "有哪些规定",
        "是什么要求",
        "如何设置",
        "如何配置",
        "如何选择",
        "是什么",
        "有哪些",
        "为什么",
        "吗",
    ]

    for suffix in suffixes:
        normalized_suffix = normalize_text(suffix)

        if core.endswith(normalized_suffix):
            core = core[:-len(normalized_suffix)]
            break

    return core


def get_char_bigrams(text):
    """提取规范化文本的相邻字符组合。"""

    text = normalize_text(text)

    if len(text) < 2:
        return set()

    return {
        text[i:i + 2]
        for i in range(len(text) - 1)
    }


def calculate_relevance(question, document):
    """计算问题与规范原文的字符二元组重合率。"""

    question_terms = get_char_bigrams(question)
    document_terms = get_char_bigrams(document)

    if not question_terms or not document_terms:
        return 0.0

    overlap = question_terms & document_terms

    return len(overlap) / len(question_terms)


def select_relevant_results(question, results, min_score=0.12):
    """
    优先使用规范化后的核心对象短语匹配。

    如果没有精确匹配，再使用字符二元组相关性筛选。
    如果二者都没有结果，则回退原始检索结果。
    """

    question_core = extract_question_core(question)

    print("\n========== 生成前相关性筛选 ==========")
    print("问题核心对象：", question_core)

    # 第一优先级：核心对象短语匹配
    if question_core:
        phrase_matches = []

        for result in results:
            document = result.get("document", "")
            normalized_document = normalize_text(document)

            if question_core in normalized_document:
                phrase_matches.append(result)

        if phrase_matches:
            print("匹配方式：核心对象短语匹配")
            print("匹配结果数量：", len(phrase_matches))

            for result in phrase_matches:
                metadata = result.get("metadata", {})
                print("匹配页码：", metadata.get("page", "未知"))

            return phrase_matches

    # 第二优先级：字符二元组相关性筛选
    print("匹配方式：字符二元组相关性筛选")

    scored_results = []

    for result in results:
        document = result.get("document", "")

        if not document:
            continue

        score = calculate_relevance(question, document)

        metadata = result.get("metadata", {})
        page = metadata.get("page", "未知")

        print(
            f"页码：{page}，"
            f"相关性分数：{score:.4f}"
        )

        scored_results.append((score, result))

    if not scored_results:
        return []

    relevant_results = [
        result
        for score, result in scored_results
        if score >= min_score
    ]

    if relevant_results:
        print("二元组筛选结果数量：", len(relevant_results))
        return relevant_results

    # 第三优先级：回退原始检索结果
    print("没有结果达到相关性阈值，回退原始检索结果。")

    return [
        result
        for _, result in scored_results
    ]


def extract_relevant_clause(document, question_core):
    """
    从原始 OCR 文本中提取包含问题核心对象的条文。

    以条文编号作为边界：
    从目标条文编号开始，到下一条文编号之前结束。

    允许条文编号首位出现 OCR 误识别字符 S，
    仅用于边界识别，不修改原始 OCR 文本。

    如果无法可靠识别条文边界，则返回原始文本。
    """

    if not document or not question_core:
        return document

    # 匹配类似 3.0.1、5.4.9、S.1.7 的条文编号。
    # 编号必须出现在新的一行开头。
    clause_pattern = re.compile(
        r"(?m)^[ \t]*([0-9S](?:[ \t]*\.[ \t]*\d+){1,2})(?=[ \t　]|$)"
    )

    matches = list(clause_pattern.finditer(document))

    if not matches:
        return document

    normalized_core = normalize_text(question_core)

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(document)

        clause_text = document[start:end].strip()

        if normalized_core in normalize_text(clause_text):
            return clause_text

    # 没有找到包含核心对象的完整条文时，不冒险截断。
    return document
def ask_gb50060_question(question, return_details=False):
    """
    基于 GB50060-2008 知识库回答问题。
    """

    question = (question or "").strip()

    if not question:
        raise ValueError("问题不能为空")

    # 1. 只检索 GB50060 独立知识库
    results = search_gb50060_documents(
        question,
        top_k=3,
    )

    if not results:
        answer = "在 GB50060-2008 知识库中没有检索到相关内容。"
        sources = []

    else:
        # 2. 规范化匹配并筛选相关结果
        selected_results = select_relevant_results(
            question,
            results,
        )

        # 3. 提取核心对象
        question_core = extract_question_core(question)

        # 4. 从筛选结果中提取相关条文
        context_parts = []
        context_results = []

        for index, result in enumerate(selected_results, start=1):
            document = result.get("document", "")
            metadata = result.get("metadata", {})
            page = metadata.get("page", "未知")

            if not document:
                continue

            clause_text = extract_relevant_clause(
                document,
                question_core,
            )

            context_parts.append(
                f"【规范原文 {index}｜PDF第 {page} 页】\n"
                f"{clause_text}"
            )

            context_results.append(result)

        context = "\n\n".join(context_parts)

        if not context.strip():
            answer = "本次检索没有获得可用于回答的规范原文。"
            sources = build_gb50060_source_list(selected_results)

        else:
            print("\n========== 实际送入模型的规范原文 ==========")
            print(context)

            # 5. 构造简洁、明确的生成要求
            grounded_context = f"""
你是 GB50060-2008《3～110kV高压配电装置设计规范》问答助手。

请严格依据下面提供的规范原文回答用户问题。

要求：
1. 只能依据提供的规范原文回答，不得引入其他标准或外部知识。
2. 如果原文直接规定了相关措施，必须提取并回答，不得回答“内容不足”。
3. OCR可能存在少量识别错误，应结合上下文理解，不要因为个别错字忽略明确的技术要求。
4. 不得编造原文没有规定的措施、数值或技术要求。
5. 使用简洁的条目回答，并标明条文编号。
6. 如果原文只规定了部分要求，只回答明确规定的部分。

【用户问题】
{question}

【规范原文】
{context}

请直接给出答案。
"""

            print("\n========== Prompt 长度 ==========")
            print("规范原文字符数：", len(context))
            print("完整上下文字符数：", len(grounded_context))

            # 6. 复用现有 Ollama 模型生成函数
            answer = generate_answer(
                question=question,
                context=grounded_context,
            )

            if not answer:
                answer = "模型未能生成有效回答，请检查检索内容。"

            # 7. 来源对应实际送入模型的结果
            sources = build_gb50060_source_list(context_results)

    if return_details:
        return {
            "answer": answer,
            "sources": sources,
        }

    return answer
