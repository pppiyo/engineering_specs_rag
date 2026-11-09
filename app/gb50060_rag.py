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
    """
    去除常见提问表达，保留问题中的核心对象描述。
    仅用于相关性匹配，不修改用户原始问题。
    """
    core = normalize_text(question)

    suffixes = [
        "需要采取哪些措施",
        "应采取哪些措施",
        "需要采取什么措施",
        "应采取什么措施",
        "有哪些措施",
        "采取哪些措施",
        "采取什么措施",
        "应满足哪些要求",
        "需要满足哪些要求",
        "应满足什么要求",
        "需要满足什么要求",
        "满足哪些要求",
        "满足什么要求",
        "有哪些方面的要求",
        "有哪些建筑要求",
        "有什么建筑要求",
        "有哪些要求",
        "有什么要求",
        "有哪些规定",
        "有什么规定",
        "是什么要求",
        "应符合什么规定",
        "应符合哪些规定",
        "如何设置",
        "如何配置",
        "如何选择",
        "应如何设置",
        "应如何配置",
        "应如何选择",
        "是什么",
        "有哪些",
        "为什么",
        "吗",
    ]

    # 反复清理，处理连续出现的提问表达。
    changed = True

    while changed and core:
        changed = False

        for suffix in suffixes:
            normalized_suffix = normalize_text(suffix)

            if core.endswith(normalized_suffix):
                core = core[:-len(normalized_suffix)]
                changed = True
                break

    return core



def detect_chapter_query(question):
    """
    判断是否为整章查询。

    支持：
    第三章
    第3章
    第 3 章

    返回章节数字；不是整章查询则返回 None。
    """
    question = unicodedata.normalize(
        "NFKC",
        str(question or ""),
    )

    match = re.search(
        r"第\s*([一二三四五六七八九十\d]+)\s*章",
        question,
    )

    if not match:
        return None

    value = match.group(1)

    chinese_numbers = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if value.isdigit():
        return int(value)

    return chinese_numbers.get(value)


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


def calculate_clause_relevance(question_core, clause_text):
    """
    计算问题核心对象与候选条文的相关性。

    优先采用完整短语匹配；
    否则采用字符二元组覆盖率。
    不修改原始规范文本。
    """
    normalized_core = normalize_text(question_core)
    normalized_clause = normalize_text(clause_text)

    if not normalized_core or not normalized_clause:
        return 0.0

    # 完整短语匹配。
    if normalized_core in normalized_clause:
        return 1.0

    core_bigrams = get_char_bigrams(normalized_core)
    clause_bigrams = get_char_bigrams(normalized_clause)

    if not core_bigrams or not clause_bigrams:
        return 0.0

    overlap = core_bigrams & clause_bigrams

    return len(overlap) / len(core_bigrams)



def deduplicate_results(results):
    """
    按页码和原文内容去重。
    不修改原始检索结果。
    """
    unique_results = []
    seen = set()

    for result in results:
        metadata = result.get("metadata", {})
        page = metadata.get("page", "未知")
        document = result.get("document", "")

        key = (str(page), document)

        if key in seen:
            continue

        seen.add(key)
        unique_results.append(result)

    return unique_results


def select_relevant_results(question, results, min_score=0.12):
    """
    综合核心短语匹配和字符二元组相关性筛选。

    不在筛选结果为空时回退全部结果，
    避免无关片段进入生成模型。
    """
    question_core = extract_question_core(question)

    print("\n========== 生成前相关性筛选 ==========")
    print("问题核心对象：", question_core)

    results = deduplicate_results(results)

    selected = []
    scored_results = []

    for result in results:
        document = result.get("document", "")

        if not document:
            continue

        metadata = result.get("metadata", {})
        page = metadata.get("page", "未知")
        normalized_document = normalize_text(document)

        phrase_match = (
            bool(question_core)
            and question_core in normalized_document
        )

        score = calculate_relevance(question, document)

        print(
            f"页码：{page}，"
            f"核心短语命中：{phrase_match}，"
            f"二元组相关性：{score:.4f}"
        )

        if phrase_match:
            selected.append(result)
        else:
            scored_results.append((score, result))

    for score, result in scored_results:
        if score >= min_score:
            selected.append(result)

    selected = deduplicate_results(selected)

    print("筛选并去重后的结果数量：", len(selected))

    for result in selected:
        metadata = result.get("metadata", {})
        print("保留页码：", metadata.get("page", "未知"))

    if not selected:
        print("没有结果达到相关性筛选条件。")

    return selected


def find_clause_boundaries(document):
    """
    识别行首条文或章节标题位置。

    支持：
    3
    3.0
    3.0.1
    5.4.9
    S.2
    S.1.7

    仅用于截取边界，不修改 OCR 原文。
    """
    clause_pattern = re.compile(
        r"(?m)^[ \t]*"
        r"((?:[0-9]+(?:[ \t]*\.[ \t]*[0-9]+){0,2})"
        r"|(?:S[ \t]*\.[ \t]*[0-9]+(?:[ \t]*\.[ \t]*[0-9]+)?))"
        r"(?=[ \t　、，.．]|$)",
        re.IGNORECASE,
    )

    return list(clause_pattern.finditer(document))


def extract_relevant_clause(document, question_core):
    """
    从原始 OCR 文本中提取相关内容。

    优先提取相关编号条文；
    如果没有编号条文，再尝试提取相关段落。

    使用统一的相关性标准，不要求核心对象完整连续出现。
    无法可靠定位时返回原始文本，由后续相关性校验决定是否使用。
    """
    if not document or not question_core:
        return document

    matches = find_clause_boundaries(document)

    # 第一种情况：从编号条文中寻找相关内容。
    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(document)

        clause_text = document[start:end].strip()

        score = calculate_clause_relevance(
            question_core,
            clause_text,
        )

        if score >= 0.50:
            print(
                "条文截取成功：",
                re.sub(r"\s+", "", match.group(1)),
                f"，相关性：{score:.4f}",
            )
            return clause_text

    # 第二种情况：尝试按段落提取。
    paragraphs = re.split(r"\n\s*\n", document)

    for index, paragraph in enumerate(paragraphs):
        score = calculate_clause_relevance(
            question_core,
            paragraph,
        )

        if score < 0.50:
            continue

        selected_parts = [paragraph]

        for following in paragraphs[index + 1:]:
            if find_clause_boundaries(following):
                break

            if following.strip():
                selected_parts.append(following)

        paragraph_text = "\n\n".join(selected_parts).strip()

        print(
            "无编号段落截取成功，"
            f"相关性：{score:.4f}"
        )

        return paragraph_text

    print("未能定位相关条文或段落，保留原始文本。")

    return document



def ask_gb50060_question(question, return_details=False):
    """
    基于 GB50060-2008 独立知识库回答问题。
    """
    question = (question or "").strip()

    if not question:
        raise ValueError("问题不能为空")

    # 1. 判断是否为整章查询。
    chapter_number = detect_chapter_query(question)
    is_chapter_query = chapter_number is not None

    if is_chapter_query:
        print(
            f"\n========== 整章问答模式：第 {chapter_number} 章 =========="
        )

    # 2. 只检索 GB50060 独立知识库。
    results = search_gb50060_documents(
        question,
        top_k=3,
    )

    if not results:
        answer = "在 GB50060-2008 知识库中没有检索到相关内容。"
        sources = []

    else:
        # 3. 整章查询不进行相关性筛选。
        # 普通问题仍使用原来的筛选逻辑。
        if is_chapter_query:
            selected_results = deduplicate_results(results)

            print(
                "整章查询：跳过相关性筛选，"
                "保留全部章节检索结果。"
            )
            print(
                "整章内容 chunks 数量：",
                len(selected_results),
            )
        else:
            selected_results = select_relevant_results(
                question,
                results,
            )

        # 4. 提取问题核心对象。
        question_core = extract_question_core(question)

        # 5. 提取相关条文或段落。
        context_parts = []
        context_results = []
        seen_clause_texts = set()

        for index, result in enumerate(
            selected_results,
            start=1,
        ):
            document = result.get("document", "")
            metadata = result.get("metadata", {})
            page = metadata.get("page", "未知")

            if not document:
                continue

            if is_chapter_query:
                # 整章查询保留每个 chunk 的完整原文，
                # 不调用单条条文提取函数。
                clause_text = document
            else:
                clause_text = extract_relevant_clause(
                    document,
                    question_core,
                )


            if not clause_text.strip():
                continue

            # 普通问题必须确认截取内容确实包含问题核心对象。
            # 如果提取失败并回退了整个原始 chunk，则不送入模型。
            if not is_chapter_query:
                normalized_core = normalize_text(question_core)
                normalized_clause = normalize_text(clause_text)

                # if (
                #     normalized_core
                #     and normalized_core not in normalized_clause
                # ):
                #     print(
                #         f"跳过无关检索结果：PDF第 {page} 页，"
                #         "未能在截取文本中找到问题核心对象。"
                #     )
                #     continue

                if not is_chapter_query:
                    relevance = calculate_clause_relevance(
                        question_core,
                        clause_text,
                    )

                    print(
                        f"条文最终相关性：PDF第 {page} 页，"
                        f"得分：{relevance:.4f}"
                    )

                    if relevance < 0.50:
                        print(
                            f"跳过低相关性结果：PDF第 {page} 页"
                        )
                        continue







            # 跳过已经加入上下文的重复条文。
            normalized_clause = normalize_text(clause_text)

            if normalized_clause in seen_clause_texts:
                print(f"跳过重复条文：PDF第 {page} 页")
                continue

            seen_clause_texts.add(normalized_clause)

            context_parts.append(
                f"【规范原文 {index}｜PDF第 {page} 页】\n"
                f"{clause_text}"
            )


            # 关键修改：
            # 保存实际送入模型的条文文本，
            # 供来源构建函数匹配 OCR 行。
            source_result = dict(result)
            source_result["highlight_text"] = clause_text
            source_result["highlight_question"] = (
                None if is_chapter_query else question
            )
            context_results.append(source_result)

        context = "\n\n".join(context_parts)

        if not context.strip():
            answer = (
                "本次检索没有获得可用的规范原文，"
                "暂时无法依据 GB50060-2008 回答。"
            )
            sources = []

        else:
            print("\n========== 实际送入模型的规范原文 ==========")
            print(context)

            # 6. 构造生成要求。
            if is_chapter_query:
                task_instruction = """
用户正在查询整章内容。

请依据提供的章节原文，按原文的章节和条文顺序进行整理。
不得只挑选其中一两条内容，也不得遗漏已提供的条文。
保留原文中的条文编号、技术术语、数值和表格信息。
如果 OCR 原文存在不清楚的地方，应明确指出，不得自行猜测。
"""
            else:
                task_instruction = """
用户正在查询具体技术问题。

请只回答原文明确支持的内容。
如果原文只规定了部分要求，只回答明确规定的部分。
"""

            grounded_context = f"""
你是 GB50060-2008《3～110kV高压配电装置设计规范》问答助手。

请严格依据下面提供的规范原文回答用户问题。

要求：
1. 只能依据提供的规范原文回答，不得引入其他标准或外部知识。
2. 不得编造技术要求、数值或措施。
3. OCR文本可能存在识别错误。不得自行猜测或改写不确定的编号、数值和技术术语。
4. 保留原文中能够确认的条文编号。
5. 如果提供了表格内容，应结合表格回答；没有提供相关表格时，不得推测表格数值。
6. 如果提供的原文与问题无关，明确说明原文不足以回答，不得用相近主题内容代替答案。

{task_instruction}

【用户问题】
{question}

【规范原文】
{context}

请直接给出答案。
"""

            print("\n========== Prompt 长度 ==========")
            print("规范原文字符数：", len(context))
            print("完整上下文字符数：", len(grounded_context))

            # 7. 复用现有 Ollama 模型生成函数。
            answer = generate_answer(
                question=question,
                context=grounded_context,
            )

            if not answer:
                answer = "模型未能生成有效回答，请检查检索内容。"

            # 8. 来源依据实际送入模型的文本重新定位高亮。
            sources = build_gb50060_source_list(context_results)

    if return_details:
        return {
            "answer": answer,
            "sources": sources,
        }

    return answer