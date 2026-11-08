import time # test latency
import requests


from app.config import (
    OLLAMA_BASE_URL,
    CHAT_MODEL,
    TOP_K,
)

import re

from app.retriever import (
    search_documents,
    search_section,
    search_clause,
    expand_clause_results
)

def detect_clause(question):
    """识别三级条文编号，例如 4.1.7。"""
    match = re.search(
        r"(?<![\d.])(\d+\.\d+\.\d+)(?![\d.])",
        question
    )
    return match.group(1) if match else None


def detect_section(question):
    """识别两级章节编号，例如 4.1。"""
    match = re.search(
        r"(?<![\d.])(\d+\.\d+)(?![\d.])",
        question
    )
    return match.group(1) if match else None


def extract_numbered_clauses(text):
    """
    提取三级条文。
    遇到下一个两级或三级编号时，结束当前条文，
    避免将章节标题并入上一条条文。
    """

    pattern = re.compile(
        r"(?m)^[ \t]*(\d+\.\d+(?:\.\d+)?)\s+"
    )

    matches = list(pattern.finditer(text))
    clauses = []

    for i, match in enumerate(matches):
        clause_number = match.group(1)

        # 只保留三级条文，例如 4.2.1
        if len(clause_number.split(".")) != 3:
            continue

        start = match.start()

        # 找到后面第一个编号标题作为结束位置
        end = len(text)

        for next_match in matches[i + 1:]:
            end = next_match.start()
            break

        clauses.append({
            "clause": clause_number,
            "text": text[start:end].strip()
        })

    return clauses



def detect_relevant_sections(question, raw_results):
    """
    从语义检索结果中识别与用户问题直接相关的两级章节标题。
    例如：问题包含“导体的选择”，检索片段出现“4.2 导体的选择”，
    则返回 4.2。
    """
    normalized_question = re.sub(r"\s+", "", question)
    sections = []

    pattern = re.compile(
        r"(?m)^[ \t]*(\d+\.\d+)[ \t]+([^\n]+)"
    )

    for item in raw_results:
        for match in pattern.finditer(item["text"]):
            section = match.group(1)
            title = re.sub(r"\s+", "", match.group(2))

            # 只匹配问题中明确出现的章节标题
            if len(title) >= 2 and title in normalized_question:
                if section not in sections:
                    sections.append(section)

    return sections


def generate_answer(question, context, section_mode=False):
    """根据检索内容生成回答，不负责引用规范原文。"""

    if section_mode:
        task_instruction = """
请对检索到的完整章节进行总结：
1. 归纳为3至5个技术主题。
2. 每个主题用完整中文句子说明具体要求。
3. 注明对应条文编号。
4. 保留原文中的电压等级、设备类型、数值、条件和限制。
5. 不逐条抄写原文，不得遗漏重要要求。
6. 不得猜测或补全 OCR 疑似错误。
7. 无法确认的内容应注明原文识别不清。
"""
    else:
        task_instruction = """
请根据检索原文回答用户问题：
1. 先概括主要规范要求，再按技术主题分点说明。
2. 每个要点必须写出具体技术要求、适用条件及数值（如原文提供）。
3. 每个要点注明对应条文编号。
4. 检查全部检索内容，不遗漏与问题直接相关的重要要求。
5. 可以合并相近条文，但必须保留所有对应编号。
6. 不得编造、猜测或补全 OCR 疑似错误。
7. 原文无法确认的内容应明确说明。
8. 检索内容不足时，应明确说明。
"""

    system_prompt = f"""你是严谨的中文电气工程设计规范助手。

只能依据提供的规范原文回答。
保留原文数值、单位、条件及“应”“宜”“可”等规范用语的区别。
不得编造或推测规范内容。
不输出分析过程，不重复完整规范原文。

{task_instruction}
"""

    user_prompt = f"""/no_think
知识库检索内容：
{context}

用户问题：
{question}

请直接给出回答。
"""

    print("\n========== Prompt 长度 ==========")
    print("System Prompt 字符数：", len(system_prompt))
    print("User Prompt 字符数：", len(user_prompt))
    print("完整 Prompt 字符数：", len(system_prompt) + len(user_prompt))

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 512,
                "temperature": 0.1
            }
        },
        timeout=(10, 600)
    )

    response.raise_for_status()
    data = response.json()

    print("\n========== Ollama 性能数据 ==========")
    print("总耗时：", round(data.get("total_duration", 0) / 1e9, 2), "秒")
    print("加载耗时：", round(data.get("load_duration", 0) / 1e9, 2), "秒")
    print("提示词处理耗时：", round(data.get("prompt_eval_duration", 0) / 1e9, 2), "秒")
    print("生成耗时：", round(data.get("eval_duration", 0) / 1e9, 2), "秒")
    print("提示词 token 数：", data.get("prompt_eval_count"))
    print("生成 token 数：", data.get("eval_count"))

    if data.get("eval_duration", 0) > 0:
        speed = data["eval_count"] / (
            data["eval_duration"] / 1e9
        )
        print("生成速度：", round(speed, 2), "tokens/s")

    return data["message"].get("content", "").strip()



def ask_question(question):
    """检索规范内容：条文编号直接返回原文，其他问题由模型解释。"""

    clause = detect_clause(question)
    section = detect_section(question)

    # ==========================================
    # 1. 三级条文编号：直接返回原文，不调用模型
    # ==========================================
    if clause:
        print(f"检测到条文编号：{clause}")
        print("正在进行精确条文检索...")

        start = time.perf_counter()
        results = search_clause(clause)

        print(
            f"精确条文检索耗时："
            f"{time.perf_counter() - start:.2f} 秒"
        )

        if not results:
            return f"知识库中没有找到条文 {clause}。"

        citation_parts = []

        for item in results:
            citation_parts.append(
                f"文件名称：{item['source']}\n"
                f"页码：第 {item['page']} 页\n"
                f"{item['text']}"
            )

        return (
            f"【条文原文：{clause}】\n\n"
            + "\n\n".join(citation_parts)
        )

    # ==========================================
    # 2. 两级章节编号：保留章节总结
    # ==========================================
    elif section:
        print(f"检测到章节编号：{section}")
        print("正在进行章节检索...")

        start = time.perf_counter()
        results = search_section(section)

        print(
            f"章节检索耗时："
            f"{time.perf_counter() - start:.2f} 秒"
        )

        if not results:
            return f"知识库中没有找到章节 {section}。"

        raw_results = []

    # ==========================================
    # 3. 自然语言问题：保留原有 RAG 流程
    # ==========================================
    else:
        print("未检测到章节或条文编号，使用向量检索...")

        start = time.perf_counter()
        raw_results = search_documents(
            question,
            top_k=TOP_K
        )

        print(
            f"向量检索耗时："
            f"{time.perf_counter() - start:.2f} 秒"
        )

        print("\n========== 原始语义检索结果 ==========")
        print("语义检索命中：", len(raw_results))

        candidates = []
        start = time.perf_counter()

        normalized_question = re.sub(r"\s+", "", question)

        for item in raw_results:
            clauses = extract_numbered_clauses(item["text"])

            for clause_item in clauses:
                text = clause_item["text"]
                normalized_text = re.sub(r"\s+", "", text)

                score = 0

                for n in range(2, min(9, len(normalized_question) + 1)):
                    for i in range(len(normalized_question) - n + 1):
                        phrase = normalized_question[i:i+n]

                        if phrase in normalized_text:
                            score += n

                if score > 0:
                    candidates.append({
                        "clause": clause_item["clause"],
                        "text": text,
                        "source": item["source"],
                        "page": item["page"],
                        "score": score
                    })

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        print("\n========== 条文匹配结果 ==========")

        for item in candidates[:5]:
            print(
                f"条文：{item['clause']}，"
                f"匹配分数：{item['score']}，"
                f"页码：{item['page']}"
            )

        results = []
        seen = set()

        for item in candidates:
            key = (item["source"], item["clause"])

            if key in seen:
                continue

            seen.add(key)

            results.append({
                "text": item["text"],
                "source": item["source"],
                "page": item["page"]
            })

            if len(results) >= 5:
                break

        if not results:
            print("没有找到可以明确定位的条文。")
            results = raw_results[:5]

        # 根据语义检索结果扩展相关完整章节
        relevant_sections = detect_relevant_sections(
            question,
            raw_results
        )

        for section_number in relevant_sections:
            print(f"识别到相关章节：{section_number}")
            print("正在扩展检索完整章节...")

            section_results = search_section(section_number)

            for item in section_results:
                results.append({
                    "text": item["text"],
                    "source": item["source"],
                    "page": item["page"]
                })

        # 对合并后的结果去重
        unique_results = []
        seen_clauses = set()

        for item in results:
            clauses = extract_numbered_clauses(item["text"])

            if not clauses:
                key = (
                    item["source"],
                    re.sub(r"\s+", "", item["text"])
                )

                if key not in seen_clauses:
                    seen_clauses.add(key)
                    unique_results.append(item)

                continue

            for clause_item in clauses:
                key = (
                    item["source"],
                    clause_item["clause"]
                )

                if key in seen_clauses:
                    continue

                seen_clauses.add(key)

                unique_results.append({
                    "text": clause_item["text"],
                    "source": item["source"],
                    "page": item["page"]
                })

        results = unique_results

        print(
            f"条文匹配及章节扩展耗时："
            f"{time.perf_counter() - start:.2f} 秒"
        )

    # ==========================================
    # 4. 章节总结或自然语言回答
    # ==========================================
    context_parts = []

    for i, item in enumerate(results, 1):
        context_parts.append(f"""
[文档片段 {i}]
文件：{item['source']}
页码：第 {item['page']} 页
内容：
{item['text']}
""")

    context = "\n".join(context_parts)

    print("检索内容总长度：", len(context))
    print("检索完成，开始调用 Ollama 生成回答...")

    start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
        section_mode=bool(section)
    )

    print("Ollama 回答生成完成。")
    print(
        f"模型生成耗时："
        f"{time.perf_counter() - start:.2f} 秒"
    )

    citation_parts = []

    for item in results:
        citation_parts.append(
            f"第 {item['page']} 页\n{item['text']}"
        )

    citation = "\n\n".join(citation_parts)

    if section:
        final_answer = (
            f"【章节总结】\n{answer}\n\n"
            f"【章节完整原文（检索原文）】\n"
            f"{citation}\n\n"
            f"【出处】\n"
            + "\n".join(
                f"文件名称：{item['source']}，"
                f"页码：第 {item['page']} 页"
                for item in results
            )
        )
    else:
        final_answer = (
            f"{answer}\n\n"
            f"【对应条文（检索原文）】\n"
            f"{citation}\n\n"
            f"【出处】\n"
            + "\n".join(
                f"文件名称：{item['source']}，"
                f"页码：第 {item['page']} 页"
                for item in results
            )
        )

    return final_answer



if __name__ == "__main__":

    print("=" * 50)
    print("本地 PDF RAG 知识库")
    print("输入 exit 退出")
    print("=" * 50)

    while True:

        question = input("\n请输入问题：").strip()

        if question.lower() in ["exit", "quit", "退出"]:
            print("再见！")
            break

        if not question:
            continue

        try:
            answer = ask_question(question)

            print("\n" + "=" * 50)
            print("回答：")
            print("=" * 50)
            print(answer)

        # except Exception as e:
        #     print(f"\n发生错误：{e}")
        except Exception:
            import traceback
            traceback.print_exc()
