
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.gb50060_rag import ask_gb50060_question

from app.gb50060_retriever import (
    search_gb50060_documents,
    build_gb50060_source_list,
)

import re


app = FastAPI(
    title="GB50060-2008 RAG API",
    description="基于 GB50060-2008 高压配电装置设计规范的检索问答 API",
    version="2.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    source: str | None = None
    page: int | None = None
    page_image: str | None = None
    bbox: list[int] | None = None
    line_boxes: list[Any] = Field(default_factory=list)
    coordinate_unit: str | None = None
    coordinate_scale: float | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)


@app.get("/")
def home():
    return {"message": "GB50060-2008 RAG API is running"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空",
        )

    try:

        # =========================
        # 判断是否输入明确条文编号
        # =========================

        clause_match = re.search(
            r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])",
            question,
        )

        # =========================
        # 路径一：编号直达原文
        # =========================

        if clause_match:

            results = search_gb50060_documents(
                question,
                top_k=3,
            )

            if not results:

                return AnswerResponse(
                    question=question,
                    answer=(
                        f"在 GB50060-2008 规范正文中"
                        f"未找到条文 {clause_match.group()}。"
                    ),
                    sources=[],
                )

            # 直接返回 Chroma 保存的原始文本
            answer = "\n\n".join(
                item["document"]
                for item in results
            )

            sources = build_gb50060_source_list(
                results
            )

            return AnswerResponse(
                question=question,
                answer=answer,
                sources=sources,
            )

        # =========================
        # 路径二：自然语言 RAG 问答
        # 保持原有逻辑
        # =========================

        result = ask_gb50060_question(
            question,
            return_details=True,
        )

        return AnswerResponse(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
