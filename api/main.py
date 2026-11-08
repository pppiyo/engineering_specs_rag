from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import ask_question


app = FastAPI(
    title="Design RAG API",
    description="电力设计规范 PDF 检索问答 API",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    source: str | None = None
    page: int | None = None
    page_image: str | None = None
    bbox: list[int] | None = None
    line_boxes: list[Any] = []
    coordinate_unit: str | None = None
    coordinate_scale: float | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse] = []


@app.get("/")
def home():
    return {"message": "Design RAG API is running"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空",
        )

    try:
        result = ask_question(
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