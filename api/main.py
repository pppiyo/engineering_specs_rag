
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import ask_question

app = FastAPI(
    title="Design RAG API",
    description="本地电气设计规范知识库问答接口",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def home():
    return {
        "message": "Design RAG API is running"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空"
        )

    try:
        answer = ask_question(question)
        return AnswerResponse(
            question=question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
