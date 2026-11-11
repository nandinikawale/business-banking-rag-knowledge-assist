from fastapi import FastAPI

from banking_rag.models import AnswerResponse, QueryRequest
from banking_rag.pipeline import RAGPipeline

app = FastAPI(title="Business Banking RAG Knowledge Service", version="0.1.0")
pipeline = RAGPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=AnswerResponse)
def query(request: QueryRequest) -> AnswerResponse:
    return pipeline.answer(request)
