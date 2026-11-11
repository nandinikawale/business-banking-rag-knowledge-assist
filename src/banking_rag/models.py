from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    document_id: str
    title: str
    category: str
    version: str
    effective_date: str
    text: str
    source_path: str


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    category: str
    section: str
    version: str
    effective_date: str
    text: str
    source_path: str


class RetrievedChunk(Chunk):
    distance: float = Field(ge=0)
    relevance: float = Field(ge=0, le=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    category: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[RetrievedChunk]
    supported: bool
    mode: str = "deterministic_grounded_synthesis"
