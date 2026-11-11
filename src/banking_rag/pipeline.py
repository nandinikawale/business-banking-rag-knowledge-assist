from __future__ import annotations

import re

from banking_rag.config import Settings, settings
from banking_rag.ingestion import load_and_chunk
from banking_rag.models import AnswerResponse, QueryRequest, RetrievedChunk
from banking_rag.store import KnowledgeStore


class RAGPipeline:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.store = KnowledgeStore(
            config.vector_dir,
            config.collection_name,
            config.embedding_dimensions,
        )

    def build_index(self) -> int:
        return self.store.rebuild(load_and_chunk(self.config.knowledge_dir))

    def ensure_index(self) -> int:
        return self.store.count() or self.build_index()

    def answer(self, request: QueryRequest) -> AnswerResponse:
        self.ensure_index()
        sources = self.store.query(request.question, request.top_k, request.category)
        supported = bool(sources and sources[0].relevance >= 0.12)
        answer = self._synthesize(sources) if supported else self._unsupported_answer()
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=sources if supported else [],
            supported=supported,
        )

    @staticmethod
    def _synthesize(sources: list[RetrievedChunk]) -> str:
        statements: list[str] = []
        for index, source in enumerate(sources[:3], start=1):
            clean = re.sub(r"\s+", " ", source.text).strip()
            sentences = re.split(r"(?<=[.!?])\s+", clean)
            excerpt = " ".join(sentences[:2])[:500].rstrip()
            statements.append(f"- {excerpt} [{index}]")
        return (
            "Based on the fictional knowledge base, the relevant guidance is:\n\n"
            + "\n".join(statements)
            + "\n\nConfirm current requirements and use RM, credit, operations, or compliance review as applicable."
        )

    @staticmethod
    def _unsupported_answer() -> str:
        return (
            "I could not find sufficient support in the fictional knowledge base. "
            "Rephrase the question or consult an authorized policy owner."
        )
