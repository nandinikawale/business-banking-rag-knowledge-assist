from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from banking_rag.embeddings import LocalHashingEmbeddings
from banking_rag.models import Chunk, RetrievedChunk


class KnowledgeStore:
    def __init__(self, vector_dir: Path, collection_name: str, dimensions: int = 384) -> None:
        self.client = chromadb.PersistentClient(path=str(vector_dir))
        self.collection_name = collection_name
        self.embeddings = LocalHashingEmbeddings(dimensions)

    def rebuild(self, chunks: list[Chunk]) -> int:
        try:
            self.client.delete_collection(self.collection_name)
        except NotFoundError:
            pass
        collection = self.client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=self.embeddings.embed([chunk.text for chunk in chunks]),
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "section": chunk.section,
                    "version": chunk.version,
                    "effective_date": chunk.effective_date,
                    "source_path": chunk.source_path,
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def count(self) -> int:
        try:
            return self.client.get_collection(self.collection_name).count()
        except NotFoundError:
            return 0

    def query(
        self, question: str, top_k: int = 4, category: str | None = None
    ) -> list[RetrievedChunk]:
        collection = self.client.get_collection(self.collection_name)
        where = {"category": category} if category else None
        result = collection.query(
            query_embeddings=self.embeddings.embed([question]),
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    distance=max(0.0, float(distance)),
                    relevance=max(0.0, min(1.0, 1.0 - float(distance))),
                    **metadata,
                )
            )
        return retrieved
