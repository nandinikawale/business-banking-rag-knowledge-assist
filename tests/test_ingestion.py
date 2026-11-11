from pathlib import Path

from banking_rag.ingestion import chunk_document, load_documents


def test_loads_fictional_documents() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = load_documents(root / "data" / "knowledge")
    assert len(documents) == 5
    assert len({document.document_id for document in documents}) == 5
    assert all("Fictional" in document.title for document in documents)


def test_chunks_retain_citation_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    document = load_documents(root / "data" / "knowledge")[0]
    chunks = chunk_document(document)
    assert chunks
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.section for chunk in chunks)
