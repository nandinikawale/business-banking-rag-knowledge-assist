from pathlib import Path

from banking_rag.config import Settings
from banking_rag.models import QueryRequest
from banking_rag.pipeline import RAGPipeline


def make_pipeline(tmp_path: Path) -> RAGPipeline:
    root = Path(__file__).resolve().parents[1]
    return RAGPipeline(
        Settings(
            knowledge_dir=root / "data" / "knowledge",
            vector_dir=tmp_path / "vectors",
            collection_name="test_knowledge",
        )
    )


def test_retrieves_line_of_credit_document(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    pipeline.build_index()
    result = pipeline.answer(
        QueryRequest(question="Which documents support a line of credit request?")
    )
    assert result.supported
    assert result.sources[0].document_id == "FIC-LOC-001"
    assert "[1]" in result.answer


def test_category_filter(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    pipeline.build_index()
    result = pipeline.answer(
        QueryRequest(
            question="What factors should be reviewed before discussing ACH origination?",
            category="Treasury",
            top_k=3,
        )
    )
    assert result.sources
    assert all(source.category == "Treasury" for source in result.sources)
