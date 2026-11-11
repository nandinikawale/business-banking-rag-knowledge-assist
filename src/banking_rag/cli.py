import argparse
import json
from pathlib import Path

from banking_rag.config import settings
from banking_rag.evaluation import evaluate_retrieval
from banking_rag.pipeline import RAGPipeline


def build_index_main() -> None:
    count = RAGPipeline().build_index()
    print(f"Indexed {count} fictional banking knowledge chunks.")


def evaluate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.knowledge_dir.parent / "evaluation" / "questions.jsonl",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_retrieval(RAGPipeline(), args.dataset)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
