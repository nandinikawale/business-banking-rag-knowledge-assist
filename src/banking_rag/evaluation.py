from __future__ import annotations

import json
from pathlib import Path

from banking_rag.pipeline import RAGPipeline


def evaluate_retrieval(pipeline: RAGPipeline, dataset_path: Path, top_k: int = 4) -> dict:
    cases = [
        json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line
    ]
    reciprocal_ranks: list[float] = []
    hits = 0
    details = []
    pipeline.ensure_index()
    for case in cases:
        results = pipeline.store.query(case["question"], top_k=top_k)
        ranked_ids = [item.document_id for item in results]
        expected = case["expected_document_id"]
        rank = ranked_ids.index(expected) + 1 if expected in ranked_ids else None
        hits += int(rank is not None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        details.append({"question": case["question"], "expected": expected, "rank": rank})
    return {
        "cases": len(cases),
        f"hit_rate_at_{top_k}": hits / len(cases),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(cases),
        "details": details,
    }
