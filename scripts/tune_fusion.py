"""Grid search over hybrid fusion parameters on an existing evaluation index.

Usage: uv run python scripts/tune_fusion.py ~/.local/share/ticket-precedent/eval/<index>.db
"""

from __future__ import annotations

import sys
from pathlib import Path

from ticket_precedent.config import EmbeddingConfig, build_embedder
from ticket_precedent.evaluation.dataset import duplicate_queries
from ticket_precedent.evaluation.runner import evaluate
from ticket_precedent.search import Searcher
from ticket_precedent.store import Store


def main(index: Path) -> None:
    store = Store(index)
    source = store.db.execute("SELECT source FROM tickets LIMIT 1").fetchone()["source"]
    model = (store.meta("embedder") or "ollama:bge-m3").removeprefix("ollama:")
    embedder = build_embedder(EmbeddingConfig(model=model))
    queries = duplicate_queries(store, source)
    searcher = Searcher(store, embedder)
    searcher.prime([query.text for query in queries])
    print(f"{source}: {len(queries)} queries")
    print("rrf_k  lexical_weight  hit@1  hit@5  hit@10  mrr@10")
    for rrf_k in (10, 30, 60):
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            searcher.rrf_k = rrf_k
            searcher.lexical_weight = weight
            (summary,), _ = evaluate(searcher, queries, ["hybrid"], source)
            print(
                f"{rrf_k:5}  {weight:14.2f}  {summary.hit_at_1:.3f}  {summary.hit_at_5:.3f}"
                f"  {summary.hit_at_10:.3f}  {summary.mrr_at_10:.3f}"
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
