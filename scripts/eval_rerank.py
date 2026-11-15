"""Benchmark a cross-encoder reranker on top of hybrid search, on an existing evaluation index.

The reranker is not a dependency of the project; pull it in for the run:

    uv run --with sentence-transformers python scripts/eval_rerank.py \
        ~/.local/share/ticket-precedent/eval/apache-zookeeper-comments-bge-m3.db \
        --candidate chunk --max-length 512 --top 50 --grid \
        --out results/rerank-zookeeper-chunk-512.json

Each query's hybrid top `--top` tickets are scored once; every variant below is derived from
those scores, so pools 10, 20 and 50 compare the same model calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sentence_transformers import CrossEncoder

from ticket_precedent.config import EmbeddingConfig, build_embedder
from ticket_precedent.evaluation.dataset import duplicate_queries
from ticket_precedent.evaluation.metrics import (
    first_relevant_rank,
    paired_difference,
    reciprocal_rank,
    summarize,
)
from ticket_precedent.search import Filters, Hit, Searcher, ticket_query
from ticket_precedent.store import Store

MODEL = "BAAI/bge-reranker-v2-m3"
POOLS = (10, 20, 50)
FUSION = {"pool": 10, "rrf_k": 10, "weight": 0.25}
"""Chosen on ZooKeeper with `--grid`, then applied to Derby unchanged."""


@dataclass(frozen=True)
class Scored:
    key: str
    relevant: frozenset[int]
    candidates: list[int]
    scores: list[float]
    rerank_ms: float


def score(args: argparse.Namespace) -> tuple[str, list[Scored]]:
    store = Store(args.index.expanduser())
    source = store.db.execute("SELECT source FROM tickets LIMIT 1").fetchone()["source"]
    embedding_model = (store.meta("embedder") or "ollama:bge-m3").removeprefix("ollama:")
    searcher = Searcher(store, build_embedder(EmbeddingConfig(model=embedding_model)))
    queries = duplicate_queries(store, source)
    searcher.prime([query.text for query in queries])
    reranker = CrossEncoder(MODEL, max_length=args.max_length, device=args.device)
    reranker.predict([("warm-up", "warm-up")])

    scored = []
    for query in queries:
        filters = Filters(
            source=source,
            created_before=query.created,
            exclude_ticket_ids=frozenset({query.ticket_id}),
        )
        hits = searcher.hybrid(query.text, filters)[: args.top]
        texts = candidate_texts(store, hits, args.candidate)
        started = time.perf_counter()
        scores = reranker.predict([(query.text, text) for text in texts], batch_size=16)
        elapsed = (time.perf_counter() - started) * 1000
        scored.append(
            Scored(
                query.key,
                query.relevant,
                [hit.ticket_id for hit in hits],
                [float(value) for value in scores],
                elapsed,
            )
        )
        print(f"{query.key}: {len(hits)} candidates in {elapsed:.0f} ms", flush=True)
    return source, scored


def candidate_texts(store: Store, hits: Sequence[Hit], candidate: str) -> list[str]:
    """`chunk`: the chunk that put the ticket in the list; `ticket`: its title and description."""
    if candidate == "chunk":
        chunks = store.chunks_by_id(hit.chunk_id for hit in hits)
        return [chunks[hit.chunk_id]["text"] for hit in hits]
    tickets = store.tickets_by_id(hit.ticket_id for hit in hits)
    return [
        ticket_query(tickets[hit.ticket_id]["title"], tickets[hit.ticket_id]["description"])
        for hit in hits
    ]


def reranked(query: Scored, pool: int) -> list[int]:
    head = sorted(
        zip(query.candidates[:pool], query.scores[:pool], strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [ticket for ticket, _ in head] + query.candidates[pool:]


def fused(query: Scored, pool: int, rrf_k: int, weight: float) -> list[int]:
    """Reciprocal rank fusion of the hybrid order and the reranker order inside the pool."""
    head = query.candidates[:pool]
    by_reranker = {ticket: rank for rank, ticket in enumerate(reranked(query, pool)[:pool], 1)}
    fusion = {
        ticket: 1 / (rrf_k + rank) + weight / (rrf_k + by_reranker[ticket])
        for rank, ticket in enumerate(head, start=1)
    }
    return sorted(head, key=fusion.__getitem__, reverse=True) + query.candidates[pool:]


def ranks(scored: Sequence[Scored], order: str, **options: float) -> list[int | None]:
    def ranking(query: Scored) -> list[int]:
        if order == "rerank":
            return reranked(query, int(options["pool"]))
        if order == "fused":
            pool, rrf_k, weight = int(options["pool"]), int(options["rrf_k"]), options["weight"]
            return fused(query, pool, rrf_k, weight)
        return query.candidates

    return [first_relevant_rank(ranking(query), query.relevant) for query in scored]


def compare(name: str, baseline: list[int | None], variant: list[int | None]) -> str:
    old = [reciprocal_rank(rank) for rank in baseline]
    new = [reciprocal_rank(rank) for rank in variant]
    mean, low, high = paired_difference(old, new)
    better = sum(1 for a, b in zip(old, new, strict=True) if b > a)
    worse = sum(1 for a, b in zip(old, new, strict=True) if b < a)
    return f"| {name} | {mean:+.3f} ({low:+.3f} to {high:+.3f}) | {better} | {worse} |"


def report(source: str, scored: list[Scored], args: argparse.Namespace) -> None:
    baseline = ranks(scored, "hybrid")
    variants = {f"rerank top {pool}": ranks(scored, "rerank", pool=pool) for pool in POOLS}
    variants = {name: r for name, r in variants.items() if int(name.split()[-1]) <= args.top}
    variants["hybrid + rerank (RRF)"] = ranks(scored, "fused", **FUSION)
    latency = statistics.median(q.rerank_ms for q in scored)

    summaries = [summarize("hybrid", baseline, [])]
    summaries += [summarize(name, r, []) for name, r in variants.items()]
    print(f"\n{source} · {len(scored)} queries · {args.candidate} · max {args.max_length} tokens")
    print(f"reranker p50 {latency:.0f} ms for the top {args.top}\n")
    print("| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 |\n|---|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s.mode} | {s.hit_at_1:.3f} | {s.hit_at_5:.3f} | {s.hit_at_10:.3f} | "
            f"{s.mrr_at_10:.3f} |"
        )
    print("\n| Method | ΔMRR@10 vs hybrid (95% CI) | Better | Worse |\n|---|---:|---:|---:|")
    for name, variant in variants.items():
        print(compare(name, baseline, variant))

    if args.grid:
        print("\npool  rrf_k  weight  mrr@10")
        for pool in (10, 20):
            for rrf_k in (10, 60):
                for weight in (0.1, 0.25, 0.5, 1.0):
                    grid = ranks(scored, "fused", pool=pool, rrf_k=rrf_k, weight=weight)
                    mrr = statistics.fmean(reciprocal_rank(rank) for rank in grid)
                    print(f"{pool:4}  {rrf_k:5}  {weight:6.2f}  {mrr:.3f}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "source": source,
                    "reranker": MODEL,
                    "candidate": args.candidate,
                    "max_length": args.max_length,
                    "top": args.top,
                    "fusion": FUSION,
                    "rerank_ms_p50": latency,
                    "summaries": [_without_latency(asdict(s)) for s in summaries],
                    "queries": [
                        {
                            "key": q.key,
                            "hybrid": baseline[i],
                            **{name: r[i] for name, r in variants.items()},
                        }
                        for i, q in enumerate(scored)
                    ],
                },
                indent=2,
            )
            + "\n"
        )


def _without_latency(summary: dict[str, object]) -> dict[str, object]:
    """Reranker latency is reported once as `rerank_ms_p50`; per summary it would read 0."""
    return {key: value for key, value in summary.items() if key != "latency_ms_p50"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("index", type=Path)
    parser.add_argument("--candidate", choices=["chunk", "ticket"], default="chunk")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--device", default="mps", help="mps, cuda or cpu")
    parser.add_argument("--grid", action="store_true", help="also print the fusion grid")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source, scored = score(args)
    report(source, scored, args)


if __name__ == "__main__":
    main()
