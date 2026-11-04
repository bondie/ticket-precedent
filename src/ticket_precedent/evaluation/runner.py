from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..search import Filters, Searcher
from .dataset import Query
from .metrics import Summary, first_relevant_rank, summarize


@dataclass(frozen=True)
class QueryResult:
    mode: str
    key: str
    rank: int | None


def evaluate(
    searcher: Searcher,
    queries: Sequence[Query],
    modes: Sequence[str],
    source: str,
    depth: int = 10,
) -> tuple[list[Summary], list[QueryResult]]:
    if any(mode != "bm25" for mode in modes):
        searcher.prime([query.text for query in queries])
    summaries: list[Summary] = []
    details: list[QueryResult] = []
    for mode in modes:
        ranks: list[int | None] = []
        latencies: list[float] = []
        for query in queries:
            filters = Filters(
                source=source,
                created_before=query.created,
                exclude_ticket_ids=frozenset({query.ticket_id}),
            )
            started = time.perf_counter()
            hits = searcher.search(query.text, mode=mode, limit=depth, filters=filters)
            latencies.append((time.perf_counter() - started) * 1000)
            rank = first_relevant_rank([hit.ticket_id for hit in hits], query.relevant)
            ranks.append(rank)
            details.append(QueryResult(mode, query.key, rank))
        summaries.append(summarize(mode, ranks, latencies))
    return summaries, details
