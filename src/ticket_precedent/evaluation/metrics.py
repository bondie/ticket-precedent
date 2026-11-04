from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Summary:
    mode: str
    queries: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    latency_ms_p50: float


def first_relevant_rank(ranked: Sequence[int], relevant: frozenset[int]) -> int | None:
    for position, ticket_id in enumerate(ranked, start=1):
        if ticket_id in relevant:
            return position
    return None


def summarize(mode: str, ranks: Sequence[int | None], latencies_ms: Sequence[float]) -> Summary:
    total = len(ranks) or 1

    def hit_at(k: int) -> float:
        return sum(1 for rank in ranks if rank is not None and rank <= k) / total

    reciprocal = sum(1 / rank for rank in ranks if rank is not None and rank <= 10)
    return Summary(
        mode=mode,
        queries=len(ranks),
        hit_at_1=hit_at(1),
        hit_at_5=hit_at(5),
        hit_at_10=hit_at(10),
        mrr_at_10=reciprocal / total,
        latency_ms_p50=statistics.median(latencies_ms) if latencies_ms else 0.0,
    )


def markdown_table(summaries: Sequence[Summary]) -> str:
    lines = [
        "| Method | Queries | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.mode} | {s.queries} | {s.hit_at_1:.3f} | {s.hit_at_5:.3f} | "
            f"{s.hit_at_10:.3f} | {s.mrr_at_10:.3f} | {s.latency_ms_p50:.0f} ms |"
        )
    return "\n".join(lines)
