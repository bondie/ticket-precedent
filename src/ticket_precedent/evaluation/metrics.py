from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

DEPTH = 10


@dataclass(frozen=True)
class Summary:
    mode: str
    queries: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    mrr_ci_low: float
    mrr_ci_high: float
    latency_ms_p50: float


def first_relevant_rank(ranked: Sequence[int], relevant: frozenset[int]) -> int | None:
    for position, ticket_id in enumerate(ranked, start=1):
        if ticket_id in relevant:
            return position
    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank > DEPTH:
        return 0.0
    return 1 / rank


def bootstrap_interval(
    values: Sequence[float], resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean; seeded so reports are reproducible."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)  # noqa: S311 - statistics, not cryptography
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


def summarize(mode: str, ranks: Sequence[int | None], latencies_ms: Sequence[float]) -> Summary:
    total = len(ranks) or 1

    def hit_at(k: int) -> float:
        return sum(1 for rank in ranks if rank is not None and rank <= k) / total

    reciprocal = [reciprocal_rank(rank) for rank in ranks]
    low, high = bootstrap_interval(reciprocal)
    return Summary(
        mode=mode,
        queries=len(ranks),
        hit_at_1=hit_at(1),
        hit_at_5=hit_at(5),
        hit_at_10=hit_at(DEPTH),
        mrr_at_10=sum(reciprocal) / total,
        mrr_ci_low=low,
        mrr_ci_high=high,
        latency_ms_p50=statistics.median(latencies_ms) if latencies_ms else 0.0,
    )


def markdown_table(summaries: Sequence[Summary]) -> str:
    lines = [
        "| Method | Queries | Hit@1 | Hit@5 | Hit@10 | MRR@10 (95% CI) | p50 latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.mode} | {s.queries} | {s.hit_at_1:.3f} | {s.hit_at_5:.3f} | "
            f"{s.hit_at_10:.3f} | {s.mrr_at_10:.3f} ({s.mrr_ci_low:.2f}-{s.mrr_ci_high:.2f}) | "
            f"{s.latency_ms_p50:.0f} ms |"
        )
    return "\n".join(lines)
