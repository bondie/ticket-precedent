"""Reference class forecasting: how long did similar, finished tickets take?"""

from __future__ import annotations

import sqlite3
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .search import (
    Filters,
    Hit,
    Searcher,
    hours,
    same_problem,
    search_or_degrade,
    ticket_query,
)

POOL = 2000


@dataclass(frozen=True)
class ReferenceClass:
    hits: list[Hit]
    hours: list[float]
    as_of: str
    warning: str | None = None

    def summary(self) -> dict[str, Any]:
        if not self.hours:
            return {"tickets": 0, "median_hours": None, "p25_hours": None, "p75_hours": None}
        p25, median, p75 = quartiles(self.hours)
        return {
            "tickets": len(self.hours),
            "median_hours": round(median, 2),
            "p25_hours": round(p25, 2),
            "p75_hours": round(p75, 2),
        }


def quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) == 1:
        return values[0], values[0], values[0]
    p25, median, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return p25, median, p75


def reference_class(
    searcher: Searcher,
    text: str,
    as_of: str | None = None,
    limit: int = 10,
    since: str | None = None,
    source: str | None = None,
    project: str | None = None,
    exclude: frozenset[int] = frozenset(),
    mode: str = "hybrid",
) -> ReferenceClass:
    """The `limit` tickets most similar to `text` whose logged time was final before `as_of`.

    `since` keeps only tickets created on or after a date, for when a process change made
    older tickets a poor guide. `mode` picks the ranking, as in `Searcher.search`.
    """
    as_of = as_of or datetime.now(UTC).replace(microsecond=0).isoformat()
    filters = Filters(
        source=source,
        project=project,
        created_after=since,
        logged_before=as_of,
        exclude_ticket_ids=exclude,
    )
    outcome = search_or_degrade(searcher, text, mode, limit, filters, pool=POOL)
    tickets = searcher.store.tickets_by_id(hit.ticket_id for hit in outcome.hits)
    spent = [hours(tickets[hit.ticket_id]["time_spent_seconds"]) for hit in outcome.hits]
    return ReferenceClass(
        hits=outcome.hits,
        hours=[value for value in spent if value is not None],
        as_of=as_of,
        warning=outcome.warning,
    )


def reference_for_ticket(
    searcher: Searcher,
    ticket: sqlite3.Row,
    limit: int = 10,
    since: str | None = None,
    source: str | None = None,
    mode: str = "hybrid",
) -> ReferenceClass:
    """What was knowable when `ticket` was created: the backtest view of an indexed ticket."""
    return reference_class(
        searcher,
        ticket_query(ticket["title"], ticket["description"]),
        as_of=ticket["created"],
        limit=limit,
        since=since,
        source=source,
        exclude=same_problem(searcher.store, ticket),
        mode=mode,
    )
