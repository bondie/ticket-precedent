from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.evaluation.dataset import duplicate_queries
from ticket_precedent.evaluation.metrics import (
    bootstrap_interval,
    first_relevant_rank,
    markdown_table,
    summarize,
)
from ticket_precedent.evaluation.runner import evaluate
from ticket_precedent.ingest import sync
from ticket_precedent.search import Searcher

from .conftest import make_ticket
from .test_search import ListConnector

CLUSTER = [
    make_ticket(
        "A-1", "Broker loses leadership after restart", "Leader election stalls.", "2020-01-01"
    ),
    make_ticket(
        "A-2",
        "Leader election stalls when broker restarts",
        "After a restart the broker never regains leadership.",
        "2021-01-01",
        links=[("Duplicate", "A-1")],
        resolution="Duplicate",
    ),
    make_ticket(
        "A-3",
        "Restarted broker cannot become leader",
        "Same as before, leadership is lost.",
        "2022-01-01",
        links=[("Duplicate", "A-2")],
        resolution="Duplicate",
    ),
    make_ticket(
        "A-4",
        "Early report that was closed as duplicate of a later one",
        "",
        "2019-01-01",
        links=[("Duplicate", "B-9")],
        resolution="Duplicate",
    ),
    make_ticket("B-9", "Unrelated later ticket", "", "2023-01-01", links=[("Relates", "A-1")]),
]


def test_queries_use_only_earlier_tickets_of_the_duplicate_cluster(store, embedder):
    sync(store, embedder, ListConnector(CLUSTER), "demo", ChunkingConfig())
    ids = {row["key"]: row["id"] for row in store.tickets("demo")}

    queries = {q.key: q for q in duplicate_queries(store, "demo")}

    assert set(queries) == {"A-2", "A-3"}
    assert queries["A-2"].relevant == {ids["A-1"]}
    assert queries["A-3"].relevant == {ids["A-1"], ids["A-2"]}
    assert queries["A-2"].text.startswith("Leader election stalls when broker restarts")


def test_metrics():
    assert first_relevant_rank([5, 7, 9], frozenset({9})) == 3
    assert first_relevant_rank([5, 7], frozenset({9})) is None
    summary = summarize("x", [1, 3, None, 12], [10.0, 20.0, 30.0, 40.0])
    assert summary.hit_at_1 == 0.25
    assert summary.hit_at_5 == 0.5
    assert summary.hit_at_10 == 0.5
    assert summary.mrr_at_10 == (1 + 1 / 3) / 4
    assert summary.latency_ms_p50 == 25.0


def test_evaluate_never_returns_the_query_or_later_tickets(store, embedder):
    sync(store, embedder, ListConnector(CLUSTER), "demo", ChunkingConfig())
    queries = duplicate_queries(store, "demo")
    summaries, details = evaluate(Searcher(store, embedder), queries, ["bm25", "hybrid"], "demo")
    assert [s.mode for s in summaries] == ["bm25", "hybrid"]
    assert all(s.hit_at_5 == 1.0 for s in summaries)
    assert {d.key for d in details} == {"A-2", "A-3"}


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    values = [1.0, 0.5, 0.0, 0.0, 1.0, 0.25, 0.0, 1.0]
    low, high = bootstrap_interval(values)
    assert low < sum(values) / len(values) < high
    assert bootstrap_interval(values) == (low, high)
    assert bootstrap_interval([]) == (0.0, 0.0)


def test_markdown_table_shows_the_interval():
    table = markdown_table([summarize("hybrid", [1, 2, None], [5.0])])
    assert "| hybrid | 3 |" in table
    assert "0.500 (" in table
