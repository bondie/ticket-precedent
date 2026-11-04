from collections.abc import Iterator, Sequence
from datetime import datetime

import pytest

from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.ingest import sync
from ticket_precedent.models import Ticket
from ticket_precedent.search import (
    Filters,
    Hit,
    Searcher,
    TicketNotFoundError,
    describe,
    find_precedents,
    resolve_ticket,
)
from ticket_precedent.store import IndexMismatchError, Store

from .conftest import make_ticket

TICKETS = [
    make_ticket(
        "D-1",
        "Invoice from ERP never appears in the shop",
        "Customer says the invoice PDF is missing although ERP exported it.",
        created="2024-01-01",
        comments=["Comparator treated the payload as unchanged; resetting it fixed the sync."],
    ),
    make_ticket(
        "D-2",
        "Checkout crashes with payment gateway timeout",
        "Card payments time out after 30 seconds and the order stays pending.",
        created="2024-02-01",
    ),
    make_ticket(
        "D-3",
        "Product images are blurry on mobile",
        "Thumbnails are scaled up from tiny sources.",
        created="2024-03-01",
        project="OTHER",
    ),
]


class ListConnector:
    def __init__(self, tickets: Sequence[Ticket]) -> None:
        self.tickets = tickets

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        return iter(self.tickets)


def keys(store: Store, hits: Sequence[Hit]) -> list[str]:
    return [result["key"] for result in describe(store, hits)]


@pytest.fixture
def indexed(store, embedder):
    sync(store, embedder, ListConnector(TICKETS), "demo", ChunkingConfig())
    return store


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_every_mode_finds_the_matching_ticket(indexed, embedder, mode):
    hits = Searcher(indexed, embedder).search("invoice missing from ERP", mode=mode, limit=3)
    assert keys(indexed, hits)[0] == "D-1"


def test_comment_text_is_searchable(indexed, embedder):
    hits = Searcher(indexed, embedder).search("comparator unchanged payload", mode="bm25")
    result = describe(indexed, hits)[0]
    assert result["key"] == "D-1"
    assert result["matched_in"] == "comment"


def test_filter_by_project(indexed, embedder):
    hits = Searcher(indexed, embedder).search(
        "payment invoice images", filters=Filters(project="OTHER")
    )
    assert keys(indexed, hits) == ["D-3"]


def test_filter_by_creation_date(indexed, embedder):
    hits = Searcher(indexed, embedder).search(
        "checkout payment timeout", filters=Filters(created_before="2024-01-15")
    )
    assert "D-2" not in keys(indexed, hits)


def test_excluded_tickets_are_skipped(indexed, embedder):
    d2 = indexed.ticket("demo", "D-2")["id"]
    hits = Searcher(indexed, embedder).search(
        "checkout payment timeout", filters=Filters(exclude_ticket_ids=frozenset({d2}))
    )
    assert "D-2" not in keys(indexed, hits)


def test_resync_embeds_only_changed_chunks(store, embedder):
    sync(store, embedder, ListConnector(TICKETS), "demo", ChunkingConfig())
    embedder.calls.clear()

    changed = make_ticket(
        "D-2", TICKETS[1].title, "Now it times out after 60 seconds.", created="2024-02-01"
    )
    stats = sync(store, embedder, ListConnector([TICKETS[0], changed]), "demo", ChunkingConfig())

    assert stats.embedded == 1
    assert embedder.calls == [[f"{changed.title}\n\nNow it times out after 60 seconds."]]
    searcher = Searcher(store, embedder)
    assert keys(store, searcher.search("60 seconds", mode="bm25"))[0] == "D-2"
    assert keys(store, searcher.search("pending", mode="bm25")) == []


def test_index_refuses_a_different_embedding_model(indexed, embedder):
    embedder.name = "fake:other"
    with pytest.raises(IndexMismatchError):
        Searcher(indexed, embedder)


def test_long_queries_keep_the_rarest_known_terms(indexed, embedder):
    searcher = Searcher(indexed, embedder, max_query_terms=3)
    filler = " ".join(f"filler{i}" for i in range(50))
    hits = searcher.search(f"{filler} the shop invoice comparator", mode="bm25")
    assert keys(indexed, hits)[0] == "D-1"


PRICES = "The order header has the wholesale price list but the items carry retail prices."


@pytest.fixture
def two_trackers(store, embedder):
    jira = [
        make_ticket("P-1", "Order items priced from the retail list", PRICES, created="2023-02-01"),
        make_ticket("P-2", "[R:200359] Wrong prices on orders", PRICES, created="2024-05-02"),
        make_ticket("P-3", "Wrong prices on orders again", PRICES, created="2025-01-01"),
        make_ticket("P-4", "Blurry product images", "Upscaled thumbnails.", created="2023-06-01"),
    ]
    helpdesk = [
        make_ticket("200359", "Wrong prices on orders", PRICES, created="2024-05-01", source="hd")
    ]
    sync(store, embedder, ListConnector(jira), "demo", ChunkingConfig())
    sync(store, embedder, ListConnector(helpdesk), "hd", ChunkingConfig())
    return store


def precedents(store: Store, embedder, key: str) -> list[str]:
    ticket = resolve_ticket(store, key)
    return keys(store, find_precedents(Searcher(store, embedder), ticket, limit=10).hits)


def test_precedents_are_older_tickets_other_than_the_query(two_trackers, embedder):
    found = precedents(two_trackers, embedder, "200359")
    assert found[0] == "P-1"
    assert {"200359", "P-2", "P-3"}.isdisjoint(found)


def test_precedents_skip_tickets_named_in_the_title(two_trackers, embedder):
    found = precedents(two_trackers, embedder, "P-2")
    assert found[0] == "P-1"
    assert "200359" not in found


def test_resolve_ticket_needs_a_source_only_for_ambiguous_keys(two_trackers):
    assert resolve_ticket(two_trackers, "200359")["source"] == "hd"
    assert resolve_ticket(two_trackers, "P-1", "demo")["key"] == "P-1"
    other = make_ticket("P-1", "Same key elsewhere", source="hd")
    two_trackers.replace_ticket(other, [], {})
    with pytest.raises(TicketNotFoundError, match="demo, hd"):
        resolve_ticket(two_trackers, "P-1")
    with pytest.raises(TicketNotFoundError, match="not indexed"):
        resolve_ticket(two_trackers, "NOPE")
    with pytest.raises(TicketNotFoundError, match="source 'hd'"):
        resolve_ticket(two_trackers, "P-4", "hd")


def test_title_only_match_shows_a_preview_instead_of_an_empty_snippet(store, embedder):
    ticket = make_ticket("E-1", "Comgate hotfix", comments=["Gateway returned PAID twice " * 3])
    sync(store, embedder, ListConnector([ticket]), "demo", ChunkingConfig())
    body = store.db.execute("SELECT id, ticket_id FROM chunks WHERE kind = 'body'").fetchone()
    (result,) = describe(store, [Hit(body["ticket_id"], body["id"], 1.0)])
    assert result["matched_in"] == "title"
    assert result["snippet"].startswith("Gateway returned PAID twice")
