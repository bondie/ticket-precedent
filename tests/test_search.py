from collections.abc import Iterator, Sequence
from datetime import datetime

import pytest

from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.ingest import sync
from ticket_precedent.models import Ticket
from ticket_precedent.search import Filters, Hit, Searcher, describe
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
