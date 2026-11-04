from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import islice
from typing import TypeVar

from .chunking import ChunkingConfig, chunk_ticket
from .connectors import Connector
from .embeddings import Embedder
from .models import Chunk, Ticket
from .store import Store, chunk_hash

T = TypeVar("T")
SINCE_MARGIN = timedelta(days=1)


@dataclass
class SyncStats:
    tickets: int = 0
    embedded: int = 0


def sync(
    store: Store,
    embedder: Embedder,
    connector: Connector,
    source: str,
    chunking: ChunkingConfig,
    full: bool = False,
    page_size: int = 50,
    progress: Callable[[SyncStats], None] | None = None,
) -> SyncStats:
    _bind(store, embedder)
    previous = store.last_updated(source)
    since = None if full or previous is None else previous - SINCE_MARGIN
    newest = previous
    stats = SyncStats()
    for page in _batched(connector.iter_tickets(since), page_size):
        planned = [(ticket, _unique(chunk_ticket(ticket, chunking))) for ticket in page]
        vectors = _embed_missing(store, embedder, planned)
        for ticket, chunks in planned:
            store.replace_ticket(ticket, chunks, vectors)
            newest = _later(newest, ticket.updated)
        stats.tickets += len(page)
        stats.embedded += len(vectors)
        if progress:
            progress(stats)
    store.mark_synced(source, newest)
    return stats


def _bind(store: Store, embedder: Embedder) -> None:
    if store.has_vectors:
        store.check_embedder(embedder.name)
        return
    store.bind_embedder(embedder.name, len(embedder.embed_documents(["dimension probe"])[0]))


def _embed_missing(
    store: Store, embedder: Embedder, planned: list[tuple[Ticket, list[Chunk]]]
) -> dict[str, list[float]]:
    missing: dict[str, str] = {}
    for ticket, chunks in planned:
        known = store.chunk_hashes(ticket.source, ticket.key)
        for chunk in chunks:
            digest = chunk_hash(chunk)
            if digest not in known:
                missing[digest] = chunk.text
    if not missing:
        return {}
    vectors = embedder.embed_documents(list(missing.values()))
    return dict(zip(missing, vectors, strict=True))


def _unique(chunks: list[Chunk]) -> list[Chunk]:
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        seen.setdefault(chunk_hash(chunk), chunk)
    return list(seen.values())


def _later(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    if current is None or candidate > current:
        return candidate
    return current


def _batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch
