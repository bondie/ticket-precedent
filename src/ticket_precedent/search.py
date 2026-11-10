from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import sqlite_vec

from .embeddings import Embedder, EmbeddingError
from .store import Store
from .text import clean, tokens

MODES = ("bm25", "vector", "hybrid")
QUERY_CHARS = 2000
KEY_MENTION = re.compile(r"\b[A-Z][A-Z0-9_]+-\d+\b|\b\d{4,}\b")
TICKET_CONDITIONS = """
    (:source IS NULL OR t.source = :source)
    AND (:sources IS NULL OR t.source IN (SELECT value FROM json_each(:sources)))
    AND (:project IS NULL OR t.project = :project)
    AND (:before IS NULL OR t.created < :before)
    AND (:after IS NULL OR t.created >= :after)
    AND (:logged_before IS NULL OR (
        t.time_spent_seconds > 0 AND COALESCE(t.last_logged, t.updated) < :logged_before
    ))
"""


@dataclass(frozen=True)
class Filters:
    source: str | None = None
    project: str | None = None
    sources: frozenset[str] | None = None
    """Only tickets from these sources, e.g. the sources of one platform."""
    created_before: str | None = None
    created_after: str | None = None
    logged_before: str | None = None
    """Only tickets with logged time whose last worklog came before this moment."""
    exclude_ticket_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Hit:
    ticket_id: int
    chunk_id: int
    score: float


@dataclass(frozen=True)
class Outcome:
    hits: list[Hit]
    warning: str | None = None


class Searcher:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        candidate_chunks: int = 200,
        max_query_terms: int = 64,
        rrf_k: int = 10,
        lexical_weight: float = 0.25,
    ) -> None:
        store.check_embedder(embedder.name)
        self.store = store
        self.embedder = embedder
        self.candidate_chunks = candidate_chunks
        self.max_query_terms = max_query_terms
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self._query_vectors: dict[str, list[float]] = {}

    def prime(self, texts: Sequence[str]) -> None:
        missing = [text for text in dict.fromkeys(texts) if text not in self._query_vectors]
        if not missing:
            return
        vectors = self.embedder.embed_queries(missing)
        self._query_vectors.update(zip(missing, vectors, strict=True))

    def search(
        self,
        text: str,
        mode: str = "hybrid",
        limit: int = 5,
        filters: Filters | None = None,
        pool: int | None = None,
    ) -> list[Hit]:
        """`pool` widens the candidate chunks for filters that discard most of them."""
        filters = filters or Filters()
        pool = pool or self.candidate_chunks
        if mode == "bm25":
            return self.lexical(text, filters, pool)[:limit]
        if mode == "vector":
            return self.semantic(text, filters, pool)[:limit]
        if mode == "hybrid":
            return self.hybrid(text, filters, pool)[:limit]
        raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")

    def hybrid(self, text: str, filters: Filters, pool: int | None = None) -> list[Hit]:
        rankings = [
            (1.0, self.semantic(text, filters, pool)),
            (self.lexical_weight, self.lexical(text, filters, pool)),
        ]
        scores: dict[int, float] = {}
        best_chunk: dict[int, int] = {}
        for weight, ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                contribution = weight / (self.rrf_k + rank)
                scores[hit.ticket_id] = scores.get(hit.ticket_id, 0.0) + contribution
                best_chunk.setdefault(hit.ticket_id, hit.chunk_id)
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        return [Hit(tid, best_chunk[tid], scores[tid]) for tid in ordered]

    def lexical(self, text: str, filters: Filters, pool: int | None = None) -> list[Hit]:
        terms = self._query_terms(text)
        if not terms:
            return []
        rows = self.store.db.execute(
            f"""
            SELECT c.id AS chunk_id, c.ticket_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN tickets t ON t.id = c.ticket_id
            WHERE chunks_fts MATCH :query AND {TICKET_CONDITIONS}
            ORDER BY score
            LIMIT :limit
            """,
            {
                **_ticket_parameters(filters),
                "query": " OR ".join(f'"{term}"' for term in terms),
                "limit": pool or self.candidate_chunks,
            },
        )
        return _best_per_ticket(
            (Hit(row["ticket_id"], row["chunk_id"], -row["score"]) for row in rows),
            filters.exclude_ticket_ids,
        )

    def semantic(self, text: str, filters: Filters, pool: int | None = None) -> list[Hit]:
        if not self.store.has_vectors:
            return []
        vector = self._query_vectors.get(text) or self.embedder.embed_query(text)
        conditions = ["embedding MATCH :vector", "k = :k"]
        if filters.source is not None:
            conditions.append("source = :source")
        elif filters.sources is not None and len(filters.sources) == 1:
            conditions.append("source = :only_source")
        if filters.project is not None:
            conditions.append("project = :project")
        rows = self.store.db.execute(
            f"""
            WITH knn AS (
                SELECT rowid, distance FROM chunks_vec WHERE {" AND ".join(conditions)}
            )
            SELECT knn.rowid AS chunk_id, c.ticket_id, knn.distance
            FROM knn
            JOIN chunks c ON c.id = knn.rowid
            JOIN tickets t ON t.id = c.ticket_id
            WHERE {TICKET_CONDITIONS}
            ORDER BY knn.distance
            """,
            {
                **_ticket_parameters(filters),
                "only_source": next(iter(filters.sources)) if filters.sources else None,
                "vector": sqlite_vec.serialize_float32(vector),
                "k": pool or self.candidate_chunks,
            },
        )
        return _best_per_ticket(
            (Hit(row["ticket_id"], row["chunk_id"], 1 - row["distance"]) for row in rows),
            filters.exclude_ticket_ids,
        )

    def _query_terms(self, text: str) -> list[str]:
        terms = tokens(text)
        if len(terms) <= self.max_query_terms:
            return terms
        frequencies = self.store.document_frequencies(terms)
        known = [term for term in terms if frequencies.get(term)]
        return sorted(known, key=frequencies.__getitem__)[: self.max_query_terms]


def _ticket_parameters(filters: Filters) -> dict[str, str | None]:
    return {
        "source": filters.source,
        "sources": None if filters.sources is None else json.dumps(sorted(filters.sources)),
        "project": filters.project,
        "before": filters.created_before,
        "after": filters.created_after,
        "logged_before": filters.logged_before,
    }


def search_or_degrade(
    searcher: Searcher,
    text: str,
    mode: str,
    limit: int,
    filters: Filters,
    pool: int | None = None,
) -> Outcome:
    """Fall back to keyword search when the embedding service is down instead of failing."""
    try:
        return Outcome(searcher.search(text, mode=mode, limit=limit, filters=filters, pool=pool))
    except EmbeddingError as error:
        hits = searcher.search(text, mode="bm25", limit=limit, filters=filters, pool=pool)
        return Outcome(hits, f"Semantic search unavailable, keyword matches only. {error}")


class TicketNotFoundError(LookupError):
    pass


def resolve_ticket(store: Store, key: str, source: str | None = None) -> sqlite3.Row:
    """Find an indexed ticket by key, in `source` or in the only source that has the key."""
    if source is not None:
        row = store.ticket(source, key)
        if row is None:
            raise TicketNotFoundError(f"Ticket {key!r} is not indexed in source {source!r}")
        return row
    rows = store.tickets_by_key([key])
    if not rows:
        raise TicketNotFoundError(f"Ticket {key!r} is not indexed")
    if len(rows) > 1:
        sources = ", ".join(row["source"] for row in rows)
        raise TicketNotFoundError(f"Ticket {key!r} exists in {sources}; pass the source")
    return rows[0]


def ticket_query(title: str, description: str | None, max_chars: int = QUERY_CHARS) -> str:
    """What the reporter wrote before anyone answered; the benchmark queries the same way."""
    return f"{title}\n\n{description or ''}".strip()[:max_chars]


def find_precedents(
    searcher: Searcher,
    ticket: sqlite3.Row,
    limit: int,
    source: str | None = None,
    sources: frozenset[str] | None = None,
) -> Outcome:
    """Tickets created before `ticket` that look like it.

    Tickets named in its title are left out: a JIRA issue titled "[R:18342] ..." was created
    from helpdesk thread 18342, so that thread is the same problem, not a precedent.
    """
    filters = Filters(
        source=source,
        sources=sources,
        created_before=ticket["created"],
        exclude_ticket_ids=same_problem(searcher.store, ticket),
    )
    text = ticket_query(ticket["title"], ticket["description"])
    return search_or_degrade(searcher, text, "hybrid", limit, filters)


def same_problem(store: Store, ticket: sqlite3.Row) -> frozenset[int]:
    """The ticket itself and the tickets its title names."""
    named = store.tickets_by_key(KEY_MENTION.findall(ticket["title"]))
    return frozenset({ticket["id"], *(row["id"] for row in named)})


def _best_per_ticket(hits: Iterable[Hit], exclude: frozenset[int]) -> list[Hit]:
    seen: dict[int, Hit] = {}
    for hit in hits:
        if hit.ticket_id in exclude or hit.ticket_id in seen:
            continue
        seen[hit.ticket_id] = hit
    return list(seen.values())


def describe(store: Store, hits: Sequence[Hit], snippet_chars: int = 300) -> list[dict[str, Any]]:
    tickets = store.tickets_by_id(hit.ticket_id for hit in hits)
    chunks = store.chunks_by_id(hit.chunk_id for hit in hits)
    results = []
    for hit in hits:
        ticket = tickets[hit.ticket_id]
        chunk = chunks[hit.chunk_id]
        snippet = chunk["text"].removeprefix(ticket["title"]).strip()
        matched_in = chunk["kind"]
        if not snippet:
            matched_in = "title"
            snippet = _preview(ticket)
        results.append(
            {
                "source": ticket["source"],
                "key": ticket["key"],
                "title": ticket["title"],
                "url": ticket["url"],
                "project": ticket["project"],
                "status": ticket["status"],
                "resolution": ticket["resolution"],
                "created": (ticket["created"] or "")[:10],
                "issue_type": ticket["issue_type"],
                "time_spent_hours": hours(ticket["time_spent_seconds"]),
                "score": round(hit.score, 4),
                "matched_in": matched_in,
                "snippet": _truncate(snippet, snippet_chars),
            }
        )
    return results


def ticket_details(store: Store, source: str, key: str) -> dict[str, Any] | None:
    ticket = store.ticket(source, key)
    if ticket is None:
        return None
    return {
        "source": ticket["source"],
        "key": ticket["key"],
        "title": ticket["title"],
        "url": ticket["url"],
        "project": ticket["project"],
        "status": ticket["status"],
        "resolution": ticket["resolution"],
        "created": ticket["created"],
        "updated": ticket["updated"],
        "time_spent_hours": hours(ticket["time_spent_seconds"]),
        "description": ticket["description"],
        "comments": json.loads(ticket["comments_json"]),
        "links": json.loads(ticket["links_json"]),
    }


def _preview(ticket: sqlite3.Row) -> str:
    description = clean(ticket["description"] or "")
    if description:
        return description
    comments = json.loads(ticket["comments_json"])
    if not comments:
        return ""
    return clean(comments[0]["body"])


def hours(seconds: int | None) -> float | None:
    if not seconds:
        return None
    return round(seconds / 3600, 2)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
