"""Fetch and index a ticket when it is asked for before any sync brought it in."""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping

import httpx

from .config import Config, ConfigError, SourceConfig, build_connector
from .connectors import TicketFetcher
from .connectors.http import RateLimitError
from .connectors.jira import KEY as JIRA_KEY
from .embeddings import Embedder, EmbeddingError
from .ingest import index_ticket
from .models import Ticket
from .search import LookupArgumentError, TicketNotFoundError, ambiguous
from .store import Store

MISS_TTL = 600.0


class OnDemandResolver:
    """Finds a ticket in the index, or fetches it from the sources that may have it.

    Misses are remembered for `miss_ttl` seconds, so an agent asking again for a key no
    source covers does not cost another round of requests. One lock serialises fetching and
    writing, so parallel tool calls asking for the same new key fetch it once.
    """

    def __init__(
        self,
        config: Config,
        store: Store,
        embedder: Embedder,
        miss_ttl: float = MISS_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.miss_ttl = miss_ttl
        self._clock = clock
        self._misses: dict[tuple[str | None, str], tuple[float, str]] = {}
        self._lock = threading.Lock()

    def resolve(self, key: str, source: str | None = None) -> sqlite3.Row:
        row = self._indexed(key, source)
        if row is not None:
            return row
        configured = [item.name for item in self.config.sources]
        if source is not None and source not in configured:
            raise LookupArgumentError(
                f"Ticket {key!r} is not indexed in source {source!r}, which is not configured "
                f"({', '.join(configured) or 'no sources'}); the key alone is enough"
            )
        with self._lock:
            row = self._indexed(key, source)
            if row is not None:
                return row
            expires, message = self._misses.get((source, key), (0.0, ""))
            if expires > self._clock():
                raise TicketNotFoundError(message)
            return self._fetch(key, source)

    def _indexed(self, key: str, source: str | None) -> sqlite3.Row | None:
        if source is not None:
            return self.store.ticket(source, key)
        rows = self.store.tickets_by_key([key])
        if len(rows) > 1:
            raise ambiguous(key, rows)
        return rows[0] if rows else None

    def _fetch(self, key: str, source: str | None) -> sqlite3.Row:
        tried: list[str] = []
        transient = False
        for candidate in self.candidates(key, source):
            try:
                connector = build_connector(candidate)
            except (ConfigError, TypeError) as error:
                tried.append(f"{candidate.name} [{error}]")
                continue
            if not isinstance(connector, TicketFetcher):
                tried.append(f"{candidate.name} [cannot fetch a single ticket]")
                continue
            try:
                ticket = connector.fetch_ticket(key)
            except (httpx.HTTPError, RateLimitError) as error:
                tried.append(f"{candidate.name} [{_reason(error)}]")
                transient = True
                continue
            if ticket is None:
                tried.append(candidate.name)
                continue
            return self._index(key, candidate, ticket)
        message = _not_found(key, tried)
        if not transient:
            self._misses[(source, key)] = (self._clock() + self.miss_ttl, message)
        raise TicketNotFoundError(message)

    def candidates(self, key: str, source: str | None = None) -> list[SourceConfig]:
        """Sources that may have `key`, those known to hold its JIRA project first."""
        if source is not None:
            return [item for item in self.config.sources if item.name == source]
        if not JIRA_KEY.fullmatch(key):
            return list(self.config.sources)
        prefix = key.rsplit("-", 1)[0]
        projects = self.store.projects_by_source()
        return sorted(self.config.sources, key=lambda item: not _covers(item, prefix, projects))

    def _index(self, key: str, source: SourceConfig, ticket: Ticket) -> sqlite3.Row:
        try:
            index_ticket(self.store, self.embedder, ticket, source.name, source.chunking)
        except (EmbeddingError, httpx.HTTPError) as error:
            reason = str(error).rstrip(".") if isinstance(error, EmbeddingError) else _reason(error)
            raise TicketNotFoundError(
                f"Ticket {key!r} was found in source {source.name!r} but could not be "
                f"embedded, so it is not indexed: {reason}"
            ) from error
        row = self.store.ticket(source.name, ticket.key)
        if row is None:
            raise TicketNotFoundError(f"Ticket {key!r} was fetched but not stored")
        return row


def _covers(source: SourceConfig, prefix: str, projects: Mapping[str, set[str]]) -> bool:
    if prefix in projects.get(source.name, set()):
        return True
    jql = str(source.options.get("jql") or "")
    return re.search(rf"\b{re.escape(prefix)}\b", jql, re.IGNORECASE) is not None


def _reason(error: Exception) -> str:
    """Short and without the request URL, which may carry more than an agent should see."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    if isinstance(error, RateLimitError):
        return "rate limit exhausted"
    return type(error).__name__


def _not_found(key: str, tried: list[str]) -> str:
    if not tried:
        return f"Ticket {key!r} is not indexed and no configured source can fetch it"
    return f"Ticket {key!r} is not indexed and none of the sources ({', '.join(tried)}) has it"
