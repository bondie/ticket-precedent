from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from .models import Chunk, Ticket
from .text import clean

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    key TEXT NOT NULL,
    project TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    url TEXT NOT NULL,
    status TEXT NOT NULL,
    resolution TEXT NOT NULL,
    created TEXT,
    updated TEXT,
    time_spent_seconds INTEGER,
    comments_json TEXT NOT NULL,
    links_json TEXT NOT NULL,
    UNIQUE (source, key)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    ticket_id INTEGER NOT NULL REFERENCES tickets (id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_ticket ON chunks (ticket_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab (chunks_fts, row);
CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT PRIMARY KEY,
    last_updated TEXT,
    synced_at TEXT NOT NULL
);
"""


class IndexMismatchError(RuntimeError):
    pass


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).replace(microsecond=0).isoformat()


def chunk_hash(chunk: Chunk) -> str:
    return hashlib.blake2b(f"{chunk.kind}\0{chunk.text}".encode(), digest_size=16).hexdigest()


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA busy_timeout = 30000")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @property
    def has_vectors(self) -> bool:
        return self.meta("embedder") is not None

    def bind_embedder(self, name: str, dimension: int) -> None:
        existing = self.meta("embedder")
        if existing == name:
            return
        if existing is not None:
            raise IndexMismatchError(
                f"Index {self.path} was built with {existing}, not {name}. "
                "Use another index_path or delete the index to rebuild it."
            )
        with self.db:
            self._set_meta("embedder", name)
            self._set_meta("dimension", str(dimension))
            self.db.execute(
                f"CREATE VIRTUAL TABLE chunks_vec USING vec0 ("
                f"embedding float[{dimension}] distance_metric=cosine, "
                f"source text, project text)"
            )

    def check_embedder(self, name: str) -> None:
        existing = self.meta("embedder")
        if existing is None or existing == name:
            return
        raise IndexMismatchError(f"Index {self.path} was built with {existing}, not {name}.")

    def chunk_hashes(self, source: str, key: str) -> set[str]:
        rows = self.db.execute(
            "SELECT c.hash FROM chunks c JOIN tickets t ON t.id = c.ticket_id "
            "WHERE t.source = ? AND t.key = ?",
            (source, key),
        )
        return {row["hash"] for row in rows}

    def replace_ticket(
        self,
        ticket: Ticket,
        chunks: Sequence[Chunk],
        vectors: Mapping[str, Sequence[float]],
    ) -> None:
        with self.db:
            ticket_id = self._upsert_ticket(ticket)
            old = {
                row["hash"]: row["id"]
                for row in self.db.execute(
                    "SELECT id, hash FROM chunks WHERE ticket_id = ?", (ticket_id,)
                )
            }
            new = {chunk_hash(chunk): chunk for chunk in chunks}
            for digest, chunk_id in old.items():
                if digest not in new:
                    self._delete_chunk(chunk_id)
            for digest, chunk in new.items():
                if digest in old:
                    continue
                self._insert_chunk(ticket_id, ticket, chunk, digest, vectors[digest])

    def _upsert_ticket(self, ticket: Ticket) -> int:
        row = self.db.execute(
            """
            INSERT INTO tickets (source, key, project, title, description, url, status,
                resolution, created, updated, time_spent_seconds, comments_json, links_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, key) DO UPDATE SET
                project = excluded.project, title = excluded.title,
                description = excluded.description, url = excluded.url,
                status = excluded.status, resolution = excluded.resolution,
                created = excluded.created, updated = excluded.updated,
                time_spent_seconds = excluded.time_spent_seconds,
                comments_json = excluded.comments_json, links_json = excluded.links_json
            RETURNING id
            """,
            (
                ticket.source,
                ticket.key,
                ticket.project,
                ticket.title,
                clean(ticket.description),
                ticket.url,
                ticket.status,
                ticket.resolution,
                iso(ticket.created),
                iso(ticket.updated),
                ticket.time_spent_seconds,
                json.dumps(
                    [
                        {"author": c.author, "created": iso(c.created), "body": clean(c.body)}
                        for c in ticket.comments
                    ],
                    ensure_ascii=False,
                ),
                json.dumps([asdict(link) for link in ticket.links], ensure_ascii=False),
            ),
        ).fetchone()
        return int(row["id"])

    def _insert_chunk(
        self, ticket_id: int, ticket: Ticket, chunk: Chunk, digest: str, vector: Sequence[float]
    ) -> None:
        chunk_id = self.db.execute(
            "INSERT INTO chunks (ticket_id, kind, ordinal, text, hash) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, chunk.kind, chunk.ordinal, chunk.text, digest),
        ).lastrowid
        self.db.execute(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (chunk_id, chunk.text)
        )
        self.db.execute(
            "INSERT INTO chunks_vec (rowid, embedding, source, project) VALUES (?, ?, ?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(list(vector)), ticket.source, ticket.project),
        )

    def _delete_chunk(self, chunk_id: int) -> None:
        self.db.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        self.db.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
        self.db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    def last_updated(self, source: str) -> datetime | None:
        row = self.db.execute(
            "SELECT last_updated FROM sync_state WHERE source = ?", (source,)
        ).fetchone()
        if row is None or row["last_updated"] is None:
            return None
        return datetime.fromisoformat(row["last_updated"])

    def synced_at(self, source: str) -> datetime | None:
        row = self.db.execute(
            "SELECT synced_at FROM sync_state WHERE source = ?", (source,)
        ).fetchone()
        return datetime.fromisoformat(row["synced_at"]) if row else None

    def mark_synced(self, source: str, last_updated: datetime | None) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO sync_state (source, last_updated, synced_at) VALUES (?, ?, ?) "
                "ON CONFLICT (source) DO UPDATE SET "
                "last_updated = excluded.last_updated, synced_at = excluded.synced_at",
                (source, iso(last_updated), iso(datetime.now(UTC))),
            )

    def tickets_by_id(self, ids: Iterable[int]) -> dict[int, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT * FROM tickets WHERE id IN ({placeholders})", ids)
        return {row["id"]: row for row in rows}

    def chunks_by_id(self, ids: Iterable[int]) -> dict[int, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids)
        return {row["id"]: row for row in rows}

    def ticket(self, source: str, key: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM tickets WHERE source = ? AND key = ?", (source, key)
        ).fetchone()
        return row

    def tickets_by_key(self, keys: Iterable[str]) -> list[sqlite3.Row]:
        keys = list(dict.fromkeys(keys))
        if not keys:
            return []
        placeholders = ",".join("?" * len(keys))
        return self.db.execute(
            f"SELECT * FROM tickets WHERE key IN ({placeholders}) ORDER BY source", keys
        ).fetchall()

    def tickets(self, source: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM tickets WHERE source = ? ORDER BY created", (source,)
        ).fetchall()

    def document_frequencies(self, terms: Sequence[str]) -> dict[str, int]:
        if not terms:
            return {}
        placeholders = ",".join("?" * len(terms))
        rows = self.db.execute(
            f"SELECT term, doc FROM chunks_vocab WHERE term IN ({placeholders})", list(terms)
        )
        return {row["term"]: row["doc"] for row in rows}

    def stats(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT t.source, COUNT(DISTINCT t.id) AS tickets, COUNT(c.id) AS chunks,
                   s.last_updated, s.synced_at
            FROM tickets t
            LEFT JOIN chunks c ON c.ticket_id = t.id
            LEFT JOIN sync_state s ON s.source = t.source
            GROUP BY t.source
            ORDER BY t.source
            """
        )
        return [dict(row) for row in rows]
