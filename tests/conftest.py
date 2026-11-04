from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ticket_precedent.embeddings import normalize
from ticket_precedent.models import Comment, Link, Ticket
from ticket_precedent.store import Store
from ticket_precedent.text import tokens

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Hashed bag of words: deterministic, offline, good enough to rank shared vocabulary."""

    name = "fake:bow"
    dimension = 64

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self.dimension] += 1.0
        return normalize(vector)


def make_ticket(
    key: str,
    title: str,
    description: str = "",
    created: str = "2024-01-01",
    comments: Sequence[str] = (),
    links: Sequence[tuple[str, str]] = (),
    resolution: str = "",
    project: str = "DEMO",
    source: str = "demo",
) -> Ticket:
    when = datetime.fromisoformat(created).replace(tzinfo=UTC)
    return Ticket(
        source=source,
        key=key,
        project=project,
        title=title,
        description=description,
        url=f"https://tracker.example/{key}",
        resolution=resolution,
        created=when,
        updated=when,
        comments=[Comment("dev", when, body) for body in comments],
        links=[Link(kind, other) for kind, other in links],
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    store = Store(tmp_path / "index.db")
    yield store
    store.close()
