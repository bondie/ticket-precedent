from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import httpx


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "bge-m3",
        url: str = "http://localhost:11434",
        batch_size: int = 32,
        max_input_chars: int = 6000,
        query_prefix: str = "",
        document_prefix: str = "",
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.url = url
        self.batch_size = batch_size
        self.max_input_chars = max_input_chars
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._client = httpx.Client(base_url=url, timeout=timeout)

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + text for text in texts])

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([self.query_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [
                text[: self.max_input_chars] for text in texts[start : start + self.batch_size]
            ]
            vectors.extend(normalize(v) for v in self._request(batch))
        return vectors

    def _request(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch, "truncate": True}
        try:
            response = self._client.post("/api/embed", json=payload)
        except httpx.ConnectError as error:
            raise EmbeddingError(
                f"Cannot reach Ollama at {self.url}. Start it with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`."
            ) from error
        if response.status_code == 404:
            raise EmbeddingError(
                f"Ollama model {self.model!r} not found: `ollama pull {self.model}`."
            )
        response.raise_for_status()
        embeddings: list[list[float]] = response.json()["embeddings"]
        return embeddings
