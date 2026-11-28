"""An optional local LLM that reads the best hits and moves those reporting the same fault first.

The prompt is the one `scripts/eval_judge.py` benchmarks: changing it changes the measured
gain, so bump `PROMPT_VERSION` and rerun the benchmark when you do.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import httpx

from .search import Hit, Outcome, ticket_query
from .store import Store

PROMPT_VERSION = 1
QUERY_CHARS = 1500
CANDIDATE_CHARS = 800
KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
THREAD = re.compile(r"\[R:\d+\]|\b\d{5,}\b")
CACHE_ENTRIES = 256

SYSTEM = """You triage bug reports in an issue tracker.
A new report arrives together with earlier reports found by search. For each earlier report,
decide whether it describes the same underlying fault as the new report, so that the new
report would be closed as its duplicate. Reports about the same component, feature or topic
that describe a different fault are not duplicates.

Score every candidate:
3 = same fault, a duplicate
2 = probably the same fault
1 = related topic, different fault
0 = unrelated"""

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "integer"},
                    "score": {"type": "integer", "enum": [0, 1, 2, 3]},
                },
                "required": ["candidate", "score"],
            },
        }
    },
    "required": ["scores"],
}

T = TypeVar("T")


class JudgeError(RuntimeError):
    pass


class Judge(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def top(self) -> int: ...

    def scores(self, query: str, candidates: Sequence[str]) -> list[int]: ...


def mask(text: str, threads: bool = False) -> str:
    """Ticket keys are hidden so the model judges the text, not a number it may remember."""
    masked = KEY.sub("[ticket]", text)
    return THREAD.sub("[ticket]", masked) if threads else masked


def user_prompt(query: str, candidates: Sequence[str], threads: bool = False) -> str:
    parts = [f"NEW REPORT\n{mask(query, threads)}", "EARLIER REPORTS"]
    parts += [
        f"[{number}]\n{mask(text, threads)}" for number, text in enumerate(candidates, start=1)
    ]
    parts.append(f"Return a score for each of the {len(candidates)} earlier reports.")
    return "\n\n".join(parts)


def chat_request(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192},
    }


def request_id(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def parse_scores(content: str, count: int) -> list[int]:
    """Scores by candidate position; a missing or malformed score counts as 0."""
    scores = [0] * count
    try:
        items = json.loads(content).get("scores", [])
    except (json.JSONDecodeError, AttributeError):
        return scores
    for item in items:
        number, score = item.get("candidate"), item.get("score")
        if isinstance(number, int) and isinstance(score, int) and 1 <= number <= count:
            scores[number - 1] = max(0, min(3, score))
    return scores


def reorder(items: Sequence[T], scores: Sequence[int]) -> list[T]:
    """Judge scores first, the original order breaks ties; items after the judged ones stay."""
    pool = len(scores)
    head = sorted(range(pool), key=lambda i: (-scores[i], i))
    return [items[i] for i in head] + list(items[pool:])


class OllamaJudge:
    def __init__(
        self,
        model: str = "gemma4:12b",
        url: str = "http://localhost:11434",
        top: int = 5,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._top = top
        self.url = url
        self._client = httpx.Client(base_url=url, timeout=timeout, transport=transport)
        self._cache: dict[str, str] = {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def top(self) -> int:
        return self._top

    def scores(self, query: str, candidates: Sequence[str]) -> list[int]:
        request = chat_request(self.model, user_prompt(query[:QUERY_CHARS], candidates))
        key = request_id(request)
        content = self._cache.get(key)
        if content is None:
            content = self._chat(request)
            if len(self._cache) >= CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = content
        return parse_scores(content, len(candidates))

    def _chat(self, request: dict[str, Any]) -> str:
        try:
            response = self._client.post("/api/chat", json=request)
        except httpx.ConnectError as error:
            raise JudgeError(f"Cannot reach Ollama at {self.url}") from error
        except httpx.TimeoutException as error:
            raise JudgeError(f"Ollama did not answer in time ({self.model})") from error
        if response.status_code == 404:
            raise JudgeError(f"Ollama model {self.model!r} not found: `ollama pull {self.model}`")
        if response.is_error:
            raise JudgeError(f"Ollama answered {response.status_code}: {response.text[:200]}")
        content: str = response.json()["message"]["content"]
        return content


@dataclass(frozen=True)
class Judged:
    hits: list[Hit]
    scores: dict[int, int] = field(default_factory=dict)
    """Judge score by ticket id, for the judged hits only."""
    warning: str | None = None


def judge_outcome(judge: Judge, store: Store, query: str, outcome: Outcome) -> Judged:
    """Reorder the top hits by the judge's verdicts; on failure keep the search order."""
    pool = outcome.hits[: judge.top]
    if len(pool) < 2:
        return Judged(outcome.hits, warning=outcome.warning)
    tickets = store.tickets_by_id(hit.ticket_id for hit in pool)
    rows = [tickets[hit.ticket_id] for hit in pool]
    texts = [ticket_query(row["title"], row["description"], CANDIDATE_CHARS) for row in rows]
    try:
        scores = judge.scores(query, texts)
    except JudgeError as error:
        warning = f"Judge unavailable, search order kept. {error}."
        return Judged(outcome.hits, warning=" ".join(filter(None, [outcome.warning, warning])))
    by_ticket = {hit.ticket_id: score for hit, score in zip(pool, scores, strict=True)}
    return Judged(reorder(outcome.hits, scores), by_ticket, outcome.warning)
