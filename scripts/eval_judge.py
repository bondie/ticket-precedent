"""Benchmark a local LLM that judges the hybrid top 5, on an existing evaluation index.

    uv run python scripts/eval_judge.py \
        ~/.local/share/ticket-precedent/eval/apache-zookeeper-comments-bge-m3.db \
        --model gemma4:12b --out results/judge-zookeeper-gemma4-12b.json

For every duplicate query the model reads the new report and the five best hybrid candidates
and scores each candidate 0-3 for "reports the same fault". Candidates are shown by title and
description only: their comments can name the duplicate outright. Ticket keys are masked so
the model cannot recall Apache tickets by number. Answers are cached on disk, so an
interrupted run resumes and a finished one is reproduced without calling the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from ticket_precedent.config import EmbeddingConfig, build_embedder, data_dir, load_config
from ticket_precedent.evaluation.dataset import duplicate_queries
from ticket_precedent.evaluation.metrics import (
    first_relevant_rank,
    paired_difference,
    reciprocal_rank,
    summarize,
)
from ticket_precedent.search import Filters, Searcher, ticket_query
from ticket_precedent.store import Store

PROMPT_VERSION = 1
QUERY_CHARS = 1500
CANDIDATE_CHARS = 800
KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
THREAD = re.compile(r"\[R:\d+\]|\b\d{5,}\b")

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


@dataclass(frozen=True)
class Judged:
    key: str
    relevant: frozenset[int]
    candidates: list[int]
    scores: list[int]
    latency_ms: float
    prompt_tokens: int
    output_tokens: int


def mask(text: str, threads: bool = False) -> str:
    masked = KEY.sub("[ticket]", text)
    return THREAD.sub("[ticket]", masked) if threads else masked


def user_prompt(query: str, candidates: Sequence[str], threads: bool = False) -> str:
    parts = [f"NEW REPORT\n{mask(query, threads)}", "EARLIER REPORTS"]
    parts += [
        f"[{number}]\n{mask(text, threads)}" for number, text in enumerate(candidates, start=1)
    ]
    parts.append(f"Return a score for each of the {len(candidates)} earlier reports.")
    return "\n\n".join(parts)


class Judge:
    def __init__(self, model: str, url: str, cache: Path) -> None:
        self.model = model
        self.client = httpx.Client(base_url=url, timeout=600)
        self.cache_path = cache
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                entry = json.loads(line)
                self.cache[entry["id"]] = entry

    def scores(self, prompt: str, count: int) -> tuple[list[int], dict[str, Any]]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "format": SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192},
        }
        entry_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        entry = self.cache.get(entry_id) or self._call(entry_id, request)
        return parse_scores(entry["content"], count), entry

    def _call(self, entry_id: str, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        response = self.client.post("/api/chat", json=request)
        response.raise_for_status()
        body = response.json()
        entry = {
            "id": entry_id,
            "content": body["message"]["content"],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "output_tokens": body.get("eval_count", 0),
        }
        with self.cache_path.open("a") as out:
            out.write(json.dumps(entry) + "\n")
        self.cache[entry_id] = entry
        return entry


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


@dataclass(frozen=True)
class Task:
    key: str
    text: str
    relevant: frozenset[int]
    filters: Filters


def apache_tasks(args: argparse.Namespace) -> tuple[str, Store, Searcher, list[Task]]:
    store = Store(args.index.expanduser())
    source = store.db.execute("SELECT source FROM tickets LIMIT 1").fetchone()["source"]
    embedding_model = (store.meta("embedder") or "ollama:bge-m3").removeprefix("ollama:")
    searcher = Searcher(store, build_embedder(EmbeddingConfig(model=embedding_model)))
    tasks = [
        Task(
            query.key,
            query.text,
            query.relevant,
            Filters(
                source=source,
                created_before=query.created,
                exclude_ticket_ids=frozenset({query.ticket_id}),
            ),
        )
        for query in duplicate_queries(store, source)
    ]
    return source, store, searcher, tasks


def linked_tasks(args: argparse.Namespace) -> tuple[str, Store, Searcher, list[Task]]:
    """Queries of `eval_linked.py` on your own index; see that script for the two tasks."""
    from eval_linked import build_queries

    config = load_config(args.config)
    _, scoped = config.scope(args.platform)
    sources = scoped or frozenset(item.name for item in config.sources)
    helpdesk = frozenset(item.name for item in config.sources if item.type == "mantis") & sources
    store = Store(config.index_path)
    searcher = Searcher(
        store,
        build_embedder(config.embedding),
        rrf_k=config.search.rrf_k,
        lexical_weight=config.search.lexical_weight,
    )
    queries = [
        query
        for query in build_queries(store, sources, helpdesk, re.compile(r"\[R:(\d+)\]"))
        if query.task == args.task and query.overlap < args.copied_below
    ]
    random.Random(7).shuffle(queries)  # noqa: S311
    searched = sources - helpdesk if args.task == "hd-to-issue" else sources
    tasks = [
        Task(
            query.key,
            query.text,
            query.relevant,
            Filters(
                sources=searched, created_before=query.created, exclude_ticket_ids=query.exclude
            ),
        )
        for query in queries
    ]
    return f"linked-{args.task}", store, searcher, tasks


def judge_all(args: argparse.Namespace) -> tuple[str, list[Judged]]:
    source, store, searcher, tasks = linked_tasks(args) if args.task else apache_tasks(args)
    queries = tasks[: args.max_queries]
    searcher.prime([query.text for query in queries])
    slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
    cache = data_dir() / "eval" / "judge-cache" / f"{source}-{slug}.jsonl"
    judge = Judge(args.model, args.ollama, cache)

    judged = []
    for query in queries:
        filters = query.filters
        hits = searcher.hybrid(query.text, filters)[:10]
        tickets = store.tickets_by_id(hit.ticket_id for hit in hits[: args.top])
        texts = [
            ticket_query(row["title"], row["description"], CANDIDATE_CHARS)
            for row in (tickets[hit.ticket_id] for hit in hits[: args.top])
        ]
        prompt = user_prompt(query.text[:QUERY_CHARS], texts, threads=bool(args.task))
        scores, entry = judge.scores(prompt, len(texts))
        judged.append(
            Judged(
                query.key,
                query.relevant,
                [hit.ticket_id for hit in hits],
                scores,
                entry["latency_ms"],
                entry["prompt_tokens"],
                entry["output_tokens"],
            )
        )
        print(f"{query.key}: {scores} in {entry['latency_ms'] / 1000:.1f} s", flush=True)
    return source, judged


def reordered(query: Judged) -> list[int]:
    """Judge scores first, hybrid order breaks ties; positions after the pool stay put."""
    pool = len(query.scores)
    head = sorted(range(pool), key=lambda i: (-query.scores[i], i))
    return [query.candidates[i] for i in head] + query.candidates[pool:]


def oracle(query: Judged) -> list[int]:
    """A perfect judge moves the first relevant candidate in the pool to the top."""
    pool = len(query.scores)
    head = sorted(range(pool), key=lambda i: (query.candidates[i] not in query.relevant, i))
    return [query.candidates[i] for i in head] + query.candidates[pool:]


def compare(name: str, baseline: list[int | None], variant: list[int | None]) -> str:
    old = [reciprocal_rank(rank) for rank in baseline]
    new = [reciprocal_rank(rank) for rank in variant]
    mean, low, high = paired_difference(old, new)
    better = sum(1 for a, b in zip(old, new, strict=True) if b > a)
    worse = sum(1 for a, b in zip(old, new, strict=True) if b < a)
    return f"| {name} | {mean:+.3f} ({low:+.3f} to {high:+.3f}) | {better} | {worse} |"


def report(source: str, judged: list[Judged], args: argparse.Namespace) -> None:
    orders = {
        "hybrid": [q.candidates for q in judged],
        f"judge top {args.top}": [reordered(q) for q in judged],
        f"oracle top {args.top}": [oracle(q) for q in judged],
    }
    ranks = {
        name: [
            first_relevant_rank(order, q.relevant) for order, q in zip(lists, judged, strict=True)
        ]
        for name, lists in orders.items()
    }
    summaries = [summarize(name, r, []) for name, r in ranks.items()]
    latency = statistics.median(q.latency_ms for q in judged)
    prompt_tokens = statistics.median(q.prompt_tokens for q in judged)
    output_tokens = statistics.median(q.output_tokens for q in judged)

    print(f"\n{source} · {len(judged)} queries · {args.model} · prompt v{PROMPT_VERSION}")
    print(
        f"judge p50 {latency / 1000:.1f} s per query, {prompt_tokens:.0f} prompt and "
        f"{output_tokens:.0f} output tokens\n"
    )
    print("| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 |\n|---|---:|---:|---:|---:|")
    for s in summaries:
        print(
            f"| {s.mode} | {s.hit_at_1:.3f} | {s.hit_at_5:.3f} | {s.hit_at_10:.3f} | "
            f"{s.mrr_at_10:.3f} |"
        )
    print("\n| Method | ΔMRR@10 vs hybrid (95% CI) | Better | Worse |\n|---|---:|---:|---:|")
    for name in list(ranks)[1:]:
        print(compare(name, ranks["hybrid"], ranks[name]))

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "source": source,
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "top": args.top,
                    "judge_ms_p50": latency,
                    "prompt_tokens_p50": prompt_tokens,
                    "output_tokens_p50": output_tokens,
                    "summaries": [
                        {k: v for k, v in asdict(s).items() if k != "latency_ms_p50"}
                        for s in summaries
                    ],
                    "queries": [
                        {
                            "key": q.key,
                            "scores": q.scores,
                            **{name: r[i] for name, r in ranks.items()},
                        }
                        for i, q in enumerate(judged)
                    ],
                },
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("index", type=Path, nargs="?", help="an evaluation index (Apache)")
    parser.add_argument("--task", choices=["earlier-issue", "hd-to-issue"], help="own index")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--platform")
    parser.add_argument(
        "--copied-below", type=float, default=1.01, help="skip queries copied this much or more"
    )
    parser.add_argument("--model", default="gemma4:12b", help="Ollama chat model")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--top", type=int, default=5, help="candidates the judge reads")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source, judged = judge_all(args)
    report(source, judged, args)


if __name__ == "__main__":
    main()
