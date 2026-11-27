"""Benchmark search on your own index, using tickets whose titles reference a helpdesk thread.

    uv run python scripts/eval_linked.py --platform ng

Many teams open an issue from a helpdesk thread and name the thread in the title, for example
"[R:18342] Wrong prices". That link is a free relevance label, and it gives two tasks:

- `hd-to-issue`: the helpdesk thread (the customer's words) is the query, and the issues
  opened from it are the answers. Issues are searched whenever they were created: the
  question is "is there already an issue for this thread?".
- `earlier-issue`: an issue (with the thread reference removed from its title) is the query,
  and earlier issues opened from the same thread, or linked to it as duplicates, are the
  answers. Like `precedents_for`, only tickets created before the query are searched and the
  referenced thread itself is left out.

Results name private tickets, so they are written outside the repository by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ticket_precedent.config import build_embedder, data_dir, load_config
from ticket_precedent.evaluation.metrics import first_relevant_rank, markdown_table, summarize
from ticket_precedent.search import KEY_MENTION, MODES, Filters, Searcher, ticket_query
from ticket_precedent.store import Store


@dataclass(frozen=True)
class LinkedQuery:
    task: str
    key: str
    text: str
    created: str | None
    relevant: frozenset[int]
    exclude: frozenset[int]
    overlap: float


def shingles(text: str, size: int = 8) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + size]) for i in range(max(len(words) - size + 1, 0))}


def copied_share(query: str, answers: Sequence[str]) -> float:
    """Share of the query's 8-word shingles that appear verbatim in an answer."""
    own = shingles(query)
    if not own:
        return 0.0
    return max(len(own & shingles(answer)) / len(own) for answer in answers)


def build_queries(
    store: Store, sources: frozenset[str], helpdesk: frozenset[str], reference: re.Pattern[str]
) -> list[LinkedQuery]:
    rows = [row for row in store.db.execute("SELECT * FROM tickets") if row["source"] in sources]
    threads = {row["key"]: row for row in rows if row["source"] in helpdesk}
    issues = [row for row in rows if row["source"] not in helpdesk]
    by_key = {row["key"]: row for row in issues}
    opened_from: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in issues:
        for thread in reference.findall(row["title"]):
            if thread in threads:
                opened_from[thread].append(row)

    queries = []
    for thread_key, opened in opened_from.items():
        thread = threads[thread_key]
        text = ticket_query(thread["title"], thread["description"])
        answers = [ticket_query(r["title"], r["description"]) for r in opened]
        queries.append(
            LinkedQuery(
                "hd-to-issue",
                thread_key,
                text,
                None,
                frozenset(r["id"] for r in opened),
                frozenset({thread["id"]}),
                copied_share(text, answers),
            )
        )

    for row in issues:
        named = reference.findall(row["title"])
        siblings = {r["id"] for t in named for r in opened_from.get(t, [])}
        duplicates = {
            by_key[link["key"]]["id"]
            for link in json.loads(row["links_json"])
            if link["type"] == "Duplicate" and link["key"] in by_key
        }
        earlier = {
            ticket_id
            for ticket_id in (siblings | duplicates) - {row["id"]}
            if (store_created(store, ticket_id) or "") < (row["created"] or "")
        }
        if not earlier:
            continue
        title = KEY_MENTION.sub(" ", reference.sub(" ", row["title"]))
        text = ticket_query(title.strip(), row["description"])
        thread_ids = {threads[t]["id"] for t in named if t in threads}
        queries.append(
            LinkedQuery(
                "earlier-issue",
                row["key"],
                text,
                row["created"],
                frozenset(earlier),
                frozenset({row["id"], *thread_ids}),
                0.0,
            )
        )
    return queries


def store_created(store: Store, ticket_id: int) -> str | None:
    row = store.db.execute("SELECT created FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return row["created"] if row else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--platform", help="evaluate the sources of one platform")
    parser.add_argument(
        "--helpdesk", default="mantis", help="source type of the helpdesk (default mantis)"
    )
    parser.add_argument("--reference", default=r"\[R:(\d+)\]", help="thread reference regex")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--max-queries", type=int, help="per task")
    parser.add_argument("--out", type=Path, default=data_dir() / "eval" / "linked.json")
    args = parser.parse_args()

    config = load_config(args.config)
    _, scoped = config.scope(args.platform)
    sources = scoped or frozenset(source.name for source in config.sources)
    helpdesk = frozenset(s.name for s in config.sources if s.type == args.helpdesk) & sources
    store = Store(config.index_path)
    searcher = Searcher(
        store,
        build_embedder(config.embedding),
        rrf_k=config.search.rrf_k,
        lexical_weight=config.search.lexical_weight,
    )
    queries = build_queries(store, sources, helpdesk, re.compile(args.reference))
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    tasks: dict[str, object] = {}

    for task in ("hd-to-issue", "earlier-issue"):
        selected = [q for q in queries if q.task == task][: args.max_queries]
        if not selected:
            continue
        if any(mode != "bm25" for mode in modes):
            searcher.prime([q.text for q in selected])
        issue_sources = sources - helpdesk if task == "hd-to-issue" else sources
        ranks: dict[str, list[int | None]] = {}
        latencies: dict[str, list[float]] = {}
        for mode in modes:
            ranks[mode] = []
            latencies[mode] = []
            for query in selected:
                filters = Filters(
                    sources=issue_sources,
                    created_before=query.created,
                    exclude_ticket_ids=query.exclude,
                )
                started = time.perf_counter()
                hits = searcher.search(query.text, mode=mode, limit=10, filters=filters)
                latencies[mode].append((time.perf_counter() - started) * 1000)
                ranks[mode].append(first_relevant_rank([h.ticket_id for h in hits], query.relevant))
        summaries = [summarize(mode, ranks[mode], latencies[mode]) for mode in modes]
        copied = sorted(q.overlap for q in selected)
        print(f"\n{task} · {len(selected)} queries")
        print(markdown_table(summaries))
        if task == "hd-to-issue":
            mostly = sum(1 for value in copied if value >= 0.5) / len(copied)
            print(
                f"\nqueries copied at least half verbatim into the issue: {mostly:.0%}; the rest:"
            )
            rest = [i for i, q in enumerate(selected) if q.overlap < 0.5]
            print(
                markdown_table(
                    [
                        summarize(m, [ranks[m][i] for i in rest], [latencies[m][i] for i in rest])
                        for m in modes
                    ]
                )
            )
        tasks[task] = {
            "summaries": [asdict(s) for s in summaries],
            "queries": [
                {"key": q.key, "overlap": q.overlap, **{m: ranks[m][i] for m in modes}}
                for i, q in enumerate(selected)
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"platform": args.platform, "tasks": tasks}, indent=2) + "\n")
    print(f"\nper-query ranks: {args.out}")


if __name__ == "__main__":
    main()
