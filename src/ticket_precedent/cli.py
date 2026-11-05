from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .autosync import sync_lock
from .chunking import ChunkingConfig
from .config import (
    Config,
    ConfigError,
    EmbeddingConfig,
    SourceConfig,
    build_connector,
    build_embedder,
    build_searcher,
    data_dir,
    load_config,
)
from .connectors.jira import JiraConnector
from .effort import reference_for_ticket
from .embeddings import EmbeddingError
from .evaluation.dataset import duplicate_queries
from .evaluation.metrics import markdown_table
from .evaluation.runner import evaluate
from .ingest import SyncStats, sync
from .search import (
    MODES,
    Filters,
    Outcome,
    Searcher,
    TicketNotFoundError,
    describe,
    find_precedents,
    resolve_ticket,
    search_or_degrade,
    ticket_details,
)
from .store import IndexMismatchError, Store

APACHE_JIRA = "https://issues.apache.org/jira"
APACHE_BOTS = frozenset(
    {
        "ASF GitHub Bot",
        "ASF subversion and git services",
        "Hudson",
        "Hadoop QA",
        "Jenkins",
        "GitBox",
    }
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (ConfigError, EmbeddingError, IndexMismatchError, TicketNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tp", description="Find precedents for new tickets.")
    parser.add_argument("--config", type=Path, help="config file (default ~/.config/...)")
    commands = parser.add_subparsers(required=True, metavar="command")

    cmd = commands.add_parser("sync", help="fetch new and updated tickets into the index")
    cmd.add_argument("--source", help="only this source")
    cmd.add_argument("--full", action="store_true", help="refetch everything")
    cmd.set_defaults(handler=_sync)

    cmd = commands.add_parser("search", help="search the index")
    cmd.add_argument("text")
    cmd.add_argument("--source")
    cmd.add_argument("--project")
    cmd.add_argument("--limit", type=int, default=5)
    cmd.add_argument("--mode", choices=MODES, default="hybrid")
    cmd.add_argument("--json", action="store_true")
    cmd.set_defaults(handler=_search)

    cmd = commands.add_parser("precedents", help="earlier tickets that look like an indexed one")
    cmd.add_argument("key", help="ticket key, e.g. PROJ-42 or a helpdesk number")
    cmd.add_argument("--source", help="source of the ticket when the key is ambiguous")
    cmd.add_argument("--only-source", help="only return tickets from this source")
    cmd.add_argument("--limit", type=int, default=5)
    cmd.add_argument("--json", action="store_true")
    cmd.set_defaults(handler=_precedents)

    cmd = commands.add_parser("effort", help="logged hours of similar tickets finished earlier")
    cmd.add_argument("key", help="ticket key, e.g. PROJ-42")
    cmd.add_argument("--source", help="source of the ticket when the key is ambiguous")
    cmd.add_argument("--since", help="only tickets created on or after YYYY-MM-DD")
    cmd.add_argument("--limit", type=int, default=10)
    cmd.add_argument("--json", action="store_true")
    cmd.set_defaults(handler=_effort)

    cmd = commands.add_parser("show", help="print one indexed ticket")
    cmd.add_argument("source")
    cmd.add_argument("key")
    cmd.set_defaults(handler=_show)

    cmd = commands.add_parser("stats", help="indexed sources")
    cmd.set_defaults(handler=_stats)

    cmd = commands.add_parser("serve", help="run the MCP server on stdio")
    cmd.set_defaults(handler=_serve)

    cmd = commands.add_parser("eval", help="benchmark on a public Apache JIRA project")
    cmd.add_argument("project", help="Apache JIRA project key, e.g. KAFKA")
    cmd.add_argument("--since", help="only index tickets created on or after YYYY-MM-DD")
    cmd.add_argument("--model", default="bge-m3", help="Ollama embedding model")
    cmd.add_argument("--query-prefix", default="")
    cmd.add_argument("--document-prefix", default="")
    cmd.add_argument("--no-comments", action="store_true", help="index title + description only")
    cmd.add_argument("--no-sync", action="store_true", help="evaluate the existing index")
    cmd.add_argument("--modes", default=",".join(MODES))
    cmd.add_argument("--max-queries", type=int)
    cmd.add_argument("--out", type=Path, help="write per-query results as JSON")
    cmd.set_defaults(handler=_eval)
    return parser


def _sync(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    sources = [config.source(args.source)] if args.source else config.sources
    if not sources:
        raise ConfigError("No sources configured; see config.example.yaml")
    with sync_lock(config.index_path) as acquired:
        if not acquired:
            print(f"error: another process is syncing {config.index_path}", file=sys.stderr)
            return 1
        _sync_sources(config, sources, full=args.full)
    return 0


def _sync_sources(config: Config, sources: list[SourceConfig], full: bool) -> None:
    store = Store(config.index_path)
    embedder = build_embedder(config.embedding)
    for source in sources:
        print(f"{source.name}: syncing", file=sys.stderr)
        stats = sync(
            store,
            embedder,
            build_connector(source),
            source.name,
            source.chunking,
            full=full,
            progress=_progress(source.name),
        )
        print(
            f"\n{source.name}: {stats.tickets} tickets, {stats.embedded} new chunks",
            file=sys.stderr,
        )


def _search(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = Store(config.index_path)
    searcher = build_searcher(config, store)
    outcome = search_or_degrade(
        searcher, args.text, args.mode, args.limit, Filters(args.source, args.project)
    )
    return _print_outcome(store, outcome, as_json=args.json)


def _precedents(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = Store(config.index_path)
    ticket = resolve_ticket(store, args.key, args.source)
    print(f"precedents for [{ticket['source']}] {ticket['key']} {ticket['title']}", file=sys.stderr)
    outcome = find_precedents(build_searcher(config, store), ticket, args.limit, args.only_source)
    return _print_outcome(store, outcome, as_json=args.json)


def _effort(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = Store(config.index_path)
    ticket = resolve_ticket(store, args.key, args.source)
    found = reference_for_ticket(build_searcher(config, store), ticket, args.limit, args.since)
    summary = found.summary()
    print(
        f"[{ticket['source']}] {ticket['key']} {ticket['title']}: {summary['tickets']} tickets "
        f"finished before {found.as_of[:10]}, median {summary['median_hours']} h, "
        f"P25-P75 {summary['p25_hours']}-{summary['p75_hours']} h",
        file=sys.stderr,
    )
    return _print_outcome(store, Outcome(found.hits, found.warning), as_json=args.json)


def _print_outcome(store: Store, outcome: Outcome, as_json: bool) -> int:
    if outcome.warning:
        print(f"warning: {outcome.warning}", file=sys.stderr)
    results = describe(store, outcome.hits)
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    for position, result in enumerate(results, start=1):
        spent = f" · {result['time_spent_hours']} h" if result["time_spent_hours"] else ""
        print(f"{position}. [{result['source']}] {result['key']} {result['title']}")
        print(f"   {result['status']} {result['resolution']}{spent} · {result['url']}")
        print(f"   {result['snippet'][:160]}".replace("\n", " "))
    return 0


def _show(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    details = ticket_details(Store(config.index_path), args.source, args.key)
    if details is None:
        print(f"{args.source} {args.key}: not indexed", file=sys.stderr)
        return 1
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return 0


def _stats(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"index: {config.index_path}")
    for row in Store(config.index_path).stats():
        print(
            f"{row['source']}: {row['tickets']} tickets, {row['chunks']} chunks, "
            f"updated up to {row['last_updated']}, synced {row['synced_at']}"
        )
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .server import create_server

    create_server(load_config(args.config)).run()
    return 0


def _eval(args: argparse.Namespace) -> int:
    project = args.project.upper()
    source = f"apache-{project.lower()}"
    variant = "nocomments" if args.no_comments else "comments"
    since = f"-since{args.since}" if args.since else ""
    slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
    index = data_dir() / "eval" / f"{source}{since}-{variant}-{slug}.db"
    store = Store(index)
    embedder = build_embedder(
        EmbeddingConfig(
            model=args.model,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
        )
    )
    if not args.no_sync:
        jql = f"project = {project}" + (f' AND created >= "{args.since}"' if args.since else "")
        print(f"{source}: syncing into {index}", file=sys.stderr)
        sync(
            store,
            embedder,
            JiraConnector(source, APACHE_JIRA, jql, page_size=100, changelog=False),
            source,
            ChunkingConfig(index_comments=not args.no_comments, skip_comment_authors=APACHE_BOTS),
            progress=_progress(source),
        )
        print(file=sys.stderr)

    queries = duplicate_queries(store, source)[: args.max_queries]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    print(f"{len(queries)} duplicate queries, modes {modes}", file=sys.stderr)
    summaries, details = evaluate(Searcher(store, embedder), queries, modes, source)
    print(f"\n{project} · {variant} · {args.model}\n")
    print(markdown_table(summaries))
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "project": project,
                    "since": args.since,
                    "variant": variant,
                    "model": args.model,
                    "summaries": [asdict(s) for s in summaries],
                    "queries": [asdict(d) for d in details],
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def _progress(name: str) -> Callable[[SyncStats], None]:
    def report(stats: SyncStats) -> None:
        print(
            f"\r{name}: {stats.tickets} tickets, {stats.embedded} chunks embedded",
            end="",
            file=sys.stderr,
            flush=True,
        )

    return report


if __name__ == "__main__":
    raise SystemExit(main())
