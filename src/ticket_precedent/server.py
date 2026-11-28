from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import __version__
from .autosync import start_background_sync
from .config import Config, ConfigError, build_judge, build_searcher
from .effort import ReferenceClass, reference_class, reference_for_ticket
from .judge import judge_outcome
from .ondemand import OnDemandResolver
from .search import (
    Filters,
    LookupArgumentError,
    Outcome,
    TicketNotFoundError,
    describe,
    find_precedents,
    search_or_degrade,
    ticket_details,
    ticket_query,
)
from .store import Store

INSTRUCTIONS = """\
Search an index of past tickets (issue tracker issues and helpdesk threads) for precedents.
When the user names a ticket (a helpdesk number or an issue key), call precedents_for with
its key; a ticket that is not indexed yet is fetched from the configured sources first.
For a symptom or an error message, call similar_tickets with the text. To estimate effort,
call effort_reference: it returns similar finished tickets with their logged hours and the
median and P25-P75 of those hours. Then open the most promising hits with get_ticket (the
key is enough) to read the discussion and how they were resolved. Results combine keyword
(BM25) and semantic (embedding) relevance, so paraphrases match too.
When sources belong to different platforms (see list_sources), results stay in one platform:
a ticket key keeps to its own platform, text uses `platform` or the configured default.
Pass the platform of the ticket at hand, or "all" to search every source.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
FETCHES = ToolAnnotations(read_only_hint=True, open_world_hint=True)
MAX_LIMIT = 20


def instructions(config: Config) -> str:
    """The base instructions plus the configured platforms, so the model knows their names."""
    lines = []
    if config.judge.enabled:
        lines.append(
            f"precedents_for and similar_tickets let a local LLM reorder the top "
            f"{config.judge.top} by whether they report the same fault, which adds 15-30 s; "
            "pass judge=false for a quick look."
        )
    platforms = config.platforms()
    if not platforms:
        return INSTRUCTIONS + "".join(line + "\n" for line in lines)
    lines.append("Configured platforms (pass one of these names as `platform`):")
    for name in platforms:
        sources = ", ".join(item.name for item in config.sources if item.platform == name)
        since = config.effort_since_for(name)
        default_since = f"; effort_reference counts tickets since {since}" if since else ""
        lines.append(f"- {name}: sources {sources}{default_since}")
    if config.default_platform:
        lines.append(f"Default platform for text: {config.default_platform}.")
    return INSTRUCTIONS + "\n".join(lines) + "\n"


def create_server(config: Config) -> MCPServer:
    store = Store(config.index_path)
    searcher = build_searcher(config, store)
    resolver = OnDemandResolver(config, store, searcher.embedder)
    llm_judge = build_judge(config)
    server: MCPServer = MCPServer(
        "ticket-precedent", instructions=instructions(config), version=__version__
    )

    def respond(outcome: Outcome, **extra: Any) -> dict[str, Any]:
        response: dict[str, Any] = {**extra, "results": describe(store, outcome.hits)}
        if outcome.warning:
            response["warning"] = outcome.warning
        return response

    def judging(wanted: bool | None) -> bool:
        return config.judge.enabled if wanted is None else wanted

    def candidates(limit: int, judged: bool) -> int:
        return max(limit, llm_judge.top) if judged else limit

    def respond_judged(text: str, outcome: Outcome, limit: int, **extra: Any) -> dict[str, Any]:
        judged = judge_outcome(llm_judge, store, text, outcome)
        hits = judged.hits[:limit]
        response: dict[str, Any] = {
            **extra,
            "results": describe(store, hits, judge_scores=judged.scores),
        }
        if judged.scores:
            response["judge"] = {"model": llm_judge.model, "judged": len(judged.scores)}
        if judged.warning:
            response["warning"] = judged.warning
        return response

    def in_platform(chosen: str | None) -> dict[str, Any]:
        return {"platform": chosen} if chosen else {}

    def lookup_error(error: TicketNotFoundError, hint: str) -> dict[str, Any]:
        if isinstance(error, LookupArgumentError):
            return {"error": str(error)}
        return {"error": f"{error}. {hint}"}

    @server.tool(annotations=READ_ONLY)
    def similar_tickets(
        text: str | None = None,
        source: str | None = None,
        project: str | None = None,
        platform: str | None = None,
        limit: int = 5,
        query: str | None = None,
        judge: bool | None = None,
    ) -> dict[str, Any]:
        """Find past tickets similar to `text` (a symptom, error message or ticket body).

        `query` is an alias of `text`; pass exactly one of them. Optionally restrict to one
        `source` (see list_sources) or `project`. When sources have a platform, results stay
        in `platform` (the configured default when omitted; "all" searches every source).
        Returns `results` with key, title, url, status, resolution, time spent and the best
        matching snippet, plus a `warning` when only keyword search was available. `judge`
        lets a local LLM reorder the top results by whether they report the same fault and
        adds its 0-3 `judge_score` (slower; the configured default when omitted).
        """
        if (text is None) == (query is None):
            return {"error": "Pass the search text as either `text` or its alias `query`."}
        limit = max(1, min(limit, MAX_LIMIT))
        try:
            chosen, sources = (None, None) if source else config.scope(platform)
        except ConfigError as error:
            return {"error": str(error)}
        filters = Filters(source, project, sources=sources)
        text = text or query or ""
        judged = judging(judge)
        outcome = search_or_degrade(searcher, text, "hybrid", candidates(limit, judged), filters)
        if judged:
            return respond_judged(text, outcome, limit, **in_platform(chosen))
        return respond(outcome, **in_platform(chosen))

    @server.tool(annotations=FETCHES)
    def precedents_for(
        key: str,
        source: str | None = None,
        only_source: str | None = None,
        platform: str | None = None,
        limit: int = 5,
        judge: bool | None = None,
    ) -> dict[str, Any]:
        """Earlier tickets resembling a ticket, e.g. helpdesk thread 18342 or PROJ-42.

        Prefer this to similar_tickets whenever the user names a ticket. A ticket that is not
        indexed yet is fetched from the configured sources and indexed first. The ticket's
        title and description are the query, and only tickets created before it are returned,
        without the ticket itself and tickets its title names. Pass `source` only when the key
        exists in several sources; `only_source` restricts the results to one source. Results
        stay in the ticket's own platform unless `platform` names another one or "all".
        `judge` lets a local LLM reorder the top results by whether they report the same
        fault and adds its 0-3 `judge_score` (slower; the configured default when omitted).
        """
        try:
            ticket = resolver.resolve(key, source)
        except TicketNotFoundError as error:
            return lookup_error(error, "Read it from the tracker and use similar_tickets.")
        try:
            chosen, sources = (
                (None, None) if only_source else config.scope(platform, ticket["source"])
            )
        except ConfigError as error:
            return {"error": str(error)}
        limit = max(1, min(limit, MAX_LIMIT))
        judged = judging(judge)
        outcome = find_precedents(searcher, ticket, candidates(limit, judged), only_source, sources)
        query = {name: ticket[name] for name in ("source", "key", "title", "created")}
        if judged:
            text = ticket_query(ticket["title"], ticket["description"])
            return respond_judged(text, outcome, limit, ticket=query, **in_platform(chosen))
        return respond(outcome, ticket=query, **in_platform(chosen))

    @server.tool(annotations=FETCHES)
    def effort_reference(
        key: str | None = None,
        text: str | None = None,
        source: str | None = None,
        since: str | None = None,
        platform: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Similar finished tickets with logged hours, and the median and P25-P75 of the hours.

        Pass `key` for a ticket (fetched and indexed first when it is not indexed yet): only
        tickets whose time was all logged before it was created count, so the answer is what
        could have been known when it was estimated. Pass `text` for a new ticket. `since`
        (YYYY-MM-DD) keeps only tickets created on or after that date, e.g. after a process
        change made older tickets slower; when omitted, the configured default of the platform
        applies, and the response's `since` says which date was used. The ticket's own
        estimate and time spent are never returned. Tickets of another platform never count:
        a key keeps to its own platform, text to `platform` or the configured default.
        """
        if (key is None) == (text is None):
            return {"error": "Pass either key or text."}
        limit = max(1, min(limit, MAX_LIMIT))
        if text is not None:
            try:
                chosen, sources = config.scope(platform)
            except ConfigError as error:
                return {"error": str(error)}
            used = since or config.effort_since_for(chosen)
            found = reference_class(searcher, text, limit=limit, since=used, sources=sources)
            return reference(found, since=used, **in_platform(chosen))
        try:
            ticket = resolver.resolve(key or "", source)
        except TicketNotFoundError as error:
            return lookup_error(error, "Read it from the tracker and pass its text instead.")
        try:
            chosen, sources = config.scope(platform, ticket["source"])
        except ConfigError as error:
            return {"error": str(error)}
        used = since or config.effort_since_for(chosen)
        found = reference_for_ticket(searcher, ticket, limit, used, sources=sources)
        query = {name: ticket[name] for name in ("source", "key", "title")}
        return reference(found, ticket=query, since=used, **in_platform(chosen))

    def reference(found: ReferenceClass, **extra: Any) -> dict[str, Any]:
        return respond(
            Outcome(found.hits, found.warning),
            **extra,
            as_of=found.as_of,
            reference_class=found.summary(),
        )

    @server.tool(annotations=FETCHES)
    def get_ticket(key: str, source: str | None = None) -> dict[str, Any]:
        """Full text of one ticket: description, comments, links and time spent.

        The key is enough; pass `source` only when the key exists in several sources. A
        ticket that is not indexed yet is fetched from the configured sources and indexed.
        """
        try:
            ticket = resolver.resolve(key, source)
        except TicketNotFoundError as error:
            return lookup_error(error, "Read it from the tracker.")
        details = ticket_details(store, ticket["source"], ticket["key"])
        if details is None:
            return {"error": f"Ticket {key!r} is not indexed"}
        return details

    @server.tool(annotations=READ_ONLY)
    def list_sources() -> list[dict[str, Any]]:
        """Indexed sources with platform, ticket and chunk counts and the last sync time."""
        return [{**row, "platform": config.platform_of(row["source"])} for row in store.stats()]

    start_background_sync(config)
    return server
