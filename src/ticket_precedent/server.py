from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import __version__
from .autosync import start_background_sync
from .config import Config, build_searcher
from .search import (
    Filters,
    Outcome,
    TicketNotFoundError,
    describe,
    find_precedents,
    resolve_ticket,
    search_or_degrade,
    ticket_details,
)
from .store import Store

INSTRUCTIONS = """\
Search an index of past tickets (issue tracker issues and helpdesk threads) for precedents.
When the user names a ticket (a helpdesk number or an issue key), call precedents_for with
its key. For a symptom, an error message or a ticket that is not indexed yet, call
similar_tickets with the text. Then open the most promising hits with get_ticket to read the
discussion and how they were resolved. Results combine keyword (BM25) and semantic
(embedding) relevance, so paraphrases match too.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
MAX_LIMIT = 20


def create_server(config: Config) -> MCPServer:
    store = Store(config.index_path)
    searcher = build_searcher(config, store)
    server: MCPServer = MCPServer(
        "ticket-precedent", instructions=INSTRUCTIONS, version=__version__
    )

    def respond(outcome: Outcome, **extra: Any) -> dict[str, Any]:
        response: dict[str, Any] = {**extra, "results": describe(store, outcome.hits)}
        if outcome.warning:
            response["warning"] = outcome.warning
        return response

    @server.tool(annotations=READ_ONLY)
    def similar_tickets(
        text: str, source: str | None = None, project: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        """Find past tickets similar to `text` (a symptom, error message or ticket body).

        Optionally restrict to one `source` (see list_sources) or `project`.
        Returns `results` with key, title, url, status, resolution, time spent and the best
        matching snippet, plus a `warning` when only keyword search was available.
        """
        limit = max(1, min(limit, MAX_LIMIT))
        return respond(search_or_degrade(searcher, text, "hybrid", limit, Filters(source, project)))

    @server.tool(annotations=READ_ONLY)
    def precedents_for(
        key: str, source: str | None = None, only_source: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        """Earlier tickets resembling an indexed ticket, e.g. helpdesk thread 18342 or PROJ-42.

        Prefer this to similar_tickets whenever the user names a ticket. The ticket's title and
        description are the query, and only tickets created before it are returned, without
        the ticket itself and tickets its title names. Pass `source` only when the key exists
        in several sources; `only_source` restricts the results to one source.
        """
        try:
            ticket = resolve_ticket(store, key, source)
        except TicketNotFoundError as error:
            return {"error": f"{error}. Read it from the tracker and use similar_tickets."}
        outcome = find_precedents(searcher, ticket, max(1, min(limit, MAX_LIMIT)), only_source)
        query = {name: ticket[name] for name in ("source", "key", "title", "created")}
        return respond(outcome, ticket=query)

    @server.tool(annotations=READ_ONLY)
    def get_ticket(source: str, key: str) -> dict[str, Any]:
        """Full indexed text of one ticket: description, comments, links and time spent."""
        details = ticket_details(store, source, key)
        if details is None:
            return {"error": f"Ticket {key!r} not found in source {source!r}"}
        return details

    @server.tool(annotations=READ_ONLY)
    def list_sources() -> list[dict[str, Any]]:
        """Indexed sources with ticket and chunk counts and the last sync time."""
        return store.stats()

    start_background_sync(config)
    return server
