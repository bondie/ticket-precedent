from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import __version__
from .autosync import start_background_sync
from .config import Config, build_searcher
from .search import Filters, describe, search_or_degrade, ticket_details
from .store import Store

INSTRUCTIONS = """\
Search an index of past tickets (issue tracker issues and helpdesk threads) for precedents.
Call similar_tickets with a symptom, an error message or a whole new ticket, then open the
most promising hits with get_ticket to read the discussion and how they were resolved.
Results combine keyword (BM25) and semantic (embedding) relevance, so paraphrases match too.
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def create_server(config: Config) -> MCPServer:
    store = Store(config.index_path)
    searcher = build_searcher(config, store)
    server: MCPServer = MCPServer(
        "ticket-precedent", instructions=INSTRUCTIONS, version=__version__
    )

    @server.tool(annotations=READ_ONLY)
    def similar_tickets(
        text: str, source: str | None = None, project: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        """Find past tickets similar to `text` (a symptom, error message or ticket body).

        Optionally restrict to one `source` (see list_sources) or `project`.
        Returns `results` with key, title, url, status, resolution, time spent and the best
        matching snippet, plus a `warning` when only keyword search was available.
        """
        outcome = search_or_degrade(
            searcher, text, "hybrid", max(1, min(limit, 20)), Filters(source, project)
        )
        response: dict[str, Any] = {"results": describe(store, outcome.hits)}
        if outcome.warning:
            response["warning"] = outcome.warning
        return response

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
