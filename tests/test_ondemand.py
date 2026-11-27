from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from ticket_precedent import cli, config, ondemand, server
from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.config import Config, ConfigError, EmbeddingConfig, SourceConfig, load_config
from ticket_precedent.embeddings import EmbeddingError
from ticket_precedent.ingest import index_ticket, sync
from ticket_precedent.models import Ticket
from ticket_precedent.ondemand import OnDemandResolver
from ticket_precedent.search import AmbiguousTicketError, LookupArgumentError, TicketNotFoundError
from ticket_precedent.store import Store

from .conftest import FakeEmbedder, make_ticket
from .test_search import TICKETS, ListConnector
from .test_server_and_cli import call


class Fetcher:
    def __init__(self, tickets: dict[str, Ticket], error: Exception | None = None) -> None:
        self.tickets = tickets
        self.error = error
        self.asked: list[str] = []

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        return iter(self.tickets.values())

    def fetch_ticket(self, key: str) -> Ticket | None:
        self.asked.append(key)
        if self.error is not None:
            raise self.error
        return self.tickets.get(key)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def source(name: str, jql: str = "", kind: str = "jira") -> SourceConfig:
    return SourceConfig(name, kind, {"url": "https://jira.example", "jql": jql})


SOURCES = [
    source("acme-jira", "project = ACME"),
    source("web-jira", "project in (WEB, SHOP)"),
    source("acme-hd", kind="mantis"),
]
NEW = make_ticket("WEB-5", "Coupon code rejected at checkout", created="2024-02-15", source="x")


@pytest.fixture
def fetchers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    connectors: dict[str, Any] = {item.name: Fetcher({}) for item in SOURCES}

    def build(item: SourceConfig) -> Any:
        connector = connectors[item.name]
        if isinstance(connector, Exception):
            raise connector
        return connector

    monkeypatch.setattr(ondemand, "build_connector", build)
    return connectors


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def resolver(tmp_path: Path, store: Store, embedder: FakeEmbedder, clock: Clock):
    loaded = Config(tmp_path / "index.db", EmbeddingConfig(), list(SOURCES))
    return OnDemandResolver(loaded, store, embedder, clock=clock)


def asked(fetchers: dict[str, Any]) -> dict[str, list[str]]:
    return {name: f.asked for name, f in fetchers.items() if isinstance(f, Fetcher) and f.asked}


def test_an_indexed_ticket_needs_no_request(resolver, store, embedder, fetchers):
    sync(store, embedder, ListConnector(TICKETS), "demo", ChunkingConfig())

    assert resolver.resolve("D-2")["title"].startswith("Checkout")
    assert resolver.resolve("D-2", "demo")["source"] == "demo"
    assert asked(fetchers) == {}


def test_sources_known_to_hold_the_project_are_asked_first(resolver, store, embedder):
    names = [item.name for item in resolver.candidates("WEB-5")]
    assert names == ["web-jira", "acme-jira", "acme-hd"]

    legacy = make_ticket("OLD-1", "Old ticket", project="OLD", source="acme-hd")
    index_ticket(store, embedder, legacy, "acme-hd", ChunkingConfig())
    assert next(item.name for item in resolver.candidates("OLD-2")) == "acme-hd"

    assert [item.name for item in resolver.candidates("18342")] == [s.name for s in SOURCES]
    assert [item.name for item in resolver.candidates("WEB-5", "acme-hd")] == ["acme-hd"]


def test_the_first_source_that_has_it_wins(resolver, store, fetchers):
    fetchers["web-jira"].tickets["WEB-5"] = NEW
    fetchers["acme-jira"].tickets["WEB-5"] = NEW

    row = resolver.resolve("WEB-5")

    assert (row["source"], row["key"]) == ("web-jira", "WEB-5")
    assert asked(fetchers) == {"web-jira": ["WEB-5"]}
    assert store.ticket("web-jira", "WEB-5") is not None
    assert store.chunk_hashes("web-jira", "WEB-5")
    assert store.synced_at("web-jira") is None


def test_the_row_is_found_under_the_key_the_tracker_returned(resolver, fetchers):
    moved = make_ticket("WEB-77", "Moved issue", source="web-jira")
    fetchers["web-jira"].tickets["WEB-5"] = moved

    assert resolver.resolve("WEB-5")["key"] == "WEB-77"


def test_a_miss_is_remembered_for_a_while(resolver, fetchers, clock):
    with pytest.raises(TicketNotFoundError) as first:
        resolver.resolve("ZED-1")
    assert str(first.value) == (
        "Ticket 'ZED-1' is not indexed and none of the sources "
        "(acme-jira, web-jira, acme-hd) has it"
    )
    assert asked(fetchers) == {name: ["ZED-1"] for name in fetchers}

    clock.now += 599
    with pytest.raises(TicketNotFoundError, match="none of the sources"):
        resolver.resolve("ZED-1")
    assert all(len(f.asked) == 1 for f in fetchers.values())

    clock.now += 2
    with pytest.raises(TicketNotFoundError):
        resolver.resolve("ZED-1")
    assert all(len(f.asked) == 2 for f in fetchers.values())


def test_a_source_without_credentials_is_skipped(resolver, fetchers):
    fetchers["web-jira"] = ConfigError("Environment variable WEB_TOKEN is not set (config or .env)")
    fetchers["acme-jira"].tickets["WEB-5"] = NEW

    assert resolver.resolve("WEB-5")["source"] == "acme-jira"

    with pytest.raises(TicketNotFoundError, match=r"web-jira \[Environment variable WEB_TOKEN"):
        resolver.resolve("WEB-6")


def test_a_failed_request_is_skipped_without_secrets_and_not_remembered(resolver, fetchers):
    request = httpx.Request("GET", "https://jira.example/rest/api/2/search?token=hunter2")
    response = httpx.Response(503, request=request)
    error = httpx.HTTPStatusError(
        "Server error for url ...token=hunter2", request=request, response=response
    )
    fetchers["web-jira"].error = error
    fetchers["acme-hd"].error = httpx.ConnectError("hunter2.example unreachable")

    with pytest.raises(TicketNotFoundError) as caught:
        resolver.resolve("WEB-5")
    message = str(caught.value)
    assert "web-jira [HTTP 503]" in message
    assert "acme-hd [ConnectError]" in message
    assert "hunter2" not in message

    with pytest.raises(TicketNotFoundError):
        resolver.resolve("WEB-5")
    assert fetchers["web-jira"].asked == ["WEB-5", "WEB-5"]


def test_a_connector_that_cannot_fetch_one_ticket_is_named(resolver, fetchers):
    fetchers["acme-hd"] = ListConnector([])
    with pytest.raises(TicketNotFoundError, match=r"acme-hd \[cannot fetch a single ticket\]"):
        resolver.resolve("18342")


def test_an_unconfigured_source_is_an_argument_error(resolver, store, embedder, fetchers):
    index_ticket(store, embedder, make_ticket("WEB-5", "Coupon"), "gone", ChunkingConfig())
    assert resolver.resolve("WEB-5", "gone")["source"] == "gone"

    with pytest.raises(LookupArgumentError) as caught:
        resolver.resolve("WEB-6", "jira")
    assert str(caught.value) == (
        "Ticket 'WEB-6' is not indexed in source 'jira', which is not configured "
        "(acme-jira, web-jira, acme-hd); the key alone is enough"
    )
    assert asked(fetchers) == {}


def test_without_sources_nothing_can_fetch(tmp_path, store, embedder):
    empty = Config(tmp_path / "index.db", EmbeddingConfig(), [])
    with pytest.raises(TicketNotFoundError, match="no configured source can fetch it"):
        OnDemandResolver(empty, store, embedder).resolve("WEB-5")


def test_indexing_waits_for_a_write_through_another_connection(resolver, fetchers, tmp_path):
    fetchers["web-jira"].tickets["WEB-5"] = NEW
    other = Store(tmp_path / "index.db")
    started = threading.Event()

    def hold_then_commit() -> None:
        other.db.execute("BEGIN IMMEDIATE")
        started.set()
        time.sleep(0.3)
        other.db.commit()
        other.close()

    thread = threading.Thread(target=hold_then_commit)
    thread.start()
    started.wait()
    try:
        assert resolver.resolve("WEB-5")["source"] == "web-jira"
    finally:
        thread.join()


def test_a_found_ticket_that_cannot_be_embedded_says_so(tmp_path, store, fetchers):
    class Down(FakeEmbedder):
        def embed_documents(self, texts: Any) -> list[list[float]]:
            raise EmbeddingError("Cannot reach Ollama at http://localhost:11434.")

    loaded = Config(tmp_path / "index.db", EmbeddingConfig(), list(SOURCES))
    resolver = OnDemandResolver(loaded, store, Down())
    fetchers["web-jira"].tickets["WEB-5"] = NEW

    for _attempt in range(2):
        with pytest.raises(
            TicketNotFoundError, match="found in source 'web-jira' but could not"
        ) as caught:
            resolver.resolve("WEB-5")
        assert str(caught.value).endswith("http://localhost:11434")
    assert fetchers["web-jira"].asked == ["WEB-5", "WEB-5"]
    assert store.ticket("web-jira", "WEB-5") is None


def test_a_key_in_several_sources_is_ambiguous(resolver, store, embedder):
    for name in ("acme-jira", "web-jira"):
        index_ticket(store, embedder, make_ticket("WEB-5", "Same key"), name, ChunkingConfig())

    with pytest.raises(AmbiguousTicketError, match="exists in acme-jira, web-jira"):
        resolver.resolve("WEB-5")
    assert resolver.resolve("WEB-5", "web-jira")["source"] == "web-jira"


def test_concurrent_lookups_fetch_once(resolver, fetchers):
    class Slow(Fetcher):
        def fetch_ticket(self, key: str) -> Ticket | None:
            time.sleep(0.05)
            return super().fetch_ticket(key)

    fetchers["web-jira"] = Slow({"WEB-5": NEW})
    found: list[str] = []
    threads = [
        threading.Thread(target=lambda: found.append(resolver.resolve("WEB-5")["key"]))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert found == ["WEB-5"] * 8
    assert fetchers["web-jira"].asked == ["WEB-5"]


ACME = "acme-jira"
INDEXED = [replace(ticket, key=ticket.key.replace("D-", "DEMO-")) for ticket in TICKETS]
FETCHED = make_ticket(
    "ACME-30",
    "[DEMO-1] Invoice from ERP missing again",
    "Customer says the invoice PDF is missing although ERP exported it.",
    created="2024-02-15",
    source=ACME,
)


@pytest.fixture
def mcp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    embedder = FakeEmbedder()
    monkeypatch.setattr(cli, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(config, "build_embedder", lambda _config: embedder)
    by_source = {"demo": ListConnector(INDEXED), ACME: Fetcher({"ACME-30": FETCHED})}
    monkeypatch.setattr(cli, "build_connector", lambda item: by_source[item.name])
    monkeypatch.setattr(ondemand, "build_connector", lambda item: by_source[item.name])
    path = tmp_path / "config.yaml"
    path.write_text(
        f"index_path: {tmp_path / 'index.db'}\n"
        "sources:\n"
        "  - {name: demo, type: jira, url: https://jira.example, jql: project = D}\n"
        f"  - {{name: {ACME}, type: jira, url: https://jira.example, jql: project = ACME}}\n"
    )
    assert cli.main(["--config", str(path), "sync", "--source", "demo"]) == 0
    return path


def test_precedents_for_fetches_an_unindexed_ticket_and_keeps_the_cutoff(mcp_config: Path):
    mcp = server.create_server(load_config(mcp_config))

    response = call(mcp, "precedents_for", key="ACME-30")

    assert response["ticket"]["source"] == ACME
    assert response["ticket"]["created"].startswith("2024-02-15")
    keys = {result["key"] for result in response["results"]}
    assert keys == {"DEMO-2"}

    details = call(mcp, "get_ticket", key="ACME-30")
    assert details["source"] == ACME
    assert details["title"] == FETCHED.title


def test_tools_that_fetch_say_so_and_explain_a_miss(mcp_config: Path):
    mcp = server.create_server(load_config(mcp_config))

    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    fetching = {
        name
        for name, tool in tools.items()
        if tool.annotations and tool.annotations.open_world_hint
    }
    assert fetching == {"precedents_for", "effort_reference", "get_ticket"}

    error = call(mcp, "precedents_for", key="ACME-31")["error"]
    assert "'ACME-31' is not indexed and none of the sources" in error
    assert error.endswith("use similar_tickets.")
    assert "pass its text" in call(mcp, "effort_reference", key="ACME-31")["error"]
    assert "not indexed" in call(mcp, "get_ticket", key="ACME-31")["error"]


def test_an_ambiguous_key_lists_its_sources_without_a_tracker_hint(mcp_config: Path):
    loaded = load_config(mcp_config)
    store = Store(loaded.index_path)
    index_ticket(store, FakeEmbedder(), replace(INDEXED[1], source=ACME), ACME, ChunkingConfig())
    store.close()
    mcp = server.create_server(loaded)

    error = call(mcp, "get_ticket", key="DEMO-2")["error"]
    assert error == "Ticket 'DEMO-2' exists in acme-jira, demo; pass the source"
    assert call(mcp, "get_ticket", key="DEMO-2", source="demo")["source"] == "demo"

    error = call(mcp, "precedents_for", key="DEMO-2", source="jira")["error"]
    assert error.endswith("the key alone is enough")


def test_effort_reference_fetches_a_ticket_by_key(mcp_config: Path):
    mcp = server.create_server(load_config(mcp_config))

    response = call(mcp, "effort_reference", key="ACME-30")

    assert response["ticket"]["key"] == "ACME-30"
    assert response["as_of"].startswith("2024-02-15")
    assert response["since"] is None


def test_cli_precedents_fetch_an_unindexed_ticket(mcp_config: Path, capsys):
    capsys.readouterr()
    assert cli.main(["--config", str(mcp_config), "precedents", "ACME-30"]) == 0
    captured = capsys.readouterr()
    assert f"precedents for [{ACME}] ACME-30" in captured.err
    assert "DEMO-2" in captured.out
    assert "DEMO-1" not in captured.out
    assert "DEMO-3" not in captured.out
