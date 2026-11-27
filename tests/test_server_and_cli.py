from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ticket_precedent import cli, config, server
from ticket_precedent.autosync import sync_lock
from ticket_precedent.config import load_config
from ticket_precedent.connectors.http import get_json
from ticket_precedent.embeddings import EmbeddingError

from .conftest import FakeEmbedder
from .test_search import TICKETS, ListConnector


class OfflineEmbedder(FakeEmbedder):
    def embed_queries(self, texts: Any) -> list[list[float]]:
        raise EmbeddingError("Cannot reach Ollama")

    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("Cannot reach Ollama")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: FakeEmbedder) -> Path:
    monkeypatch.setattr(cli, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(config, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(cli, "build_connector", lambda _source: ListConnector(TICKETS))
    path = tmp_path / "config.yaml"
    path.write_text(
        f"index_path: {tmp_path / 'index.db'}\n"
        "sources:\n"
        "  - {name: demo, type: jira, url: https://jira.example, jql: project = DEMO}\n"
    )
    return path


def test_cli_sync_search_show_and_stats(config_file: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main(["--config", str(config_file), "sync"]) == 0
    assert "demo: 3 tickets" in capsys.readouterr().err

    assert cli.main(["--config", str(config_file), "search", "invoice ERP", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert results[0]["key"] == "D-1"

    assert cli.main(["--config", str(config_file), "search", "invoice ERP"]) == 0
    assert capsys.readouterr().out.startswith("1. [demo] D-1 Invoice from ERP")

    assert cli.main(["--config", str(config_file), "show", "demo", "D-1"]) == 0
    assert json.loads(capsys.readouterr().out)["comments"][0]["author"] == "dev"

    assert cli.main(["--config", str(config_file), "show", "demo", "NOPE"]) == 1

    assert cli.main(["--config", str(config_file), "precedents", "D-2", "--json"]) == 0
    captured = capsys.readouterr()
    assert "precedents for [demo] D-2" in captured.err
    assert [result["key"] for result in json.loads(captured.out)] == ["D-1"]

    assert cli.main(["--config", str(config_file), "precedents", "NOPE"]) == 1
    assert "'NOPE' is not indexed" in capsys.readouterr().err

    assert cli.main(["--config", str(config_file), "stats"]) == 0
    assert "demo: 3 tickets" in capsys.readouterr().out


def test_cli_reports_configuration_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    missing = tmp_path / "missing.yaml"
    assert cli.main(["--config", str(missing), "sync"]) == 1
    assert "No sources configured" in capsys.readouterr().err


def call(mcp: Any, tool: str, **arguments: Any) -> Any:
    result = asyncio.run(mcp.call_tool(tool, arguments))
    assert not result.is_error
    content = result.structured_content
    return content.get("result", content)


def test_mcp_tools(config_file: Path):
    cli.main(["--config", str(config_file), "sync"])
    mcp = server.create_server(load_config(config_file))

    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    assert set(tools) == {
        "similar_tickets",
        "precedents_for",
        "effort_reference",
        "get_ticket",
        "list_sources",
    }
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools.values())

    response = call(mcp, "similar_tickets", text="payment gateway timeout", limit=2)
    assert response["results"][0]["key"] == "D-2"
    assert len(response["results"]) <= 2
    assert "warning" not in response

    response = call(mcp, "precedents_for", key="D-3", limit=5)
    assert response["ticket"]["key"] == "D-3"
    assert {result["key"] for result in response["results"]} <= {"D-1", "D-2"}
    assert "not indexed" in call(mcp, "precedents_for", key="D-9")["error"]

    assert call(mcp, "get_ticket", source="demo", key="D-2")["title"].startswith("Checkout")
    assert "error" in call(mcp, "get_ticket", source="demo", key="X")
    assert call(mcp, "list_sources")[0]["tickets"] == 3


def test_search_degrades_to_keywords_when_embeddings_are_down(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    cli.main(["--config", str(config_file), "sync"])
    offline = OfflineEmbedder()
    monkeypatch.setattr(config, "build_embedder", lambda _config: offline)

    response = call(server.create_server(load_config(config_file)), "similar_tickets", text="ERP")
    assert response["results"][0]["key"] == "D-1"
    assert "keyword matches only" in response["warning"]

    capsys.readouterr()
    assert cli.main(["--config", str(config_file), "search", "invoice ERP"]) == 0
    captured = capsys.readouterr()
    assert "keyword matches only" in captured.err
    assert "D-1" in captured.out


def test_get_json_retries_transient_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ticket_precedent.connectors.http.time.sleep", lambda _s: None)
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = httpx.Client(
        base_url="https://x.example", transport=httpx.MockTransport(lambda _r: next(responses))
    )
    assert get_json(client, "/api") == {"ok": True}


def test_get_json_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ticket_precedent.connectors.http.time.sleep", lambda _s: None)
    client = httpx.Client(
        base_url="https://x.example", transport=httpx.MockTransport(lambda _r: httpx.Response(503))
    )
    with pytest.raises(httpx.HTTPStatusError):
        get_json(client, "/api", attempts=2)


def test_cli_sync_refuses_to_run_while_another_sync_holds_the_lock(
    config_file: Path, capsys: pytest.CaptureFixture[str]
):
    index = load_config(config_file).index_path
    with sync_lock(index) as acquired:
        assert acquired
        assert cli.main(["--config", str(config_file), "sync"]) == 1
    assert "another process is syncing" in capsys.readouterr().err


def test_similar_tickets_takes_query_as_an_alias_of_text(config_file: Path):
    cli.main(["--config", str(config_file), "sync"])
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "similar_tickets", query="payment gateway timeout")
    assert response["results"][0]["key"] == "D-2"

    for arguments in ({}, {"text": "a", "query": "b"}):
        error = call(mcp, "similar_tickets", **arguments)["error"]
        assert "either `text` or its alias `query`" in error


def test_get_ticket_needs_only_the_key(config_file: Path):
    cli.main(["--config", str(config_file), "sync"])
    mcp = server.create_server(load_config(config_file))

    assert call(mcp, "get_ticket", key="D-2")["source"] == "demo"
    assert call(mcp, "get_ticket", key="D-2", source="demo")["title"].startswith("Checkout")
    assert "not configured (demo)" in call(mcp, "get_ticket", key="D-2", source="other")["error"]


def test_cli_show_takes_the_key_alone_or_after_the_source(
    config_file: Path, capsys: pytest.CaptureFixture[str]
):
    cli.main(["--config", str(config_file), "sync"])
    capsys.readouterr()

    assert cli.main(["--config", str(config_file), "show", "D-1"]) == 0
    assert json.loads(capsys.readouterr().out)["key"] == "D-1"

    assert cli.main(["--config", str(config_file), "show", "D-1", "--source", "demo"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "demo"

    assert cli.main(["--config", str(config_file), "show", "demo", "D-1"]) == 0
    assert json.loads(capsys.readouterr().out)["key"] == "D-1"

    assert cli.main(["--config", str(config_file), "show", "demo", "D-1", "--source", "x"]) == 1
    assert "not both" in capsys.readouterr().err

    assert cli.main(["--config", str(config_file), "show", "D-9"]) == 1
    assert "'D-9' is not indexed and none of the sources" in capsys.readouterr().err
