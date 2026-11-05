import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ticket_precedent import cli, server
from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.config import load_config
from ticket_precedent.effort import (
    ReferenceClass,
    quartiles,
    reference_class,
    reference_for_ticket,
)
from ticket_precedent.ingest import sync
from ticket_precedent.models import Ticket
from ticket_precedent.search import Searcher
from ticket_precedent.store import Store

from .conftest import make_ticket
from .test_search import ListConnector
from .test_server_and_cli import call

SLOW_EXPORT = "The nightly price export to the marketplace times out on large catalogues."


def finished(
    key: str,
    created: str,
    logged: str,
    hours: float,
    title: str = "Price export times out",
    project: str = "DEMO",
) -> Ticket:
    ticket = make_ticket(key, title, SLOW_EXPORT, created=created, project=project)
    return replace(
        ticket,
        time_spent_seconds=int(hours * 3600),
        initial_estimate_seconds=1800,
        last_logged=datetime.fromisoformat(logged).replace(tzinfo=UTC),
    )


TICKETS = [
    finished("E-1", "2024-01-01", "2024-01-05", 4),
    finished("E-2", "2024-02-01", "2024-02-03", 6),
    finished("E-3", "2025-03-01", "2025-03-02", 2),
    finished("E-4", "2026-01-10", "2026-01-11", 1),
    finished("E-5", "2026-02-01", "2026-03-20", 8),
    make_ticket("E-6", "Price export times out again", SLOW_EXPORT, created="2026-03-01"),
    finished("E-7", "2024-03-01", "2024-03-02", 3, title="Blurry images", project="OTHER"),
]


@pytest.fixture
def searcher(store, embedder):
    sync(store, embedder, ListConnector(TICKETS), "demo", ChunkingConfig())
    return Searcher(store, embedder)


def keys(store: Store, found: ReferenceClass) -> list[str]:
    tickets = store.tickets_by_id(hit.ticket_id for hit in found.hits)
    return sorted(tickets[hit.ticket_id]["key"] for hit in found.hits)


def test_quartiles():
    assert quartiles([2.0]) == (2.0, 2.0, 2.0)
    assert quartiles([1.0, 2.0, 3.0, 4.0, 5.0]) == (2.0, 3.0, 4.0)


def test_backtest_sees_only_tickets_finished_before_the_target_was_created(searcher, store):
    found = reference_for_ticket(searcher, store.ticket("demo", "E-6"), limit=10)

    assert keys(store, found) == ["E-1", "E-2", "E-3", "E-4", "E-7"]
    assert found.as_of.startswith("2026-03-01")
    assert found.summary() == {
        "tickets": 5,
        "median_hours": 3.0,
        "p25_hours": 2.0,
        "p75_hours": 4.0,
    }


def test_since_limits_the_reference_class_to_recent_tickets(searcher, store):
    found = reference_for_ticket(searcher, store.ticket("demo", "E-6"), since="2026-01-01")
    assert keys(store, found) == ["E-4"]


def test_keyword_ranking_applies_the_same_cut_off(searcher, store):
    found = reference_for_ticket(searcher, store.ticket("demo", "E-6"), mode="bm25")
    assert keys(store, found) == ["E-1", "E-2", "E-3", "E-4", "E-7"]


def test_text_query_uses_everything_finished_by_now(searcher, store):
    found = reference_class(searcher, "price export timeout", limit=3, project="DEMO")
    assert len(found.hits) == 3
    assert "E-6" not in keys(store, found)
    assert found.summary()["tickets"] == 3


def test_empty_reference_class(searcher):
    found = reference_class(searcher, "price export", as_of="2020-01-01T00:00:00+00:00")
    assert found.summary() == {
        "tickets": 0,
        "median_hours": None,
        "p25_hours": None,
        "p75_hours": None,
    }


def test_estimate_fields_survive_the_index(store, searcher):
    row = store.ticket("demo", "E-2")
    assert row["initial_estimate_seconds"] == 1800
    assert row["last_logged"] == "2024-02-03T00:00:00+00:00"


def test_old_index_gains_the_new_columns(tmp_path: Path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE tickets (id INTEGER PRIMARY KEY, source TEXT, key TEXT)")
    db.commit()
    db.close()

    columns = {row["name"] for row in Store(path).db.execute("PRAGMA table_info(tickets)")}

    assert {"issue_type", "initial_estimate_seconds", "last_logged"} <= columns


@pytest.fixture
def config_file(tmp_path, monkeypatch, embedder):
    monkeypatch.setattr(cli, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(cli, "build_connector", lambda _source: ListConnector(TICKETS))
    monkeypatch.setattr("ticket_precedent.config.build_embedder", lambda _config: embedder)
    path = tmp_path / "config.yaml"
    path.write_text(
        f"index_path: {tmp_path / 'index.db'}\n"
        "sources:\n"
        "  - {name: demo, type: jira, url: https://jira.example, jql: project = DEMO}\n"
    )
    cli.main(["--config", str(path), "sync"])
    return path


def test_effort_reference_tool(config_file):
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "effort_reference", key="E-6", since="2026-01-01")
    assert response["ticket"]["key"] == "E-6"
    assert response["reference_class"]["median_hours"] == 1.0
    assert [result["key"] for result in response["results"]] == ["E-4"]

    response = call(mcp, "effort_reference", text="price export times out", limit=2)
    assert response["reference_class"]["tickets"] == 2

    assert "either key or text" in call(mcp, "effort_reference")["error"]
    assert "not indexed" in call(mcp, "effort_reference", key="E-99")["error"]


def test_effort_command(config_file, capsys):
    assert cli.main(["--config", str(config_file), "effort", "E-6", "--json"]) == 0
    captured = capsys.readouterr()
    assert "5 tickets finished before 2026-03-01, median 3.0 h, P25-P75 2.0-4.0 h" in captured.err
    assert {result["key"] for result in json.loads(captured.out)} == {
        "E-1",
        "E-2",
        "E-3",
        "E-4",
        "E-7",
    }
