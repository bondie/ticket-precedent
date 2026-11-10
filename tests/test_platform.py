from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ticket_precedent import cli, config, server
from ticket_precedent.config import ConfigError, load_config
from ticket_precedent.models import Ticket

from .conftest import FakeEmbedder, make_ticket
from .test_search import ListConnector
from .test_server_and_cli import call

INVOICE = "Customer says the invoice PDF from the ERP is missing in the shop."
SHOP = "shop-jira"
LEGACY = "legacy-jira"


def logged(ticket: Ticket, when: str, hours: float) -> Ticket:
    return replace(
        ticket,
        time_spent_seconds=int(hours * 3600),
        last_logged=datetime.fromisoformat(when).replace(tzinfo=UTC),
    )


BY_SOURCE = {
    SHOP: [
        logged(
            make_ticket(
                "S-1", "Invoice from ERP never appears", INVOICE, "2024-01-01", source=SHOP
            ),
            "2024-01-05",
            2.0,
        ),
        make_ticket("S-2", "Checkout times out at the gateway", created="2024-02-01", source=SHOP),
    ],
    LEGACY: [
        logged(
            make_ticket(
                "L-1", "ERP invoice missing in legacy", INVOICE, "2024-01-10", source=LEGACY
            ),
            "2024-01-12",
            9.0,
        ),
        make_ticket("L-2", "Invoice from ERP not visible", INVOICE, "2024-06-01", source=LEGACY),
    ],
}


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    embedder = FakeEmbedder()
    monkeypatch.setattr(cli, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(config, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(
        cli, "build_connector", lambda source: ListConnector(BY_SOURCE[source.name])
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        f"index_path: {tmp_path / 'index.db'}\n"
        "default_platform: shop\n"
        "sources:\n"
        "  - {name: shop-jira, type: jira, url: https://jira.example, jql: x, platform: shop}\n"
        "  - {name: legacy-jira, type: jira, url: https://jira.example, jql: y, platform: legacy}\n"
    )
    assert cli.main(["--config", str(path), "sync"]) == 0
    return path


def result_keys(response: dict[str, Any]) -> set[str]:
    return {result["key"] for result in response["results"]}


def test_scope_prefers_explicit_then_ticket_then_default(config_file: Path):
    loaded = load_config(config_file)
    assert loaded.platform_of("legacy-jira") == "legacy"
    assert loaded.scope() == ("shop", frozenset({"shop-jira"}))
    assert loaded.scope(ticket_source="legacy-jira") == ("legacy", frozenset({"legacy-jira"}))
    assert loaded.scope("shop", ticket_source="legacy-jira") == ("shop", frozenset({"shop-jira"}))
    assert loaded.scope("all") == (None, None)
    with pytest.raises(ConfigError, match="Unknown platform 'erp'"):
        loaded.scope("erp")


def test_sources_without_platform_search_everything(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("sources:\n  - {name: demo, type: jira, url: https://jira.example, jql: x}\n")
    assert load_config(path).scope() == (None, None)


def test_platform_all_is_reserved(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("sources:\n  - {name: demo, type: jira, url: u, jql: x, platform: all}\n")
    with pytest.raises(ConfigError, match="reserved"):
        load_config(path)


def test_text_search_stays_in_the_default_or_named_platform(config_file: Path):
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "similar_tickets", text="invoice ERP missing")
    assert response["platform"] == "shop"
    assert result_keys(response) <= {"S-1", "S-2"}

    response = call(mcp, "similar_tickets", text="invoice ERP missing", platform="legacy")
    assert result_keys(response) == {"L-1", "L-2"}

    response = call(mcp, "similar_tickets", text="invoice ERP missing", platform="all")
    assert "platform" not in response
    assert {"S-1", "L-1"} <= result_keys(response)

    assert "Unknown platform" in call(mcp, "similar_tickets", text="x", platform="erp")["error"]


def test_precedents_keep_to_the_tickets_own_platform(config_file: Path):
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "precedents_for", key="L-2")
    assert response["platform"] == "legacy"
    assert result_keys(response) == {"L-1"}

    response = call(mcp, "precedents_for", key="L-2", platform="all")
    assert {"S-1", "L-1"} <= result_keys(response)


def test_effort_reference_ignores_hours_from_another_platform(config_file: Path):
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "effort_reference", key="L-2")
    assert response["platform"] == "legacy"
    assert response["reference_class"]["median_hours"] == 9.0

    response = call(mcp, "effort_reference", text=INVOICE)
    assert response["platform"] == "shop"
    assert response["reference_class"]["median_hours"] == 2.0


def test_list_sources_shows_the_platform(config_file: Path):
    mcp = server.create_server(load_config(config_file))
    platforms = {row["source"]: row["platform"] for row in call(mcp, "list_sources")}
    assert platforms == {"shop-jira": "shop", "legacy-jira": "legacy"}


def test_cli_platform_options(config_file: Path, capsys: pytest.CaptureFixture[str]):
    capsys.readouterr()
    assert (
        cli.main(["--config", str(config_file), "search", "invoice ERP", "--platform", "legacy"])
        == 0
    )
    out = capsys.readouterr().out
    assert "[legacy-jira]" in out
    assert "[shop-jira]" not in out

    assert cli.main(["--config", str(config_file), "precedents", "L-2"]) == 0
    out = capsys.readouterr().out
    assert "L-1" in out
    assert "S-1" not in out

    assert cli.main(["--config", str(config_file), "effort", "L-2", "--platform", "shop"]) == 0
    assert "median 2.0 h" in capsys.readouterr().err

    assert cli.main(["--config", str(config_file), "stats"]) == 0
    assert "legacy-jira [legacy]:" in capsys.readouterr().out
