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


def test_platform_names_match_regardless_of_case(config_file: Path):
    loaded = load_config(config_file)
    assert loaded.scope("SHOP") == ("shop", frozenset({"shop-jira"}))
    assert loaded.scope("ALL") == (None, None)

    mcp = server.create_server(loaded)
    response = call(mcp, "similar_tickets", text="invoice ERP missing", platform="Legacy")
    assert response["platform"] == "legacy"
    assert result_keys(response) == {"L-1", "L-2"}

    error = call(mcp, "similar_tickets", text="x", platform="webshop")["error"]
    assert error == "Unknown platform 'webshop'; configured: ['legacy', 'shop']"


def test_server_instructions_name_the_platforms(config_file: Path):
    with_since = config_file.read_text() + "effort_since: {Legacy: 2024-02-01}\n"
    config_file.write_text(with_since)
    text = server.instructions(load_config(config_file))

    assert "- legacy: sources legacy-jira; effort_reference counts tickets since 2024-02-01" in text
    assert "- shop: sources shop-jira\n" in text
    assert "Default platform for text: shop." in text

    no_platforms = replace(load_config(config_file), sources=[])
    assert server.instructions(no_platforms) == server.INSTRUCTIONS


def test_effort_since_defaults_per_platform_and_yields_to_an_explicit_since(
    config_file: Path, capsys: pytest.CaptureFixture[str]
):
    config_file.write_text(config_file.read_text() + "effort_since:\n  LEGACY: 2024-02-01\n")
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "effort_reference", key="L-2")
    assert response["since"] == "2024-02-01"
    assert response["reference_class"]["tickets"] == 0

    response = call(mcp, "effort_reference", key="L-2", since="2024-01-01")
    assert response["since"] == "2024-01-01"
    assert response["reference_class"]["median_hours"] == 9.0

    response = call(mcp, "effort_reference", text=INVOICE)
    assert response["platform"] == "shop"
    assert response["since"] is None
    assert response["reference_class"]["median_hours"] == 2.0

    response = call(mcp, "effort_reference", text=INVOICE, platform="legacy")
    assert response["since"] == "2024-02-01"

    capsys.readouterr()
    assert cli.main(["--config", str(config_file), "effort", "L-2"]) == 0
    assert "0 tickets created since 2024-02-01, finished before 2024-06-01" in (
        capsys.readouterr().err
    )
    assert cli.main(["--config", str(config_file), "effort", "L-2", "--since", "2024-01-01"]) == 0
    assert "created since 2024-01-01, finished before 2024-06-01, median 9.0 h" in (
        capsys.readouterr().err
    )


def test_one_effort_since_for_every_platform(config_file: Path):
    config_file.write_text(config_file.read_text() + "effort_since: '2024-01-05'\n")
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "effort_reference", text=INVOICE)
    assert response["since"] == "2024-01-05"
    assert response["reference_class"]["tickets"] == 0

    response = call(mcp, "effort_reference", text=INVOICE, platform="all")
    assert response["since"] == "2024-01-05"
    assert response["reference_class"]["median_hours"] == 9.0
