from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from ticket_precedent import cli, config, server
from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.config import ConfigError, build_judge, load_config
from ticket_precedent.ingest import sync
from ticket_precedent.judge import (
    SCHEMA,
    JudgeError,
    OllamaJudge,
    judge_outcome,
    mask,
    parse_scores,
    reorder,
)
from ticket_precedent.search import Filters, Outcome, Searcher

from .conftest import FakeEmbedder
from .test_search import TICKETS, ListConnector
from .test_server_and_cli import call


class FakeJudge:
    """Scores a candidate 3 when it contains `same`, else 1."""

    model = "fake-judge"
    top = 3

    def __init__(self, same: str = "payment", fail: bool = False) -> None:
        self.same = same
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    def scores(self, query: str, candidates: Sequence[str]) -> list[int]:
        self.calls.append((query, list(candidates)))
        if self.fail:
            raise JudgeError("Cannot reach Ollama at http://localhost:11434")
        return [3 if self.same in text.lower() else 1 for text in candidates]


def test_parse_scores_tolerates_missing_and_malformed_answers():
    content = json.dumps(
        {
            "scores": [
                {"candidate": 2, "score": 3},
                {"candidate": 9, "score": 3},
                {"candidate": 1, "score": 7},
                {"candidate": 3, "score": "high"},
            ]
        }
    )
    assert parse_scores(content, 3) == [3, 3, 0]
    assert parse_scores("not json", 2) == [0, 0]
    assert parse_scores("[]", 2) == [0, 0]


def test_reorder_keeps_the_original_order_for_ties_and_unjudged_items():
    assert reorder(["a", "b", "c", "d", "e"], [1, 3, 1, 3]) == ["b", "d", "a", "c", "e"]


def test_mask_hides_ticket_keys_and_optionally_thread_numbers():
    assert mask("Same as SHOP-412, see [R:18342]") == "Same as [ticket], see [R:18342]"
    assert mask("Same as SHOP-412, see [R:18342]", threads=True) == "Same as [ticket], see [ticket]"


def ollama(handler: Any) -> tuple[OllamaJudge, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response: httpx.Response = handler(request)
        return response

    judge = OllamaJudge(model="gemma4:12b", transport=httpx.MockTransport(record))
    return judge, requests


def test_ollama_judge_asks_for_scores_and_caches_the_answer():
    answer = {"scores": [{"candidate": 1, "score": 1}, {"candidate": 2, "score": 3}]}
    judge, requests = ollama(
        lambda _request: httpx.Response(200, json={"message": {"content": json.dumps(answer)}})
    )

    assert judge.scores("Checkout fails SHOP-9", ["Images blurry", "Payment timeout"]) == [1, 3]
    assert judge.scores("Checkout fails SHOP-9", ["Images blurry", "Payment timeout"]) == [1, 3]

    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "gemma4:12b"
    assert request["format"] == SCHEMA
    assert request["options"]["temperature"] == 0
    prompt = request["messages"][1]["content"]
    assert "SHOP-9" not in prompt
    assert "[2]\nPayment timeout" in prompt


def test_ollama_judge_explains_a_missing_model_and_an_unreachable_server():
    judge, _ = ollama(lambda _request: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(JudgeError, match=r"ollama pull gemma4:12b"):
        judge.scores("query", ["a", "b"])

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    judge, _ = ollama(refuse)
    with pytest.raises(JudgeError, match="Cannot reach Ollama"):
        judge.scores("query", ["a", "b"])


@pytest.fixture
def indexed(store, embedder):
    sync(store, embedder, ListConnector(TICKETS), "demo", ChunkingConfig())
    return store


def test_judge_outcome_moves_the_same_fault_first(indexed, embedder):
    hits = Searcher(indexed, embedder).search("order", limit=3, filters=Filters())
    judge = FakeJudge(same="payment")

    judged = judge_outcome(judge, indexed, "Card payment fails", Outcome(hits))

    first = indexed.tickets_by_id([judged.hits[0].ticket_id])[judged.hits[0].ticket_id]
    assert first["key"] == "D-2"
    assert judged.scores[judged.hits[0].ticket_id] == 3
    assert set(judged.scores) == {hit.ticket_id for hit in hits}
    assert judge.calls[0][0] == "Card payment fails"
    assert judged.warning is None


def test_judge_outcome_keeps_the_search_order_when_the_judge_fails(indexed, embedder):
    hits = Searcher(indexed, embedder).search("order", limit=3, filters=Filters())

    judged = judge_outcome(FakeJudge(fail=True), indexed, "query", Outcome(hits, "keywords only"))

    assert judged.hits == hits
    assert judged.scores == {}
    assert judged.warning is not None
    assert judged.warning.startswith("keywords only Judge unavailable, search order kept.")


def test_judge_outcome_skips_a_single_hit(indexed, embedder):
    hits = Searcher(indexed, embedder).search("order", limit=1, filters=Filters())
    judge = FakeJudge()

    assert judge_outcome(judge, indexed, "query", Outcome(hits)).hits == hits
    assert judge.calls == []


@pytest.fixture
def judge() -> FakeJudge:
    return FakeJudge(same="invoice")


@pytest.fixture
def config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: FakeEmbedder, judge: FakeJudge
) -> Path:
    monkeypatch.setattr(cli, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(config, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(cli, "build_connector", lambda _source: ListConnector(TICKETS))
    monkeypatch.setattr(cli, "build_judge", lambda _config: judge)
    monkeypatch.setattr(server, "build_judge", lambda _config: judge)
    path = tmp_path / "config.yaml"
    path.write_text(
        f"index_path: {tmp_path / 'index.db'}\n"
        "sources:\n"
        "  - {name: demo, type: jira, url: https://jira.example, jql: project = DEMO}\n"
    )
    cli.main(["--config", str(path), "sync"])
    return path


def enable_judge(path: Path) -> None:
    path.write_text(path.read_text() + "judge: {enabled: true, top: 3}\n")


def test_mcp_tools_judge_only_when_asked_or_enabled(config_file: Path, judge: FakeJudge):
    mcp = server.create_server(load_config(config_file))

    plain = call(mcp, "similar_tickets", text="card payment timeout", limit=1)
    assert "judge" not in plain
    assert "judge_score" not in plain["results"][0]
    assert judge.calls == []

    judged = call(mcp, "similar_tickets", text="card payment timeout", limit=1, judge=True)
    assert judged["results"][0]["key"] == "D-1"
    assert judged["results"][0]["judge_score"] == 3
    assert len(judged["results"]) == 1
    assert judged["judge"] == {"model": "fake-judge", "judged": 3}
    assert len(judge.calls[0][1]) == 3

    precedents = call(mcp, "precedents_for", key="D-3", judge=True)
    assert precedents["results"][0]["key"] == "D-1"
    assert judge.calls[-1][0].startswith("Product images are blurry on mobile")

    enable_judge(config_file)
    enabled = server.create_server(load_config(config_file))
    assert call(enabled, "similar_tickets", text="card payment timeout")["judge"]["judged"] == 3
    quick = call(enabled, "similar_tickets", text="card payment timeout", judge=False)
    assert "judge" not in quick
    assert "pass judge=false" in server.instructions(load_config(config_file))


def test_mcp_tool_reports_an_unavailable_judge(config_file: Path, judge: FakeJudge):
    judge.fail = True
    mcp = server.create_server(load_config(config_file))

    response = call(mcp, "similar_tickets", text="card payment timeout", judge=True)

    assert response["results"][0]["key"] == "D-2"
    assert "Judge unavailable" in response["warning"]
    assert "judge" not in response


def test_cli_judge_flag(config_file: Path, capsys: pytest.CaptureFixture[str]):
    capsys.readouterr()
    assert cli.main(["--config", str(config_file), "search", "card payment", "--judge"]) == 0
    captured = capsys.readouterr()
    assert "judge: fake-judge reads the top 3" in captured.err
    assert captured.out.startswith("1. [demo] D-1 Invoice from ERP")
    assert "judge 3/3" in captured.out

    enable_judge(config_file)
    assert cli.main(["--config", str(config_file), "precedents", "D-3", "--json"]) == 0
    assert "judge_score" in json.loads(capsys.readouterr().out)[0]
    assert cli.main(["--config", str(config_file), "search", "card payment", "--no-judge"]) == 0
    assert "judge" not in capsys.readouterr().out


def test_judge_configuration(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("embedding: {url: http://gpu:11434}\njudge: {model: qwen3:8b}\n")
    loaded = load_config(path)
    assert loaded.judge.enabled is False
    built = build_judge(loaded)
    assert (built.model, built.url, built.top) == ("qwen3:8b", "http://gpu:11434", 5)

    path.write_text("judge: {top: 1}\n")
    with pytest.raises(ConfigError, match=r"judge\.top"):
        load_config(path)

    path.write_text("judge: {modle: qwen3:8b}\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)
