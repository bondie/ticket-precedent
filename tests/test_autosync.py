from __future__ import annotations

import fcntl
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ticket_precedent import autosync
from ticket_precedent.autosync import due_sources, run_due_syncs, start_background_sync
from ticket_precedent.config import Config, EmbeddingConfig, SourceConfig
from ticket_precedent.store import Store

from .conftest import FakeEmbedder
from .test_search import TICKETS, ListConnector


def connector_for(source: SourceConfig) -> ListConnector:
    return ListConnector([replace(TICKETS[0], source=source.name)])


def synced(config: Config) -> set[str]:
    store = Store(config.index_path)
    return {row["source"] for row in store.stats()}


def make_config(tmp_path: Path, hours: float | None = 6) -> Config:
    sources = [
        SourceConfig(name, "jira", {"url": "https://jira.example", "jql": "x"})
        for name in ("fresh", "stale", "new")
    ]
    return Config(tmp_path / "index.db", EmbeddingConfig(), sources, auto_sync_hours=hours)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    embedder = FakeEmbedder()
    monkeypatch.setattr(autosync, "build_embedder", lambda _config: embedder)
    monkeypatch.setattr(autosync, "build_connector", connector_for)
    config = make_config(tmp_path)
    store = Store(config.index_path)
    store.mark_synced("fresh", None)
    store.mark_synced("stale", None)
    store.db.execute(
        "UPDATE sync_state SET synced_at = ? WHERE source = 'stale'",
        ((datetime.now(UTC) - timedelta(hours=7)).isoformat(),),
    )
    store.db.commit()
    store.close()
    return config


def test_only_stale_and_never_synced_sources_are_due(config: Config):
    store = Store(config.index_path)
    due = due_sources(store, config.sources, timedelta(hours=6))
    assert [source.name for source in due] == ["stale", "new"]


def test_run_due_syncs_indexes_due_sources(config: Config):
    run_due_syncs(config)
    assert synced(config) == {"stale", "new"}
    assert due_sources(Store(config.index_path), config.sources, timedelta(hours=6)) == []


def test_concurrent_sync_is_skipped(config: Config):
    lock_path = config.index_path.with_name(config.index_path.name + ".sync.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_due_syncs(config)
    assert synced(config) == set()


def test_failing_source_does_not_stop_the_others(config: Config, monkeypatch: pytest.MonkeyPatch):
    def connector(source: SourceConfig) -> ListConnector:
        if source.name == "stale":
            raise RuntimeError("tracker is down")
        return connector_for(source)

    monkeypatch.setattr(autosync, "build_connector", connector)
    run_due_syncs(config)
    assert synced(config) == {"new"}


def test_background_sync_runs_in_a_thread(config: Config):
    thread = start_background_sync(config)
    assert thread is not None
    thread.join(timeout=10)
    assert synced(config) == {"stale", "new"}


def test_background_sync_is_off_by_default(tmp_path: Path):
    assert start_background_sync(make_config(tmp_path, hours=None)) is None
