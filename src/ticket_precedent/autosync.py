from __future__ import annotations

import fcntl
import logging
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import Config, SourceConfig, build_connector, build_embedder
from .ingest import sync
from .store import Store

logger = logging.getLogger(__name__)


def due_sources(
    store: Store, sources: Sequence[SourceConfig], max_age: timedelta
) -> list[SourceConfig]:
    now = datetime.now(UTC)
    return [
        source
        for source in sources
        if (synced := store.synced_at(source.name)) is None or now - synced >= max_age
    ]


@contextmanager
def sync_lock(index_path: Path) -> Iterator[bool]:
    """Yield True when this process may write to the index, False if another one is syncing."""
    lock_path = index_path.with_name(index_path.name + ".sync.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def run_due_syncs(config: Config) -> None:
    if not config.auto_sync_hours:
        return
    with sync_lock(config.index_path) as acquired:
        if not acquired:
            logger.info("another process is already syncing %s", config.index_path)
            return
        _sync_due(config, timedelta(hours=config.auto_sync_hours))


def _sync_due(config: Config, max_age: timedelta) -> None:
    store = Store(config.index_path)
    try:
        embedder = build_embedder(config.embedding)
        for source in due_sources(store, config.sources, max_age):
            try:
                stats = sync(store, embedder, build_connector(source), source.name, source.chunking)
            except Exception:
                logger.exception("auto-sync of %s failed", source.name)
                continue
            logger.info(
                "auto-sync %s: %d tickets, %d chunks", source.name, stats.tickets, stats.embedded
            )
    finally:
        store.close()


def start_background_sync(config: Config) -> threading.Thread | None:
    if not config.auto_sync_hours or not config.sources:
        return None
    thread = threading.Thread(target=run_due_syncs, args=(config,), name="auto-sync", daemon=True)
    thread.start()
    return thread
