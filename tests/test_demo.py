import gzip
import hashlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from ticket_precedent import cli, config
from ticket_precedent.chunking import ChunkingConfig
from ticket_precedent.config import load_config
from ticket_precedent.demo import APACHE_JIRA, DemoError, DemoIndex, download, write_config
from ticket_precedent.ingest import sync
from ticket_precedent.models import Ticket
from ticket_precedent.store import Store

from .conftest import FakeEmbedder, make_ticket

PAYLOAD = b"SQLite format 3\x00" + b"x" * 5000
PACKED = gzip.compress(PAYLOAD)


def demo_index(digest: str = hashlib.sha256(PACKED).hexdigest()) -> DemoIndex:
    return DemoIndex("Demo", "DEMO", "https://downloads.example/index.db.gz", digest, 1, 2)


def client_serving(body: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
    )


def test_download_checks_the_digest_and_unpacks(tmp_path: Path):
    target = tmp_path / "demo" / "index.db"
    seen: list[tuple[int, int | None]] = []

    download(
        client_serving(PACKED), demo_index(), target, lambda got, total: seen.append((got, total))
    )

    assert target.read_bytes() == PAYLOAD
    assert seen[-1] == (len(PACKED), len(PACKED))
    assert list(target.parent.iterdir()) == [target]


def test_download_rejects_a_corrupted_file_and_leaves_nothing_behind(tmp_path: Path):
    target = tmp_path / "demo" / "index.db"

    with pytest.raises(DemoError, match="checksum"):
        download(client_serving(PACKED[:-10]), demo_index(), target)

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_demo_configuration_syncs_from_apache_without_credentials(tmp_path: Path):
    path = tmp_path / "demo.yaml"
    write_config(demo_index(), tmp_path / "index.db", path)

    loaded = load_config(path)

    assert loaded.index_path == tmp_path / "index.db"
    assert loaded.embedding.model == "bge-m3"
    [source] = loaded.sources
    assert source.name == "apache-demo"
    assert source.resolved_options() == {
        "url": APACHE_JIRA,
        "jql": "project = DEMO",
        "page_size": 100,
        "changelog": False,
    }
    assert "Jenkins" in source.chunking.skip_comment_authors


class Tickets:
    def __init__(self, tickets: list[Ticket]) -> None:
        self.tickets = tickets

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        yield from self.tickets


def test_cli_demo_downloads_once_and_prints_next_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    embedder = FakeEmbedder()
    monkeypatch.setattr(config, "build_embedder", lambda _config: embedder)
    downloads: list[Path] = []

    def fake_download(_client, index, target, _progress=None):
        downloads.append(target)
        tickets = [
            make_ticket(
                "Z-1", "Server will not start with an empty transaction log", source=index.source
            ),
            make_ticket("Z-2", "Client hangs after OutOfMemoryError", source=index.source),
        ]
        sync(Store(target), embedder, Tickets(tickets), index.source, ChunkingConfig())

    monkeypatch.setattr(cli, "download", fake_download)

    assert cli.main(["demo"]) == 0
    assert cli.main(["demo"]) == 0

    assert len(downloads) == 1
    out = capsys.readouterr().out
    assert "Z-1 Server will not start with an empty transaction log" in out
    config_path = tmp_path / "config" / "ticket-precedent" / "demo.yaml"
    assert f"TICKET_PRECEDENT_CONFIG={config_path}" in out
    assert load_config(config_path).index_path == downloads[0]
