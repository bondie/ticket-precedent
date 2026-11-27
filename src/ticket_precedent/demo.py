from __future__ import annotations

import gzip
import hashlib
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

APACHE_JIRA = "https://issues.apache.org/jira"
APACHE_BOTS = frozenset(
    {
        "ASF GitHub Bot",
        "ASF subversion and git services",
        "Hudson",
        "Hadoop QA",
        "Jenkins",
        "GitBox",
    }
)


class DemoError(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoIndex:
    name: str
    project: str
    url: str
    sha256: str
    size_mb: int
    tickets: int

    @property
    def source(self) -> str:
        return f"apache-{self.project.lower()}"


ZOOKEEPER = DemoIndex(
    name="Apache ZooKeeper",
    project="ZOOKEEPER",
    url=(
        "https://github.com/bondie/ticket-precedent/releases/download/"
        "demo-index-1/apache-zookeeper-bge-m3.db.gz"
    ),
    sha256="31481ce7a956e045b449cd092fbbc1217a5547bce6cbfdb217eb96c23b253389",
    size_mb=122,
    tickets=4922,
)


def download(
    client: httpx.Client,
    index: DemoIndex,
    target: Path,
    progress: Callable[[int, int | None], None] | None = None,
) -> None:
    """Fetch the gzipped index, check its digest and only then move it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        packed = Path(work) / "index.db.gz"
        digest = hashlib.sha256()
        with client.stream("GET", index.url, follow_redirects=True) as response:
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
            received = 0
            with packed.open("wb") as out:
                for block in response.iter_bytes(1 << 20):
                    out.write(block)
                    digest.update(block)
                    received += len(block)
                    if progress:
                        progress(received, total)
        if digest.hexdigest() != index.sha256:
            raise DemoError(
                f"Downloaded demo index does not match its checksum ({digest.hexdigest()}); "
                "try again or report it"
            )
        unpacked = Path(work) / "index.db"
        with gzip.open(packed, "rb") as source, unpacked.open("wb") as out:
            shutil.copyfileobj(source, out, 1 << 20)
        unpacked.replace(target)


def config_document(index: DemoIndex, index_path: Path) -> dict[str, object]:
    """A normal configuration, so `tp sync` keeps the demo index current from Apache JIRA."""
    return {
        "index_path": str(index_path),
        "embedding": {"model": "bge-m3"},
        "sources": [
            {
                "name": index.source,
                "type": "jira",
                "url": APACHE_JIRA,
                "jql": f"project = {index.project}",
                "page_size": 100,
                "changelog": False,
                "chunking": {"skip_comment_authors": sorted(APACHE_BOTS)},
            }
        ],
    }


def write_config(index: DemoIndex, index_path: Path, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Written by `tp demo`: public Apache JIRA tickets, no credentials needed.\n"
    body = yaml.safe_dump(config_document(index, index_path), sort_keys=False)
    config_path.write_text(header + body)
