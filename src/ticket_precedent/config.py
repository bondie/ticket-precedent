from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .chunking import ChunkingConfig
from .connectors import Connector
from .connectors.jira import JiraConnector
from .connectors.mantis import MantisConnector
from .embeddings import OllamaEmbedder
from .search import Searcher
from .store import Store

APP = "ticket-precedent"
_VARIABLE = re.compile(r"\$\{(\w+)\}")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "bge-m3"
    url: str = "http://localhost:11434"
    batch_size: int = 32
    query_prefix: str = ""
    document_prefix: str = ""


@dataclass(frozen=True)
class SearchConfig:
    rrf_k: int = 10
    lexical_weight: float = 0.25
    candidate_chunks: int = 200


@dataclass(frozen=True)
class SourceConfig:
    name: str
    type: str
    options: dict[str, Any]
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    env: dict[str, str] = field(default_factory=dict, repr=False)

    def resolved_options(self) -> dict[str, Any]:
        """Secrets are resolved only when a connector is built, so search works without them."""
        resolved: dict[str, Any] = _expand(self.options, self.env)
        return resolved


@dataclass(frozen=True)
class Config:
    index_path: Path
    embedding: EmbeddingConfig
    sources: list[SourceConfig]
    search: SearchConfig = field(default_factory=SearchConfig)
    auto_sync_hours: float | None = None

    def source(self, name: str) -> SourceConfig:
        for source in self.sources:
            if source.name == name:
                return source
        raise ConfigError(f"Unknown source {name!r}; configured: {[s.name for s in self.sources]}")


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / APP


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / APP


def default_config_path() -> Path:
    override = os.environ.get("TICKET_PRECEDENT_CONFIG")
    return Path(override).expanduser() if override else config_dir() / "config.yaml"


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        return Config(index_path=data_dir() / "index.db", embedding=EmbeddingConfig(), sources=[])
    document = yaml.safe_load(path.read_text()) or {}
    env: dict[str, str] = {}
    for env_file in document.get("env_files") or []:
        env.update(read_dotenv(Path(env_file).expanduser()))
    env.update(read_dotenv(path.parent / ".env"))
    env.update(os.environ)
    sources = document.pop("sources", None) or []
    raw = _expand(document, env)
    return Config(
        index_path=Path(raw.get("index_path") or data_dir() / "index.db").expanduser(),
        embedding=_build(EmbeddingConfig, raw.get("embedding") or {}, "embedding"),
        sources=[_source(item, env) for item in sources],
        search=_build(SearchConfig, raw.get("search") or {}, "search"),
        auto_sync_hours=raw.get("auto_sync_hours"),
    )


def read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def build_connector(source: SourceConfig) -> Connector:
    if source.type == "jira":
        return JiraConnector(source.name, **source.resolved_options())
    if source.type == "mantis":
        return MantisConnector(source.name, **source.resolved_options())
    raise ConfigError(f"Source {source.name!r}: unknown type {source.type!r} (jira, mantis)")


def build_searcher(config: Config, store: Store) -> Searcher:
    return Searcher(
        store,
        build_embedder(config.embedding),
        candidate_chunks=config.search.candidate_chunks,
        rrf_k=config.search.rrf_k,
        lexical_weight=config.search.lexical_weight,
    )


def build_embedder(config: EmbeddingConfig) -> OllamaEmbedder:
    return OllamaEmbedder(
        model=config.model,
        url=config.url,
        batch_size=config.batch_size,
        query_prefix=config.query_prefix,
        document_prefix=config.document_prefix,
    )


def _source(raw: dict[str, Any], env: dict[str, str]) -> SourceConfig:
    options = dict(raw)
    name = options.pop("name", None)
    kind = options.pop("type", None)
    if not name or not kind:
        raise ConfigError(f"Every source needs `name` and `type`: {raw}")
    chunking = dict(options.pop("chunking", None) or {})
    if "skip_comment_authors" in chunking:
        chunking["skip_comment_authors"] = frozenset(chunking["skip_comment_authors"])
    return SourceConfig(
        name=name,
        type=kind,
        options=options,
        chunking=_build(ChunkingConfig, chunking, f"sources.{name}.chunking"),
        env=env,
    )


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**values)


def _expand(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, env) for item in value]
    if isinstance(value, str):
        return _VARIABLE.sub(lambda match: _lookup(match.group(1), env), value)
    return value


def _lookup(name: str, env: dict[str, str]) -> str:
    if name not in env:
        raise ConfigError(f"Environment variable {name} is not set (config or .env)")
    return env[name]
