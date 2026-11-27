from pathlib import Path

import pytest

from ticket_precedent.config import (
    ConfigError,
    SourceConfig,
    build_connector,
    config_dir,
    data_dir,
    load_config,
)
from ticket_precedent.connectors.github import GitHubConnector
from ticket_precedent.connectors.jira_cloud import JiraCloudConnector


def test_expands_variables_from_env_file_and_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TRACKER_TOKEN='from-dotenv'\nJIRA_PASSWORD=dotenv\n")
    (tmp_path / "config.yaml").write_text(
        """
index_path: {index}
embedding:
  model: bge-m3
sources:
  - name: helpdesk
    type: mantis
    url: https://helpdesk.example
    token: ${{TRACKER_TOKEN}}
    project_ids: [3]
  - name: jira
    type: jira
    url: https://jira.example
    jql: project = ACME
    username: ${{JIRA_USERNAME}}
    password: ${{JIRA_PASSWORD}}
    chunking:
      skip_comment_authors: [Jenkins]
      max_chars: 1000
""".format(index=tmp_path / "x.db")
    )
    monkeypatch.setenv("JIRA_USERNAME", "alice")
    monkeypatch.setenv("JIRA_PASSWORD", "from-environment")

    config = load_config(tmp_path / "config.yaml")

    assert config.index_path == tmp_path / "x.db"
    assert config.source("helpdesk").resolved_options()["token"] == "from-dotenv"
    jira = config.source("jira")
    assert jira.resolved_options()["password"] == "from-environment"
    assert jira.chunking.max_chars == 1000
    assert jira.chunking.skip_comment_authors == frozenset({"Jenkins"})


def test_env_files_are_read_with_lowest_precedence(tmp_path, monkeypatch):
    shared = tmp_path / "shared.env"
    shared.write_text("TOKEN=shared\nOTHER=shared\n")
    (tmp_path / ".env").write_text("OTHER=local\n")
    (tmp_path / "config.yaml").write_text(
        f"env_files: [{shared}]\n"
        "sources:\n"
        "  - {name: a, type: mantis, url: u, token: '${TOKEN}-${OTHER}', project_ids: [1]}\n"
    )
    monkeypatch.delenv("TOKEN", raising=False)
    monkeypatch.delenv("OTHER", raising=False)

    config = load_config(tmp_path / "config.yaml")

    assert config.source("a").resolved_options()["token"] == "shared-local"


def test_missing_secret_fails_only_when_the_source_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    (tmp_path / "config.yaml").write_text(
        "sources:\n  - {name: a, type: mantis, url: u, token: '${NOPE}', project_ids: [1]}\n"
    )
    config = load_config(tmp_path / "config.yaml")
    with pytest.raises(ConfigError, match="NOPE"):
        config.source("a").resolved_options()


def test_unknown_chunking_key_is_an_error(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "sources:\n  - {name: a, type: jira, url: u, jql: x, chunking: {max_char: 5}}\n"
    )
    with pytest.raises(ConfigError, match="max_char"):
        load_config(tmp_path / "config.yaml")


def test_without_config_file_defaults_apply(tmp_path):
    config = load_config(tmp_path / "missing.yaml")
    assert config.sources == []
    assert config.embedding.model == "bge-m3"


def test_empty_xdg_variables_count_as_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("XDG_DATA_HOME", "")

    assert config_dir() == Path("~/.config/ticket-precedent").expanduser()
    assert data_dir() == Path("~/.local/share/ticket-precedent").expanduser()


def test_builds_cloud_and_github_connectors():
    cloud = build_connector(
        SourceConfig(
            "cloud",
            "jira-cloud",
            {"url": "https://acme.atlassian.net/", "jql": "project = ACME", "page_size": 500},
        )
    )
    assert isinstance(cloud, JiraCloudConnector)
    assert cloud.base_url == "https://acme.atlassian.net"
    assert cloud.page_size == 100

    github = build_connector(SourceConfig("gh", "github", {"repo": ["acme/widget", "acme/gear"]}))
    assert isinstance(github, GitHubConnector)
    assert github.repos == ["acme/widget", "acme/gear"]
    assert github.base_url == "https://api.github.com"


def test_unknown_source_type_lists_every_type():
    with pytest.raises(ConfigError, match="jira, jira-cloud, mantis, github"):
        build_connector(SourceConfig("x", "gitlab", {}))
