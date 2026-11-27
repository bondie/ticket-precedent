# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

### Added

- `tp demo` downloads a ready-made index of the 4,922 public Apache ZooKeeper tickets with
  their comments, checks its SHA-256, writes a configuration that keeps it current from
  Apache JIRA without credentials, and runs a first search. Without Ollama the demo falls
  back to keyword search and says so.
- Published on PyPI: `uvx ticket-precedent demo` runs it without cloning the repository.
  The `ticket-precedent` command is an alias of `tp`.
- JIRA Cloud connector (`type: jira-cloud`): token-paged `/rest/api/2/search/jql` with
  plain-text bodies, `email` + `api_token` or anonymous access, and the full changelog and
  comment thread fetched when search results cut them short.
- GitHub Issues connector (`type: github`): one or more repositories, pull requests skipped,
  the whole comment thread on every sync, keys like `acme/widget#123`, GitHub Enterprise
  through `url`, and waiting for the rate limit reset up to `rate_limit_wait` seconds.

## [0.1.0] - 2026-11-15

First public version, used from a clone of the repository.

### Added

- Hybrid search (SQLite FTS5 BM25 and sqlite-vec embeddings from Ollama, fused with
  Reciprocal Rank Fusion) over JIRA Server / Data Center and MantisBT tickets and their
  comments, with incremental sync.
- MCP server with `precedents_for`, `similar_tickets`, `effort_reference`, `get_ticket`
  and `list_sources`, and the same operations on the command line.
- `effort_reference`: logged hours of similar finished tickets, cut off at the moment the
  ticket was created, with the median and P25–P75.
- Platforms keep the tickets of different products apart.
- `tp eval`: benchmark on public Apache JIRA projects with tickets closed as duplicates,
  with bootstrap confidence intervals; results for ZooKeeper and Derby.
- A cross-encoder reranker benchmark, documented as a negative result.
