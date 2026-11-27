# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `scripts/eval_judge.py`: a local LLM (`gemma4:12b` through Ollama) judges whether each of
  the top 5 candidates reports the same fault, and the run measures the reordering. It raises
  MRR@10 by 0.04–0.07 on Apache duplicates and on private Czech trackers, with cached and
  reproducible answers.
- `scripts/eval_linked.py`: a benchmark on your own index, built from issues whose titles
  name the helpdesk thread they came from, with two tasks (thread to issue, earlier issue).

## [0.3.0] - 2026-11-27

### Added

- On-demand indexing: `precedents_for`, `effort_reference` with a key and `get_ticket` (and
  `tp precedents`, `tp effort` and `tp show`) fetch a ticket that is not indexed yet from
  the configured sources, index it and answer as for any indexed ticket, still with only
  earlier tickets and without the tickets its title names. Sources that already hold the
  key's JIRA project, or name it in their JQL, are asked first; a key that no source has
  is remembered for 10 minutes, and the error names the sources that were asked.
- `effort_since` in the configuration: the default `since` of `effort_reference`, one date
  for every platform or a date per platform. An explicit `since` still wins, and the
  response (and the `tp effort` summary) says which date was used.
- The MCP server tells the model the configured platforms, their sources and the default
  platform when it starts, so it passes valid platform names.
- `similar_tickets` accepts `query` as an alias of `text`.

### Changed

- `get_ticket(key, source?)`: the key is enough, `source` is needed only when two sources
  share the key. `tp show KEY [--source S]` likewise; `tp show SOURCE KEY` still works.
- Platform names match regardless of case (`SHOP` finds `shop`).
- `precedents_for`, `effort_reference` and `get_ticket` are marked as reaching outside the
  index (`open_world_hint`), because they may fetch a ticket from its tracker.

### Fixed

- Parallel MCP tool calls no longer share one SQLite connection across worker threads,
  which could fail with `sqlite3.InterfaceError`; every thread now has its own connection.

## [0.2.0] - 2026-11-27

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
