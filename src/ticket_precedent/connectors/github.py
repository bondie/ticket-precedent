from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from ..models import Comment, Ticket
from .http import get_response

API_URL = "https://api.github.com"
PAGE_SIZE = 100
KEY = re.compile(r"^([\w.-]+/[\w.-]+)#(\d+)$")
MISSING_STATUSES = {404, 410}


class GitHubConnector:
    """GitHub Issues REST API; pull requests are skipped."""

    def __init__(
        self,
        source: str,
        repo: str | Sequence[str],
        token: str | None = None,
        url: str = API_URL,
        timeout: float = 60.0,
        rate_limit_wait: float = 3600.0,
    ) -> None:
        self.source = source
        self.repos = [repo] if isinstance(repo, str) else list(repo)
        self.base_url = url.rstrip("/")
        self.rate_limit_wait = rate_limit_wait
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        for repo in self.repos:
            yield from self._repo_tickets(repo, since)

    def fetch_ticket(self, key: str) -> Ticket | None:
        """Without waiting for a rate-limit reset: an agent is waiting for the answer."""
        match = KEY.fullmatch(key)
        if match is None:
            return None
        repo = next((r for r in self.repos if r.lower() == match.group(1).lower()), None)
        if repo is None:
            return None
        number = int(match.group(2))
        try:
            response = get_response(
                self._client, f"/repos/{repo}/issues/{number}", attempts=1, rate_limit_wait=0
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code in MISSING_STATUSES:
                return None
            raise
        issue = response.json()
        if "pull_request" in issue:
            return None
        comments = self._issue_comments(repo, number, wait=0) if issue.get("comments") else []
        return issue_to_ticket(issue, self.source, repo, comments)

    def _repo_tickets(self, repo: str, since: datetime | None) -> Iterator[Ticket]:
        # A stored ticket's comments are replaced, so a `since`-filtered stream would drop old ones.
        known = self._comments_by_issue(repo) if since is None else {}
        params: dict[str, Any] = {
            "state": "all",
            "sort": "updated",
            "direction": "asc",
            "per_page": PAGE_SIZE,
        }
        if since is not None:
            params["since"] = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for issue in self._paginate(f"/repos/{repo}/issues", params):
            if "pull_request" in issue:
                continue
            comments = known.get(issue["number"], [])
            if len(comments) != issue.get("comments", 0):
                comments = self._issue_comments(repo, issue["number"])
            yield issue_to_ticket(issue, self.source, repo, comments)

    def _comments_by_issue(self, repo: str) -> dict[int, list[Comment]]:
        grouped: dict[int, list[Comment]] = defaultdict(list)
        params = {"sort": "created", "direction": "asc", "per_page": PAGE_SIZE}
        for item in self._paginate(f"/repos/{repo}/issues/comments", params):
            number = item.get("issue_url", "").rsplit("/", 1)[-1]
            if number.isdigit():
                grouped[int(number)].append(_comment(item))
        return grouped

    def _issue_comments(self, repo: str, number: int, wait: float | None = None) -> list[Comment]:
        path = f"/repos/{repo}/issues/{number}/comments"
        return [_comment(item) for item in self._paginate(path, {"per_page": PAGE_SIZE}, wait)]

    def _paginate(
        self, path: str, params: dict[str, Any], wait: float | None = None
    ) -> Iterator[dict[str, Any]]:
        url: str | None = path
        query: dict[str, Any] | None = params
        rate_limit_wait = self.rate_limit_wait if wait is None else wait
        while url:
            response = get_response(self._client, url, query, rate_limit_wait=rate_limit_wait)
            yield from response.json()
            url = response.links.get("next", {}).get("url")
            query = None


def issue_to_ticket(
    issue: dict[str, Any], source: str, repo: str, comments: list[Comment]
) -> Ticket:
    closed = issue.get("state") == "closed"
    return Ticket(
        source=source,
        key=f"{repo}#{issue['number']}",
        project=repo,
        title=issue.get("title") or "",
        description=issue.get("body") or "",
        url=issue.get("html_url") or "",
        status=issue.get("state") or "",
        resolution=(issue.get("state_reason") or "") if closed else "",
        created=_parse(issue.get("created_at")),
        updated=_parse(issue.get("updated_at")),
        issue_type=(issue.get("type") or {}).get("name") or "",
        comments=comments,
    )


def _comment(item: dict[str, Any]) -> Comment:
    return Comment(
        author=(item.get("user") or {}).get("login") or "",
        created=_parse(item.get("created_at")),
        body=item.get("body") or "",
    )


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)
