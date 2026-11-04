from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import httpx

from ..models import Comment, Link, Ticket
from .http import get_json

FIELDS = (
    "summary,description,comment,status,resolution,created,updated,"
    "issuelinks,timespent,aggregatetimespent,project"
)


class JiraConnector:
    """JIRA Server / Data Center REST API v2."""

    def __init__(
        self,
        source: str,
        url: str,
        jql: str,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        page_size: int = 100,
        timeout: float = 60.0,
    ) -> None:
        self.source = source
        self.base_url = url.rstrip("/")
        self.jql = jql
        self.page_size = page_size
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        auth = (username, password) if username and password else None
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, auth=auth, timeout=timeout
        )

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        jql = f"({self.jql})"
        if since is not None:
            jql += f' AND updated >= "{since.strftime("%Y/%m/%d %H:%M")}"'
        jql += " ORDER BY updated ASC"
        start_at = 0
        while True:
            data = get_json(
                self._client,
                "/rest/api/2/search",
                {"jql": jql, "startAt": start_at, "maxResults": self.page_size, "fields": FIELDS},
            )
            issues = data.get("issues", [])
            for issue in issues:
                yield self._ticket(issue)
            start_at += len(issues)
            if not issues or start_at >= data.get("total", 0):
                return

    def _ticket(self, issue: dict[str, Any]) -> Ticket:
        ticket = issue_to_ticket(issue, self.source, self.base_url)
        field = issue["fields"].get("comment") or {}
        if field.get("total", 0) <= len(field.get("comments", [])):
            return ticket
        ticket.comments = self._all_comments(issue["key"])
        return ticket

    def _all_comments(self, key: str) -> list[Comment]:
        comments: list[Comment] = []
        while True:
            data = get_json(
                self._client,
                f"/rest/api/2/issue/{key}/comment",
                {"startAt": len(comments), "maxResults": 500},
            )
            batch = data.get("comments", [])
            comments.extend(_comment(c) for c in batch)
            if not batch or len(comments) >= data.get("total", 0):
                return comments


def issue_to_ticket(issue: dict[str, Any], source: str, base_url: str) -> Ticket:
    fields = issue["fields"]
    key = issue["key"]
    return Ticket(
        source=source,
        key=key,
        project=(fields.get("project") or {}).get("key") or key.split("-")[0],
        title=fields.get("summary") or "",
        description=fields.get("description") or "",
        url=f"{base_url}/browse/{key}",
        status=(fields.get("status") or {}).get("name", ""),
        resolution=(fields.get("resolution") or {}).get("name", ""),
        created=parse_time(fields.get("created")),
        updated=parse_time(fields.get("updated")),
        time_spent_seconds=fields.get("aggregatetimespent") or fields.get("timespent"),
        comments=[_comment(c) for c in (fields.get("comment") or {}).get("comments", [])],
        links=list(_links(fields.get("issuelinks") or [])),
    )


def _comment(comment: dict[str, Any]) -> Comment:
    author = comment.get("author") or {}
    return Comment(
        author=author.get("displayName") or author.get("name") or "",
        created=parse_time(comment.get("created")),
        body=comment.get("body") or "",
    )


def _links(issue_links: list[dict[str, Any]]) -> Iterator[Link]:
    for link in issue_links:
        other = link.get("outwardIssue") or link.get("inwardIssue")
        if other:
            yield Link(type=link["type"]["name"], key=other["key"])


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)
