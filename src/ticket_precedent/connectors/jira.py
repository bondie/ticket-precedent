from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any, NamedTuple

import httpx

from ..models import Comment, Link, Ticket
from .http import get_json

FIELDS = (
    "summary,description,comment,status,resolution,created,updated,"
    "issuelinks,timespent,aggregatetimespent,project,issuetype,timeoriginalestimate"
)
TRACKED_CHANGES = frozenset({"timeoriginalestimate", "timespent"})


class Change(NamedTuple):
    at: datetime
    field: str
    before: str | None
    after: str | None


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
        page_size: int = 50,
        timeout: float = 120.0,
        changelog: bool = True,
    ) -> None:
        self.source = source
        self.base_url = url.rstrip("/")
        self.jql = jql
        self.page_size = page_size
        self.changelog = changelog
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
        params: dict[str, Any] = {"jql": jql, "maxResults": self.page_size, "fields": FIELDS}
        if self.changelog:
            params["expand"] = "changelog"
        start_at = 0
        while True:
            data = get_json(self._client, "/rest/api/2/search", {**params, "startAt": start_at})
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
    changes = tracked_changes(issue)
    own_time = fields.get("timespent")
    total_time = fields.get("aggregatetimespent") or own_time
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
        time_spent_seconds=total_time,
        issue_type=(fields.get("issuetype") or {}).get("name", ""),
        initial_estimate_seconds=initial_estimate(fields.get("timeoriginalestimate"), changes),
        # Sub-task worklogs are not in the parent's changelog, so its last entry may be too early.
        last_logged=last_logged(changes) if total_time == own_time else None,
        comments=[_comment(c) for c in (fields.get("comment") or {}).get("comments", [])],
        links=list(_links(fields.get("issuelinks") or [])),
    )


def tracked_changes(issue: dict[str, Any]) -> list[Change] | None:
    """Estimate and time-spent changes in time order, or None without a complete changelog."""
    changelog = issue.get("changelog")
    if changelog is None:
        return None
    histories = changelog.get("histories", [])
    if changelog.get("total", 0) > len(histories):
        return None
    changes = [
        Change(at, item["field"], item.get("from"), item.get("to"))
        for history in histories
        if (at := parse_time(history.get("created"))) is not None
        for item in history.get("items", [])
        if item.get("field") in TRACKED_CHANGES
    ]
    return sorted(changes, key=lambda change: change.at)


def initial_estimate(current: int | None, changes: Sequence[Change] | None) -> int | None:
    """The original estimate as it stood when the first time was logged.

    Trackers often raise the estimate to match the time spent once the work is done, so the
    current value is a record of the effort rather than a forecast of it.
    """
    if changes is None:
        return None
    estimates = [change for change in changes if change.field == "timeoriginalestimate"]
    if not estimates:
        return current or None
    first_logged = next((c.at for c in changes if c.field == "timespent"), None)
    value = estimates[0].before
    for change in estimates:
        if first_logged is not None and change.at >= first_logged:
            break
        value = change.after
    return int(value) if value else None


def last_logged(changes: Sequence[Change] | None) -> datetime | None:
    """When time was last logged: from then on the ticket's total effort was known."""
    if changes is None:
        return None
    return next((c.at for c in reversed(changes) if c.field == "timespent"), None)


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
