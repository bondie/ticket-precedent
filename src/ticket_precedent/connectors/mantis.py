from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

import httpx

from ..models import Comment, Link, Ticket
from .http import get_json


class MantisConnector:
    """MantisBT REST API (/api/rest). Issues come newest-updated first."""

    def __init__(
        self,
        source: str,
        url: str,
        token: str,
        project_ids: Sequence[int],
        page_size: int = 50,
        include_private_notes: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.source = source
        self.base_url = url.rstrip("/")
        self.project_ids = list(project_ids)
        self.page_size = page_size
        self.include_private_notes = include_private_notes
        self._client = httpx.Client(
            base_url=self.base_url, headers={"Authorization": token}, timeout=timeout
        )

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        for project_id in self.project_ids:
            yield from self._project_tickets(project_id, since)

    def _project_tickets(self, project_id: int, since: datetime | None) -> Iterator[Ticket]:
        page = 1
        while True:
            data = get_json(
                self._client,
                "/api/rest/issues",
                {"project_id": project_id, "page_size": self.page_size, "page": page},
            )
            issues = data.get("issues", [])
            tickets = [
                issue_to_ticket(issue, self.source, self.base_url, self.include_private_notes)
                for issue in issues
            ]
            fresh = [t for t in tickets if since is None or t.updated is None or t.updated >= since]
            yield from fresh
            if len(issues) < self.page_size or not fresh:
                return
            page += 1


def issue_to_ticket(
    issue: dict[str, Any], source: str, base_url: str, include_private_notes: bool = True
) -> Ticket:
    issue_id = str(issue["id"])
    notes = [
        note
        for note in issue.get("notes", [])
        if include_private_notes or (note.get("view_state") or {}).get("name") == "public"
    ]
    parts = (
        issue.get("description"),
        issue.get("steps_to_reproduce"),
        issue.get("additional_information"),
    )
    return Ticket(
        source=source,
        key=issue_id,
        project=(issue.get("project") or {}).get("name", ""),
        title=issue.get("summary") or "",
        description="\n\n".join(part for part in parts if part),
        url=f"{base_url}/view.php?id={issue_id}",
        status=(issue.get("status") or {}).get("name", ""),
        resolution=(issue.get("resolution") or {}).get("name", ""),
        created=_parse(issue.get("created_at")),
        updated=_parse(issue.get("updated_at")),
        time_spent_seconds=sum(_duration(note) for note in notes) or None,
        comments=[_comment(note) for note in notes],
        links=[
            Link(type=(rel.get("type") or {}).get("name", ""), key=str(rel["issue"]["id"]))
            for rel in issue.get("relationships", [])
            if rel.get("issue")
        ],
    )


def _comment(note: dict[str, Any]) -> Comment:
    reporter = note.get("reporter") or {}
    return Comment(
        author=reporter.get("real_name") or reporter.get("name") or "",
        created=_parse(note.get("created_at")),
        body=note.get("text") or "",
    )


def _duration(note: dict[str, Any]) -> int:
    value = (note.get("time_tracking") or {}).get("duration") or ""
    hours, _, minutes = value.partition(":")
    if not hours.isdigit() or not minutes.isdigit():
        return 0
    return int(hours) * 3600 + int(minutes) * 60


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)
