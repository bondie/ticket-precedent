from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from ..models import Ticket
from .http import get_json
from .jira import JiraConnector

MAX_PAGE_SIZE = 100


class JiraCloudConnector(JiraConnector):
    """JIRA Cloud REST API v2 with token-paged `/search/jql`; bodies stay plain text, not ADF."""

    search_path = "/rest/api/2/search/jql"

    def __init__(
        self,
        source: str,
        url: str,
        jql: str,
        email: str | None = None,
        api_token: str | None = None,
        page_size: int = 50,
        timeout: float = 120.0,
        changelog: bool = True,
    ) -> None:
        super().__init__(
            source,
            url,
            jql,
            username=email,
            password=api_token,
            page_size=min(page_size, MAX_PAGE_SIZE),
            timeout=timeout,
            changelog=changelog,
        )

    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]:
        params = self._search_params(since)
        while True:
            data = get_json(self._client, self.search_path, params)
            for issue in data.get("issues", []):
                yield self._ticket(issue)
            token = data.get("nextPageToken")
            if data.get("isLast") or not token:
                return
            params = {**params, "nextPageToken": token}

    def _ticket(self, issue: dict[str, Any]) -> Ticket:
        changelog = issue.get("changelog")
        if changelog and changelog.get("total", 0) > len(changelog.get("histories", [])):
            issue = {**issue, "changelog": self._full_changelog(issue["key"])}
        return super()._ticket(issue)

    def _full_changelog(self, key: str) -> dict[str, Any]:
        """Search results cut long changelogs short (at 40 histories when tested)."""
        histories: list[dict[str, Any]] = []
        while True:
            data = get_json(
                self._client,
                f"/rest/api/2/issue/{key}/changelog",
                {"startAt": len(histories), "maxResults": MAX_PAGE_SIZE},
            )
            batch = data.get("values", [])
            histories.extend(batch)
            if not batch or data.get("isLast") or len(histories) >= data.get("total", 0):
                return {"total": len(histories), "histories": histories}
