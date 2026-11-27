from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ticket_precedent.connectors import http
from ticket_precedent.connectors.github import GitHubConnector, issue_to_ticket
from ticket_precedent.connectors.http import RateLimitError
from ticket_precedent.models import Comment

API = "https://api.github.com"


def gh_issue(number: int, comments: int = 0, **fields: Any) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Crash {number}",
        "body": "Steps to reproduce",
        "html_url": f"https://github.com/acme/widget/issues/{number}",
        "state": "closed",
        "state_reason": "not_planned",
        "created_at": "2024-05-01T09:30:00Z",
        "updated_at": "2024-05-02T08:00:00Z",
        "comments": comments,
        "user": {"login": "reporter"},
        **fields,
    }


def gh_comment(number: int, body: str, created: str = "2024-05-01T10:00:00Z") -> dict[str, Any]:
    return {
        "issue_url": f"{API}/repos/acme/widget/issues/{number}",
        "user": {"login": f"user-{body}"},
        "created_at": created,
        "body": body,
    }


def connector_with(handler, **options: Any) -> GitHubConnector:
    connector = GitHubConnector("gh", options.pop("repo", "acme/widget"), **options)
    connector._client = httpx.Client(base_url=API, transport=httpx.MockTransport(handler))
    return connector


def page(items: list[dict[str, Any]], next_url: str | None = None) -> httpx.Response:
    headers = {"Link": f'<{next_url}>; rel="next"'} if next_url else {}
    return httpx.Response(200, json=items, headers=headers)


def test_github_issue_mapping():
    comment = Comment("alice", datetime(2024, 5, 1, 10, tzinfo=UTC), "Same here")
    issue = gh_issue(7, body=None, type={"name": "Bug"})
    ticket = issue_to_ticket(issue, "gh", "acme/widget", [comment])

    assert ticket.key == "acme/widget#7"
    assert ticket.project == "acme/widget"
    assert ticket.url == "https://github.com/acme/widget/issues/7"
    assert ticket.description == ""
    assert (ticket.status, ticket.resolution) == ("closed", "not_planned")
    assert ticket.issue_type == "Bug"
    assert ticket.created == datetime(2024, 5, 1, 9, 30, tzinfo=UTC)
    assert ticket.comments == [comment]

    reopened = issue_to_ticket(gh_issue(8, state="open", state_reason="reopened"), "gh", "a/b", [])
    assert (reopened.status, reopened.resolution) == ("open", "")


def test_full_sync_reads_the_comment_stream_once_and_skips_pull_requests():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/repos/acme/widget/issues/comments":
            if request.url.params.get("page") == "2":
                return page([gh_comment(1, "b", "2024-05-03T10:00:00Z")])
            stream = [gh_comment(1, "a"), gh_comment(3, "pr"), gh_comment(2, "c")]
            return page(stream, f"{API}/repos/acme/widget/issues/comments?page=2")
        if path == "/repos/acme/widget/issues":
            if request.url.params.get("page") == "2":
                return page([gh_issue(2, comments=1)])
            pull = gh_issue(3, comments=1, pull_request={"url": "x"})
            return page([gh_issue(1, comments=2), pull], f"{API}/repos/acme/widget/issues?page=2")
        raise AssertionError(f"unexpected request {request.url}")

    tickets = list(connector_with(handler).iter_tickets())

    assert [t.key for t in tickets] == ["acme/widget#1", "acme/widget#2"]
    assert [c.body for c in tickets[0].comments] == ["a", "b"]
    assert [c.body for c in tickets[1].comments] == ["c"]
    first = next(r for r in requests if r.url.path == "/repos/acme/widget/issues").url.params
    assert (first["state"], first["sort"], first["direction"]) == ("all", "updated", "asc")
    assert "since" not in first
    assert not any(r.url.path.endswith(("/1/comments", "/2/comments")) for r in requests)


def test_full_sync_refetches_an_issue_whose_comment_count_changed():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/acme/widget/issues/comments":
            return page([gh_comment(1, "a")])
        if path == "/repos/acme/widget/issues/1/comments":
            return page([gh_comment(1, "a"), gh_comment(1, "late")])
        return page([gh_issue(1, comments=2)])

    [ticket] = connector_with(handler).iter_tickets()

    assert [c.body for c in ticket.comments] == ["a", "late"]


def test_incremental_sync_carries_every_comment_of_a_changed_issue():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/repos/acme/widget/issues":
            return page([gh_issue(5, comments=2), gh_issue(6)])
        if path == "/repos/acme/widget/issues/5/comments":
            old = gh_comment(5, "old", "2023-01-01T10:00:00Z")
            return page([old, gh_comment(5, "new", "2024-05-02T07:00:00Z")])
        raise AssertionError(f"unexpected request {request.url}")

    since = datetime(2024, 5, 1, 14, 0, tzinfo=UTC)
    tickets = list(connector_with(handler).iter_tickets(since))

    assert [c.body for c in tickets[0].comments] == ["old", "new"]
    assert tickets[1].comments == []
    assert requests[0].url.params["since"] == "2024-05-01T14:00:00Z"
    assert [r.url.path for r in requests] == [
        "/repos/acme/widget/issues",
        "/repos/acme/widget/issues/5/comments",
    ]


def test_every_configured_repository_is_synced():
    def plain(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/issues"):
            return page([gh_issue(1)])
        return page([])

    connector = connector_with(plain, repo=["acme/widget", "acme/gear"])
    keys = [t.key for t in connector.iter_tickets()]

    assert keys == ["acme/widget#1", "acme/gear#1"]


def exhausted(reset: datetime, status: int = 403) -> httpx.Response:
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset.timestamp()))}
    return httpx.Response(status, json={"message": "API rate limit exceeded"}, headers=headers)


@pytest.fixture
def slept(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(http, "time", SimpleNamespace(sleep=calls.append))
    return calls


@pytest.mark.parametrize("status", [403, 429])
def test_waits_for_a_rate_limit_reset_that_is_near(slept, status):
    responses = [exhausted(datetime.now(UTC).replace(microsecond=0), status), page([]), page([])]
    connector = connector_with(lambda _request: responses.pop(0), rate_limit_wait=600)

    assert list(connector.iter_tickets()) == []
    assert len(slept) == 1
    assert 0 < slept[0] <= 2


def test_rate_limit_reset_beyond_the_wait_fails_with_a_clear_message(slept):
    reset = datetime(2099, 1, 1, tzinfo=UTC)
    connector = connector_with(lambda _request: exhausted(reset), rate_limit_wait=600)

    with pytest.raises(RateLimitError, match=r"rate limit exhausted until 209.*token"):
        list(connector.iter_tickets())
    assert slept == []


def test_forbidden_without_rate_limit_headers_is_an_http_error(slept):
    connector = connector_with(lambda _request: httpx.Response(403, json={}))

    with pytest.raises(httpx.HTTPStatusError):
        list(connector.iter_tickets())


def test_secondary_rate_limit_retries_after_the_given_delay(slept):
    responses = [httpx.Response(403, headers={"Retry-After": "7"}), page([]), page([])]
    connector = connector_with(lambda _request: responses.pop(0))

    assert list(connector.iter_tickets()) == []
    assert slept == [7.0]
