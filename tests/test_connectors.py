from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from ticket_precedent.connectors.jira import JiraConnector, issue_to_ticket
from ticket_precedent.connectors.jira_cloud import JiraCloudConnector
from ticket_precedent.connectors.mantis import MantisConnector
from ticket_precedent.connectors.mantis import issue_to_ticket as mantis_ticket
from ticket_precedent.models import Link, Ticket


def jira_issue(key: str, updated: str = "2024-05-02T08:00:00.000+0000", comments_total=None):
    comments = [
        {
            "author": {"displayName": "Alice", "name": "alice"},
            "created": "2024-05-01T10:00:00.000+0000",
            "body": "Same here",
        }
    ]
    return {
        "key": key,
        "fields": {
            "project": {"key": "ACME"},
            "summary": f"Summary {key}",
            "description": "h3. Steps\nDo things",
            "status": {"name": "Resolved"},
            "resolution": {"name": "Duplicate"},
            "created": "2024-05-01T09:30:00.000+0000",
            "updated": updated,
            "aggregatetimespent": 5400,
            "comment": {
                "comments": comments,
                "total": comments_total or len(comments),
                "maxResults": 1,
            },
            "issuelinks": [
                {"type": {"name": "Duplicate"}, "outwardIssue": {"key": "ACME-1"}},
                {"type": {"name": "Relates"}, "inwardIssue": {"key": "ACME-7"}},
            ],
        },
    }


def test_jira_issue_mapping():
    ticket = issue_to_ticket(jira_issue("ACME-9"), "jira", "https://jira.example")
    assert ticket.key == "ACME-9"
    assert ticket.project == "ACME"
    assert ticket.url == "https://jira.example/browse/ACME-9"
    assert ticket.resolution == "Duplicate"
    assert ticket.created == datetime(2024, 5, 1, 9, 30, tzinfo=UTC)
    assert ticket.time_spent_seconds == 5400
    assert ticket.comments[0].author == "Alice"
    assert ticket.links == [Link("Duplicate", "ACME-1"), Link("Relates", "ACME-7")]


def test_jira_pages_through_search_and_fetches_truncated_comments():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/comment"):
            body = [{"author": {"displayName": f"U{i}"}, "body": f"c{i}"} for i in range(3)]
            return httpx.Response(200, json={"comments": body, "total": 3})
        start = int(request.url.params["startAt"])
        issues = [jira_issue("ACME-1", comments_total=3)] if start == 0 else [jira_issue("ACME-2")]
        return httpx.Response(200, json={"issues": issues, "total": 2})

    connector = JiraConnector("jira", "https://jira.example", "project = ACME", page_size=1)
    connector._client = httpx.Client(
        base_url="https://jira.example", transport=httpx.MockTransport(handler)
    )

    since = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
    tickets = list(connector.iter_tickets(since))

    assert [t.key for t in tickets] == ["ACME-1", "ACME-2"]
    assert [c.author for c in tickets[0].comments] == ["U0", "U1", "U2"]
    jql = requests[0].url.params["jql"]
    assert jql == '(project = ACME) AND updated >= "2024/05/01 12:00" ORDER BY updated ASC'


def mantis_issue(issue_id: int, updated: str):
    return {
        "id": issue_id,
        "summary": "Faktura se nepropsala",
        "description": "Klient nevidí fakturu.",
        "additional_information": "ERP ji odeslalo.",
        "project": {"name": "Shop"},
        "status": {"name": "resolved"},
        "resolution": {"name": "fixed"},
        "created_at": "2024-03-01T10:00:00+01:00",
        "updated_at": updated,
        "notes": [
            {
                "reporter": {"name": "dev", "real_name": "Dev One"},
                "text": "Reset comparatoru pomohl.",
                "view_state": {"name": "private"},
                "created_at": "2024-03-02T10:00:00+01:00",
                "time_tracking": {"duration": "01:30"},
            },
            {
                "reporter": {"name": "client"},
                "text": "Díky, funguje.",
                "view_state": {"name": "public"},
                "created_at": "2024-03-03T10:00:00+01:00",
            },
        ],
        "relationships": [{"type": {"name": "duplicate-of"}, "issue": {"id": 41}}],
    }


def test_mantis_issue_mapping():
    ticket = mantis_ticket(
        mantis_issue(42, "2024-03-03T10:00:00+01:00"), "hd", "https://hd.example"
    )
    assert ticket.key == "42"
    assert ticket.url == "https://hd.example/view.php?id=42"
    assert ticket.description == "Klient nevidí fakturu.\n\nERP ji odeslalo."
    assert ticket.time_spent_seconds == 5400
    assert [c.author for c in ticket.comments] == ["Dev One", "client"]
    assert ticket.links == [Link("duplicate-of", "41")]

    public_only = mantis_ticket(
        mantis_issue(42, "2024-03-03T10:00:00+01:00"), "hd", "https://hd.example", False
    )
    assert [c.author for c in public_only.comments] == ["client"]


def test_mantis_stops_paging_once_issues_are_older_than_since():
    pages = {
        1: [
            mantis_issue(3, "2024-03-10T10:00:00+00:00"),
            mantis_issue(2, "2024-03-05T10:00:00+00:00"),
        ],
        2: [
            mantis_issue(1, "2024-01-01T10:00:00+00:00"),
            mantis_issue(0, "2023-12-01T10:00:00+00:00"),
        ],
        3: [mantis_issue(9, "2023-01-01T10:00:00+00:00")],
    }
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        requested.append(page)
        return httpx.Response(200, json={"issues": pages[page]})

    connector = MantisConnector("hd", "https://hd.example", "token", [5], page_size=2)
    connector._client = httpx.Client(
        base_url="https://hd.example", transport=httpx.MockTransport(handler)
    )

    tickets = list(connector.iter_tickets(datetime(2024, 3, 1, tzinfo=UTC)))

    assert [t.key for t in tickets] == ["3", "2"]
    assert requested == [1, 2]


def history(at: str, field: str, before: str | None, after: str | None) -> dict[str, Any]:
    return {
        "created": f"{at}.000+0000",
        "items": [{"field": field, "from": before, "to": after}],
    }


def estimated_issue(
    histories: list[dict[str, Any]],
    estimate: int | None = None,
    spent: int = 3600,
    aggregate: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = jira_issue("ACME-5")
    issue["fields"].update(
        timeoriginalestimate=estimate,
        timespent=spent,
        aggregatetimespent=aggregate or spent,
        issuetype={"name": "Bug"},
    )
    count = len(histories) if total is None else total
    issue["changelog"] = {"total": count, "histories": histories}
    return issue


def ticket_of(issue: dict[str, Any]) -> Ticket:
    return issue_to_ticket(issue, "jira", "https://jira.example")


def test_estimate_entered_with_the_ticket_and_never_changed():
    ticket = ticket_of(
        estimated_issue([history("2024-05-02T10:00:00", "timespent", None, "3600")], 1800)
    )
    assert ticket.initial_estimate_seconds == 1800
    assert ticket.issue_type == "Bug"
    assert ticket.last_logged == datetime(2024, 5, 2, 10, tzinfo=UTC)


def test_estimate_raised_after_work_started_keeps_the_forecast():
    issue = estimated_issue(
        [
            history("2024-05-01T10:00:00", "timeoriginalestimate", None, "1800"),
            history("2024-05-01T12:00:00", "timeoriginalestimate", "1800", "2700"),
            history("2024-05-02T10:00:00", "timespent", None, "3600"),
            history("2024-05-03T10:00:00", "timeoriginalestimate", "2700", "3600"),
        ],
        estimate=3600,
    )
    assert ticket_of(issue).initial_estimate_seconds == 2700


def test_estimate_entered_at_creation_and_changed_later_keeps_the_first_value():
    issue = estimated_issue(
        [
            history("2024-05-02T10:00:00", "timespent", None, "3600"),
            history("2024-05-02T11:00:00", "timeoriginalestimate", "1800", "3600"),
        ],
        estimate=3600,
    )
    assert ticket_of(issue).initial_estimate_seconds == 1800


def test_estimate_first_entered_after_work_started_is_no_forecast():
    issue = estimated_issue(
        [
            history("2024-05-02T10:00:00", "timespent", None, "3600"),
            history("2024-05-02T11:00:00", "timeoriginalestimate", None, "3600"),
        ],
        estimate=3600,
    )
    assert ticket_of(issue).initial_estimate_seconds is None


def test_incomplete_changelog_gives_no_estimate_or_log_time():
    histories = [history("2024-05-02T10:00:00", "timespent", None, "3600")]
    ticket = ticket_of(estimated_issue(histories, 1800, total=120))
    assert ticket.initial_estimate_seconds is None
    assert ticket.last_logged is None


def test_time_logged_on_subtasks_leaves_the_log_time_unknown():
    histories = [history("2024-05-02T10:00:00", "timespent", None, "3600")]
    ticket = ticket_of(estimated_issue(histories, 1800, spent=3600, aggregate=7200))
    assert ticket.time_spent_seconds == 7200
    assert ticket.last_logged is None


def test_last_logged_is_the_latest_worklog():
    histories = [
        history("2024-05-03T10:00:00", "timespent", "3600", "5400"),
        history("2024-05-02T10:00:00", "timespent", None, "3600"),
    ]
    assert ticket_of(estimated_issue(histories)).last_logged == datetime(2024, 5, 3, 10, tzinfo=UTC)


def test_changelog_is_requested_unless_disabled():
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.params.get("expand"))
        return httpx.Response(200, json={"issues": [], "total": 0})

    for changelog in (True, False):
        connector = JiraConnector("jira", "https://jira.example", "x", changelog=changelog)
        connector._client = httpx.Client(
            base_url="https://jira.example", transport=httpx.MockTransport(handler)
        )
        list(connector.iter_tickets())

    assert requested == ["changelog", None]


def cloud_connector(handler, **options: Any) -> JiraCloudConnector:
    connector = JiraCloudConnector(
        "cloud", "https://acme.atlassian.net", "project = ACME", **options
    )
    connector._client = httpx.Client(
        base_url="https://acme.atlassian.net",
        auth=connector._client.auth,
        transport=httpx.MockTransport(handler),
    )
    return connector


def test_jira_cloud_pages_by_token_with_basic_auth():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/comment"):
            body = [{"author": {"displayName": f"U{i}"}, "body": f"c{i}"} for i in range(3)]
            return httpx.Response(200, json={"comments": body, "total": 3})
        if "nextPageToken" not in request.url.params:
            issues = [jira_issue("ACME-1", comments_total=3)]
            return httpx.Response(200, json={"issues": issues, "nextPageToken": "t2"})
        return httpx.Response(200, json={"issues": [jira_issue("ACME-2")], "isLast": True})

    connector = cloud_connector(handler, email="dev@example.com", api_token="secret")
    since = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
    tickets = list(connector.iter_tickets(since))

    assert [t.key for t in tickets] == ["ACME-1", "ACME-2"]
    assert tickets[0].url == "https://acme.atlassian.net/browse/ACME-1"
    assert [c.author for c in tickets[0].comments] == ["U0", "U1", "U2"]
    searches = [r for r in requests if r.url.path == "/rest/api/2/search/jql"]
    assert len(searches) == 2
    assert searches[1].url.params["nextPageToken"] == "t2"
    jql = searches[0].url.params["jql"]
    assert jql == '(project = ACME) AND updated >= "2024/05/01 12:00" ORDER BY updated ASC'
    assert searches[0].url.params["expand"] == "changelog"
    assert all(r.headers["Authorization"].startswith("Basic ") for r in requests)


def test_jira_cloud_fetches_a_changelog_the_search_cut_short():
    inline = [history("2024-05-01T10:00:00", "timeoriginalestimate", None, "1800")]
    full = [
        *inline,
        history("2024-05-01T12:00:00", "timeoriginalestimate", "1800", "2700"),
        history("2024-05-02T10:00:00", "timespent", None, "3600"),
    ]
    issue = estimated_issue(inline, estimate=2700, total=3)
    pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/changelog"):
            start = int(request.url.params["startAt"])
            pages.append(start)
            values = full[start : start + 2]
            return httpx.Response(
                200, json={"values": values, "total": 3, "isLast": start + 2 >= 3}
            )
        return httpx.Response(200, json={"issues": [issue], "isLast": True})

    [ticket] = cloud_connector(handler).iter_tickets()

    assert pages == [0, 2]
    assert ticket.initial_estimate_seconds == 2700
    assert ticket.last_logged == datetime(2024, 5, 2, 10, tzinfo=UTC)


def jira_connector(handler, cls: type[JiraConnector] = JiraConnector) -> JiraConnector:
    connector = cls("jira", "https://jira.example", "project in (ACME, WEB)")
    connector._client = httpx.Client(
        base_url="https://jira.example", transport=httpx.MockTransport(handler)
    )
    return connector


@pytest.mark.parametrize(
    ("cls", "path"),
    [(JiraConnector, "/rest/api/2/search"), (JiraCloudConnector, "/rest/api/2/search/jql")],
)
def test_jira_fetches_one_ticket_within_the_query(cls, path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/comment"):
            body = [{"author": {"displayName": f"U{i}"}, "body": f"c{i}"} for i in range(3)]
            return httpx.Response(200, json={"comments": body, "total": 3})
        return httpx.Response(200, json={"issues": [jira_issue("ACME-9", comments_total=3)]})

    ticket = jira_connector(handler, cls).fetch_ticket("ACME-9")

    assert ticket is not None
    assert ticket.key == "ACME-9"
    assert ticket.source == "jira"
    assert [c.author for c in ticket.comments] == ["U0", "U1", "U2"]
    search = requests[0]
    assert search.url.path == path
    assert search.url.params["jql"] == (
        '(project in (ACME, WEB)) AND key = "ACME-9" ORDER BY updated ASC'
    )
    assert search.url.params["maxResults"] == "1"
    assert search.url.params["expand"] == "changelog"
    assert "timeoriginalestimate" in search.url.params["fields"]


def test_jira_fetch_outside_the_query_or_unknown_key_is_none():
    responses = [
        httpx.Response(200, json={"issues": [], "total": 0}),
        httpx.Response(400, json={"errorMessages": ["An issue with key 'ZED-1' does not exist"]}),
    ]
    connector = jira_connector(lambda _request: responses.pop(0))

    assert connector.fetch_ticket("OTHER-1") is None
    assert connector.fetch_ticket("ZED-1") is None
    assert responses == []


def test_jira_fetch_skips_keys_of_another_shape_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    connector = jira_connector(handler)
    for key in ("18342", "acme-9", "A-1", "acme/widget#3", 'ACME-1" OR key = "X-1'):
        assert connector.fetch_ticket(key) is None


def test_jira_fetch_raises_on_server_errors():
    connector = jira_connector(lambda _request: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        connector.fetch_ticket("ACME-1")


def mantis_connector(handler) -> MantisConnector:
    connector = MantisConnector("hd", "https://hd.example", "token", project_ids=[3])
    connector._client = httpx.Client(
        base_url="https://hd.example", transport=httpx.MockTransport(handler)
    )
    return connector


def test_mantis_fetches_one_issue_of_a_configured_project():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        issue = mantis_issue(42, "2024-03-03T10:00:00+01:00")
        project = 3 if request.url.path.endswith("/42") else 7
        issue["project"] = {"id": project, "name": "Shop"}
        return httpx.Response(200, json={"issues": [issue]})

    connector = mantis_connector(handler)
    ticket = connector.fetch_ticket("42")

    assert ticket is not None
    assert ticket.key == "42"
    assert ticket.time_spent_seconds == 5400
    assert requests[0].url.path == "/api/rest/issues/42"
    assert connector.fetch_ticket("43") is None


def test_mantis_fetch_of_a_missing_issue_or_another_shape_is_none():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404, json={"message": "Issue #9 not found"})

    connector = mantis_connector(handler)
    assert connector.fetch_ticket("9") is None
    assert connector.fetch_ticket("ACME-9") is None
    assert connector.fetch_ticket("acme/widget#9") is None
    assert len(requests) == 1


def test_mantis_fetch_keeps_private_notes_out_when_configured():
    def handler(_request: httpx.Request) -> httpx.Response:
        issue = {**mantis_issue(42, "2024-03-03T10:00:00+01:00"), "project": {"id": 3}}
        return httpx.Response(200, json={"issues": [issue]})

    connector = mantis_connector(handler)
    connector.include_private_notes = False
    ticket = connector.fetch_ticket("42")

    assert ticket is not None
    assert [c.body for c in ticket.comments] == ["Díky, funguje."]
