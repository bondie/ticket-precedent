from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from ..search import QUERY_CHARS, ticket_query
from ..store import Store


@dataclass(frozen=True)
class Query:
    ticket_id: int
    key: str
    text: str
    created: str
    relevant: frozenset[int]


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def duplicate_queries(
    store: Store,
    source: str,
    link_type: str = "Duplicate",
    resolution: str = "Duplicate",
    max_chars: int = QUERY_CHARS,
) -> list[Query]:
    """Each ticket closed as a duplicate becomes a query.

    Relevant answers are the earlier tickets in its duplicate cluster: the precedent
    someone could have found at the time the duplicate was reported.
    """
    rows = store.tickets(source)
    by_key = {row["key"]: row for row in rows}
    clusters = _DisjointSet()
    for row in rows:
        for link in json.loads(row["links_json"]):
            if link["type"] == link_type and link["key"] in by_key:
                clusters.union(row["key"], link["key"])
    members: dict[str, set[str]] = defaultdict(set)
    for key in clusters.parent:
        members[clusters.find(key)].add(key)

    queries = []
    for row in rows:
        if row["resolution"] != resolution or row["key"] not in clusters.parent:
            continue
        relevant = frozenset(
            by_key[key]["id"]
            for key in members[clusters.find(row["key"])]
            if key != row["key"] and (by_key[key]["created"] or "") < (row["created"] or "")
        )
        if not relevant:
            continue
        text = ticket_query(row["title"], row["description"], max_chars)
        queries.append(Query(row["id"], row["key"], text, row["created"], relevant))
    return queries
