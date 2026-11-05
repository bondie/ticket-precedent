from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Comment:
    author: str
    created: datetime | None
    body: str


@dataclass(frozen=True)
class Link:
    type: str
    key: str


@dataclass
class Ticket:
    source: str
    key: str
    project: str
    title: str
    description: str
    url: str
    status: str = ""
    resolution: str = ""
    created: datetime | None = None
    updated: datetime | None = None
    time_spent_seconds: int | None = None
    issue_type: str = ""
    initial_estimate_seconds: int | None = None
    last_logged: datetime | None = None
    comments: list[Comment] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    kind: str
    ordinal: int
    text: str
