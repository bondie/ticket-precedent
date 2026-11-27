from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..models import Ticket


class Connector(Protocol):
    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]: ...


@runtime_checkable
class TicketFetcher(Protocol):
    def fetch_ticket(self, key: str) -> Ticket | None:
        """One ticket by key, or None when it does not exist or is outside the source's scope."""
        ...
