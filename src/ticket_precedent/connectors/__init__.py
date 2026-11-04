from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Protocol

from ..models import Ticket


class Connector(Protocol):
    def iter_tickets(self, since: datetime | None = None) -> Iterator[Ticket]: ...
