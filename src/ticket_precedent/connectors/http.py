from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}
RATE_LIMIT_STATUSES = {403, 429}

logger = logging.getLogger(__name__)


class RateLimitError(RuntimeError):
    pass


def get_json(
    client: httpx.Client,
    path: str,
    params: dict[str, Any] | None = None,
    attempts: int = 5,
    rate_limit_wait: float | None = None,
) -> Any:
    return get_response(client, path, params, attempts, rate_limit_wait).json()


def get_response(
    client: httpx.Client,
    path: str,
    params: dict[str, Any] | None = None,
    attempts: int = 5,
    rate_limit_wait: float | None = None,
) -> httpx.Response:
    """GET with retries. `rate_limit_wait` (seconds) turns on `x-ratelimit-*` handling."""
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(path, params=params)
        except httpx.TransportError:
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
            continue
        reset = _rate_limit_reset(response) if rate_limit_wait is not None else None
        if reset is not None:
            _wait_until(reset, rate_limit_wait or 0.0, last=attempt == attempts)
            continue
        if _retryable(response, rate_limit_wait) and attempt < attempts:
            time.sleep(_retry_after(response) or 2**attempt)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("unreachable")


def _retryable(response: httpx.Response, rate_limit_wait: float | None) -> bool:
    if response.status_code in RETRY_STATUSES:
        return True
    # GitHub's secondary rate limit is a 403 with Retry-After and requests left.
    delay = _retry_after(response)
    return (
        rate_limit_wait is not None
        and response.status_code in RATE_LIMIT_STATUSES
        and delay is not None
        and delay <= rate_limit_wait
    )


def _rate_limit_reset(response: httpx.Response) -> datetime | None:
    if response.status_code not in RATE_LIMIT_STATUSES:
        return None
    if response.headers.get("x-ratelimit-remaining") != "0":
        return None
    value = response.headers.get("x-ratelimit-reset", "")
    if not value.isdigit():
        return None
    return datetime.fromtimestamp(int(value), UTC)


def _wait_until(reset: datetime, limit: float, last: bool) -> None:
    seconds = max((reset - datetime.now(UTC)).total_seconds(), 0.0) + 1
    if last or seconds > limit:
        local = (reset + timedelta(seconds=1)).astimezone()
        raise RateLimitError(
            f"API rate limit exhausted until {local:%Y-%m-%d %H:%M %Z} "
            f"(in {seconds / 60:.0f} min; rate_limit_wait is {limit:.0f} s). "
            "Configure a token for a higher limit, or raise rate_limit_wait to wait for it."
        )
    logger.warning("API rate limit exhausted, waiting %.0f s for the reset", seconds)
    time.sleep(seconds)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value and value.isdigit():
        return float(value)
    return None
