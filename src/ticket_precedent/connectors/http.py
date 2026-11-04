from __future__ import annotations

import time
from typing import Any

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


def get_json(
    client: httpx.Client, path: str, params: dict[str, Any] | None = None, attempts: int = 5
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(path, params=params)
        except httpx.TransportError:
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
            continue
        if response.status_code in RETRY_STATUSES and attempt < attempts:
            time.sleep(_retry_after(response) or 2**attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value and value.isdigit():
        return float(value)
    return None
