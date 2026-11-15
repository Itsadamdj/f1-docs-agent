"""Shared HTTP client: retries with backoff, per-host rate limiting, disk cache.

Every external call (ESPN, Kalshi, Polymarket) goes through `get_json` so that
  * transient failures (429 / 5xx / dropped connections) are retried with
    exponential backoff instead of crashing a long backfill,
  * we never hammer one host faster than `min_interval` seconds per request,
  * repeated calls for immutable data (settled markets, finished games) are
    served from data/cache/http instead of the network.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

# Conservative defaults. Kalshi/Polymarket publish much higher limits, but a
# backfill has no reason to go fast and being polite avoids 429s entirely.
DEFAULT_MIN_INTERVAL = {
    "site.api.espn.com": 0.25,
    "api.elections.kalshi.com": 0.12,
    "gamma-api.polymarket.com": 0.05,
    "clob.polymarket.com": 0.05,
}

_session: requests.Session | None = None
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.0,  # sleeps 1s, 2s, 4s, 8s ... between attempts
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "nfl-win-probability research client"
        _session = s
    return _session


def _throttle(host: str) -> None:
    gap = DEFAULT_MIN_INTERVAL.get(host, 0.2)
    with _lock:
        wait = _last_call.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def _cache_path(url: str, params: dict | None) -> Path:
    key = url + "?" + urlencode(sorted((params or {}).items()))
    h = hashlib.sha1(key.encode()).hexdigest()
    host = urlparse(url).netloc.replace(":", "_")
    return config.HTTP_CACHE_DIR / host / f"{h}.json"


def get_json(
    url: str,
    params: dict | None = None,
    *,
    cache: bool = True,
    ttl_seconds: float | None = None,
    timeout: float = 30,
) -> Any:
    """GET a JSON endpoint.

    cache=True with ttl_seconds=None means "cache forever": use that only for
    data that can no longer change (finished games, settled markets).
    """
    path = _cache_path(url, params)
    if cache and path.exists():
        age = time.time() - path.stat().st_mtime
        if ttl_seconds is None or age < ttl_seconds:
            return json.loads(path.read_text(encoding="utf-8"))

    _throttle(urlparse(url).netloc)
    resp = session().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)  # atomic on both Windows and POSIX
    return data
