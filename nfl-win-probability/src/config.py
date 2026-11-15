"""Central place for paths and constants.

Everything uses pathlib so the same code runs on Windows and macOS.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # untouched downloads (one parquet per season)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned tables built from raw
HTTP_CACHE_DIR = DATA_DIR / "cache" / "http"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# nflverse play-by-play starts in 1999. Wall-clock play timestamps only exist
# from 2011 on, but older seasons are still useful for training EP/WP.
FIRST_PBP_SEASON = 1999

GAME_STATES_PATH = PROCESSED_DIR / "game_states.parquet"

# Public API base URLs (verified against the docs in Nov 2026).
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB = "https://clob.polymarket.com"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, HTTP_CACHE_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
