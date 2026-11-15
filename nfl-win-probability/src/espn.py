"""ESPN public NFL endpoints (no key needed).

    scoreboard: {ESPN_BASE}/scoreboard?dates=YYYYMMDD   -> games on a date
    summary:    {ESPN_BASE}/summary?event={espn_id}      -> drives -> plays

Each play in summary['drives']['previous'][i]['plays'] carries:
    period.number, clock.displayValue ("M:SS"), wallclock (ISO UTC),
    start.down / start.distance / start.yardsToEndzone, start.team.id,
    homeScore / awayScore (AFTER the play), type.text, text.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import config
from .http_utils import get_json


def scoreboard(date_yyyymmdd: str, ttl_seconds: float | None = 30) -> dict[str, Any]:
    return get_json(f"{config.ESPN_BASE}/scoreboard", {"dates": date_yyyymmdd}, ttl_seconds=ttl_seconds)


def summary(event_id: str, *, final: bool) -> dict[str, Any]:
    """Finished games never change, so cache them forever; live games 5s."""
    return get_json(
        f"{config.ESPN_BASE}/summary", {"event": str(event_id)},
        ttl_seconds=None if final else 5,
    )


def clock_to_seconds(display: str) -> int | None:
    try:
        m, s = display.split(":")
        return int(m) * 60 + int(float(s))
    except (AttributeError, ValueError):
        return None


def plays_frame(summary_json: dict[str, Any]) -> pd.DataFrame:
    """Flatten an ESPN summary into one row per play."""
    rows = []
    drives = (summary_json.get("drives") or {}).get("previous", []) or []
    current = (summary_json.get("drives") or {}).get("current")
    if current:
        drives = drives + [current]
    for d in drives:
        for p in d.get("plays", []) or []:
            start = p.get("start") or {}
            qtr = (p.get("period") or {}).get("number")
            clock_s = clock_to_seconds((p.get("clock") or {}).get("displayValue"))
            rows.append({
                "espn_play_id": p.get("id"),
                "sequence": int(p.get("sequenceNumber") or 0),
                "qtr": qtr,
                "clock_seconds": clock_s,
                # Same convention as nflverse: regulation counts down from 3600;
                # in OT this is seconds left in the OT period.
                "game_seconds_remaining": (
                    (4 - qtr) * 900 + clock_s if qtr and qtr <= 4 and clock_s is not None else clock_s
                ),
                "down": start.get("down"),
                "distance": start.get("distance"),
                "yards_to_endzone": start.get("yardsToEndzone"),
                "offense_team_id": ((start.get("team") or {}).get("id")),
                "home_score_after": p.get("homeScore"),
                "away_score_after": p.get("awayScore"),
                "play_type": (p.get("type") or {}).get("text"),
                "text": p.get("text"),
                "wallclock": pd.to_datetime(p.get("wallclock"), utc=True, errors="coerce"),
                "modified": pd.to_datetime(p.get("modified"), utc=True, errors="coerce"),
            })
    return pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True) if rows else pd.DataFrame()
