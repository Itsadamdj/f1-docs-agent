"""Unit tests for the game-state table (src/features.py).

They use tiny hand-built frames, so they run in under a second with no
network or downloaded data:  pytest -q
"""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.features import PBP_COLUMNS, build_game_states


def make_pbp(rows: list[dict]) -> pl.DataFrame:
    base = {c: None for c in PBP_COLUMNS}
    base.update({
        "game_id": "2025_01_AWY_HOM", "season": 2025, "week": 1, "season_type": "REG",
        "home_team": "HOM", "away_team": "AWY", "play_type": "run", "desc": "x",
        "qtr": 1, "game_half": "Half1", "down": 1, "ydstogo": 10, "yardline_100": 75, "goal_to_go": 0,
        "game_seconds_remaining": 3600, "half_seconds_remaining": 1800,
        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
        "home_timeouts_remaining": 3, "away_timeouts_remaining": 3,
        "home_opening_kickoff": 1, "spread_line": 3.0, "total_line": 45.0,
        "ep": 0.5, "wp": 0.5, "vegas_wp": 0.5, "home_wp": 0.5, "vegas_home_wp": 0.5,
        "time_of_day": "2025-09-07T17:03:40.463Z", "end_clock_time": "2025-09-07T17:03:45.000Z",
        "posteam_score": 0, "defteam_score": 0, "score_differential": 0,
    })
    out = []
    for i, r in enumerate(rows):
        row = dict(base)
        row["play_id"] = i + 1
        row.update(r)
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


def make_sched(home_score=24, away_score=17, spread=3.0) -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["2025_01_AWY_HOM"], "espn": ["401"], "gameday": ["2025-09-07"], "gametime": ["13:00"],
        "spread_line": [spread], "total_line": [45.0], "home_score": [home_score], "away_score": [away_score],
    })


def test_home_and_away_perspective():
    pbp = make_pbp([
        {"posteam": "HOM", "defteam": "AWY", "posteam_score": 7, "defteam_score": 3, "score_differential": 4},
        {"posteam": "AWY", "defteam": "HOM", "posteam_score": 3, "defteam_score": 7, "score_differential": -4},
    ])
    s = build_game_states(pbp, make_sched()).sort("play_id")
    home, away = s.row(0, named=True), s.row(1, named=True)

    assert home["posteam_is_home"] == 1 and away["posteam_is_home"] == 0
    # Home leads 7-3 in both rows, whoever has the ball.
    assert home["home_score_pre"] == 7 and home["away_score_pre"] == 3
    assert away["home_score_pre"] == 7 and away["away_score_pre"] == 3
    # Spread +3 = home favoured by 3 -> posteam_spread is +3 for home, -3 for away.
    assert home["posteam_spread"] == 3.0 and away["posteam_spread"] == -3.0
    # Home won 24-17.
    assert home["home_win"] == 1.0 and home["posteam_win"] == 1.0
    assert away["home_win"] == 1.0 and away["posteam_win"] == 0.0
    assert away["posteam_final_margin"] == -7


def test_tie_is_half():
    pbp = make_pbp([{"posteam": "HOM", "defteam": "AWY"}, {"posteam": "AWY", "defteam": "HOM"}])
    s = build_game_states(pbp, make_sched(20, 20))
    assert s["is_tie"].to_list() == [1, 1]
    assert s["posteam_win"].to_list() == [0.5, 0.5]
    assert s["home_win"].to_list() == [0.5, 0.5]


@pytest.mark.parametrize(
    "home_opening_kickoff, posteam, qtr, expected",
    [
        (1, "HOM", 1, 0),  # home got the ball first -> away gets it after halftime
        (1, "AWY", 2, 1),
        (0, "HOM", 2, 1),
        (0, "AWY", 1, 0),
        (0, "HOM", 3, 0),  # meaningless in the 2nd half -> always 0
        (1, "AWY", 4, 0),
    ],
)
def test_receive_2h_kickoff(home_opening_kickoff, posteam, qtr, expected):
    defteam = "AWY" if posteam == "HOM" else "HOM"
    pbp = make_pbp([{"posteam": posteam, "defteam": defteam, "qtr": qtr, "home_opening_kickoff": home_opening_kickoff}])
    assert build_game_states(pbp, make_sched())["receive_2h_ko"][0] == expected


def test_timestamp_parsing_formats():
    pbp = make_pbp([
        {"posteam": "HOM", "defteam": "AWY", "time_of_day": "2025-09-07T17:03:40.463Z"},  # with ms
        {"posteam": "HOM", "defteam": "AWY", "time_of_day": "2025-09-07T17:04:12Z"},      # without ms
        {"posteam": "HOM", "defteam": "AWY", "time_of_day": "13:05:22"},                  # 2001-02 style: no date
        {"posteam": "HOM", "defteam": "AWY", "time_of_day": None},
    ])
    ts = build_game_states(pbp, make_sched()).sort("play_id")["snap_time_utc"].to_list()
    assert ts[0] == datetime(2025, 9, 7, 17, 3, 40, 463000, tzinfo=timezone.utc)
    assert ts[1] == datetime(2025, 9, 7, 17, 4, 12, tzinfo=timezone.utc)
    assert ts[2] is None  # never guess a date/timezone
    assert ts[3] is None


def test_drops_admin_rows_and_overtime_flag():
    pbp = make_pbp([
        {"posteam": None, "defteam": None, "play_type": None},          # GAME start row
        {"posteam": "HOM", "defteam": "AWY", "play_type": None},         # timeout / 2-min warning
        {"posteam": "HOM", "defteam": "AWY", "play_type": "kickoff", "down": None},
        {"posteam": "AWY", "defteam": "HOM", "qtr": 5, "game_seconds_remaining": 600},
    ])
    s = build_game_states(pbp, make_sched()).sort("play_id")
    assert s.height == 2
    assert s["is_overtime"].to_list() == [0, 1]
    assert s["play_index"].to_list() == [0, 1]


def test_spread_falls_back_to_pbp_when_schedule_missing():
    pbp = make_pbp([{"posteam": "AWY", "defteam": "HOM", "spread_line": -2.5}])
    sched = make_sched().with_columns(pl.lit(None, pl.Float64).alias("spread_line"))
    s = build_game_states(pbp, sched)
    assert s["home_spread"][0] == -2.5
    assert s["posteam_spread"][0] == 2.5  # away team favoured by 2.5
