"""Build the clean game-state table: one row per play.

Usage:
    python -m src.features            # reads data/raw, writes data/processed/game_states.parquet

Conventions (read these before using the table):
  * Every game-state column describes the situation BEFORE the snap
    (nflverse convention). That's what a model can know at that moment.
  * "posteam" view: score_diff = posteam score - defteam score.
  * Spreads: nflverse `spread_line` is positive when the HOME team is favoured
    (e.g. +6.5 = home favoured by 6.5). We store it as `home_spread` and also
    `posteam_spread` = expected final margin for the team with the ball.
  * Labels: home_win / posteam_win are 1, 0, or 0.5 for a tie (16 ties since
    1999). `is_tie` flags them so later phases can drop or half-weight them.
  * nflverse's own ep / wp / vegas_wp are kept with a `bench_` prefix. They are
    for benchmarking only and must NEVER be used as model inputs.
  * Timestamps: `snap_time_utc` comes from nflverse `time_of_day`. The data
    dictionary calls it "HH:MM:SS", but from 2003 on it is a full ISO-8601 UTC
    timestamp (sometimes with milliseconds). 2001-02 holds only HH:MM:SS (no
    date or timezone), and 1999-2000 holds nothing; both are left null.
    `end_time_utc` comes from `end_clock_time` (mostly 2022+), which looks like
    the wall-clock time the play ended, despite its description ("game time
    at the end of a play"). Phase 1's report checks both against ESPN.
"""

from __future__ import annotations

import polars as pl

from . import config
from .ingest import load_pbp, load_schedules

# Play types we keep. Rows with no play_type are administrative (timeouts,
# two-minute warning, end of quarter); the next real play already carries the
# updated state, so they'd only add duplicates.
KEEP_PLAY_TYPES = [
    "pass", "run", "kickoff", "punt", "field_goal", "extra_point",
    "no_play", "qb_kneel", "qb_spike",
]

ISO_FMT = "%Y-%m-%dT%H:%M:%S%.fZ"  # %.f also matches "no fractional part"


def parse_utc(col: str) -> pl.Expr:
    """Parse an nflverse ISO-8601 'Z' timestamp string to a UTC datetime.

    Anything that isn't a full ISO timestamp (e.g. bare "13:05:22" in
    2001-02) becomes null instead of being guessed at.
    """
    return (
        pl.col(col)
        .cast(pl.String)
        .str.strptime(pl.Datetime("ms", "UTC"), ISO_FMT, strict=False)
    )


def _schedule_cols(schedules: pl.DataFrame) -> pl.DataFrame:
    """Per-game info taken from schedules (the authoritative source for
    lines and final scores)."""
    return schedules.select(
        "game_id",
        pl.col("espn").cast(pl.String).alias("espn_game_id"),
        pl.col("gameday").cast(pl.String),
        pl.col("gametime").cast(pl.String),
        pl.col("spread_line").cast(pl.Float64).alias("sched_spread_line"),
        pl.col("total_line").cast(pl.Float64).alias("sched_total_line"),
        pl.col("home_score").cast(pl.Float64).alias("home_final"),
        pl.col("away_score").cast(pl.Float64).alias("away_final"),
    )


def build_game_states(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    df = pbp.filter(
        pl.col("posteam").is_not_null()
        & (pl.col("posteam") != "")
        & pl.col("play_type").is_in(KEEP_PLAY_TYPES)
    )

    df = df.join(_schedule_cols(schedules), on="game_id", how="left")

    is_home = pl.col("posteam") == pl.col("home_team")
    qtr = pl.col("qtr").cast(pl.Int8)

    df = df.with_columns(
        is_home.cast(pl.Int8).alias("posteam_is_home"),
        qtr.alias("qtr"),
        (qtr >= 5).cast(pl.Int8).alias("is_overtime"),
        # Lines: prefer schedules, fall back to the copy inside pbp.
        pl.coalesce("sched_spread_line", pl.col("spread_line").cast(pl.Float64)).alias("home_spread"),
        pl.coalesce("sched_total_line", pl.col("total_line").cast(pl.Float64)).alias("total_line_"),
    ).with_columns(
        pl.when(is_home).then(pl.col("home_spread")).otherwise(-pl.col("home_spread")).alias("posteam_spread"),
    )

    # Pre-snap scores. nflverse's posteam_score/defteam_score are pre-play.
    # ~0.02% of rows have them null (mostly 1999-2000); those rows are dropped
    # below because a WP row without a score is useless.
    df = df.with_columns(
        pl.col("score_differential").cast(pl.Float64).alias("score_diff"),
        pl.when(is_home).then(pl.col("posteam_score")).otherwise(pl.col("defteam_score")).cast(pl.Float64).alias("home_score_pre"),
        pl.when(is_home).then(pl.col("defteam_score")).otherwise(pl.col("posteam_score")).cast(pl.Float64).alias("away_score_pre"),
    )

    # Receives the 2nd-half kickoff: only meaningful in the first half.
    # home_opening_kickoff = 1 means home received the opening kickoff, so the
    # AWAY team receives the second-half kickoff (and vice versa).
    home_got_opening = pl.col("home_opening_kickoff").cast(pl.Int8) == 1
    df = df.with_columns(
        pl.when(qtr <= 2)
        .then(
            pl.when(home_got_opening).then(~is_home).otherwise(is_home).cast(pl.Int8)
        )
        .otherwise(pl.lit(0, pl.Int8))
        .alias("receive_2h_ko"),
    )

    # Final result + labels (ties = 0.5).
    home_margin = pl.col("home_final") - pl.col("away_final")
    home_win = (
        pl.when(home_margin > 0).then(1.0)
        .when(home_margin < 0).then(0.0)
        .when(home_margin == 0).then(0.5)
        .otherwise(None)
    )
    df = df.with_columns(
        home_margin.alias("home_margin"),
        home_win.alias("home_win"),
        (home_margin == 0).cast(pl.Int8).alias("is_tie"),
    ).with_columns(
        pl.when(is_home).then(pl.col("home_win")).otherwise(1.0 - pl.col("home_win")).alias("posteam_win"),
        pl.when(is_home).then(pl.col("home_margin")).otherwise(-pl.col("home_margin")).alias("posteam_final_margin"),
    )

    # Wall-clock timestamps.
    df = df.with_columns(
        parse_utc("time_of_day").alias("snap_time_utc"),
        parse_utc("end_clock_time").alias("end_time_utc"),
    )

    out = df.select(
        # identifiers
        "game_id", "espn_game_id", "season", pl.col("week").cast(pl.Int16), "season_type",
        pl.col("play_id").cast(pl.Int64), "play_type", "desc",
        "home_team", "away_team", "posteam", "defteam", "posteam_is_home",
        # score
        "score_diff", "home_score_pre", "away_score_pre",
        # field position / down
        pl.col("down").cast(pl.Int8), pl.col("ydstogo").cast(pl.Int16),
        pl.col("yardline_100").cast(pl.Float64), pl.col("goal_to_go").cast(pl.Int8),
        # clock
        "qtr", "is_overtime", "game_half",
        pl.col("game_seconds_remaining").cast(pl.Float64),
        pl.col("half_seconds_remaining").cast(pl.Float64),
        # timeouts
        pl.col("posteam_timeouts_remaining").cast(pl.Int8).alias("posteam_timeouts"),
        pl.col("defteam_timeouts_remaining").cast(pl.Int8).alias("defteam_timeouts"),
        pl.col("home_timeouts_remaining").cast(pl.Int8).alias("home_timeouts"),
        pl.col("away_timeouts_remaining").cast(pl.Int8).alias("away_timeouts"),
        "receive_2h_ko",
        # pregame lines
        "home_spread", "posteam_spread", pl.col("total_line_").alias("total_line"),
        # outcome
        "home_final", "away_final", "home_margin", "posteam_final_margin",
        "home_win", "posteam_win", "is_tie",
        # nflverse benchmarks (NOT model inputs)
        pl.col("ep").alias("bench_ep"),
        pl.col("wp").alias("bench_wp"),
        pl.col("vegas_wp").alias("bench_vegas_wp"),
        pl.col("home_wp").alias("bench_home_wp"),
        pl.col("vegas_home_wp").alias("bench_vegas_home_wp"),
        # timing
        "gameday", "gametime", "snap_time_utc", "end_time_utc",
    )

    out = out.filter(pl.col("score_diff").is_not_null() & pl.col("game_seconds_remaining").is_not_null())
    out = out.sort("season", "game_id", "play_id")
    out = out.with_columns(pl.int_range(pl.len()).over("game_id").cast(pl.Int32).alias("play_index"))
    return out


# Raw pbp columns this module reads (keeps memory low).
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "play_id", "play_type", "desc",
    "home_team", "away_team", "posteam", "defteam", "qtr", "game_half",
    "score_differential", "posteam_score", "defteam_score",
    "down", "ydstogo", "yardline_100", "goal_to_go",
    "game_seconds_remaining", "half_seconds_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "home_timeouts_remaining", "away_timeouts_remaining",
    "home_opening_kickoff", "spread_line", "total_line",
    "ep", "wp", "vegas_wp", "home_wp", "vegas_home_wp",
    "time_of_day", "end_clock_time",
]


def main() -> None:
    from .ingest import cached_pbp_seasons

    config.ensure_dirs()
    schedules = load_schedules()
    # One season at a time: peak memory stays at a single season's worth.
    parts = [
        build_game_states(load_pbp([s], columns=PBP_COLUMNS), schedules)
        for s in cached_pbp_seasons()
    ]
    states = pl.concat(parts, how="vertical_relaxed")
    states.write_parquet(config.GAME_STATES_PATH)
    print(f"Wrote {states.height:,} plays from {states['game_id'].n_unique():,} games -> {config.GAME_STATES_PATH}")


if __name__ == "__main__":
    main()
