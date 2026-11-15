"""Phase 1 report: data coverage, sanity checks, and timestamp validation.

Usage:
    python -m src.phase1_report            # writes reports/phase1/*

The key question this answers for later phases: when nflverse says a play
happened at time T, is that really when it happened? We check it against an
independent source (ESPN's per-play `wallclock`) and against the scheduled
kickoff time.
"""

from __future__ import annotations

import json
import random
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # no display needed (works headless, on Windows and macOS)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from . import config, espn

OUT = config.REPORTS_DIR / "phase1"
ET = ZoneInfo("America/New_York")


def load_states() -> pd.DataFrame:
    return pl.read_parquet(config.GAME_STATES_PATH).to_pandas()


# ---------------------------------------------------------------- coverage
def coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("season")
    cov = pd.DataFrame({
        "games": g["game_id"].nunique(),
        "plays": g.size(),
        "pct_snap_time": g["snap_time_utc"].apply(lambda s: s.notna().mean() * 100),
        "pct_end_time": g["end_time_utc"].apply(lambda s: s.notna().mean() * 100),
        "pct_spread": g["home_spread"].apply(lambda s: s.notna().mean() * 100),
    })
    return cov.round(1)


def plot_coverage(cov: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].bar(cov.index, cov["games"], color="#4C72B0")
    ax[0].set_title("Games per season in the game-state table")
    ax[0].set_xlabel("season")
    ax[1].plot(cov.index, cov["pct_snap_time"], "o-", label="snap_time_utc (time_of_day)")
    ax[1].plot(cov.index, cov["pct_end_time"], "s-", label="end_time_utc (end_clock_time)")
    ax[1].axvspan(2023.5, cov.index.max() + 0.5, color="orange", alpha=0.15, label="seasons with market data")
    ax[1].set_ylim(-3, 103)
    ax[1].set_title("% of plays with a wall-clock timestamp")
    ax[1].set_xlabel("season")
    ax[1].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------- timestamp validation
def kickoff_offsets(df: pd.DataFrame) -> pd.Series:
    """Minutes between the first snap and the scheduled kickoff (schedules
    give gameday + gametime in US Eastern). Should be ~0-10 minutes."""
    first = df[df["snap_time_utc"].notna() & df["gametime"].notna()].groupby("game_id").agg(
        first_snap=("snap_time_utc", "min"), gameday=("gameday", "first"), gametime=("gametime", "first"),
        season=("season", "first"),
    )
    sched = pd.to_datetime(first["gameday"] + " " + first["gametime"]).dt.tz_localize(ET).dt.tz_convert("UTC")
    off = (first["first_snap"] - sched).dt.total_seconds() / 60
    return off.rename("minutes").to_frame().assign(season=first["season"])


def espn_comparison(df: pd.DataFrame, n_games: int = 40, seed: int = 7) -> pd.DataFrame:
    """Match nflverse plays to ESPN plays for a random sample of 2024+ games
    on (quarter, game clock, down, yards-to-endzone) and compare timestamps.
    Only keys that are unique on BOTH sides are used, so ambiguous matches
    (e.g. a penalty and the replayed down at the same clock) are skipped."""
    recent = df[(df["season"] >= 2024) & df["espn_game_id"].notna()]
    games = sorted(recent["game_id"].unique())
    random.Random(seed).shuffle(games)
    out = []
    for gid in games[:n_games]:
        g = recent[recent["game_id"] == gid]
        try:
            e = espn.plays_frame(espn.summary(g["espn_game_id"].iloc[0], final=True))
        except Exception as exc:  # network hiccup: skip the game, don't fail the report
            print(f"  ESPN fetch failed for {gid}: {exc}")
            continue
        if e.empty:
            continue
        e = e.dropna(subset=["down", "wallclock"])
        g = g.dropna(subset=["down", "snap_time_utc"])
        key = ["qtr", "game_seconds_remaining", "down", "yardline_100"]
        e = e.rename(columns={"yards_to_endzone": "yardline_100"})
        for frame in (e, g):
            frame["qtr"] = frame["qtr"].astype(int)
            frame["down"] = frame["down"].astype(int)
            frame["game_seconds_remaining"] = frame["game_seconds_remaining"].astype(int)
            frame["yardline_100"] = frame["yardline_100"].astype(int)
        e = e.drop_duplicates(key, keep=False)
        g = g.drop_duplicates(key, keep=False)
        m = g.merge(e[key + ["wallclock", "modified"]], on=key, how="inner")
        m["game_id"] = gid
        out.append(m[["game_id", "season", "play_id", "qtr", "snap_time_utc", "end_time_utc", "wallclock", "modified"]])
    m = pd.concat(out, ignore_index=True)
    m["espn_minus_snap_s"] = (m["wallclock"] - m["snap_time_utc"]).dt.total_seconds()
    m["espn_modified_minus_snap_s"] = (m["modified"] - m["snap_time_utc"]).dt.total_seconds()
    m["end_minus_snap_s"] = (m["end_time_utc"] - m["snap_time_utc"]).dt.total_seconds()
    return m


def order_check(df: pd.DataFrame) -> dict:
    """Are snap times non-decreasing in play order? Out-of-order stamps would
    poison any time-based join with market prices."""
    d = df[(df["season"] >= 2024) & df["snap_time_utc"].notna()].sort_values(["game_id", "play_id"])
    gap = d.groupby("game_id")["snap_time_utc"].diff().dt.total_seconds()
    return {
        "plays_checked": int(gap.notna().sum()),
        "pct_backwards": round(float((gap < 0).mean() * 100), 3),
        "pct_backwards_more_than_60s": round(float((gap < -60).mean() * 100), 3),
        "median_seconds_between_snaps": float(gap.median()),
        "p95_seconds_between_snaps": float(gap.quantile(0.95)),
    }, gap


def plot_timestamps(m: pd.DataFrame, ko: pd.DataFrame, gap: pd.Series) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    x = m["espn_minus_snap_s"].clip(-30, 30)
    ax[0, 0].hist(x, bins=np.arange(-30, 31, 1), color="#4C72B0")
    ax[0, 0].set_title(f"ESPN wallclock − nflverse snap time (n={len(m):,} plays)")
    ax[0, 0].set_xlabel("seconds (clipped to ±30)")

    y = m["end_minus_snap_s"].dropna().clip(0, 40)
    ax[0, 1].hist(y, bins=np.arange(0, 41, 1), color="#55A868")
    ax[0, 1].set_title("nflverse end_time − snap time (play duration)")
    ax[0, 1].set_xlabel("seconds (clipped to 0-40)")

    k = ko["minutes"].clip(-30, 60)
    ax[1, 0].hist(k[ko["season"] >= 2024], bins=np.arange(-30, 61, 1), color="#C44E52")
    ax[1, 0].set_title("First snap − scheduled kickoff (2024+ games)")
    ax[1, 0].set_xlabel("minutes (clipped)")

    ax[1, 1].hist(gap.dropna().clip(-60, 300), bins=np.arange(-60, 301, 5), color="#8172B2")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Seconds between consecutive snaps (2024+)")
    ax[1, 1].set_xlabel("seconds (clipped to −60..300)")
    fig.tight_layout()
    fig.savefig(OUT / "timestamp_checks.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------- sanity checks
def sanity(df: pd.DataFrame) -> dict:
    games = df.groupby("game_id").first()
    # Spread sanity: home_spread should predict home_margin with slope ~1.
    ok = games["home_spread"].notna() & games["home_margin"].notna()
    slope, intercept = np.polyfit(games.loc[ok, "home_spread"], games.loc[ok, "home_margin"], 1)
    fav_win = ((games.loc[ok, "home_spread"] > 0) == (games.loc[ok, "home_margin"] > 0))[games.loc[ok, "home_spread"] != 0].mean()
    return {
        "plays": len(df),
        "games": int(df["game_id"].nunique()),
        "seasons": f"{df['season'].min()}-{df['season'].max()}",
        "postseason_games": int(games["season_type"].eq("POST").sum()),
        "tie_games": int(games["is_tie"].sum()),
        "overtime_plays": int(df["is_overtime"].sum()),
        "games_with_overtime": int(df.loc[df["is_overtime"] == 1, "game_id"].nunique()),
        "home_win_rate": round(float(games["home_win"].mean()), 4),
        "spread_vs_margin_slope": round(float(slope), 3),
        "spread_vs_margin_intercept": round(float(intercept), 3),
        "favourite_win_rate": round(float(fav_win), 4),
        "receive_2h_ko_rate_first_half": round(float(df.loc[df["qtr"] <= 2, "receive_2h_ko"].mean()), 4),
    }


def plot_distributions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    ax[0, 0].hist(df["score_diff"].clip(-35, 35), bins=71, color="#4C72B0")
    ax[0, 0].set_title("Score differential (posteam view)")
    ax[0, 1].hist(df["yardline_100"].dropna(), bins=np.arange(0.5, 100.5, 1), color="#55A868")
    ax[0, 1].set_title("yardline_100 (yards to opponent end zone)")
    dc = df["down"].fillna(0).astype(int).value_counts().sort_index()
    ax[0, 2].bar(["none\n(kickoff/PAT)", "1", "2", "3", "4"], dc.values, color="#8172B2")
    ax[0, 2].set_title("Down")
    games = df.groupby("game_id").first()
    ax[1, 0].scatter(games["home_spread"], games["home_margin"], s=3, alpha=0.25)
    ax[1, 0].plot([-20, 20], [-20, 20], "r--", lw=1)
    ax[1, 0].set_xlabel("pregame home spread (+ = home favoured)")
    ax[1, 0].set_ylabel("final home margin")
    ax[1, 0].set_title("Spread vs final margin (each dot = 1 game)")
    # Calibration of the spread itself: bin by spread, show home win rate.
    b = pd.cut(games["home_spread"], bins=np.arange(-17.5, 18, 2.5))
    wr = games.groupby(b, observed=True)["home_win"].agg(["mean", "size"])
    ax[1, 1].plot([iv.mid for iv in wr.index], wr["mean"], "o-")
    ax[1, 1].set_xlabel("home spread bin")
    ax[1, 1].set_ylabel("home win rate")
    ax[1, 1].set_title("Home win rate by pregame spread")
    ax[1, 1].grid(alpha=0.3)
    # How nflverse's benchmark WP evolves: mean bench_vegas_wp by quarter for
    # eventual winners vs losers (sanity that labels line up with posteam).
    d = df[df["is_tie"] == 0]
    piv = d.groupby(["qtr", "posteam_win"])["bench_vegas_wp"].mean().unstack()
    piv.plot(ax=ax[1, 2], marker="o")
    ax[1, 2].set_title("Mean nflverse vegas_wp by quarter\n(posteam eventually won=1 / lost=0)")
    ax[1, 2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "distributions.png", dpi=120)
    plt.close(fig)


def plot_example_game(df: pd.DataFrame, game_id: str) -> None:
    g = df[df["game_id"] == game_id].sort_values("play_id")
    fig, ax1 = plt.subplots(figsize=(13, 4.5))
    ax1.step(g["snap_time_utc"], g["home_score_pre"] - g["away_score_pre"], where="post", color="#4C72B0", label="home score diff (pre-snap)")
    ax1.set_ylabel("home − away score")
    ax2 = ax1.twinx()
    ax2.plot(g["snap_time_utc"], g["bench_vegas_home_wp"], color="#C44E52", lw=1.2, label="nflverse vegas_home_wp (benchmark)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("home WP")
    ax1.set_title(f"{game_id}: one row per play on a wall-clock axis (UTC). This is the timeline Phase 4 joins to market prices.")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "example_game_timeline.png", dpi=120)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_states()

    cov = coverage(df)
    cov.to_csv(OUT / "coverage_by_season.csv")
    plot_coverage(cov)

    san = sanity(df)
    plot_distributions(df)

    ko = kickoff_offsets(df)
    order, gap = order_check(df)
    print("Fetching ESPN summaries for timestamp validation (cached after first run)...")
    m = espn_comparison(df)
    m.to_csv(OUT / "espn_timestamp_matches.csv", index=False)
    plot_timestamps(m, ko, gap)

    ex = df[(df["season"] == 2025) & (df["season_type"] == "REG")]["game_id"].iloc[0]
    plot_example_game(df, ex)

    q = lambda s, p: round(float(s.quantile(p)), 2)
    ts = {
        "espn_matched_plays": len(m),
        "espn_games": int(m["game_id"].nunique()),
        "espn_wallclock_minus_snap_s": {"median": q(m["espn_minus_snap_s"], .5), "p05": q(m["espn_minus_snap_s"], .05), "p95": q(m["espn_minus_snap_s"], .95),
                                         "pct_within_2s": round(float((m["espn_minus_snap_s"].abs() <= 2).mean() * 100), 1)},
        "espn_modified_minus_snap_s": {"median": q(m["espn_modified_minus_snap_s"], .5)},
        "end_minus_snap_s (play duration)": {"median": q(m["end_minus_snap_s"].dropna(), .5), "p95": q(m["end_minus_snap_s"].dropna(), .95)},
        "first_snap_minus_scheduled_kickoff_min_2024+": {"median": q(ko.loc[ko.season >= 2024, "minutes"], .5),
                                                         "p05": q(ko.loc[ko.season >= 2024, "minutes"], .05), "p95": q(ko.loc[ko.season >= 2024, "minutes"], .95)},
        "snap_order": order,
    }
    summary = {"sanity": san, "timestamps": ts}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(cov.to_string())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
