"""Download and cache nflverse play-by-play and schedules as parquet.

Usage:
    python -m src.ingest                 # all seasons 1999..current, skip cached
    python -m src.ingest --seasons 2023 2024
    python -m src.ingest --refresh       # re-download everything

Finished seasons never change, so they are downloaded once. The current
season is re-downloaded if the cached copy is older than 12 hours (nflverse
updates pbp nightly during the season).
"""

from __future__ import annotations

import argparse
import time

import nflreadpy as nfl
import polars as pl

from . import config

CURRENT_SEASON_MAX_AGE_S = 12 * 3600


def pbp_path(season: int):
    return config.RAW_DIR / f"pbp_{season}.parquet"


def schedules_path():
    return config.RAW_DIR / "schedules.parquet"


def _is_fresh(path, season: int, current: int) -> bool:
    if not path.exists():
        return False
    if season < current:
        return True
    return (time.time() - path.stat().st_mtime) < CURRENT_SEASON_MAX_AGE_S


def download_pbp(seasons: list[int], refresh: bool = False) -> None:
    current = nfl.get_current_season()
    for season in seasons:
        path = pbp_path(season)
        if not refresh and _is_fresh(path, season, current):
            print(f"  pbp {season}: cached")
            continue
        df = nfl.load_pbp(season)
        if df.height == 0:
            print(f"  pbp {season}: no data published yet, skipping")
            continue
        df.write_parquet(path)
        print(f"  pbp {season}: {df.height:,} plays -> {path.name}")


def download_schedules(refresh: bool = False) -> None:
    path = schedules_path()
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < CURRENT_SEASON_MAX_AGE_S:
        print("  schedules: cached")
        return
    df = nfl.load_schedules(True)  # all seasons, includes spread_line / total_line / espn id
    df.write_parquet(path)
    print(f"  schedules: {df.height:,} games -> {path.name}")


def cached_pbp_seasons() -> list[int]:
    return sorted(int(f.stem.split("_")[1]) for f in config.RAW_DIR.glob("pbp_*.parquet"))


def load_pbp(seasons: list[int] | None = None, columns: list[str] | None = None) -> pl.DataFrame:
    """Read cached pbp.

    Pass `columns` whenever possible: the full table is 372 columns x 1.3M rows
    and won't fit comfortably in a laptop's memory. Seasons can have slightly
    different column types, so concat with diagonal_relaxed (missing columns
    become null, types are upcast).
    """
    seasons = seasons or cached_pbp_seasons()
    files = [pbp_path(s) for s in seasons if pbp_path(s).exists()]
    if not files:
        raise FileNotFoundError("No cached pbp found. Run: python -m src.ingest")
    frames = []
    for f in files:
        if columns:
            available = set(pl.read_parquet_schema(f))
            frames.append(pl.read_parquet(f, columns=[c for c in columns if c in available]))
        else:
            frames.append(pl.read_parquet(f))
    return pl.concat(frames, how="diagonal_relaxed")


def load_schedules() -> pl.DataFrame:
    return pl.read_parquet(schedules_path())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="*", help="default: 1999..current")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    args = ap.parse_args()

    config.ensure_dirs()
    current = nfl.get_current_season()
    seasons = args.seasons or list(range(config.FIRST_PBP_SEASON, current + 1))
    print(f"Downloading pbp for {seasons[0]}..{seasons[-1]} into {config.RAW_DIR}")
    download_pbp(seasons, refresh=args.refresh)
    download_schedules(refresh=args.refresh)


if __name__ == "__main__":
    main()
