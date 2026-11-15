# NFL Live Win Probability + Prediction-Market Backtest

Estimates each NFL team's chance of winning, updated after every play, using
**expected points (EP)** as the core game-state signal. It then tests those
estimates against Kalshi (and Polymarket) game-winner markets, both on
accuracy and on whether trading the disagreements would have made money
after fees.

This folder is a self-contained project. It shares nothing with the rest of
the repository it sits in, and can be moved into a repository of its own as is.

**Status:** Phase 1 (data pipeline) is done. Phases 2–5 haven't started.

---

## Setup (Windows and macOS)

Requires Python 3.11 or newer.

```bash
cd nfl-win-probability
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

No API keys are needed. Every data source used here is public and read-only.
`.env.example` shows where a key would go if that ever changes.

## Running Phase 1

```bash
python -m src.ingest          # download pbp 1999..now + schedules into data/raw/ (~310 MB, about 1 minute)
python -m src.features        # build data/processed/game_states.parquet (one row per play)
python -m src.phase1_report   # coverage, sanity and timestamp checks -> reports/phase1/
pytest -q                     # unit tests (no network needed)
```

Finished seasons are downloaded once. The current season is re-downloaded if
the cached copy is more than 12 hours old. Add `--refresh` to force a fresh download.

---

## Folder structure

```
nfl-win-probability/
├── README.md
├── requirements.txt
├── .env.example            # optional keys (none needed today)
├── data/                   # git-ignored, rebuilt by the pipeline
│   ├── raw/                #   pbp_<season>.parquet, schedules.parquet
│   ├── processed/          #   game_states.parquet (+ model features later)
│   └── cache/http/         #   cached ESPN / Kalshi / Polymarket responses
├── models/                 # trained EP / WP models (Phases 2-3)
├── reports/                # charts, CSVs, HTML report per phase
├── src/
│   ├── config.py           # paths + API base URLs
│   ├── http_utils.py       # retries, backoff, per-host rate limit, disk cache
│   ├── ingest.py           # nflverse download + parquet cache            [Phase 1]
│   ├── features.py         # clean game-state table                       [Phase 1]
│   ├── espn.py             # ESPN scoreboard/summary client               [Phase 1, used in 5]
│   ├── phase1_report.py    # Phase 1 checks and charts                    [Phase 1]
│   ├── ep_model.py         # next-score multinomial model -> EP           [Phase 2]
│   ├── wp_model.py         # win-probability model                        [Phase 3]
│   ├── markets/
│   │   ├── kalshi.py       # series -> events -> markets, candles, trades [Phase 4]
│   │   └── polymarket.py   # gamma events + CLOB prices-history           [Phase 4]
│   ├── backtest.py         # timestamp matching, accuracy, paper trading  [Phase 4]
│   └── live/               # live poller, SQLite logger, terminal view    [Phase 5]
└── tests/
    ├── test_features.py    # game-state building                          [Phase 1]
    └── test_matching.py    # play <-> market-price timestamp matching     [Phase 4]
```

---

## Data sources (checked against the live APIs and docs, Nov 2026)

| Source | What we use | Notes |
|---|---|---|
| **nflverse** via `nflreadpy` 0.1.5 | `load_pbp`, `load_schedules` | Returns Polars. Play-by-play from 1999. |
| **ESPN** (no key) | `/scoreboard?dates=YYYYMMDD`, `/summary?event={id}` | Each play has a `wallclock` (UTC), down, distance, yards to end zone. nflverse schedules include the ESPN game id (`espn`). |
| **Kalshi** (no key for market data) | Series `KXNFLGAME` ("NFL Game") → one event per game → **two markets per game, one per team** (e.g. `KXNFLGAME-25SEP07ARINO-ARI`, `...-NO`) | Games start Aug 2025, so the 2025 and 2026 seasons (preseason included). Markets that settled before `GET /historical/cutoff` (currently 2026-09-16) only exist under `/historical/markets/...`. 1-minute candles carry bid/ask OHLC, at most 5,000 candles per request. `/historical/trades` has microsecond timestamps. |
| **Kalshi fees** | `fee_type = quadratic_with_maker_fees`, multiplier 1 | Taker fee = `ceil_to_cent(0.07 × contracts × P × (1−P))`. Buying at the ask is a taker trade. Maker fees apply on this series too, but we only simulate taker trades. |
| **Polymarket** (no key) | Gamma `/events?slug=nfl-<away>-<home>-<date>` → moneyline market → CLOB `/prices-history` | **Resolved games still have 1-minute history** when you pass `startTs`/`endTs` (tested back to Sept 2024). `interval=max` returns *nothing* for resolved markets, which is where the "coarse data" reputation comes from. Only one price series is available: **no historical bid/ask**. Gamma `/events` pages over 50 rows must be requested with gzip (`requests` does this automatically). |

---

## Phase 1 results: the game-state table

`data/processed/game_states.parquet`: **1,211,675 plays, 7,418 games, 1999–2026** (309 postseason games, 16 ties, 456 games with overtime).

One row per play, describing the situation **before the snap**:

| Group | Columns |
|---|---|
| ids | `game_id, espn_game_id, season, week, season_type, play_id, play_index, play_type, desc` |
| teams | `home_team, away_team, posteam, defteam, posteam_is_home` |
| score | `score_diff` (posteam − defteam), `home_score_pre, away_score_pre` |
| situation | `down, ydstogo, yardline_100, goal_to_go` |
| clock | `qtr, is_overtime, game_half, game_seconds_remaining, half_seconds_remaining` |
| timeouts | `posteam_timeouts, defteam_timeouts, home_timeouts, away_timeouts` |
| kickoff | `receive_2h_ko` (1 if this team gets the 2nd-half kickoff; always 0 after halftime) |
| lines | `home_spread` (+ = home favoured), `posteam_spread` (expected margin for team with ball), `total_line` |
| outcome | `home_final, away_final, home_margin, posteam_final_margin, home_win, posteam_win` (1 / 0 / **0.5 for a tie**), `is_tie` |
| benchmarks | `bench_ep, bench_wp, bench_vegas_wp, bench_home_wp, bench_vegas_home_wp`. These are nflverse's own numbers, kept **for comparison only and never used as inputs** |
| time | `snap_time_utc`, `end_time_utc`, `gameday`, `gametime` |

Key decisions:
- **Administrative rows are dropped**: timeouts, two-minute warnings and end-of-quarter markers. The next real play already carries the updated state, so they'd only duplicate it.
- **Lines and final scores come from schedules**, which is nflverse's authoritative per-game table.
- **Overtime is flagged** (`is_overtime`), not removed. In OT, `game_seconds_remaining` counts down the OT period itself.

### Which field is the wall-clock timestamp?

- **`time_of_day` is the snap time, in UTC.** The data dictionary says it's "HH:MM:SS", but from 2003 on it is a full ISO-8601 timestamp. 2001–02 has only HH:MM:SS with no date or timezone, so it's left null. 2024 onward has 100% coverage.
- **`end_clock_time` is the wall-clock time the play ended** (median 4.9 seconds after the snap), not "game time" as the dictionary says. Coverage is only about 90%, from 2022 on.
- Checked against an independent source: on 5,009 plays from 40 random 2024–26 games, ESPN's per-play `wallclock` differs from nflverse's snap time by a **median of −0.09 seconds, and 97.6% are within 2 seconds**. The remaining 2.4% differ by 30–60 seconds, mostly around penalties. Phase 4 will test both sources against when the market price actually moves.
- The first snap comes a median of **2.6 minutes after the scheduled kickoff**, which confirms the timezone handling.
- Only **0.03%** of consecutive snaps go backwards in time. Those will be flagged when joining to market prices.
- Important for Phase 4: these are times the play *happened*. The world learns the result of a play at about `end_time_utc` plus broadcast or data-feed delay. That is why the backtest compares prices at the play time and at +30, +60 and +120 seconds.

Sanity checks:
- Pregame spread vs final margin has a slope of 1.045 and an intercept of −0.05, so the sign convention is right.
- Favourites won 66.5% of games, and the home team won 56.2%.

Charts are in `reports/phase1/`.

---

## How the model works (plain English)

*This section grows as each phase is built.*

1. **Expected points (Phase 2).** From any situation (down, distance, field position, time left), a classifier estimates the chance that the *next* score in the half is each of: our TD, FG or safety; their TD, FG or safety; or no score. Multiply each chance by its point value and add them up to get EP: how many points this possession is "worth" on average.
2. **Win probability (Phase 3).** Score difference + EP = the expected margin right now. Combine that with time remaining, timeouts, who gets the 2nd-half kickoff and the pregame spread to predict the chance of winning. The spread's weight fades as the game goes on, because late in a game the scoreboard matters far more than the pregame opinion.
3. **Market comparison (Phase 4).** Line each play up with the Kalshi / Polymarket price at that moment and see who was closer to the truth, and whether buying when the model and the market disagreed would have made money after fees.
