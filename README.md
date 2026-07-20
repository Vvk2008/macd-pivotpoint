# macd-pivotpoint

Backtest for a pivot-point + MACD confluence strategy, built around EUR/USD H1
data.

## Strategy

- **Long entry:** MACD line crosses above its signal line while price trades
  above the daily pivot (PP).
- **Short entry:** MACD line crosses below its signal line while price trades
  below PP.
- **Exit:** opposite MACD crossover, or a pivot-based stop/target, whichever
  comes first.
  - Stop = nearest pivot level on the wrong side of entry.
  - Target = nearest pivot level in the direction of the trade.
- Daily pivots (PP, R1-R3, S1-S3) are computed from the *previous* trading
  day's high/low/close using the standard formula, so every level used by a
  bar is knowable before that bar trades (no lookahead).

## Why EUR/USD / forex over stocks

Pivot points were designed for continuous, near-24h markets. Forex majors
don't gap the way stocks do at the open (earnings, overnight news), so
pivot levels get tested and respected rather than skipped. EUR/USD H1 was
chosen as the first liquid, tight-spread instrument to validate the concept
before extending to other pairs or index futures/CFDs.

## Project layout

```
data/loader.py               generic OHLCV CSV loader
indicators/macd.py           EMA-based MACD (line, signal, histogram)
indicators/pivot_points.py   standard daily pivot points, no-lookahead
strategy/pivot_macd_strategy.py   entry/exit signal generation
backtester/engine.py         event-driven, bar-by-bar backtest engine
backtester/metrics.py        win rate, profit factor, Sharpe, CAGR, drawdown
scripts/run_backtest.py      CLI entry point
scripts/visualize.py         interactive HTML backtest visualizer
tests/                       pytest unit + integration tests
```

## Visualizing a backtest

```bash
python scripts/visualize.py data/raw/EURUSD240.csv --start 2023-01-01 --end 2023-04-01
```

Writes a self-contained HTML file (`<pair>_<strategy>.html`) with three
linked panels -- panning/zooming one scrolls the others to match:

- **Equity** -- full-history equity curve
- **Drawdown %** -- full-history drawdown
- **Price** -- candlesticks for the requested window (`--start`/`--end`, or
  the last `--window-days` by default) over a thin full-history close-price
  line, with pivot levels (PP solid, R1-R3/S1-S3 toggleable via the legend)
  and trade markers: green/red triangles at long/short entries, X's at
  win/loss exits, dashed connecting lines, hover text with entry/exit
  price, exit reason, and P&L.

`--strategy {bounce,macd_only,breakout}` and the same tolerance/window/
level/MACD/risk flags as `run_backtest.py` select what gets backtested and
plotted. Without `--start`/`--end` it defaults to the last 90 days of data.

## Data format

Place an OHLCV CSV at `data/raw/eurusd_h1.csv` (or anywhere else). Required
columns (case-insensitive):

- one timestamp column: `timestamp`, `datetime`, `date`, or `time`
- `open`, `high`, `low`, `close`
- `volume` is optional (defaults to 0 if missing)

```
timestamp,open,high,low,close,volume
2023-01-02 00:00:00,1.0700,1.0712,1.0695,1.0705,1234
...
```

Timestamps are parsed as UTC. If your broker's H1 candles close on the NY
5pm session boundary rather than UTC midnight, pass `--session-start-hour 17`
so pivots roll over at the right point (see below).

## Running a backtest

```bash
pip install -r requirements.txt
python scripts/run_backtest.py data/raw/eurusd_h1.csv --session-start-hour 17
```

Useful flags:

- `--macd-fast/--macd-slow/--macd-signal` (default 12/26/9)
- `--risk-per-trade` fraction of equity risked per trade (default 0.01)
- `--spread` round-trip spread cost in price units, e.g. `0.0002` = 2 pips
- `--commission` fixed commission per closed trade
- `--initial-capital` starting equity

## Tests

```bash
pytest
```

33 tests cover indicator correctness (including a no-lookahead check on
pivots), signal logic for all three strategy variants, backtester mechanics
(stop/target/sizing), the metrics module, and an end-to-end smoke test on
synthetic data.

## Strategy variants

- `strategy/pivot_macd_strategy.py` ("breakout") -- the original design:
  MACD crosses while price is already on one side of PP. Its stop ends up
  at the nearest pivot level, which is almost always the level just
  crossed -- essentially no room, so it gets stopped out constantly. It's
  been established as fundamentally broken regardless of timeframe or
  tuning; the code stays for reference but `validate.py` and
  `iterate_bounce.py` no longer run it.
- `strategy/pivot_bounce_strategy.py` ("bounce", the CLI default) -- price
  must touch a support/resistance level first, then a MACD crossover within
  a confirmation window confirms the bounce. Stop sits `stop_levels` pivot
  steps beyond the touched level; target sits `target_levels` steps beyond
  it in the trade's direction (both tuned -- see below).
- `strategy/macd_only_strategy.py` -- plain MACD crossover with an
  ATR-based stop/target, no pivot dependency at all. Exists purely as a
  baseline to check whether the pivot confluence adds anything over MACD by
  itself.

`Backtester` also blocks opening a new position on the same bar an existing
one just closed (`block_same_bar_reversal`, default on) -- a signal exit and
the opposite entry are often the same MACD crossover event, so without this
a losing trade would immediately flip into another trade off no new
information.

## Timeframe matters a lot: H1 vs H4

The original validation (H1, all 5 pairs) found zero edge anywhere --
every strategy, every pair, every period came back profit factor < 1.
Resampling the exact same strategies to H4 data (`data/raw/*240.csv`)
changed that materially: MACD's classic 12/26/9 was designed for daily
charts, and H1 turned out to be too noisy for it. H4 is the timeframe
everything below uses.

## Tuning the bounce strategy (in-sample only)

`scripts/iterate_bounce.py` is a scorecard harness used to tune the bounce
strategy against the in-sample period only (everything before the last 5
years) across all 5 H4 pairs at once -- deliberately never touching
out-of-sample data while iterating, to avoid tuning into a leak:

```bash
python scripts/iterate_bounce.py
```

Starting from the original 1-level stop/1-level target design (aggregate
PF 0.81, avg return -16.9%), three changes were tested and kept because
they held up under a robustness check (profit factor recomputed after
excluding each pair's best 3 trades, to catch results that only look good
because of one or two lucky outliers):

1. **`target_levels=3`** instead of 1 -- target-hit rate was already
   ~97-100%, so the fix was to let winners run further rather than filter
   entries (a minimum reward:risk filter was tried first and made things
   *worse* -- it disproportionately cut high-win-rate quick bounces).
2. **`tolerance_pips=8`** instead of 5 -- slightly looser touch detection,
   more trades, modest further improvement.
3. **`confirmation_window=5`** instead of 3 -- more bars allowed between
   the pivot touch and the MACD confirmation.

A wider-stop variant (`stop_levels=2, target_levels=4`) looked even better
headline-wise (aggregate PF 1.07) but failed the robustness check hard --
removing the top 3 trades per pair dropped most pairs well below
breakeven, meaning that result was a few lucky trades, not a real edge. It
was discarded. A MACD-parameter sweep (faster/slower than 12/26/9) made
things worse across the board and was also discarded.

Final in-sample scorecard, all 5 pairs, H4, ~11 years:

| Pair | Trades | Win rate | PF | Return | PF (excl. best 3 trades) |
|---|---|---|---|---|---|
| EUR/USD | 644 | 45.5% | 1.13 | +40.7% | 1.09 |
| GBP/USD | 604 | 46.0% | 1.11 | +27.5% | 1.06 |
| USD/CAD | 716 | 42.7% | 0.90 | -24.1% | 0.85 |
| USD/CHF | 715 | 41.4% | 0.90 | -26.0% | 0.83 |
| USD/JPY | 682 | 43.3% | 0.92 | -19.1% | 0.85 |
| **Aggregate** | | | **0.99** | **-0.2%** | |

Up from PF 0.81 to 0.99 (essentially breakeven in aggregate) with large,
consistent sample sizes and gains that survive removing the best few
trades per pair -- 2 of 5 pairs individually clear PF 1.0, the other 3
are close. This is real, in-sample-only progress, not a flip to
demonstrated positive edge: 3 of 5 pairs are still net negative, and the
aggregate is a rounding error from breakeven, not comfortably above it.

**Out-of-sample was checked exactly once**, incidentally, while smoke-testing
the tuned CLI defaults (not as part of the tuning loop): on EUR/USD it was
PF 0.87 / -15.2% out-of-sample vs. PF 1.13 / +40.7% in-sample -- a real
degradation, as expected for parameters tuned only on the in-sample window.
The tuning loop deliberately never used this number to adjust anything
further.

## Round 2: pivots that persist longer than a day (`--pivot-period`)

`compute_daily_pivots` originally recomputed fresh levels every single
session -- a support level that had been respected for a week would still
vanish and get replaced at midnight, which doesn't match how S/R actually
behaves. `period` (`'D'`/`'W'`/`'M'`, a pandas period alias) makes this
configurable: `'W'` holds the *previous week's* pivots for the whole
following week instead of recomputing daily; `'M'` does the same monthly.

Tested the same way as before -- IS-only scorecard across all 5 pairs,
kept only if the aggregate improved *and* survived excluding each pair's
best trades:

- **Weekly pivots** (`period='W'`) alone: aggregate PF 0.81(D) -> 0.99(D,
  tuned) -> **1.01**, with drawdowns cut dramatically across every pair
  (e.g. USD/JPY -36.5% -> -12.3%). Kept.
- **Monthly pivots**: worse (aggregate PF 0.87, only 1/5 pairs positive) --
  levels too coarse/stale for intraday touches. Discarded.
- Re-tuned `tolerance_pips` (20, up from 8 -- wider levels need a wider
  touch band), `confirmation_window` (back down to 3), `target_levels`
  (down to 2) and `stop_levels` (up to 2) specifically for the weekly
  regime, each kept only after passing the same top-N-trade-exclusion
  robustness check as round 1.

Final in-sample scorecard, all 5 pairs, H4, weekly pivots, ~11 years:

| Pair | Trades | Win rate | PF | Return | PF (excl. best 5 trades) |
|---|---|---|---|---|---|
| EUR/USD | 424 | 44.6% | 1.41 | +26.6% | 1.30 |
| GBP/USD | 407 | 41.3% | 1.22 | +14.0% | 1.11 |
| USD/CAD | 452 | 36.3% | 1.01 | +1.0% | 0.93 |
| USD/CHF | 492 | 35.0% | 0.93 | -6.3% | 0.85 |
| USD/JPY | 463 | 37.1% | 1.06 | +4.5% | 0.91 |
| **Aggregate** | | | **1.13** | **+7.9%** | |

**4 of 5 pairs individually profitable in-sample** (up from 2/5), aggregate
PF 1.13 with a genuinely positive average return, and every pair holds up
gracefully after excluding its best 5 trades -- no collapse, unlike the
wider-stop variant discarded in round 1. This is the strongest in-sample
result so far. It is still an in-sample result: out-of-sample has not been
re-checked since this round, and USD/CHF remains the weakest pair
throughout every iteration of this project so far, on every timeframe and
pivot period tried.

CLI defaults (`run_backtest.py`, `validate.py`, `visualize.py`,
`iterate_bounce.py`) now reflect this round: `--pivot-period W
--tolerance-pips 20 --confirmation-window 3 --stop-levels 2
--target-levels 2`.
