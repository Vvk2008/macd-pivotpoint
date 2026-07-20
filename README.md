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
tests/                       pytest unit + integration tests
```

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
  crossed -- essentially no room, so it gets stopped out constantly.
- `strategy/pivot_bounce_strategy.py` ("bounce", the CLI default) -- price
  must touch a support/resistance level first, then a MACD crossover within
  a confirmation window confirms the bounce. Stop/target sit one pivot step
  beyond the touched level, giving the stop real room.
- `strategy/macd_only_strategy.py` -- plain MACD crossover with an
  ATR-based stop/target, no pivot dependency at all. Exists purely as a
  baseline to check whether the pivot confluence adds anything over MACD by
  itself.

## Validation: does it actually work?

`scripts/validate.py` splits the data chronologically and runs all three
strategies plus a buy-and-hold baseline on each half independently, so a
result has to hold up out-of-sample to mean anything. Use `--oos-years N`
for a fixed-date cutoff (recommended when comparing multiple instruments
with different bar counts) or `--split FRACTION` for a bar-count split:

```bash
python scripts/validate.py data/raw/EURUSD60.csv --oos-years 5
```

Result across all 5 pairs in `data/raw/` (hourly, 2010-07 to 2026-07,
last 5 years held out as out-of-sample, ~11 years in-sample):

| Pair | IS PF (breakout/bounce/macd) | OOS PF (breakout/bounce/macd) | OOS buy&hold |
|---|---|---|---|
| EUR/USD | 0.78 / 0.85 / 0.96 | 0.68 / 0.86 / 0.85 | -3.1% |
| GBP/USD | 0.72 / 0.90 / 0.98 | 0.55 / 0.83 / 0.92 | -2.3% |
| USD/CAD | 0.59 / 0.87 / 0.88 | 0.49 / 0.78 / 0.84 | +11.2% |
| USD/CHF | 0.62 / 0.86 / 0.93 | 0.47 / 0.75 / 0.81 | -12.1% |
| USD/JPY | 0.72 / 0.87 / 0.92 | 0.84 / 0.88 / 0.95 | +47.6% |

**Conclusion: this rule family does not have edge, full stop.** 30 out of 30
strategy/pair/period combinations (3 strategies x 5 pairs x 2 periods) came
back with profit factor < 1. This isn't a single-instrument fluke or a
curve-fit to one time window -- it's consistent across 5 different
currency pairs and two independent multi-year periods (~11 years in-sample,
5 years out-of-sample). Buy-and-hold, doing nothing, beats every active
variant in every single case, sometimes by a wide margin (USD/JPY OOS:
+47.6% doing nothing vs. -24.7% to -66.4% trading).

The ranking is consistent too: bounce > macd_only > breakout everywhere,
confirming the fixes made earlier (real stop room, rejecting overshot
entries) are real improvements -- they just aren't enough to flip a
fundamentally negative-edge rule set into a positive one. `macd_only`
losing as badly as it does (no pivot dependency at all) confirms the
problem isn't specific to how pivots were used here; it's underlying
whipsaw in the MACD crossover as an entry signal at this timeframe.

What hasn't been tried: other timeframes (this is H1 only -- MACD crossover
strategies are frequently pitched at H4/daily instead, where noise-to-signal
is better), a trend/regime filter to skip choppy periods, and walk-forward
re-optimization rather than one fixed parameter set across 16 years. Given
how uniform the negative result already is across 5 pairs and 2 periods,
though, the honest expectation is that these would soften the loss (as the
bounce fix did) rather than produce a genuinely profitable system.
