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

`scripts/validate.py` splits the data chronologically (default 70/30) and
runs all three strategies plus a buy-and-hold baseline on each half
independently, so a result has to hold up out-of-sample to mean anything:

```bash
python scripts/validate.py data/raw/EURUSD_1H_2020-2024.csv
```

Result on EUR/USD H1, 2020-01 to 2024-08 (in-sample = first ~3.25 years,
out-of-sample = last ~1.4 years):

| Strategy | In-sample PF | In-sample return | OOS PF | OOS return |
|---|---|---|---|---|
| buy & hold | -- | -2.7% | -- | +1.3% |
| breakout | 0.68 | -77.3% | 0.58 | -58.3% |
| bounce | 0.87 | -19.3% | 0.89 | -9.3% |
| macd_only | 0.84 | -75.1% | 0.84 | -46.6% |

**Conclusion: this rule family does not have edge on EUR/USD H1.** All three
variants have profit factor < 1 in *both* independent periods -- not a
one-off curve-fit result, a consistent negative expectancy. Buy-and-hold
(literally doing nothing) beats every active variant by a wide margin in
both windows. The bounce fix meaningfully reduced *how badly* the strategy
loses (tighter drawdowns, better win rate) by giving stops real room, but
it never crossed into positive expectancy, and the MACD-only baseline
confirms the problem isn't specific to the pivot filter -- plain MACD
crossover trading loses money here too.

Before concluding "pivot + MACD can never work," the untested degrees of
freedom that remain: other instruments/timeframes (this is EUR/USD H1
only), a trend/regime filter (avoid trading MACD crossovers during chop,
which is where most of the remaining losses concentrate), and walk-forward
re-optimization rather than a single fixed parameter set. None of those
have been tried yet.
