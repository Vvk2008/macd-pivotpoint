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

19 tests cover indicator correctness (including a no-lookahead check on
pivots), signal logic, backtester mechanics (stop/target/sizing), and an
end-to-end smoke test on synthetic data.

## Validating the strategy (next step)

Once real EUR/USD H1 data is in `data/raw/`, the honest next steps before
trusting any result:

1. Compare against baselines: buy-and-hold, MACD-only (no pivot filter),
   pivot-only (no MACD).
2. Split into in-sample/out-of-sample (or walk-forward) periods -- don't
   judge on the same data you tuned parameters against.
3. Re-run on at least one more instrument/timeframe to check the edge isn't
   curve-fit to EUR/USD H1 specifically.

The synthetic-data run used during development (`scripts/run_backtest.py`
against a randomly generated series) is a plumbing check only -- it confirms
the code runs end-to-end, not that the strategy has edge. Random-walk paths
can look profitable to a trend-following signal by pure chance in a single
run; that number is not evidence of anything.
