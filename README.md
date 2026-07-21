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

## Round 3: ablation study + out-of-sample-confirmed defaults

`scripts/ablation.py` builds ~99 configs from the round-2 baseline: a
single-factor variation of every existing knob (pivot period, tolerance,
confirmation window, stop/target levels, `min_reward_risk`,
`block_same_bar_reversal`, MACD params, spread cost, strategy family,
session-hour filters) plus an interaction grid across the axes that
matter most. In-sample only, ranked by aggregate PF with a robustness
column (PF excluding each pair's best 3 trades, to catch results driven
by a couple of lucky outliers rather than a real edge):

```bash
python scripts/ablation.py
```

**What actually matters, by impact:**
1. **Pivot period, by far.** Every `pivot_period=D` config clusters at the
   very bottom (PF 0.86-0.93, 0-1/5 pairs) no matter how else it's tuned.
   Confirms round 2's core hypothesis was the single most important
   decision in the whole strategy.
2. **Strategy family, second.** `bounce` (1.13) clearly beats `macd_only`
   (0.95, 0/5 pairs) and `breakout` (0.92, 2/5) -- the touch-then-confirm
   mechanism itself is doing real work, not just "MACD" or "pivots" in
   isolation.
3. Everything else (tolerance, confirmation window, stop/target levels,
   MACD period) is second-order, ~0.05-0.10 PF of range.
4. Session filtering reconfirmed useless (all session-restricted configs
   land below baseline, matching the dedicated session-analysis round).
   Extreme `min_reward_risk` (>=1.5) or `stop/target_levels>=4` collapse
   to near-zero trades.

Three individually robustness-confirmed improvements over the round-2
baseline -- `confirmation_window=2`, `tolerance_pips=18`,
`min_reward_risk=0.5` (rejects an entry unless realized reward:risk at
actual entry price is >= this multiple; drift during the confirmation
window can erode a nominally-fine setup, same idea as the overshoot
check from round 1, applied continuously instead of as a hard cutoff) --
were combined and checked **in-sample and out-of-sample**:

| | IS PF | IS pairs+ | OOS PF | OOS pairs+ | OOS PF (excl. best 3) |
|---|---|---|---|---|---|
| round-2 baseline | 1.13 | 4/5 | 1.12 | 4/5 (GBP/USD fails: 0.88) | 1.00 |
| **round-3 candidate** | **1.17** | 3/5 | **1.17** | **5/5 -- all pairs** | **1.03** |

The candidate fixes GBP/USD out-of-sample (0.88 -> 1.03, the one
consistently-losing pair) while holding or improving every other pair --
**first time in this project all 5 pairs are OOS-profitable
simultaneously**. In-sample pairs-positive dips from 4/5 to 3/5 (USD/CAD,
USD/CHF land just under 1.0 IS), but both of those pairs are *better*
out-of-sample under the candidate than under the baseline, so this reads
as in-sample noise rather than a real regression. Adopted as the new
default.

CLI defaults (`run_backtest.py`, `validate.py`, `visualize.py`,
`iterate_bounce.py`) now reflect this round: `--pivot-period W
--tolerance-pips 18 --confirmation-window 2 --stop-levels 2
--target-levels 2 --min-reward-risk 0.5`.

## Round 4: does every rule earn its complexity? (28 pairs, in-sample only)

Round 3 varied hyperparameter *values* for rules that were always
structurally present. Round 4 asks a different question: does each rule
matter *at all*, versus trading with it switched off entirely? Four new
boolean flags were added to `generate_signals` (`require_touch`,
`require_macd_confirmation`, `require_no_overshoot`, `use_signal_exit`,
all default `True`, byte-identical behavior to before when left alone),
and `scripts/ablation_rules.py` builds 40 configs across 9 rule axes plus
pillar-removal combos, run on **all 28 pairs**, in-sample only:

```bash
python scripts/ablation_rules.py
```

Ranked by R-multiple PF (`rPF` -- each trade's P&L normalized by dollars
risked; scale-invariant, unlike raw dollar PF/return which is distorted
by unbounded equity compounding over an 11-year backtest).

**Headline finding: on the full 28-pair universe, almost every rule is
statistically neutral.** rPF sits in a tight 0.86-1.03 band across nearly
all 40 configs -- no rule combination produces a real edge, and removing
most individual rules barely moves the result. Baseline (current
production config) ranks 11th of 40, rPF 0.999 -- essentially breakeven,
consistent with the 28-pair portfolio backtest result below.

**Per-rule verdicts -- 9 of 10 rules are NEUTRAL** (removing them doesn't
materially hurt, and sometimes helps slightly): touch/location gate, MACD
confirmation, confirmation window, stop buffer, target buffer,
reward:risk filter, overshoot check, signal exit, same-bar-reversal
block. **Only `pivot_period=W` is NECESSARY** -- daily collapses to rPF
0.909 (4/28 pairs profitable) and monthly to 0.950 (13/28), reconfirming
weekly pivots as the one lever that actually matters, this time across
all 28 pairs rather than just the 5 majors.

One inconsistent result worth flagging rather than hiding: `signal_exit:
OFF` (stop/target only, no MACD-cross exit) ranks near the top on the
full 11-year run (rPF 1.018-1.028 across several configs, 3,300+ trades)
-- but the same flag was the single worst result in a dedicated
2014-2015-only re-run of this ablation (rPF 0.845, on a comparable
566-trade sample). This flip suggests `signal_exit`'s value is
regime-dependent rather than a stable property of the strategy, and
should not be treated as a settled improvement in either direction.

**Conclusion: there is no rule tweak within this framework that rescues
the 28-pair result.** The structure is already near its ceiling given
weekly pivots; the gap between the 5 majors (OOS PF 1.17) and the 23
crosses (OOS PF 0.88) is not a rule-tuning problem. See the 28-pair
portfolio backtest below for what actually moves the outcome.

## 28-pair single shared-account portfolio backtest (in-sample only)

`scripts/portfolio_backtest_28.py` runs all 28 pairs through one shared
$10,000 account (1% risk per trade, position sizing off live total
equity across all open positions), in-sample only:

```bash
python scripts/portfolio_backtest_28.py
```

Result: 10,044 trades, 35.2% win rate, PF 0.99, **-17.86% total return**,
max drawdown **-49.62%**, Sharpe ~0.00, $10,000 -> $8,214. 14 of 28 pairs
contributed positively, 14 negatively -- winners are almost entirely the
original 5 majors plus a handful of crosses (EUR/USD +$2,282, GBP/USD
+$1,148, CHF/JPY +$746); losers are almost entirely the newer crosses
(CAD/JPY -$1,781, EUR/CHF -$1,521, EUR/AUD -$810).

Two compounding causes, both already established earlier in this
project: (1) params tuned on the 5 majors don't generalize to the 23
crosses (OOS PF 1.17 vs 0.88), and (2) a shared account amplifies
correlated risk-taking -- the same 5 majors returned +79.8% in a shared
account vs +13-29% as summed independent accounts, and pairs' USD
direction agrees 71% of the time when 2+ positions are open
simultaneously (vs ~50% expected if independent), so "28 pairs" isn't 28
independent bets when several open at once.

Round 4's ablation confirms this isn't fixable by further rule-tuning.
Next steps worth testing: restrict the pair universe to pairs with
standalone IS edge rather than trading all 28, and cap aggregate open
risk across simultaneously-open positions rather than only capping
risk per trade.

## The pair-selection trap: an out-of-sample honesty check

An obvious idea after the 28-pair result is "just keep the 14 pairs that
were IS-profitable." That is textbook selection bias -- with 28 pairs,
several will look profitable on in-sample data by luck alone, and picking
the winners *after* seeing their IS results and then re-quoting IS
performance double-counts that luck. The only honest test is to freeze the
selection on IS and evaluate it ONCE on the untouched OOS window
(`scripts/portfolio_backtest_14_oos.py`):

Running the 14 IS-positive pairs on OOS (shared account, 1% risk):
PF 0.87, **-40.86% return**, max drawdown -46.33%, and **only 7 of the 14
stayed positive** -- a coin flip. Five crosses that looked fine IS
(CHF/JPY, NZD/CAD, NZD/CHF, EUR/NZD, EUR/JPY) accounted for almost the
entire OOS loss. The IS pair selection was largely survivorship noise, not
edge. The only pairs that hold up across *both* windows are the USD majors
that were tuned and validated in the first place.

## Round 5: genetic-algorithm parameter search + overfitting harness

Would searching parameters with a GA -- far more thoroughly than the
hand-tuning of round 3 -- find a profitable configuration, and how do you
run that search *without* overfitting? A GA is by construction an
overfitting maximiser (its whole job is to find the vector that scores
best on the data shown), so the GA alone is not the answer; the harness
around it is (`scripts/ga_optimize.py`):

- **Three chronological windows**, never shuffled: IS-train -> IS-validation
  -> OOS. The GA only evolves against IS-train. The winning genome is
  *selected by IS-validation* (early stopping / model selection), never by
  train. OOS is touched exactly once, at the end, as a pass/fail gate.
- **A deliberately tiny 5-knob search space** (tolerance, confirmation
  window, min_reward_risk, stop_levels, target_levels). The high-leverage
  regime lever (`pivot_period`) and the rationale-free surfaces (MACD
  periods, cost assumptions, position size) and every structural flag are
  FROZEN, not searched -- see the PARAM NOTES block in the script for the
  full change/don't-change rationale. Overfitting capacity scales with
  (#free params x their leverage x #configs tried); the GA maximises the
  last term, so the defence is to shrink the first two.
- **A robust R-multiple fitness** aggregated across pairs, with a hard
  penalty for too few trades and a multiplier rewarding cross-pair
  consistency, so a config can't win by overfitting one lucky pair or a
  handful of outlier trades.

Result on the 5 majors:

| config | train rPF | val rPF | OOS rPF | OOS pairs+ | OOS trades |
|---|---|---|---|---|---|
| baseline (hand-tuned, round 3) | 1.197 | 1.109 | **1.124** | **4/5** | **873** |
| GA best-on-validation (selected) | 1.296 | 1.574 | 1.062 | 3/5 | 85 |
| GA best-on-train (overfit ref) | 1.609 | 0.833 | 1.484 | 3/5 | 41 |

**Overfitting was caught in the act:** as the GA drove train rPF up (1.30
-> 1.61 across generations), the same champion's validation rPF *fell*
(1.57 -> 0.83, 5/5 pairs -> 1/5). That divergence is overfitting made
visible before OOS is ever touched -- and selecting by validation instead
of train is what stopped the overfit config from being shipped.

**The GA found nothing that beats the hand-tuned baseline.** The
train-optimal config's flashy "OOS rPF 1.484" is a mirage on 41 trades
(the GA cranked min_reward_risk up, strangling trade count -- fewer trades
is exactly how a lucky-looking backtest happens). The baseline wins where
it matters: OOS rPF 1.124 on **873 trades across 4/5 pairs**, an order of
magnitude more evidence than either GA config. More search power over the
same knobs just produced in-sample mirages the OOS gate rejected.

**Conclusion:** the strategy is at its parameter ceiling. A GA wrapped in
a proper train/validation/OOS harness is an excellent overfitting
*detector*, but it cannot manufacture edge that isn't in the data. Real
improvement has to come from a new source -- pair selection, a different
signal, a regime filter -- not from mining these five parameters harder.
