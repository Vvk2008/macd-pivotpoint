"""Ablation study on the bounce strategy: vary one component at a time from
the current best config, plus a modest interaction grid, to see which
design decisions actually drive performance.

In-sample only (last 5 years held out), same discipline as the rest of the
tuning -- this is meant to explain the *existing* IS results, not to
squeeze out a new OOS-untested config.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.atr import compute_atr
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.macd_only_strategy import generate_signals as generate_macd_only_signals
from strategy.pivot_bounce_strategy import generate_signals as generate_bounce_signals
from strategy.pivot_macd_strategy import generate_signals as generate_breakout_signals
from utils import infer_pip_size

PAIRS = ["EURUSD240", "GBPUSD240", "USDCAD240", "USDCHF240", "USDJPY240"]
OOS_YEARS = 5

BASELINE = dict(
    strategy="bounce",
    pivot_period="W",
    tolerance_pips=20.0,
    confirmation_window=3,
    stop_levels=2,
    target_levels=2,
    min_reward_risk=0.0,
    block_same_bar_reversal=True,
    macd_params=(12, 26, 9),
    spread_pips=2.0,
    session_hours=None,
)

MACD_ALTS = {
    "default(12,26,9)": (12, 26, 9),
    "fast(8,17,9)": (8, 17, 9),
    "fast(5,13,6)": (5, 13, 6),
    "slow(19,39,9)": (19, 39, 9),
    "slow(24,52,18)": (24, 52, 18),
    "vfast(3,10,16)": (3, 10, 16),
}

SESSION_ALTS = {
    "all": None,
    "drop20": {0, 4, 8, 12, 16},
    "core8-16": {8, 12, 16},
    "london(8,12)": {8, 12},
    "newyork(16,20)": {16, 20},
    "overlap(12,16)": {12, 16},
}

# --- build the ~100 config list -------------------------------------------
configs = {}


def add(label, **overrides):
    cfg = {**BASELINE, **overrides}
    configs[label] = cfg


add("baseline (current best)")

for v in ["D", "W", "M"]:
    if v != BASELINE["pivot_period"]:
        add(f"pivot_period={v}", pivot_period=v)

for v in [3, 5, 8, 10, 12, 14, 16, 18, 25, 30, 35, 40]:
    add(f"tolerance_pips={v}", tolerance_pips=v)

for v in [1, 2, 4, 5, 6, 8, 10, 12, 15]:
    add(f"confirmation_window={v}", confirmation_window=v)

for v in [1, 3, 4]:
    add(f"stop_levels={v}", stop_levels=v)

for v in [1, 3, 4, 5]:
    add(f"target_levels={v}", target_levels=v)

for v in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
    add(f"min_reward_risk={v}", min_reward_risk=v)

add("block_same_bar_reversal=False", block_same_bar_reversal=False)

for name, params in MACD_ALTS.items():
    if params != BASELINE["macd_params"]:
        add(f"macd={name}", macd_params=params)

for v in [0, 1, 3, 4, 5, 8, 10]:
    add(f"spread_pips={v}", spread_pips=v)

add("strategy=macd_only", strategy="macd_only")
add("strategy=breakout", strategy="breakout")

for name, hours in SESSION_ALTS.items():
    if name != "all":
        add(f"session={name}", session_hours=hours)

# interaction grids
for pp, tol, cw in product(["D", "W"], [10, 20, 30], [1, 3, 8]):
    add(f"INTERACT pivot={pp},tol={tol},cw={cw}", pivot_period=pp, tolerance_pips=tol, confirmation_window=cw)

for sl, tl in product([1, 2, 3], [1, 2, 3]):
    add(f"INTERACT stop={sl},target={tl}", stop_levels=sl, target_levels=tl)

for tol, mrr in product([10, 20, 30], [0, 0.5, 1.0]):
    add(f"INTERACT tol={tol},mrr={mrr}", tolerance_pips=tol, min_reward_risk=mrr)

for name, params in [("default", (12, 26, 9)), ("fast", (8, 17, 9)), ("slow", (19, 39, 9))]:
    for pp in ["D", "W"]:
        add(f"INTERACT macd={name},pivot={pp}", macd_params=params, pivot_period=pp)

print(f"Built {len(configs)} configs")

# --- caching layer ----------------------------------------------------------
_pivot_cache = {}
_macd_cache = {}
_atr_cache = {}
_df_cache = {}


def in_sample_only(df):
    cutoff = df.index[-1] - pd.DateOffset(years=OOS_YEARS)
    return df[df.index < cutoff]


def get_df(pair):
    if pair not in _df_cache:
        _df_cache[pair] = in_sample_only(load_ohlcv_csv(f"data/raw/{pair}.csv"))
    return _df_cache[pair]


def get_pivots(pair, period):
    key = (pair, period)
    if key not in _pivot_cache:
        _pivot_cache[key] = compute_daily_pivots(get_df(pair), period=period)
    return _pivot_cache[key]


def get_macd(pair, params):
    key = (pair, params)
    if key not in _macd_cache:
        _macd_cache[key] = compute_macd(get_df(pair)["close"], *params)
    return _macd_cache[key]


def get_atr(pair):
    if pair not in _atr_cache:
        _atr_cache[pair] = compute_atr(get_df(pair))
    return _atr_cache[pair]


def pf_excl_top(trades, n):
    remaining = sorted(trades, key=lambda t: -t.pnl)[n:]
    if not remaining:
        return float("nan")
    gp = sum(t.pnl for t in remaining if t.pnl > 0)
    gl = -sum(t.pnl for t in remaining if t.pnl <= 0)
    return gp / gl if gl > 0 else float("inf")


def run_config(cfg):
    pfs, rets, n_trades_total, excl3s = [], [], 0, []
    for pair in PAIRS:
        df = get_df(pair)
        pip = infer_pip_size(df["close"])
        macd = get_macd(pair, cfg["macd_params"])

        if cfg["strategy"] == "bounce":
            pivots = get_pivots(pair, cfg["pivot_period"])
            signals = generate_bounce_signals(
                df, macd, pivots,
                tolerance=cfg["tolerance_pips"] * pip,
                confirmation_window=cfg["confirmation_window"],
                min_reward_risk=cfg["min_reward_risk"],
                stop_levels=cfg["stop_levels"],
                target_levels=cfg["target_levels"],
            )
        elif cfg["strategy"] == "macd_only":
            atr = get_atr(pair)
            signals = generate_macd_only_signals(df, macd, atr)
        elif cfg["strategy"] == "breakout":
            pivots = get_pivots(pair, cfg["pivot_period"])
            signals = generate_breakout_signals(df, macd, pivots)
        else:
            raise ValueError(cfg["strategy"])

        if cfg["session_hours"] is not None:
            signals = signals.copy()
            disallowed = ~np.isin(signals.index.hour, list(cfg["session_hours"]))
            signals.loc[disallowed, "long_entry"] = False
            signals.loc[disallowed, "short_entry"] = False

        bt = Backtester(
            initial_capital=10_000,
            risk_per_trade=0.01,
            spread=cfg["spread_pips"] * pip,
            commission=0.0,
            block_same_bar_reversal=cfg["block_same_bar_reversal"],
        )
        result = bt.run(df, signals)
        m = compute_metrics(result["trades"], result["equity_curve"], 10_000)
        pf = m["profit_factor"]
        pfs.append(pf if np.isfinite(pf) else 0.0)
        rets.append(m["total_return_pct"])
        n_trades_total += m["n_trades"]
        excl3s.append(pf_excl_top(result["trades"], 3))

    return dict(
        agg_pf=float(np.mean(pfs)),
        agg_ret=float(np.mean(rets)),
        n_trades=n_trades_total,
        n_pairs_positive=sum(1 for pf in pfs if pf >= 1.0),
        agg_pf_excl3=float(np.nanmean(excl3s)),
        per_pair_pf=pfs,
    )


def main():
    rows = []
    for i, (label, cfg) in enumerate(configs.items(), 1):
        r = run_config(cfg)
        rows.append({"label": label, **r})
        print(f"[{i}/{len(configs)}] {label:<45} PF={r['agg_pf']:.3f}  ret={r['agg_ret']:+7.2f}%  "
              f"trades={r['n_trades']:5d}  pairs+={r['n_pairs_positive']}/5  PFexT3={r['agg_pf_excl3']:.3f}",
              flush=True)

    df_results = pd.DataFrame(rows).sort_values("agg_pf", ascending=False).reset_index(drop=True)
    df_results.to_csv("/tmp/ablation_results.csv", index=False)

    print("\n\n" + "=" * 100)
    print("RANKED RESULTS (highest to lowest aggregate PF)")
    print("=" * 100)
    print(f"{'rank':<5}{'label':<45}{'PF':>7}{'ret%':>9}{'trades':>8}{'pairs+':>8}{'PFexT3':>8}")
    print("-" * 100)
    for i, row in df_results.iterrows():
        print(f"{i+1:<5}{row['label']:<45}{row['agg_pf']:>7.3f}{row['agg_ret']:>9.2f}"
              f"{row['n_trades']:>8}{row['n_pairs_positive']:>6}/5{row['agg_pf_excl3']:>8.3f}")


if __name__ == "__main__":
    main()
