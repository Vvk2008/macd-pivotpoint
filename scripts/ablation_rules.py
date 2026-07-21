"""Core-rules ablation: does each structural rule in the bounce strategy
earn its complexity, or is it just an extra knob that doesn't move the
result?

Unlike scripts/ablation.py (which sweeps *values* of parameters that are
always structurally present), this varies whether each *rule itself* is
present at all -- e.g. does requiring MACD confirmation help versus just
trading the pivot touch directly, not just "what confirmation_window value
is best."

In-sample only (last 5 years held out), across all 28 pairs in data/raw/,
same discipline as every other round -- this explains why the current
IS/OOS numbers look the way they do, it does not re-open OOS.
"""
import sys
import glob
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.pivot_bounce_strategy import generate_signals
from utils import infer_pip_size

PAIRS = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(Path(__file__).resolve().parents[1] / "data/raw/*240.csv")))
OOS_YEARS = 5

BASELINE = dict(
    pivot_period="W",
    tolerance_pips=18.0,
    confirmation_window=2,
    stop_levels=2,
    target_levels=2,
    min_reward_risk=0.5,
    require_touch=True,
    require_macd_confirmation=True,
    require_no_overshoot=True,
    use_signal_exit=True,
    block_same_bar_reversal=True,
    spread_pips=2.0,
)

# label -> (rule name, overrides)
configs = {}
rule_of = {}  # label -> rule name, for the per-rule verdict summary


def add(label, rule, **overrides):
    configs[label] = {**BASELINE, **overrides}
    rule_of[label] = rule


add("baseline (current production config)", "baseline")

# 1. touch / location filter
add("touch: OFF (no location gate)", "touch", require_touch=False)
add("touch: tolerance=5 (tight)", "touch", tolerance_pips=5.0)
add("touch: tolerance=10", "touch", tolerance_pips=10.0)
add("touch: tolerance=25", "touch", tolerance_pips=25.0)
add("touch: tolerance=35 (loose)", "touch", tolerance_pips=35.0)
add("touch: tolerance=45 (very loose)", "touch", tolerance_pips=45.0)

# 2. MACD confirmation
add("macd_confirm: OFF (pure pivot bounce)", "macd_confirmation", require_macd_confirmation=False)

# 3. confirmation window (how long a touch stays valid)
add("confirm_window=1 (same-bar only)", "confirmation_window", confirmation_window=1)
add("confirm_window=3", "confirmation_window", confirmation_window=3)
add("confirm_window=5", "confirmation_window", confirmation_window=5)
add("confirm_window=20", "confirmation_window", confirmation_window=20)
add("confirm_window=100000 (~unlimited)", "confirmation_window", confirmation_window=100_000)

# 4. stop buffer
add("stop_levels=0 (OFF -- stop AT touched level)", "stop_buffer", stop_levels=0)
add("stop_levels=1", "stop_buffer", stop_levels=1)
add("stop_levels=3", "stop_buffer", stop_levels=3)
add("stop_levels=4", "stop_buffer", stop_levels=4)

# 5. target buffer
add("target_levels=0 (OFF -- target AT touched level)", "target_buffer", target_levels=0)
add("target_levels=1", "target_buffer", target_levels=1)
add("target_levels=3", "target_buffer", target_levels=3)
add("target_levels=4", "target_buffer", target_levels=4)

# 6. min reward:risk filter
add("min_reward_risk=0.0 (OFF)", "reward_risk_filter", min_reward_risk=0.0)
add("min_reward_risk=0.2", "reward_risk_filter", min_reward_risk=0.2)
add("min_reward_risk=1.0", "reward_risk_filter", min_reward_risk=1.0)
add("min_reward_risk=1.5", "reward_risk_filter", min_reward_risk=1.5)

# 7. overshoot / bounds check
add("overshoot_check: OFF", "overshoot_check", require_no_overshoot=False)

# 8. signal-based exit
add("signal_exit: OFF (stop/target only)", "signal_exit", use_signal_exit=False)

# 9. same-bar-reversal block
add("same_bar_block: OFF", "same_bar_block", block_same_bar_reversal=False)

# 10. pivot period
add("pivot_period=D", "pivot_period", pivot_period="D")
add("pivot_period=M", "pivot_period", pivot_period="M")

# --- pillar-removal / interaction sanity checks -----------------------------
add("PILLAR: touch OFF + macd OFF (no filter at all)", "combo",
    require_touch=False, require_macd_confirmation=False)
add("PILLAR: macd OFF + signal_exit OFF (pure pivot, no MACD anywhere)", "combo",
    require_macd_confirmation=False, use_signal_exit=False)
add("PILLAR: touch OFF + stop_levels=0", "combo",
    require_touch=False, stop_levels=0)
add("PILLAR: overshoot OFF + min_reward_risk=0 (both safety filters off)", "combo",
    require_no_overshoot=False, min_reward_risk=0.0)
add("PILLAR: touch OFF + overshoot OFF", "combo",
    require_touch=False, require_no_overshoot=False)
add("PILLAR: refinements off, core (touch+macd) kept", "combo",
    require_no_overshoot=False, min_reward_risk=0.0, stop_levels=1, target_levels=1)
add("PILLAR: bare minimum (every rule off)", "combo",
    require_touch=False, require_macd_confirmation=False, require_no_overshoot=False,
    min_reward_risk=0.0, use_signal_exit=False)
add("PILLAR: signal_exit OFF + same_bar_block OFF", "combo",
    use_signal_exit=False, block_same_bar_reversal=False)
add("PILLAR: macd OFF + overshoot OFF", "combo",
    require_macd_confirmation=False, require_no_overshoot=False)
add("PILLAR: touch OFF + macd OFF + min_reward_risk=0", "combo",
    require_touch=False, require_macd_confirmation=False, min_reward_risk=0.0)

print(f"Built {len(configs)} configs across {len(PAIRS)} pairs")

# --- caching layer -----------------------------------------------------------
_df_cache, _pivot_cache, _macd_cache = {}, {}, {}


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


def get_macd(pair):
    if pair not in _macd_cache:
        _macd_cache[pair] = compute_macd(get_df(pair)["close"])
    return _macd_cache[pair]


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
        macd = get_macd(pair)
        pivots = get_pivots(pair, cfg["pivot_period"])

        signals = generate_signals(
            df, macd, pivots,
            tolerance=cfg["tolerance_pips"] * pip,
            confirmation_window=cfg["confirmation_window"],
            min_reward_risk=cfg["min_reward_risk"],
            stop_levels=cfg["stop_levels"],
            target_levels=cfg["target_levels"],
            require_touch=cfg["require_touch"],
            require_macd_confirmation=cfg["require_macd_confirmation"],
            require_no_overshoot=cfg["require_no_overshoot"],
            use_signal_exit=cfg["use_signal_exit"],
        )

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

    n_pairs = len(PAIRS)
    return dict(
        agg_pf=float(np.mean(pfs)),
        agg_ret=float(np.mean(rets)),
        n_trades=n_trades_total,
        n_pairs_positive=sum(1 for pf in pfs if pf >= 1.0),
        n_pairs=n_pairs,
        agg_pf_excl3=float(np.nanmean(excl3s)),
    )


def main():
    rows = []
    for i, (label, cfg) in enumerate(configs.items(), 1):
        r = run_config(cfg)
        rows.append({"label": label, "rule": rule_of[label], **r})
        print(
            f"[{i}/{len(configs)}] {label:<55} PF={r['agg_pf']:.3f}  ret={r['agg_ret']:+7.2f}%  "
            f"trades={r['n_trades']:6d}  pairs+={r['n_pairs_positive']}/{r['n_pairs']}  PFexT3={r['agg_pf_excl3']:.3f}",
            flush=True,
        )

    df_results = pd.DataFrame(rows).sort_values("agg_pf", ascending=False).reset_index(drop=True)
    df_results.to_csv("/tmp/ablation_rules_results.csv", index=False)

    print("\n\n" + "=" * 108)
    print("RANKED RESULTS (highest to lowest aggregate PF)")
    print("=" * 108)
    print(f"{'rank':<5}{'label':<55}{'PF':>7}{'ret%':>9}{'trades':>9}{'pairs+':>9}{'PFexT3':>8}")
    print("-" * 108)
    for i, row in df_results.iterrows():
        pairs_str = f"{row['n_pairs_positive']}/{row['n_pairs']}"
        print(
            f"{i+1:<5}{row['label']:<55}{row['agg_pf']:>7.3f}{row['agg_ret']:>9.2f}"
            f"{row['n_trades']:>9}{pairs_str:>9}{row['agg_pf_excl3']:>8.3f}"
        )

    baseline_row = df_results[df_results["label"] == "baseline (current production config)"].iloc[0]

    print("\n\n" + "=" * 108)
    print("PER-RULE VERDICT (baseline PF = {:.3f}, PFexT3 = {:.3f})".format(
        baseline_row["agg_pf"], baseline_row["agg_pf_excl3"]))
    print("=" * 108)
    for rule in df_results["rule"].unique():
        if rule in ("baseline", "combo"):
            continue
        sub = df_results[df_results["rule"] == rule]
        best = sub.loc[sub["agg_pf"].idxmax()]
        off_rows = sub[sub["label"].str.contains("OFF")]
        delta_pf = best["agg_pf"] - baseline_row["agg_pf"]
        delta_robust = best["agg_pf_excl3"] - baseline_row["agg_pf_excl3"]
        if delta_pf > 0.02 and delta_robust > 0.01:
            verdict = "IMPROVES on baseline"
        elif abs(delta_pf) <= 0.02 and abs(delta_robust) <= 0.01:
            verdict = "NEUTRAL -- unnecessary complexity if just this axis"
        else:
            verdict = "baseline setting is NECESSARY (variants worse)"
        print(f"\n[{rule}]")
        print(f"  best variant: {best['label']:<50} PF={best['agg_pf']:.3f}  PFexT3={best['agg_pf_excl3']:.3f}  (delta {delta_pf:+.3f})")
        for _, r in off_rows.iterrows():
            print(f"  OFF variant : {r['label']:<50} PF={r['agg_pf']:.3f}  PFexT3={r['agg_pf_excl3']:.3f}  (delta {r['agg_pf']-baseline_row['agg_pf']:+.3f})")
        print(f"  VERDICT: {verdict}")

    print("\n\n=== PILLAR-REMOVAL COMBOS ===")
    combos = df_results[df_results["rule"] == "combo"]
    for _, r in combos.sort_values("agg_pf", ascending=False).iterrows():
        print(f"  {r['label']:<55} PF={r['agg_pf']:.3f}  ret={r['agg_ret']:+7.2f}%  pairs+={r['n_pairs_positive']}/{r['n_pairs']}  PFexT3={r['agg_pf_excl3']:.3f}")


if __name__ == "__main__":
    main()
