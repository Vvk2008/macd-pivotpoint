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


def r_multiples(trades):
    """Each trade's P&L normalized by the dollar amount actually risked on
    it (size * stop distance) -- e.g. +2.0 means the trade won 2x what it
    risked. Unlike raw $ PF, this is invariant to how much the account had
    compounded by the time of that trade, so it isolates entry/exit *rule
    quality* from pure position-sizing snowball effects. Position sizing
    that compounds on current equity with no cap (as this backtester does)
    can otherwise produce enormous, meaningless dollar returns once trade
    count and edge combine over many years -- $ PF/return can look
    spectacular even when the per-trade quality is only modestly better.
    """
    out = []
    for t in trades:
        risked = t.size * abs(t.entry_price - t.stop)
        if risked > 0:
            out.append(t.pnl / risked)
    return out


def r_pf(rs):
    if not rs:
        return float("nan")
    gp = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r <= 0)
    return gp / gl if gl > 0 else float("inf")


def run_config(cfg):
    pfs, rets, n_trades_total, excl3s = [], [], 0, []
    all_r = []
    per_pair_rpf = []
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

        pair_r = r_multiples(result["trades"])
        all_r.extend(pair_r)
        pair_rpf = r_pf(pair_r)
        per_pair_rpf.append(pair_rpf if np.isfinite(pair_rpf) else 0.0)

    n_pairs = len(PAIRS)
    return dict(
        agg_pf=float(np.mean(pfs)),
        agg_ret=float(np.mean(rets)),
        n_trades=n_trades_total,
        n_pairs_positive=sum(1 for pf in pfs if pf >= 1.0),
        n_pairs=n_pairs,
        agg_pf_excl3=float(np.nanmean(excl3s)),
        r_pf=r_pf(all_r),
        r_pf_avg_pair=float(np.mean(per_pair_rpf)) if per_pair_rpf else float("nan"),
        n_pairs_r_positive=sum(1 for x in per_pair_rpf if x >= 1.0),
    )


def main():
    rows = []
    for i, (label, cfg) in enumerate(configs.items(), 1):
        r = run_config(cfg)
        rows.append({"label": label, "rule": rule_of[label], **r})
        print(
            f"[{i}/{len(configs)}] {label:<55} rPF={r['r_pf']:.3f}  $PF={r['agg_pf']:.3f}  "
            f"trades={r['n_trades']:6d}  pairs+(r)={r['n_pairs_r_positive']}/{r['n_pairs']}",
            flush=True,
        )

    df_results = pd.DataFrame(rows)
    df_results.to_csv("/tmp/ablation_rules_results.csv", index=False)

    print("\n\n" + "=" * 108)
    print("!! Dollar PF/return columns are DISTORTED by unbounded equity compounding: position")
    print("!! size scales with current equity with no cap, so configs with more trades and any")
    print("!! edge snowball into meaningless multi-million-percent returns over 11 years. The")
    print("!! trustworthy, scale-invariant metric is rPF (profit factor of each trade's P&L")
    print("!! normalized by what was actually risked on it -- pure trade quality, no compounding).")
    print("=" * 108)
    print("RANKED RESULTS (highest to lowest R-multiple PF -- the trustworthy metric)")
    print("=" * 108)
    ranked = df_results.sort_values("r_pf", ascending=False).reset_index(drop=True)
    print(f"{'rank':<5}{'label':<55}{'rPF':>7}{'$PF':>8}{'trades':>9}{'pairs+(r)':>11}")
    print("-" * 108)
    for i, row in ranked.iterrows():
        pairs_str = f"{row['n_pairs_r_positive']}/{row['n_pairs']}"
        print(f"{i+1:<5}{row['label']:<55}{row['r_pf']:>7.3f}{row['agg_pf']:>8.3f}{row['n_trades']:>9}{pairs_str:>11}")

    baseline_row = df_results[df_results["label"] == "baseline (current production config)"].iloc[0]

    print("\n\n" + "=" * 108)
    print("PER-RULE VERDICT (baseline rPF = {:.3f}, {} trades)".format(baseline_row["r_pf"], baseline_row["n_trades"]))
    print("=" * 108)
    for rule in df_results["rule"].unique():
        if rule in ("baseline", "combo"):
            continue
        sub = df_results[df_results["rule"] == rule]
        best = sub.loc[sub["r_pf"].idxmax()]
        off_rows = sub[sub["label"].str.contains("OFF")]
        delta = best["r_pf"] - baseline_row["r_pf"]
        if delta > 0.02:
            verdict = "IMPROVES on baseline"
        elif abs(delta) <= 0.02:
            verdict = "NEUTRAL -- unnecessary complexity if just this axis"
        else:
            verdict = "baseline setting is NECESSARY (variants worse)"
        print(f"\n[{rule}]")
        print(f"  best variant: {best['label']:<50} rPF={best['r_pf']:.3f}  trades={best['n_trades']:6d}  (delta {delta:+.3f})")
        for _, r in off_rows.iterrows():
            print(f"  OFF variant : {r['label']:<50} rPF={r['r_pf']:.3f}  trades={r['n_trades']:6d}  (delta {r['r_pf']-baseline_row['r_pf']:+.3f})")
        print(f"  VERDICT: {verdict}")

    print("\n\n=== PILLAR-REMOVAL COMBOS (by rPF) ===")
    combos = df_results[df_results["rule"] == "combo"]
    for _, r in combos.sort_values("r_pf", ascending=False).iterrows():
        print(f"  {r['label']:<55} rPF={r['r_pf']:.3f}  $PF={r['agg_pf']:.3f}  trades={r['n_trades']:6d}  pairs+(r)={r['n_pairs_r_positive']}/{r['n_pairs']}")


if __name__ == "__main__":
    main()
