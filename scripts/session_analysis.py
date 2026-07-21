"""Which trading session(s) should the bounce strategy take entries in?

In-sample only (last 5 years held out), same anti-overfit discipline as the
rest of the tuning. The data is H4, so there are only 6 possible entry hours
per day (00, 04, 08, 12, 16, 20 UTC) -- a "session" here is just a subset of
those. Session windows are UTC standard-time approximations; DST is ignored
(a real caveat -- London/NY shift by an hour seasonally).

Entries are masked to a session's hours; exits (stop/target/signal) still
fire on any bar, so a position opened in-session can close out afterward.
"""
import sys
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

PAIRS = ["EURUSD240", "GBPUSD240", "USDCAD240", "USDCHF240", "USDJPY240"]
OOS_YEARS = 5
HOURS = [0, 4, 8, 12, 16, 20]

# Session -> allowed H4 entry hours (UTC). Windows (approx, standard time):
#   Sydney   ~21:00-06:00   Tokyo ~00:00-09:00
#   London   ~08:00-16:00   New York ~13:00-21:00
SESSIONS = {
    "all (baseline)":  {0, 4, 8, 12, 16, 20},
    "sydney":          {20, 0},
    "tokyo":           {0, 4},
    "london":          {8, 12},
    "newyork":         {16, 20},
    "asian (syd+tok)": {20, 0, 4},
    "western(ldn+ny)": {8, 12, 16, 20},
    "ldn/ny overlap":  {12, 16},
    "drop 20:00":      {0, 4, 8, 12, 16},
    "core 8-16":       {8, 12, 16},
}


def in_sample_only(df):
    cutoff = df.index[-1] - pd.DateOffset(years=OOS_YEARS)
    return df[df.index < cutoff]


def signals_for_pair(pair):
    df = in_sample_only(load_ohlcv_csv(f"data/raw/{pair}.csv"))
    pip = infer_pip_size(df["close"])
    macd = compute_macd(df["close"])
    pivots = compute_daily_pivots(df, period="W")
    signals = generate_signals(
        df, macd, pivots, tolerance=20 * pip, confirmation_window=3, stop_levels=2, target_levels=2
    )
    return df, signals, pip


def mask_entries(signals, allowed_hours):
    s = signals.copy()
    disallowed = ~np.isin(s.index.hour, list(allowed_hours))
    s.loc[disallowed, "long_entry"] = False
    s.loc[disallowed, "short_entry"] = False
    return s


def pf_excl_top(trades, n):
    remaining = sorted(trades, key=lambda t: -t.pnl)[n:]
    if not remaining:
        return float("nan")
    gp = sum(t.pnl for t in remaining if t.pnl > 0)
    gl = -sum(t.pnl for t in remaining if t.pnl <= 0)
    return gp / gl if gl > 0 else float("inf")


def run(allowed_hours, per_pair):
    pfs, rets, n_trades_total = [], [], 0
    excl3s = []
    for pair in PAIRS:
        df, signals, pip = per_pair[pair]
        masked = mask_entries(signals, allowed_hours)
        bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=2 * pip, commission=0.0)
        result = bt.run(df, masked)
        m = compute_metrics(result["trades"], result["equity_curve"], 10_000)
        pfs.append(m["profit_factor"] if not np.isnan(m["profit_factor"]) else 0.0)
        rets.append(m["total_return_pct"])
        n_trades_total += m["n_trades"]
        excl3s.append(pf_excl_top(result["trades"], 3))
    agg_pf = sum(pfs) / len(pfs)
    agg_ret = sum(rets) / len(rets)
    agg_excl3 = np.nanmean(excl3s)
    n_pos = sum(1 for pf in pfs if pf >= 1.0)
    return agg_pf, agg_ret, n_trades_total, n_pos, agg_excl3


def main():
    per_pair = {pair: signals_for_pair(pair) for pair in PAIRS}

    print("=== per-hour, in-sample (entries only on that single H4 hour) ===")
    hdr = f"{'entry hour':<14}{'agg PF':>9}{'agg ret%':>10}{'trades':>9}{'pairs+':>8}{'PF exT3':>9}"
    print(hdr); print("-" * len(hdr))
    for h in HOURS:
        pf, ret, n, npos, e3 = run({h}, per_pair)
        print(f"{f'{h:02d}:00 UTC':<14}{pf:>9.2f}{ret:>10.2f}{n:>9}{npos:>6}/5{e3:>9.2f}")

    print("\n=== named sessions / combos, in-sample ===")
    print(hdr); print("-" * len(hdr))
    for name, hours in SESSIONS.items():
        pf, ret, n, npos, e3 = run(hours, per_pair)
        print(f"{name:<14}{pf:>9.2f}{ret:>10.2f}{n:>9}{npos:>6}/5{e3:>9.2f}")


if __name__ == "__main__":
    main()
