"""IS-only scorecard for iterating on the bounce strategy.

Deliberately never touches the out-of-sample window (last 5 years) --
only the in-sample period is used while tuning, so we don't overfit to
data we later validate against. Run this after every change to the bounce
strategy and compare the aggregate row to the previous iteration.
"""
import sys
from pathlib import Path

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


def in_sample_only(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = df.index[-1] - pd.DateOffset(years=OOS_YEARS)
    return df[df.index < cutoff]


def run_pair(pair: str, tolerance_pips=20.0, confirmation_window=3, macd_params=(12, 26, 9), pivot_period="W", **strategy_kwargs):
    df = load_ohlcv_csv(f"data/raw/{pair}.csv")
    df = in_sample_only(df)
    pip = infer_pip_size(df["close"])

    macd = compute_macd(df["close"], *macd_params)
    pivots = compute_daily_pivots(df, period=pivot_period)
    signals = generate_signals(
        df, macd, pivots, tolerance=tolerance_pips * pip, confirmation_window=confirmation_window, **strategy_kwargs
    )

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=2 * pip, commission=0.0)
    result = bt.run(df, signals)
    metrics = compute_metrics(result["trades"], result["equity_curve"], 10_000)
    return metrics, result


def scorecard(label="", **kwargs):
    print(f"\n=== {label or 'bounce, in-sample only'} ===")
    header = f"{'pair':<12}{'trades':>8}{'win_rate':>10}{'PF':>8}{'return%':>10}{'maxDD%':>10}{'sharpe':>9}"
    print(header)
    print("-" * len(header))

    pfs, rets = [], []
    for pair in PAIRS:
        m, _ = run_pair(pair, **kwargs)
        pfs.append(m["profit_factor"])
        rets.append(m["total_return_pct"])
        print(
            f"{pair:<12}{m['n_trades']:>8}{m['win_rate']:>10.1%}{m['profit_factor']:>8.2f}"
            f"{m['total_return_pct']:>10.2f}{m['max_drawdown_pct']:>10.2f}{m['sharpe']:>9.2f}"
        )

    print("-" * len(header))
    n_positive = sum(1 for pf in pfs if pf >= 1.0)
    print(f"{'AGGREGATE':<12}{'':>8}{'':>10}{sum(pfs)/len(pfs):>8.2f}{sum(rets)/len(rets):>10.2f}"
          f"{'':>10}{'':>9}   ({n_positive}/5 pairs PF>=1)")
    return pfs, rets


if __name__ == "__main__":
    scorecard()
