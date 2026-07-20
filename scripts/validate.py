import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.baseline import buy_and_hold_equity_curve
from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.atr import compute_atr
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.macd_only_strategy import generate_signals as generate_macd_only_signals
from strategy.pivot_bounce_strategy import generate_signals as generate_bounce_signals
from strategy.pivot_macd_strategy import generate_signals as generate_breakout_signals

PIP = 0.0001


def run_strategy(name, df, macd, pivots, atr, bt_kwargs, tolerance, confirmation_window):
    if name == "bounce":
        signals = generate_bounce_signals(df, macd, pivots, tolerance=tolerance, confirmation_window=confirmation_window)
    elif name == "breakout":
        signals = generate_breakout_signals(df, macd, pivots)
    elif name == "macd_only":
        signals = generate_macd_only_signals(df, macd, atr)
    else:
        raise ValueError(name)

    bt = Backtester(**bt_kwargs)
    result = bt.run(df, signals)
    return compute_metrics(result["trades"], result["equity_curve"], bt_kwargs["initial_capital"])


def buy_and_hold_metrics(df, initial_capital):
    equity = buy_and_hold_equity_curve(df, initial_capital)
    return compute_metrics([], equity, initial_capital)


def print_period(label, df, args, bt_kwargs):
    macd = compute_macd(df["close"], args.macd_fast, args.macd_slow, args.macd_signal)
    pivots = compute_daily_pivots(df, session_start_hour=args.session_start_hour)
    atr = compute_atr(df)

    print(f"\n=== {label}  ({df.index[0]} -> {df.index[-1]}, {len(df)} bars) ===")
    header = f"{'strategy':<12}{'trades':>8}{'win_rate':>10}{'PF':>8}{'return%':>10}{'maxDD%':>10}{'sharpe':>9}"
    print(header)
    print("-" * len(header))

    bh = buy_and_hold_metrics(df, args.initial_capital)
    print(f"{'buy&hold':<12}{'-':>8}{'-':>10}{'-':>8}{bh['total_return_pct']:>10.2f}{bh['max_drawdown_pct']:>10.2f}{bh['sharpe']:>9.2f}")

    for name in ["breakout", "bounce", "macd_only"]:
        m = run_strategy(name, df, macd, pivots, atr, bt_kwargs, args.tolerance_pips * PIP, args.confirmation_window)
        print(
            f"{name:<12}{m['n_trades']:>8}{m['win_rate']:>10.1%}{m['profit_factor']:>8.2f}"
            f"{m['total_return_pct']:>10.2f}{m['max_drawdown_pct']:>10.2f}{m['sharpe']:>9.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare breakout/bounce/macd-only strategies against buy-and-hold, "
        "split into in-sample and out-of-sample periods."
    )
    parser.add_argument("csv_path", help="Path to OHLCV CSV")
    parser.add_argument("--split", type=float, default=0.7, help="Fraction of bars used as in-sample (default 0.7)")
    parser.add_argument("--session-start-hour", type=int, default=0)
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--tolerance-pips", type=float, default=5.0)
    parser.add_argument("--confirmation-window", type=int, default=3)
    parser.add_argument("--initial-capital", type=float, default=10_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--commission", type=float, default=0.0)
    args = parser.parse_args()

    df = load_ohlcv_csv(args.csv_path)
    split_idx = int(len(df) * args.split)
    in_sample, out_of_sample = df.iloc[:split_idx], df.iloc[split_idx:]

    bt_kwargs = dict(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        spread=args.spread,
        commission=args.commission,
    )

    print_period("IN-SAMPLE", in_sample, args, bt_kwargs)
    print_period("OUT-OF-SAMPLE", out_of_sample, args, bt_kwargs)


if __name__ == "__main__":
    main()
