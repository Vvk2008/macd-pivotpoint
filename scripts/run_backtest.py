import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.pivot_macd_strategy import generate_signals as generate_breakout_signals
from strategy.pivot_bounce_strategy import generate_signals as generate_bounce_signals
from utils import infer_pip_size


def main():
    parser = argparse.ArgumentParser(description="Backtest the pivot-point + MACD strategy.")
    parser.add_argument("csv_path", help="Path to OHLCV CSV (timestamp, open, high, low, close[, volume])")
    parser.add_argument(
        "--strategy",
        choices=["bounce", "breakout"],
        default="bounce",
        help="'bounce' = touch a support/resistance level, MACD confirms within a window (default). "
        "'breakout' = MACD crosses while price is already on one side of PP (original, prone to "
        "near-zero stop distances).",
    )
    parser.add_argument(
        "--tolerance-pips",
        type=float,
        default=20.0,
        help="[bounce] how close (in pips) a close must be to a pivot level to count as a touch",
    )
    parser.add_argument(
        "--confirmation-window",
        type=int,
        default=3,
        help="[bounce] max bars between a pivot touch and the MACD crossover that confirms it",
    )
    parser.add_argument(
        "--stop-levels",
        type=int,
        default=2,
        help="[bounce] how many pivot steps beyond the touched level the stop sits",
    )
    parser.add_argument(
        "--target-levels",
        type=int,
        default=2,
        help="[bounce] how many pivot steps beyond the touched level the target sits",
    )
    parser.add_argument(
        "--pivot-period",
        default="W",
        help="Pandas period alias for how often pivot levels recompute: 'D' (daily, resets every "
        "session), 'W' (weekly, default -- a level persists for the whole following week), 'M' (monthly)",
    )
    parser.add_argument(
        "--session-start-hour",
        type=int,
        default=0,
        help="UTC hour where the pivot trading day rolls over (0=midnight, 17=NY 5pm close). Only "
        "affects --pivot-period D.",
    )
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--initial-capital", type=float, default=10_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Fraction of equity risked per trade")
    parser.add_argument("--spread-pips", type=float, default=2.0, help="Round-trip spread cost in pips")
    parser.add_argument("--commission", type=float, default=0.0, help="Fixed commission per closed trade")
    args = parser.parse_args()

    df = load_ohlcv_csv(args.csv_path)
    pip = infer_pip_size(df["close"])
    macd = compute_macd(df["close"], args.macd_fast, args.macd_slow, args.macd_signal)
    pivots = compute_daily_pivots(df, session_start_hour=args.session_start_hour, period=args.pivot_period)

    if args.strategy == "bounce":
        signals = generate_bounce_signals(
            df,
            macd,
            pivots,
            tolerance=args.tolerance_pips * pip,
            confirmation_window=args.confirmation_window,
            stop_levels=args.stop_levels,
            target_levels=args.target_levels,
        )
    else:
        signals = generate_breakout_signals(df, macd, pivots)

    bt = Backtester(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        spread=args.spread_pips * pip,
        commission=args.commission,
    )
    result = bt.run(df, signals)
    metrics = compute_metrics(result["trades"], result["equity_curve"], args.initial_capital)

    print(f"Strategy:        {args.strategy}")
    print(f"Bars:            {len(df)}  ({df.index[0]} -> {df.index[-1]})")
    print(f"Trades:          {metrics['n_trades']}")
    print(f"Win rate:        {metrics['win_rate']:.1%}")
    print(f"Profit factor:   {metrics['profit_factor']:.2f}")
    print(f"Total return:    {metrics['total_return_pct']:.2f}%")
    print(f"CAGR:            {metrics['cagr_pct']:.2f}%")
    print(f"Max drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe:          {metrics['sharpe']:.2f}")
    print(f"Final equity:    {metrics['final_equity']:.2f}")


if __name__ == "__main__":
    main()
