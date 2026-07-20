import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.pivot_macd_strategy import generate_signals


def main():
    parser = argparse.ArgumentParser(description="Backtest the pivot-point + MACD strategy.")
    parser.add_argument("csv_path", help="Path to OHLCV CSV (timestamp, open, high, low, close[, volume])")
    parser.add_argument(
        "--session-start-hour",
        type=int,
        default=0,
        help="UTC hour where the pivot trading day rolls over (0=midnight, 17=NY 5pm close)",
    )
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--initial-capital", type=float, default=10_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Fraction of equity risked per trade")
    parser.add_argument("--spread", type=float, default=0.0002, help="Round-trip spread cost in price units")
    parser.add_argument("--commission", type=float, default=0.0, help="Fixed commission per closed trade")
    args = parser.parse_args()

    df = load_ohlcv_csv(args.csv_path)
    macd = compute_macd(df["close"], args.macd_fast, args.macd_slow, args.macd_signal)
    pivots = compute_daily_pivots(df, session_start_hour=args.session_start_hour)
    signals = generate_signals(df, macd, pivots)

    bt = Backtester(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        spread=args.spread,
        commission=args.commission,
    )
    result = bt.run(df, signals)
    metrics = compute_metrics(result["trades"], result["equity_curve"], args.initial_capital)

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
