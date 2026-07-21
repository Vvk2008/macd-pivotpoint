"""Interactive backtest visualizer.

Renders a self-contained HTML report: full-history equity curve, full-history
drawdown, and a price panel (close-price line across the whole run plus
candlesticks + pivot levels + trade markers for a zoomable default window).
All three panels share a linked x-axis, so panning/zooming the price panel
also scrolls the equity/drawdown panels to match, and vice versa.

Usage:
    python scripts/visualize.py data/raw/EURUSD240.csv
    python scripts/visualize.py data/raw/EURUSD240.csv --strategy macd_only
    python scripts/visualize.py data/raw/EURUSD240.csv --start 2023-01-01 --end 2023-06-01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

PIVOT_COLORS = {
    "PP": "#888888",
    "R1": "#d62728",
    "R2": "#d62728",
    "R3": "#d62728",
    "S1": "#2ca02c",
    "S2": "#2ca02c",
    "S3": "#2ca02c",
}


def build_signals(strategy, df, macd, pivots, atr, args, pip):
    if strategy == "bounce":
        return generate_bounce_signals(
            df,
            macd,
            pivots,
            tolerance=args.tolerance_pips * pip,
            confirmation_window=args.confirmation_window,
            stop_levels=args.stop_levels,
            target_levels=args.target_levels,
            min_reward_risk=args.min_reward_risk,
        )
    if strategy == "macd_only":
        return generate_macd_only_signals(df, macd, atr)
    if strategy == "breakout":
        return generate_breakout_signals(df, macd, pivots)
    raise ValueError(strategy)


def resolve_window(df, args):
    if args.start or args.end:
        start = pd.Timestamp(args.start, tz="UTC") if args.start else df.index[0]
        end = pd.Timestamp(args.end, tz="UTC") if args.end else df.index[-1]
    else:
        end = df.index[-1]
        start = end - pd.Timedelta(days=args.window_days)
    return start, end


def add_equity_and_drawdown(fig, equity_curve):
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity", line=dict(color="#1f77b4"), fill="tozeroy"),
        row=1,
        col=1,
    )
    running_max = equity_curve.cummax()
    drawdown_pct = (equity_curve - running_max) / running_max * 100
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            name="Drawdown %",
            line=dict(color="#d62728"),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.3)",
        ),
        row=2,
        col=1,
    )


def add_price_panel(fig, df, pivots, start, end, show_pivots):
    fig.add_trace(
        go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#999999", width=1), opacity=0.6),
        row=3,
        col=1,
    )

    windowed = df[(df.index >= start) & (df.index <= end)]
    if len(windowed):
        fig.add_trace(
            go.Candlestick(
                x=windowed.index,
                open=windowed["open"],
                high=windowed["high"],
                low=windowed["low"],
                close=windowed["close"],
                name="Price",
                increasing_line_color="#2ca02c",
                decreasing_line_color="#d62728",
            ),
            row=3,
            col=1,
        )

    if show_pivots and len(windowed):
        windowed_pivots = pivots.loc[windowed.index]
        for level, color in PIVOT_COLORS.items():
            series = windowed_pivots[level]
            if series.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        name=level,
                        line=dict(color=color, width=1, dash="dot", shape="hv"),
                        opacity=0.7,
                        visible="legendonly" if level in ("S3", "R3") else True,
                    ),
                    row=3,
                    col=1,
                )


def add_trade_markers(fig, trades):
    if not trades:
        return

    def hover(t):
        return (
            f"{t.direction.upper()}<br>entry {t.entry_price:.5f} @ {t.entry_time}"
            f"<br>exit {t.exit_price:.5f} @ {t.exit_time}<br>reason: {t.exit_reason}"
            f"<br>pnl: {t.pnl:+.2f}"
        )

    for direction, symbol, color in [("long", "triangle-up", "#2ca02c"), ("short", "triangle-down", "#d62728")]:
        dtrades = [t for t in trades if t.direction == direction]
        if not dtrades:
            continue
        fig.add_trace(
            go.Scatter(
                x=[t.entry_time for t in dtrades],
                y=[t.entry_price for t in dtrades],
                mode="markers",
                name=f"{direction} entry",
                marker=dict(symbol=symbol, size=9, color=color),
                text=[hover(t) for t in dtrades],
                hoverinfo="text",
            ),
            row=3,
            col=1,
        )

    win_trades = [t for t in trades if t.pnl and t.pnl > 0]
    loss_trades = [t for t in trades if t.pnl is not None and t.pnl <= 0]
    for label, subset, color in [("win exit", win_trades, "#2ca02c"), ("loss exit", loss_trades, "#d62728")]:
        if not subset:
            continue
        fig.add_trace(
            go.Scatter(
                x=[t.exit_time for t in subset],
                y=[t.exit_price for t in subset],
                mode="markers",
                name=label,
                marker=dict(symbol="x", size=7, color=color),
                text=[hover(t) for t in subset],
                hoverinfo="text",
            ),
            row=3,
            col=1,
        )

    for t in trades:
        fig.add_trace(
            go.Scatter(
                x=[t.entry_time, t.exit_time],
                y=[t.entry_price, t.exit_price],
                mode="lines",
                line=dict(color="#2ca02c" if t.pnl and t.pnl > 0 else "#d62728", width=1, dash="dash"),
                opacity=0.4,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )


def main():
    parser = argparse.ArgumentParser(description="Render an interactive HTML backtest visualization.")
    parser.add_argument("csv_path", help="Path to OHLCV CSV")
    parser.add_argument("--strategy", choices=["bounce", "macd_only", "breakout"], default="bounce")
    parser.add_argument("--start", default=None, help="Price-panel window start (YYYY-MM-DD). Default: last --window-days")
    parser.add_argument("--end", default=None, help="Price-panel window end (YYYY-MM-DD). Default: last bar")
    parser.add_argument("--window-days", type=int, default=90, help="Default price-panel window size if --start/--end unset")
    parser.add_argument("--no-pivots", action="store_true", help="Don't draw pivot levels in the price panel")
    parser.add_argument("--pivot-period", default="W", help="'D', 'W' (default), or 'M' -- how often pivots recompute")
    parser.add_argument("--session-start-hour", type=int, default=0, help="Only affects --pivot-period D")
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--tolerance-pips", type=float, default=18.0)
    parser.add_argument("--confirmation-window", type=int, default=2)
    parser.add_argument("--stop-levels", type=int, default=2)
    parser.add_argument("--target-levels", type=int, default=2)
    parser.add_argument("--min-reward-risk", type=float, default=0.5, help="0 = disabled")
    parser.add_argument("--initial-capital", type=float, default=10_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--spread-pips", type=float, default=2.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--output", default=None, help="Output HTML path (default: <csv-stem>_<strategy>.html)")
    args = parser.parse_args()

    df = load_ohlcv_csv(args.csv_path)
    pip = infer_pip_size(df["close"])
    macd = compute_macd(df["close"], args.macd_fast, args.macd_slow, args.macd_signal)
    pivots = compute_daily_pivots(df, session_start_hour=args.session_start_hour, period=args.pivot_period)
    atr = compute_atr(df)
    signals = build_signals(args.strategy, df, macd, pivots, atr, args, pip)

    bt = Backtester(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        spread=args.spread_pips * pip,
        commission=args.commission,
    )
    result = bt.run(df, signals)
    metrics = compute_metrics(result["trades"], result["equity_curve"], args.initial_capital)

    start, end = resolve_window(df, args)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.22, 0.13, 0.65],
        vertical_spacing=0.03,
        subplot_titles=("Equity", "Drawdown %", f"Price ({args.strategy})"),
    )

    add_equity_and_drawdown(fig, result["equity_curve"])
    add_price_panel(fig, df, pivots, start, end, show_pivots=not args.no_pivots)
    add_trade_markers(fig, result["trades"])

    title = (
        f"{Path(args.csv_path).stem} | {args.strategy} | trades={metrics['n_trades']} "
        f"win_rate={metrics['win_rate']:.1%} PF={metrics['profit_factor']:.2f} "
        f"return={metrics['total_return_pct']:.1f}% maxDD={metrics['max_drawdown_pct']:.1f}% "
        f"sharpe={metrics['sharpe']:.2f}"
    )
    fig.update_layout(
        title=title,
        height=900,
        xaxis3_rangeslider_visible=False,
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[start, end], row=3, col=1)

    # The full-history close-price line otherwise forces the y-axis to fit
    # 16 years of range, squashing the zoomed candlesticks into a sliver.
    windowed = df[(df.index >= start) & (df.index <= end)]
    if len(windowed):
        pad = (windowed["high"].max() - windowed["low"].min()) * 0.05
        fig.update_yaxes(
            range=[windowed["low"].min() - pad, windowed["high"].max() + pad], row=3, col=1
        )

    output = args.output or f"{Path(args.csv_path).stem}_{args.strategy}.html"
    fig.write_html(output, include_plotlyjs=True)
    print(f"Wrote {output}")
    print(title)


if __name__ == "__main__":
    main()
