import numpy as np
import pandas as pd

from backtester.engine import Backtester
from indicators.atr import compute_atr
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from scripts.visualize import add_equity_and_drawdown, add_price_panel, add_trade_markers, resolve_window
from strategy.pivot_bounce_strategy import generate_signals


def _synthetic_df(n=24 * 30):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close * (1 + rng.uniform(0, 0.0008, n))
    low = close * (1 - rng.uniform(0, 0.0008, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx)


def test_visualizer_helpers_run_without_error():
    from plotly.subplots import make_subplots

    df = _synthetic_df()
    macd = compute_macd(df["close"])
    pivots = compute_daily_pivots(df)
    atr = compute_atr(df)
    signals = generate_signals(df, macd, pivots, tolerance=0.001, confirmation_window=5, target_levels=3)

    bt = Backtester()
    result = bt.run(df, signals)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    add_equity_and_drawdown(fig, result["equity_curve"])
    add_price_panel(fig, df, pivots, df.index[0], df.index[-1], show_pivots=True)
    add_trade_markers(fig, result["trades"])

    assert len(fig.data) > 0


def test_resolve_window_defaults_to_last_n_days():
    class Args:
        start = None
        end = None
        window_days = 5

    df = _synthetic_df()
    start, end = resolve_window(df, Args())

    assert end == df.index[-1]
    assert (end - start).days == 5


def test_resolve_window_uses_explicit_range():
    class Args:
        start = "2024-01-05"
        end = "2024-01-10"
        window_days = 90

    df = _synthetic_df()
    start, end = resolve_window(df, Args())

    assert start == pd.Timestamp("2024-01-05", tz="UTC")
    assert end == pd.Timestamp("2024-01-10", tz="UTC")
