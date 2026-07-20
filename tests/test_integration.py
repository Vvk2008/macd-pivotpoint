import numpy as np
import pandas as pd

from backtester.engine import Backtester
from backtester.metrics import compute_metrics
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.pivot_macd_strategy import generate_signals


def test_full_pipeline_runs_end_to_end_on_synthetic_data():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.0015, len(idx))
    close = 1.10 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.001, len(idx)))
    low = close * (1 - rng.uniform(0, 0.001, len(idx)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx)

    macd = compute_macd(df["close"])
    pivots = compute_daily_pivots(df)
    signals = generate_signals(df, macd, pivots)

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=0.0002, commission=0.0)
    result = bt.run(df, signals)
    metrics = compute_metrics(result["trades"], result["equity_curve"], 10_000)

    assert len(result["equity_curve"]) == len(df)
    expected_keys = {
        "n_trades",
        "win_rate",
        "profit_factor",
        "total_return_pct",
        "cagr_pct",
        "max_drawdown_pct",
        "sharpe",
        "final_equity",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["n_trades"] >= 0
