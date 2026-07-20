import numpy as np
import pandas as pd

from backtester.metrics import compute_metrics


def test_return_reflects_equity_curve_even_with_no_discrete_trades():
    # A buy-and-hold-style equity curve: no trades, but the curve itself
    # moves. total_return must come from the curve, not from summing an
    # empty trade list (which would wrongly report 0% regardless of what
    # actually happened to equity).
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    equity = pd.Series([10_000, 10_500, 9_800, 9_000, 8_800], index=idx)

    metrics = compute_metrics([], equity, initial_capital=10_000)

    assert metrics["n_trades"] == 0
    assert np.isclose(metrics["final_equity"], 8_800)
    assert np.isclose(metrics["total_return_pct"], -12.0)
    assert metrics["max_drawdown_pct"] < 0
