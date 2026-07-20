import numpy as np
import pandas as pd

from indicators.atr import compute_atr


def _index(n):
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def test_atr_is_zero_for_flat_price():
    n = 30
    df = pd.DataFrame({"open": 1.10, "high": 1.10, "low": 1.10, "close": 1.10}, index=_index(n))
    atr = compute_atr(df, period=14)
    assert np.allclose(atr.dropna(), 0.0, atol=1e-12)


def test_atr_matches_manual_true_range_for_constant_range():
    n = 30
    idx = _index(n)
    close = pd.Series(1.10, index=idx)
    high = close + 0.001
    low = close - 0.001
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    atr = compute_atr(df, period=14)
    # constant true range of 0.002 every bar -> ATR converges to 0.002
    assert np.isclose(atr.iloc[-1], 0.002, atol=1e-9)


def test_atr_has_nan_before_period_bars():
    n = 20
    df = pd.DataFrame({"open": 1.10, "high": 1.12, "low": 1.08, "close": 1.10}, index=_index(n))
    atr = compute_atr(df, period=14)
    assert atr.iloc[:13].isna().all()
    assert atr.iloc[13:].notna().all()
