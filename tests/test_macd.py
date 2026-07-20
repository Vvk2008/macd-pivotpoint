import numpy as np
import pandas as pd

from indicators.macd import compute_macd


def _index(n):
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def test_macd_flat_price_is_zero():
    close = pd.Series(1.10, index=_index(100))
    macd = compute_macd(close)
    assert np.allclose(macd["macd"], 0.0, atol=1e-9)
    assert np.allclose(macd["hist"], 0.0, atol=1e-9)


def test_macd_positive_during_sustained_uptrend():
    close = pd.Series(np.linspace(1.0, 1.5, 100), index=_index(100))
    macd = compute_macd(close)
    assert macd["macd"].iloc[-1] > 0


def test_macd_negative_during_sustained_downtrend():
    close = pd.Series(np.linspace(1.5, 1.0, 100), index=_index(100))
    macd = compute_macd(close)
    assert macd["macd"].iloc[-1] < 0


def test_macd_columns_and_length():
    close = pd.Series(np.random.default_rng(0).normal(1.1, 0.01, 50), index=_index(50))
    macd = compute_macd(close)
    assert list(macd.columns) == ["macd", "signal", "hist"]
    assert len(macd) == len(close)
