import pandas as pd

from strategy.macd_only_strategy import generate_signals


def _index(n):
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def test_long_entry_on_bull_cross_with_atr_stop_and_target():
    n = 4
    idx = _index(n)
    df = pd.DataFrame({"close": [1.10, 1.10, 1.10, 1.10]}, index=idx)
    macd = pd.DataFrame({"macd": [-0.1, -0.1, 0.1, 0.1], "signal": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    macd["hist"] = macd["macd"] - macd["signal"]
    atr = pd.Series([0.001, 0.001, 0.001, 0.001], index=idx)

    signals = generate_signals(df, macd, atr, stop_atr_mult=1.5, target_atr_mult=3.0)

    assert signals["long_entry"].tolist() == [False, False, True, False]
    assert signals.loc[idx[2], "stop"] == 1.10 - 0.0015
    assert signals.loc[idx[2], "target"] == 1.10 + 0.003


def test_short_entry_on_bear_cross_with_atr_stop_and_target():
    n = 4
    idx = _index(n)
    df = pd.DataFrame({"close": [1.10, 1.10, 1.10, 1.10]}, index=idx)
    macd = pd.DataFrame({"macd": [0.1, 0.1, -0.1, -0.1], "signal": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    macd["hist"] = macd["macd"] - macd["signal"]
    atr = pd.Series([0.001, 0.001, 0.001, 0.001], index=idx)

    signals = generate_signals(df, macd, atr, stop_atr_mult=1.5, target_atr_mult=3.0)

    assert signals["short_entry"].tolist() == [False, False, True, False]
    assert signals.loc[idx[2], "stop"] == 1.10 + 0.0015
    assert signals.loc[idx[2], "target"] == 1.10 - 0.003


def test_no_entry_when_atr_not_yet_available():
    n = 3
    idx = _index(n)
    df = pd.DataFrame({"close": [1.10, 1.10, 1.10]}, index=idx)
    macd = pd.DataFrame({"macd": [-0.1, 0.1, 0.1], "signal": [0.0, 0.0, 0.0]}, index=idx)
    macd["hist"] = macd["macd"] - macd["signal"]
    atr = pd.Series([None, None, None], index=idx, dtype=float)

    signals = generate_signals(df, macd, atr)

    assert not signals["long_entry"].any()
