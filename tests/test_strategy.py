import pandas as pd

from strategy.pivot_macd_strategy import generate_signals


def _index(n):
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def _pivots(n):
    idx = _index(n)
    return pd.DataFrame(
        {
            "PP": [1.02] * n,
            "R1": [1.06] * n,
            "R2": [1.10] * n,
            "R3": [1.14] * n,
            "S1": [0.98] * n,
            "S2": [0.94] * n,
            "S3": [0.90] * n,
        },
        index=idx,
    )


def test_long_entry_requires_macd_cross_and_price_above_pivot():
    idx = _index(5)
    df = pd.DataFrame({"close": [1.00, 1.00, 1.05, 1.05, 1.05]}, index=idx)
    macd = pd.DataFrame(
        {"macd": [-0.1, -0.1, 0.1, 0.1, 0.1], "signal": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    macd["hist"] = macd["macd"] - macd["signal"]

    signals = generate_signals(df, macd, _pivots(5))

    assert signals["long_entry"].tolist() == [False, False, True, False, False]
    assert signals.loc[idx[2], "stop"] == 1.02
    assert signals.loc[idx[2], "target"] == 1.06


def test_short_entry_requires_macd_cross_and_price_below_pivot():
    idx = _index(5)
    df = pd.DataFrame({"close": [1.05, 1.05, 0.99, 0.99, 0.99]}, index=idx)
    macd = pd.DataFrame(
        {"macd": [0.1, 0.1, -0.1, -0.1, -0.1], "signal": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    macd["hist"] = macd["macd"] - macd["signal"]

    signals = generate_signals(df, macd, _pivots(5))

    assert signals["short_entry"].tolist() == [False, False, True, False, False]
    assert signals.loc[idx[2], "stop"] == 1.02
    assert signals.loc[idx[2], "target"] == 0.98


def test_no_entry_when_macd_crosses_but_price_on_wrong_side_of_pivot():
    idx = _index(5)
    df = pd.DataFrame({"close": [1.00, 1.00, 0.99, 0.99, 0.99]}, index=idx)
    macd = pd.DataFrame(
        {"macd": [-0.1, -0.1, 0.1, 0.1, 0.1], "signal": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx
    )
    macd["hist"] = macd["macd"] - macd["signal"]

    signals = generate_signals(df, macd, _pivots(5))

    assert not signals["long_entry"].any()
