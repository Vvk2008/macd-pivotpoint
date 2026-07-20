import numpy as np
import pandas as pd

from strategy.pivot_bounce_strategy import generate_signals


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


def _flat_macd(n, cross_at):
    """MACD below signal until `cross_at`, then above (bull cross at that bar)."""
    macd_vals = [-0.1] * n
    signal_vals = [0.0] * n
    for i in range(cross_at, n):
        macd_vals[i] = 0.1
    macd = pd.DataFrame({"macd": macd_vals, "signal": signal_vals}, index=_index(n))
    macd["hist"] = macd["macd"] - macd["signal"]
    return macd


def test_long_entry_on_macd_confirmation_after_support_touch():
    n = 6
    idx = _index(n)
    # Bar 1 touches S1 (0.98); MACD confirms (bull cross) two bars later at bar 3.
    close = [1.02, 0.981, 1.00, 1.00, 1.00, 1.00]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=3)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3)

    assert signals["long_entry"].tolist() == [False, False, False, True, False, False]
    assert signals.loc[idx[3], "stop"] == 0.94  # one level below touched S1 -> S2
    assert signals.loc[idx[3], "target"] == 1.02  # one level above touched S1 -> PP


def test_no_entry_once_confirmation_window_expires():
    n = 7
    idx = _index(n)
    close = [1.02, 0.981, 1.00, 1.00, 1.00, 1.00, 1.00]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=5)  # crosses 4 bars after the touch

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3)

    assert not signals["long_entry"].any()


def test_no_entry_without_any_support_touch():
    n = 5
    idx = _index(n)
    close = [1.20, 1.20, 1.20, 1.20, 1.20]  # nowhere near any level
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=2)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3)

    assert not signals["long_entry"].any()


def test_no_trade_when_touched_level_has_no_room_beyond_it():
    n = 4
    idx = _index(n)
    close = [0.901, 0.901, 0.901, 0.901]  # touches S3, the lowest support level
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=1)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3)

    assert not signals["long_entry"].any()


def test_no_entry_when_price_already_overshot_target_before_confirmation():
    n = 5
    idx = _index(n)
    # Bar 1 touches S1 (0.98); by the time MACD confirms at bar 3, price has
    # already run past the target (PP=1.02) -- the trade would open already
    # on the wrong side of its own target.
    close = [1.02, 0.981, 1.01, 1.03, 1.03]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=3)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3)

    assert not signals["long_entry"].any()


def test_no_crash_when_pivots_not_yet_available():
    # First trading day: pivots are all NaN until the prior day's H/L/C exists.
    n = 4
    idx = _index(n)
    df = pd.DataFrame({"close": [1.00, 1.00, 1.00, 1.00]}, index=idx)
    macd = _flat_macd(n, cross_at=2)
    pivots = pd.DataFrame(
        {col: [np.nan] * n for col in ["PP", "R1", "R2", "R3", "S1", "S2", "S3"]}, index=idx
    )

    signals = generate_signals(df, macd, pivots, tolerance=0.003, confirmation_window=3)

    assert not signals["long_entry"].any()
    assert not signals["short_entry"].any()
