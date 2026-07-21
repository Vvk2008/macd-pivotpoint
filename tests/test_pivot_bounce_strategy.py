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

    signals = generate_signals(
        df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, target_levels=1, stop_levels=1,
        min_reward_risk=0.0,
    )

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


def test_min_reward_risk_rejects_unfavorable_realized_distance():
    n = 6
    idx = _index(n)
    # Same setup as the basic confirmation test: entry at px=1.00, stop=S2=0.94
    # (risk 0.06), target=PP=1.02 (reward 0.02) -> R:R = 0.33, well under 1.0.
    close = [1.02, 0.981, 1.00, 1.00, 1.00, 1.00]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=3)

    signals = generate_signals(
        df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, min_reward_risk=1.0, target_levels=1
    )

    assert not signals["long_entry"].any()


def test_min_reward_risk_allows_favorable_realized_distance():
    n = 6
    idx = _index(n)
    close = [1.02, 0.981, 1.00, 1.00, 1.00, 1.00]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=3)

    # min_reward_risk below the actual 0.33 ratio should still allow the trade.
    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, min_reward_risk=0.2)

    assert signals["long_entry"].tolist() == [False, False, False, True, False, False]


def test_no_entry_when_price_already_overshot_target_before_confirmation():
    n = 5
    idx = _index(n)
    # Bar 1 touches S1 (0.98); by the time MACD confirms at bar 3, price has
    # already run past the target (PP=1.02) -- the trade would open already
    # on the wrong side of its own target.
    close = [1.02, 0.981, 1.01, 1.03, 1.03]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=3)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, target_levels=1)

    assert not signals["long_entry"].any()


def test_confirmation_window_of_one_requires_same_bar_cross():
    n = 5
    idx = _index(n)
    # Touch happens at bar 1; MACD confirms one bar later at bar 2 --
    # with confirmation_window=1, that's too late (must be the same bar).
    close = [1.02, 0.981, 1.00, 1.00, 1.00]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=2)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=1)

    assert not signals["long_entry"].any()


def test_confirmation_window_of_one_allows_same_bar_cross():
    n = 4
    idx = _index(n)
    # Touch and MACD cross both land on bar 1.
    close = [1.02, 0.981, 1.00, 1.00]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=1)

    signals = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=1)

    assert signals["long_entry"].tolist() == [False, True, False, False]


def test_require_touch_false_ignores_location_gate():
    n = 5
    idx = _index(n)
    # 1.03 is 1 cent from PP (1.02) -- outside a 0.003 tolerance, so never
    # "touched" under the normal gate, but still resolves to a nearest
    # level (PP) when the gate is off.
    close = [1.03, 1.03, 1.03, 1.03, 1.03]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=2)

    gated = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, require_touch=True)
    ungated = generate_signals(df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, require_touch=False)

    assert not gated["long_entry"].any()
    assert ungated["long_entry"].tolist() == [False, False, True, False, False]


def test_require_macd_confirmation_false_fires_on_touch_alone():
    n = 4
    idx = _index(n)
    close = [1.05, 0.981, 1.00, 1.00]  # bar 0 touches nothing; bar 1 touches S1
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=10)  # MACD never actually crosses within range

    confirmed = generate_signals(
        df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, require_macd_confirmation=True
    )
    unconfirmed = generate_signals(
        df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, require_macd_confirmation=False
    )

    assert not confirmed["long_entry"].any()
    assert unconfirmed["long_entry"].tolist() == [False, True, False, False]


def test_require_no_overshoot_false_allows_entry_past_stop():
    n = 4
    idx = _index(n)
    # Bar 1 touches S1 (0.98); by the confirmation bar, price has already
    # dropped *through* the would-be stop (S2=0.94) rather than past the
    # target -- this keeps the realized reward positive, so the R:R filter
    # (independent of the overshoot check) doesn't also reject it.
    close = [1.05, 0.981, 0.93, 0.93]
    df = pd.DataFrame({"close": close}, index=idx)
    macd = _flat_macd(n, cross_at=3)

    checked = generate_signals(
        df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, stop_levels=1, target_levels=1,
        min_reward_risk=0.0, require_no_overshoot=True,
    )
    unchecked = generate_signals(
        df, macd, _pivots(n), tolerance=0.003, confirmation_window=3, stop_levels=1, target_levels=1,
        min_reward_risk=0.0, require_no_overshoot=False,
    )

    assert not checked["long_entry"].any()
    assert unchecked["long_entry"].tolist() == [False, False, False, True]


def test_use_signal_exit_false_disables_macd_exit():
    n = 5
    idx = _index(n)
    close = [1.00] * n
    df = pd.DataFrame({"close": close}, index=idx)
    # MACD crosses bullish at bar 1 then bearish at bar 3 (a long_exit event).
    macd_vals = [-0.1, 0.1, 0.1, -0.1, -0.1]
    signal_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
    macd = pd.DataFrame({"macd": macd_vals, "signal": signal_vals}, index=idx)
    macd["hist"] = macd["macd"] - macd["signal"]

    with_exit = generate_signals(df, macd, _pivots(n), use_signal_exit=True)
    without_exit = generate_signals(df, macd, _pivots(n), use_signal_exit=False)

    assert with_exit["long_exit"].iloc[3]
    assert not without_exit["long_exit"].any()
    assert not without_exit["short_exit"].any()


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
