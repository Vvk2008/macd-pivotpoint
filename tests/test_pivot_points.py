import pandas as pd

from indicators.pivot_points import compute_daily_pivots


def _two_day_df():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    day1, day2 = idx[:24], idx[24:]
    df = pd.DataFrame(index=idx, columns=["open", "high", "low", "close"], dtype=float)
    df.loc[day1, ["open", "high", "low", "close"]] = [1.10, 1.12, 1.08, 1.11]
    df.loc[day2, ["open", "high", "low", "close"]] = [1.11, 1.13, 1.09, 1.10]
    return df, day1, day2


def test_pivot_formula_matches_manual_calculation():
    df, day1, day2 = _two_day_df()
    pivots = compute_daily_pivots(df)

    expected_pp = (1.12 + 1.08 + 1.11) / 3
    expected_r1 = 2 * expected_pp - 1.08
    expected_s1 = 2 * expected_pp - 1.12
    expected_r2 = expected_pp + (1.12 - 1.08)
    expected_s2 = expected_pp - (1.12 - 1.08)

    day2_pivots = pivots.loc[day2]
    assert day2_pivots["PP"].round(6).eq(round(expected_pp, 6)).all()
    assert day2_pivots["R1"].round(6).eq(round(expected_r1, 6)).all()
    assert day2_pivots["S1"].round(6).eq(round(expected_s1, 6)).all()
    assert day2_pivots["R2"].round(6).eq(round(expected_r2, 6)).all()
    assert day2_pivots["S2"].round(6).eq(round(expected_s2, 6)).all()


def test_no_lookahead_first_day_has_no_pivots():
    df, day1, day2 = _two_day_df()
    pivots = compute_daily_pivots(df)
    assert pivots.loc[day1].isna().all().all()


def test_pivot_levels_are_ordered():
    df, day1, day2 = _two_day_df()
    pivots = compute_daily_pivots(df)
    row = pivots.loc[day2[0]]
    ordered = [row["S3"], row["S2"], row["S1"], row["PP"], row["R1"], row["R2"], row["R3"]]
    assert ordered == sorted(ordered)


def test_session_start_hour_shifts_day_boundary():
    df, day1, day2 = _two_day_df()
    pivots_midnight = compute_daily_pivots(df, session_start_hour=0)
    pivots_5pm = compute_daily_pivots(df, session_start_hour=17)
    assert not pivots_midnight["PP"].equals(pivots_5pm["PP"])
