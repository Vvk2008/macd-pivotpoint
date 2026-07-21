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


def _three_week_df():
    idx = pd.date_range("2024-01-01", periods=24 * 21, freq="h", tz="UTC")  # Mon 2024-01-01 -> 3 weeks
    week1, week2, week3 = idx[:24 * 7], idx[24 * 7 : 24 * 14], idx[24 * 14 :]
    df = pd.DataFrame(index=idx, columns=["open", "high", "low", "close"], dtype=float)
    df.loc[week1, ["open", "high", "low", "close"]] = [1.10, 1.15, 1.05, 1.12]
    df.loc[week2, ["open", "high", "low", "close"]] = [1.12, 1.20, 1.08, 1.14]
    df.loc[week3, ["open", "high", "low", "close"]] = [1.14, 1.18, 1.10, 1.16]
    return df, week1, week2, week3


def test_weekly_period_holds_one_level_for_the_whole_week():
    df, week1, week2, week3 = _three_week_df()
    pivots = compute_daily_pivots(df, period="W")

    assert pivots.loc[week2, "PP"].nunique() == 1
    assert pivots.loc[week3, "PP"].nunique() == 1
    assert pivots.loc[week1].isna().all().all()  # no prior week yet


def test_weekly_pivots_use_prior_week_high_low_close():
    df, week1, week2, week3 = _three_week_df()
    pivots = compute_daily_pivots(df, period="W")

    expected_pp = (1.15 + 1.05 + 1.12) / 3  # week1's H/L/C feeds week2's pivot
    assert round(pivots.loc[week2, "PP"].iloc[0], 6) == round(expected_pp, 6)


def test_daily_period_is_the_default_and_unchanged():
    df, day1, day2 = _two_day_df()
    explicit = compute_daily_pivots(df, period="D")
    default = compute_daily_pivots(df)
    assert explicit["PP"].equals(default["PP"])
