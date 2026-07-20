import pandas as pd

PIVOT_LEVELS_ASC = ["S3", "S2", "S1", "PP", "R1", "R2", "R3"]


def compute_daily_pivots(df: pd.DataFrame, session_start_hour: int = 0) -> pd.DataFrame:
    """Compute standard daily pivot points (PP, R1-R3, S1-S3) from OHLC bars.

    Each trading day's pivots are derived from the *previous* day's high,
    low, and close, then held constant across every intraday bar of the
    current day -- so a level is always knowable before any bar that uses
    it trades (no lookahead). `session_start_hour` sets where one trading
    "day" ends and the next begins, in UTC: 0 for a calendar-day rollover,
    17 for the common forex convention of rolling at the NY 5pm close.
    """
    session_day = (df.index - pd.Timedelta(hours=session_start_hour)).floor("D")

    daily = df.groupby(session_day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    daily = daily.shift(1)

    pp = (daily["high"] + daily["low"] + daily["close"]) / 3
    r1 = 2 * pp - daily["low"]
    s1 = 2 * pp - daily["high"]
    r2 = pp + (daily["high"] - daily["low"])
    s2 = pp - (daily["high"] - daily["low"])
    r3 = daily["high"] + 2 * (pp - daily["low"])
    s3 = daily["low"] - 2 * (daily["high"] - pp)

    daily_pivots = pd.DataFrame({"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3})

    out = daily_pivots.reindex(session_day)
    out.index = df.index
    return out
