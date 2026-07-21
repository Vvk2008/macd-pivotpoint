import pandas as pd

PIVOT_LEVELS_ASC = ["S3", "S2", "S1", "PP", "R1", "R2", "R3"]


def compute_daily_pivots(df: pd.DataFrame, session_start_hour: int = 0, period: str = "D") -> pd.DataFrame:
    """Compute standard pivot points (PP, R1-R3, S1-S3) from OHLC bars.

    Each session's pivots are derived from the *previous* session's high,
    low, and close, then held constant across every intraday bar of the
    current session -- so a level is always knowable before any bar that
    uses it trades (no lookahead). `session_start_hour` sets where one
    session ends and the next begins, in UTC: 0 for a calendar rollover, 17
    for the common forex convention of rolling at the NY 5pm close (`period`
    'D' only -- coarser periods roll over naturally at their own boundary).

    `period` sets how long a level stays in force before being replaced:
    'D' (default) recomputes daily -- a support/resistance level is
    forgotten every session even if price has been respecting it for a
    week. 'W' or 'M' hold the previous week's/month's levels for the whole
    following week/month instead, which is closer to how these levels are
    meant to behave (a real S/R zone doesn't reset at midnight).
    """
    shifted = df.index - pd.Timedelta(hours=session_start_hour)
    if period == "D":
        session_day = shifted.floor("D")
    else:
        tz = shifted.tz
        session_day = shifted.tz_localize(None).to_period(period).start_time.tz_localize(tz)

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
