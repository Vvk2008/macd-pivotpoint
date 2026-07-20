import numpy as np
import pandas as pd

LEVELS = ["S3", "S2", "S1", "PP", "R1", "R2", "R3"]
SUPPORT_LEVELS = ["S3", "S2", "S1", "PP"]
RESISTANCE_LEVELS = ["PP", "R1", "R2", "R3"]


def generate_signals(
    df: pd.DataFrame,
    macd: pd.DataFrame,
    pivots: pd.DataFrame,
    tolerance: float = 0.0005,
    confirmation_window: int = 3,
) -> pd.DataFrame:
    """Pivot bounce, MACD-confirmed.

    A "touch" is any bar whose close lands within `tolerance` of a
    support-capable level (S1-S3 or PP) or resistance-capable level (PP or
    R1-R3). A long triggers when MACD crosses bullish within
    `confirmation_window` bars of a support touch -- the touch establishes
    location (price is at a level that should hold), the crossover confirms
    the bounce is actually happening. Short is the mirror image at
    resistance.

    Unlike a plain PP-side filter, the stop and target are anchored to the
    *specific* level touched, one pivot step beyond it in each direction --
    e.g. touching S1 gives a stop at S2 and a target at PP -- so the stop
    always has real room instead of sitting at the level price just left.
    """
    close = df["close"]
    macd_line, signal_line = macd["macd"], macd["signal"]
    bull_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    bear_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    touched_support = _touch_level(close, pivots, SUPPORT_LEVELS, tolerance)
    touched_resistance = _touch_level(close, pivots, RESISTANCE_LEVELS, tolerance)

    hold_bars = max(confirmation_window - 1, 0)
    active_support = touched_support.ffill(limit=hold_bars)
    active_resistance = touched_resistance.ffill(limit=hold_bars)

    long_entry = bull_cross & active_support.notna()
    short_entry = bear_cross & active_resistance.notna()

    stop = pd.Series(np.nan, index=df.index)
    target = pd.Series(np.nan, index=df.index)

    for i in np.where(long_entry.to_numpy())[0]:
        idx = LEVELS.index(active_support.iloc[i])
        if idx - 1 < 0:
            long_entry.iloc[i] = False  # touched S3: no level below to use as a stop
            continue
        lvl_stop = pivots[LEVELS[idx - 1]].iloc[i]
        lvl_target = pivots[LEVELS[idx + 1]].iloc[i]
        px = close.iloc[i]
        if not (lvl_stop < px < lvl_target):
            # price already ran past the target (or through the stop) during
            # the confirmation window, before the trade could even open
            long_entry.iloc[i] = False
            continue
        stop.iloc[i] = lvl_stop
        target.iloc[i] = lvl_target

    for i in np.where(short_entry.to_numpy())[0]:
        idx = LEVELS.index(active_resistance.iloc[i])
        if idx + 1 > len(LEVELS) - 1:
            short_entry.iloc[i] = False  # touched R3: no level above to use as a stop
            continue
        lvl_stop = pivots[LEVELS[idx + 1]].iloc[i]
        lvl_target = pivots[LEVELS[idx - 1]].iloc[i]
        px = close.iloc[i]
        if not (lvl_target < px < lvl_stop):
            short_entry.iloc[i] = False
            continue
        stop.iloc[i] = lvl_stop
        target.iloc[i] = lvl_target

    out = pd.DataFrame(index=df.index)
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    out["long_exit"] = bear_cross
    out["short_exit"] = bull_cross
    out["stop"] = stop
    out["target"] = target
    return out


def _touch_level(close: pd.Series, pivots: pd.DataFrame, candidate_levels: list, tolerance: float) -> pd.Series:
    """Nearest candidate level within `tolerance` of each bar's close, or NaN."""
    levels = pivots[candidate_levels]
    valid = pivots["PP"].notna()
    # idxmin errors on all-NaN rows (the first trading day, before any pivots
    # exist); fill with +inf so it always resolves, then the `valid` mask
    # below nulls those rows out anyway.
    diffs = levels.sub(close, axis=0).abs().fillna(np.inf)
    nearest_level = diffs.idxmin(axis=1)
    nearest_dist = diffs.min(axis=1)
    return nearest_level.where(valid & (nearest_dist <= tolerance))
