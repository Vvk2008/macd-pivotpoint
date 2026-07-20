import numpy as np
import pandas as pd

from indicators.pivot_points import PIVOT_LEVELS_ASC


def generate_signals(df: pd.DataFrame, macd: pd.DataFrame, pivots: pd.DataFrame) -> pd.DataFrame:
    """Combine a MACD crossover with pivot-point structure.

    Long entry: MACD line crosses above the signal line while price trades
    above the daily pivot (PP) -- momentum confirming an already-bullish
    structural level.
    Short entry: mirror image, MACD crosses down while price is below PP.
    Signal exit: the opposite MACD crossover.
    Stop: nearest pivot level on the wrong side of entry.
    Target: nearest pivot level in the direction of the trade.
    """
    macd_line, signal_line = macd["macd"], macd["signal"]
    bull_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    bear_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    close = df["close"]
    above_pivot = close > pivots["PP"]
    below_pivot = close < pivots["PP"]

    long_entry = bull_cross & above_pivot & pivots["PP"].notna()
    short_entry = bear_cross & below_pivot & pivots["PP"].notna()

    stop, target = _nearest_levels(close, pivots, long_entry, short_entry)

    out = pd.DataFrame(index=df.index)
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    out["long_exit"] = bear_cross
    out["short_exit"] = bull_cross
    out["stop"] = stop
    out["target"] = target
    return out


def _nearest_levels(close: pd.Series, pivots: pd.DataFrame, long_entry: pd.Series, short_entry: pd.Series):
    levels = pivots[PIVOT_LEVELS_ASC]
    stop = pd.Series(np.nan, index=close.index)
    target = pd.Series(np.nan, index=close.index)

    for i in np.where(long_entry.to_numpy())[0]:
        row = levels.iloc[i]
        px = close.iloc[i]
        below = row[row < px]
        above = row[row > px]
        if len(below):
            stop.iloc[i] = below.max()
        if len(above):
            target.iloc[i] = above.min()

    for i in np.where(short_entry.to_numpy())[0]:
        row = levels.iloc[i]
        px = close.iloc[i]
        above = row[row > px]
        below = row[row < px]
        if len(above):
            stop.iloc[i] = above.min()
        if len(below):
            target.iloc[i] = below.max()

    return stop, target
