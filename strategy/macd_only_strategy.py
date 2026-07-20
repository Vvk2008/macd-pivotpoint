import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    macd: pd.DataFrame,
    atr: pd.Series,
    stop_atr_mult: float = 1.5,
    target_atr_mult: float = 3.0,
) -> pd.DataFrame:
    """Plain MACD crossover, no pivot filter at all.

    Long on a bullish crossover, short on a bearish one. Stop/target are
    ATR-based (volatility-scaled) rather than tied to any price level, so
    this strategy has no dependency on pivot points whatsoever -- it exists
    purely as a baseline to check whether the pivot confluence in the other
    strategies is adding anything over MACD alone.
    """
    close = df["close"]
    macd_line, signal_line = macd["macd"], macd["signal"]
    bull_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    bear_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    has_atr = atr.notna()
    long_entry = bull_cross & has_atr
    short_entry = bear_cross & has_atr

    stop = pd.Series(np.nan, index=df.index)
    target = pd.Series(np.nan, index=df.index)

    stop_dist = atr * stop_atr_mult
    target_dist = atr * target_atr_mult
    stop.loc[long_entry] = (close - stop_dist)[long_entry]
    target.loc[long_entry] = (close + target_dist)[long_entry]
    stop.loc[short_entry] = (close + stop_dist)[short_entry]
    target.loc[short_entry] = (close - target_dist)[short_entry]

    out = pd.DataFrame(index=df.index)
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    out["long_exit"] = bear_cross
    out["short_exit"] = bull_cross
    out["stop"] = stop
    out["target"] = target
    return out
