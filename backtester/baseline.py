import pandas as pd


def buy_and_hold_equity_curve(df: pd.DataFrame, initial_capital: float = 10_000) -> pd.Series:
    """Equity curve for buying at the first close and holding to the last bar."""
    close = df["close"]
    equity = initial_capital * (close / close.iloc[0])
    equity.index.name = "timestamp"
    return equity
