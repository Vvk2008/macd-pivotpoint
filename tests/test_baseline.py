import pandas as pd

from backtester.baseline import buy_and_hold_equity_curve


def test_buy_and_hold_starts_at_initial_capital_and_tracks_price():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"close": [1.00, 1.10, 0.90, 1.05]}, index=idx)

    equity = buy_and_hold_equity_curve(df, initial_capital=10_000)

    assert equity.iloc[0] == 10_000
    assert equity.iloc[1] == 11_000
    assert equity.iloc[2] == 9_000
    assert equity.iloc[3] == 10_500
