import numpy as np
import pandas as pd

from backtester.engine import Backtester


def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 0.0
    return df


def _empty_signals(idx):
    return pd.DataFrame(
        {
            "long_entry": False,
            "short_entry": False,
            "long_exit": False,
            "short_exit": False,
            "stop": np.nan,
            "target": np.nan,
        },
        index=idx,
    )


def test_long_trade_hits_target():
    df = _bars(
        [
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.16, 1.09, 1.15],
            [1.15, 1.15, 1.15, 1.15],
        ]
    )
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], ["long_entry", "stop", "target"]] = [True, 1.05, 1.15]

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=0.0, commission=0.0)
    result = bt.run(df, signals)

    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade.direction == "long"
    assert trade.exit_reason == "target"
    assert trade.pnl > 0


def test_long_trade_hits_stop():
    df = _bars(
        [
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.03, 1.04],
            [1.04, 1.04, 1.04, 1.04],
        ]
    )
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], ["long_entry", "stop", "target"]] = [True, 1.05, 1.20]

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=0.0, commission=0.0)
    result = bt.run(df, signals)

    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade.exit_reason == "stop"
    assert trade.pnl < 0


def test_risk_per_trade_sizing():
    df = _bars([[1.10, 1.10, 1.10, 1.10], [1.10, 1.10, 1.05, 1.06]])
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], ["long_entry", "stop", "target"]] = [True, 1.05, 1.30]

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.02, spread=0.0, commission=0.0)
    result = bt.run(df, signals)
    trade = result["trades"][0]

    assert np.isclose(trade.size, 200 / 0.05)


def test_no_entry_without_valid_stop():
    df = _bars([[1.10, 1.10, 1.10, 1.10], [1.10, 1.10, 1.10, 1.10]])
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], "long_entry"] = True  # stop left as NaN

    bt = Backtester()
    result = bt.run(df, signals)

    assert len(result["trades"]) == 0


def test_blocks_same_bar_reversal_by_default():
    df = _bars([[1.10, 1.10, 1.10, 1.10], [1.10, 1.10, 1.10, 1.10], [1.10, 1.10, 1.10, 1.10]])
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], ["long_entry", "stop", "target"]] = [True, 1.05, 1.20]
    # Bar 1: the same event closes the long (signal exit) and would open a
    # short -- should be blocked, since it's not a fresh setup.
    signals.loc[df.index[1], ["long_exit", "short_entry", "stop", "target"]] = [True, True, 1.15, 1.00]

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=0.0, commission=0.0)
    result = bt.run(df, signals)

    assert len(result["trades"]) == 1
    assert result["trades"][0].exit_reason == "signal"


def test_allows_reversal_on_a_later_bar_with_block_enabled():
    df = _bars(
        [
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.10, 1.10],
        ]
    )
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], ["long_entry", "stop", "target"]] = [True, 1.05, 1.20]
    signals.loc[df.index[1], "long_exit"] = True  # closes with no new entry this bar
    signals.loc[df.index[2], ["short_entry", "stop", "target"]] = [True, 1.15, 1.00]  # fresh setup, later bar
    signals.loc[df.index[3], "short_exit"] = True  # close it out so it lands in `trades`

    bt = Backtester(initial_capital=10_000, risk_per_trade=0.01, spread=0.0, commission=0.0)
    result = bt.run(df, signals)

    assert len(result["trades"]) == 2
    assert result["trades"][1].direction == "short"


def test_allows_same_bar_reversal_when_disabled():
    df = _bars(
        [
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.10, 1.10],
            [1.10, 1.10, 1.10, 1.10],
        ]
    )
    signals = _empty_signals(df.index)
    signals.loc[df.index[0], ["long_entry", "stop", "target"]] = [True, 1.05, 1.20]
    signals.loc[df.index[1], ["long_exit", "short_entry", "stop", "target"]] = [True, True, 1.15, 1.00]
    signals.loc[df.index[2], "short_exit"] = True  # close it out so it lands in `trades`

    bt = Backtester(
        initial_capital=10_000, risk_per_trade=0.01, spread=0.0, commission=0.0, block_same_bar_reversal=False
    )
    result = bt.run(df, signals)

    assert len(result["trades"]) == 2
    assert result["trades"][1].direction == "short"
