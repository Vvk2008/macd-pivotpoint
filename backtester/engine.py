from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # "long" or "short"
    stop: float
    target: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None

    def close(self, time: pd.Timestamp, price: float, reason: str) -> None:
        self.exit_time = time
        self.exit_price = price
        self.exit_reason = reason
        if self.direction == "long":
            self.pnl = (price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - price) * self.size


class Backtester:
    """Event-driven, bar-by-bar backtester. One open position at a time,
    no lookahead: entries/exits for bar t only ever use bar t's own OHLC
    and signals computed from data available at or before t.
    """

    def __init__(
        self,
        initial_capital: float = 10_000,
        risk_per_trade: float = 0.01,
        spread: float = 0.0002,
        commission: float = 0.0,
        block_same_bar_reversal: bool = True,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread = spread
        self.commission = commission
        self.block_same_bar_reversal = block_same_bar_reversal

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> dict:
        equity = self.initial_capital
        equity_curve = {}
        trades: list[Trade] = []
        position: Optional[Trade] = None

        for t, bar in df.iterrows():
            sig = signals.loc[t]
            just_closed = False

            if position is not None:
                exit_price, reason = self._check_exit(position, bar, sig)
                if exit_price is not None:
                    position.close(t, exit_price, reason)
                    equity += position.pnl - self.commission
                    trades.append(position)
                    position = None
                    just_closed = True

            # A signal exit and the opposite-direction entry are often the
            # *same* crossover event -- closing a long and opening a short
            # off one bar's bear-cross, with no fresh setup in between.
            # Blocking same-bar reversal forces a genuinely new signal
            # before re-entering.
            if position is None and not (just_closed and self.block_same_bar_reversal):
                if bool(sig["long_entry"]) and not np.isnan(sig["stop"]):
                    position = self._open(t, bar, "long", sig, equity)
                elif bool(sig["short_entry"]) and not np.isnan(sig["stop"]):
                    position = self._open(t, bar, "short", sig, equity)

            unrealized = 0.0
            if position is not None:
                if position.direction == "long":
                    unrealized = (bar["close"] - position.entry_price) * position.size
                else:
                    unrealized = (position.entry_price - bar["close"]) * position.size
            equity_curve[t] = equity + unrealized

        equity_series = pd.Series(equity_curve)
        equity_series.index.name = "timestamp"
        return {"trades": trades, "equity_curve": equity_series, "final_equity": equity_series.iloc[-1] if len(equity_series) else equity}

    def _open(self, t: pd.Timestamp, bar: pd.Series, direction: str, sig: pd.Series, equity: float) -> Optional[Trade]:
        half_spread = self.spread / 2
        entry_price = bar["close"] + half_spread if direction == "long" else bar["close"] - half_spread
        stop = sig["stop"]
        target = sig["target"]
        stop_distance = abs(entry_price - stop)
        if stop_distance <= 0 or np.isnan(stop_distance):
            return None
        risk_amount = equity * self.risk_per_trade
        size = risk_amount / stop_distance
        return Trade(entry_time=t, entry_price=entry_price, direction=direction, stop=stop, target=target, size=size)

    def _check_exit(self, position: Trade, bar: pd.Series, sig: pd.Series):
        if position.direction == "long":
            if bar["low"] <= position.stop:
                return position.stop, "stop"
            if not np.isnan(position.target) and bar["high"] >= position.target:
                return position.target, "target"
            if bool(sig["long_exit"]):
                return bar["close"], "signal"
        else:
            if bar["high"] >= position.stop:
                return position.stop, "stop"
            if not np.isnan(position.target) and bar["low"] <= position.target:
                return position.target, "target"
            if bool(sig["short_exit"]):
                return bar["close"], "signal"
        return None, None
