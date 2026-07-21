"""One shared account trading all 5 pairs simultaneously, instead of 5
independent per-pair accounts summed together.

Key difference from backtester/engine.py's single-instrument Backtester:
position size for a new trade is 1% of the account's CURRENT total equity
(realized cash + unrealized P&L across every other currently-open
position, across all pairs), not 1% of that pair's own isolated capital.
Up to 5 positions can be open at once, one per pair.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from strategy.pivot_bounce_strategy import generate_signals
from utils import infer_pip_size

PAIRS = ["EURUSD240", "GBPUSD240", "USDCAD240", "USDCHF240", "USDJPY240"]


@dataclass
class PortfolioTrade:
    pair: str
    entry_time: pd.Timestamp
    entry_price: float
    direction: str
    stop: float
    target: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None

    def close(self, time, price, reason):
        self.exit_time = time
        self.exit_price = price
        self.exit_reason = reason
        if self.direction == "long":
            self.pnl = (price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - price) * self.size


def check_exit(pos, bar, sig):
    if pos.direction == "long":
        if bar["low"] <= pos.stop:
            return pos.stop, "stop"
        if not np.isnan(pos.target) and bar["high"] >= pos.target:
            return pos.target, "target"
        if bool(sig["long_exit"]):
            return bar["close"], "signal"
    else:
        if bar["high"] >= pos.stop:
            return pos.stop, "stop"
        if not np.isnan(pos.target) and bar["low"] <= pos.target:
            return pos.target, "target"
        if bool(sig["short_exit"]):
            return bar["close"], "signal"
    return None, None


class PortfolioBacktester:
    def __init__(self, initial_capital=10_000, risk_per_trade=0.01, commission=0.0, block_same_bar_reversal=True):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.block_same_bar_reversal = block_same_bar_reversal

    def run(self, data: dict):
        """data: {pair: (df, signals, spread)}"""
        global_index = pd.DatetimeIndex(sorted(set().union(*[df.index for df, _, _ in data.values()])))
        # forward-filled close per pair on the *global* timeline, used only to mark unrealized
        # P&L between that pair's own actual bars (e.g. before its data history starts)
        close_ffilled = {
            pair: df["close"].reindex(global_index, method="ffill") for pair, (df, _, _) in data.items()
        }

        cash = self.initial_capital
        open_pos = {}  # pair -> PortfolioTrade
        trades = []
        equity_curve = {}

        def unrealized(exclude_pair=None):
            total = 0.0
            for pair, pos in open_pos.items():
                if pair == exclude_pair:
                    continue
                px = close_ffilled[pair].get(equity_curve_t, pos.entry_price)
                if pd.isna(px):
                    px = pos.entry_price
                total += (px - pos.entry_price) * pos.size if pos.direction == "long" else (pos.entry_price - px) * pos.size
            return total

        for t in global_index:
            equity_curve_t = t
            just_closed = set()

            # exits first, across every pair that has a real bar right now
            for pair, (df, signals, spread) in data.items():
                if t not in df.index or pair not in open_pos:
                    continue
                bar, sig = df.loc[t], signals.loc[t]
                exit_price, reason = check_exit(open_pos[pair], bar, sig)
                if exit_price is not None:
                    pos = open_pos.pop(pair)
                    pos.close(t, exit_price, reason)
                    cash += pos.pnl - self.commission
                    trades.append(pos)
                    just_closed.add(pair)

            # entries, sized off current TOTAL equity (cash + unrealized on other open positions)
            for pair, (df, signals, spread) in data.items():
                if t not in df.index or pair in open_pos:
                    continue
                if pair in just_closed and self.block_same_bar_reversal:
                    continue
                bar, sig = df.loc[t], signals.loc[t]
                half_spread = spread / 2
                current_equity = cash + unrealized()

                if bool(sig["long_entry"]) and not np.isnan(sig["stop"]):
                    entry_price = bar["close"] + half_spread
                    stop_distance = abs(entry_price - sig["stop"])
                    if stop_distance > 0:
                        size = (current_equity * self.risk_per_trade) / stop_distance
                        open_pos[pair] = PortfolioTrade(pair, t, entry_price, "long", sig["stop"], sig["target"], size)
                elif bool(sig["short_entry"]) and not np.isnan(sig["stop"]):
                    entry_price = bar["close"] - half_spread
                    stop_distance = abs(entry_price - sig["stop"])
                    if stop_distance > 0:
                        size = (current_equity * self.risk_per_trade) / stop_distance
                        open_pos[pair] = PortfolioTrade(pair, t, entry_price, "short", sig["stop"], sig["target"], size)

            equity_curve[t] = cash + unrealized()

        eq = pd.Series(equity_curve)
        eq.index.name = "timestamp"
        return {"trades": trades, "equity_curve": eq, "final_equity": eq.iloc[-1] if len(eq) else cash}


def main():
    CAPITAL = 10_000
    RISK = 0.01

    data = {}
    for pair in PAIRS:
        df = load_ohlcv_csv(f"data/raw/{pair}.csv")
        pip = infer_pip_size(df["close"])
        macd = compute_macd(df["close"])
        pivots = compute_daily_pivots(df, period="W")
        signals = generate_signals(
            df, macd, pivots, tolerance=18 * pip, confirmation_window=2, min_reward_risk=0.5,
            stop_levels=2, target_levels=2,
        )
        data[pair] = (df, signals, 2 * pip)

    bt = PortfolioBacktester(initial_capital=CAPITAL, risk_per_trade=RISK, commission=0.0)
    result = bt.run(data)
    m = compute_metrics(result["trades"], result["equity_curve"], CAPITAL)

    print(f"ONE SHARED ACCOUNT trading all 5 pairs, {CAPITAL:,} start, {RISK:.0%} risk/trade")
    print(f"  Trades         : {m['n_trades']}")
    print(f"  Win rate       : {m['win_rate']:.1%}")
    print(f"  Profit factor  : {m['profit_factor']:.2f}")
    print(f"  Total return   : {m['total_return_pct']:+.2f}%")
    print(f"  CAGR           : {m['cagr_pct']:.2f}%")
    print(f"  Max drawdown   : {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe         : {m['sharpe']:.2f}")
    print(f"  Final equity   : ${m['final_equity']:,.2f}")

    by_pair = {}
    for t in result["trades"]:
        by_pair.setdefault(t.pair, []).append(t.pnl)
    print("\n  Per-pair trade count / P&L contribution:")
    for pair, pnls in by_pair.items():
        print(f"    {pair:<10} n={len(pnls):4d}  total_pnl=${sum(pnls):+9.2f}")

    eq = result["equity_curve"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(eq.index, eq.values, color="#1f4e8c", linewidth=1.1)
    ax.fill_between(eq.index, CAPITAL, eq.values, where=(eq.values >= CAPITAL), color="#1f4e8c", alpha=0.12)
    ax.fill_between(eq.index, CAPITAL, eq.values, where=(eq.values < CAPITAL), color="#c0392b", alpha=0.12)
    ax.axhline(CAPITAL, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"Single Shared-Account Portfolio -- one ${CAPITAL:,} account trading all 5 pairs\n"
        f"H4, bounce strategy, weekly pivots, 2010-2026  |  "
        f"${CAPITAL:,.0f} -> ${m['final_equity']:,.0f}  ({m['total_return_pct']:+.1f}%)  |  "
        f"Max DD {m['max_drawdown_pct']:.1f}%  |  PF {m['profit_factor']:.2f}",
        fontsize=12,
    )
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "/tmp/claude-0/-home-user-macd-pivotpoint/c65123b3-4cfd-5cba-b859-a0aee813ec6d/scratchpad/single_account_equity_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
