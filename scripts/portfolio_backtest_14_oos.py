"""Out-of-sample honesty check for post-hoc pair selection.

The 28-pair IN-SAMPLE portfolio backtest (scripts/portfolio_backtest_28.py)
had 14 pairs that contributed positively. Picking those 14 *because* they
were IS-positive and then re-reporting IS performance would be textbook
selection bias -- with 28 pairs, several will look profitable by luck alone.

The only honest way to judge that selection is to freeze it on IS and then
test the resulting portfolio ONCE on the untouched OOS window (the same
last-5-years hold-out used for parameters everywhere else in this project).
If the 14 still work OOS, the selection captured real edge; if they
collapse, the IS profitability was survivorship noise.

Same shared-account engine, config, and 1% risk as portfolio_backtest_28.py
-- only the pair list (14 IS-positive) and the date filter (OOS instead of
IS) differ.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.metrics import compute_metrics
from data.loader import load_ohlcv_csv
from indicators.macd import compute_macd
from indicators.pivot_points import compute_daily_pivots
from scripts.portfolio_backtest import PortfolioBacktester
from strategy.pivot_bounce_strategy import generate_signals
from utils import infer_pip_size

# The 14 pairs that contributed POSITIVELY in the 28-pair IN-SAMPLE backtest.
# This list is frozen from IS results -- selection happened before OOS is
# ever touched, which is the whole point of the test.
PAIRS = [
    "EURUSD240", "GBPUSD240", "CHFJPY240", "AUDUSD240", "AUDJPY240",
    "EURGBP240", "EURJPY240", "USDJPY240", "EURNZD240", "GBPNZD240",
    "NZDCAD240", "NZDCHF240", "USDCHF240", "CADCHF240",
]
OOS_YEARS = 5
CAPITAL = 10_000
RISK = 0.01


def main():
    data = {}
    for pair in PAIRS:
        df = load_ohlcv_csv(f"data/raw/{pair}.csv")
        cutoff = df.index[-1] - pd.DateOffset(years=OOS_YEARS)
        df = df[df.index >= cutoff]  # OUT-OF-SAMPLE only (the held-out tail)

        pip = infer_pip_size(df["close"])
        macd = compute_macd(df["close"])
        pivots = compute_daily_pivots(df, period="W")
        signals = generate_signals(
            df, macd, pivots, tolerance=18 * pip, confirmation_window=2, min_reward_risk=0.5,
            stop_levels=2, target_levels=2,
        )
        data[pair] = (df, signals, 2 * pip)

    print(f"OOS window per pair: e.g. {PAIRS[0]} {data[PAIRS[0]][0].index[0].date()} -> {data[PAIRS[0]][0].index[-1].date()}")

    bt = PortfolioBacktester(initial_capital=CAPITAL, risk_per_trade=RISK, commission=0.0)
    result = bt.run(data)
    m = compute_metrics(result["trades"], result["equity_curve"], CAPITAL)

    print(f"\nONE SHARED ${CAPITAL:,} ACCOUNT, {len(PAIRS)} IS-positive pairs, {RISK:.0%} risk/trade, OUT-OF-SAMPLE ONLY")
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
    print("\n  Per-pair trade count / P&L contribution (sorted by contribution):")
    n_pos = 0
    for pair, pnls in sorted(by_pair.items(), key=lambda kv: -sum(kv[1])):
        tot = sum(pnls)
        n_pos += tot > 0
        print(f"    {pair:<12} n={len(pnls):4d}  total_pnl=${tot:+10.2f}")
    print(f"\n  Pairs still positive OUT-OF-SAMPLE: {n_pos}/{len(by_pair)}")
    print("  (If far fewer than 14 stay positive, the IS selection was largely luck.)")

    eq = result["equity_curve"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(eq.index, eq.values, color="#1f4e8c", linewidth=1.1)
    ax.fill_between(eq.index, CAPITAL, eq.values, where=(eq.values >= CAPITAL), color="#1f4e8c", alpha=0.12)
    ax.fill_between(eq.index, CAPITAL, eq.values, where=(eq.values < CAPITAL), color="#c0392b", alpha=0.12)
    ax.axhline(CAPITAL, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"OOS honesty check -- {len(PAIRS)} IS-positive pairs, one ${CAPITAL:,} shared account (OUT-OF-SAMPLE)\n"
        f"H4, bounce strategy, weekly pivots, {eq.index[0].date()} -> {eq.index[-1].date()}  |  "
        f"${CAPITAL:,.0f} -> ${m['final_equity']:,.0f}  ({m['total_return_pct']:+.1f}%)  |  "
        f"Max DD {m['max_drawdown_pct']:.1f}%  |  PF {m['profit_factor']:.2f}  |  Sharpe {m['sharpe']:.2f}",
        fontsize=11,
    )
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = "/tmp/claude-0/-home-user-macd-pivotpoint/c65123b3-4cfd-5cba-b859-a0aee813ec6d/scratchpad/portfolio_14pairs_OOS_equity_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
