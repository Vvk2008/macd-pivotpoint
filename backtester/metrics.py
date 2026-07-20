import numpy as np
import pandas as pd


def compute_metrics(trades: list, equity_curve: pd.Series, initial_capital: float) -> dict:
    n_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_rate = len(wins) / n_trades if n_trades else np.nan
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Derived from the equity curve itself, not summed trade PnL: a curve
    # also reflects any position still open on the final bar (mark-to-market),
    # and it's the only source of truth when there are no discrete trades at
    # all (e.g. a buy-and-hold baseline).
    final_equity = equity_curve.iloc[-1] if len(equity_curve) else initial_capital
    total_return = final_equity / initial_capital - 1

    returns = equity_curve.pct_change().dropna()
    sharpe = np.nan
    if len(returns) > 1 and returns.std() > 0:
        periods_per_year = _infer_periods_per_year(equity_curve.index)
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) else np.nan

    n_years = (
        (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25 if len(equity_curve) > 1 else np.nan
    )
    cagr = (final_equity / initial_capital) ** (1 / n_years) - 1 if n_years and n_years > 0 else np.nan

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100 if cagr is not None and not np.isnan(cagr) else np.nan,
        "max_drawdown_pct": max_drawdown * 100 if not np.isnan(max_drawdown) else np.nan,
        "sharpe": sharpe,
        "final_equity": final_equity,
    }


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_delta = pd.Series(index).diff().median()
    seconds = median_delta.total_seconds()
    if seconds <= 0:
        return 252.0
    return (365.25 * 24 * 3600) / seconds
