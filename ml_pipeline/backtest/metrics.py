"""Portfolio performance metrics for backtest reporting.

All functions take an equity curve (pd.Series with DatetimeIndex)
or a trade log (pd.DataFrame) and return scalar or dict metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def total_return(equity: pd.Series) -> float:
    """Cumulative return from first to last equity value."""
    if len(equity) < 2:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) - 1.0


def annualized_return(equity: pd.Series, trading_days: int = 252) -> float:
    """CAGR assuming `trading_days` per year."""
    if len(equity) < 2:
        return 0.0
    n = len(equity) - 1
    return (equity.iloc[-1] / equity.iloc[0]) ** (trading_days / n) - 1.0


def annualized_volatility(equity: pd.Series, trading_days: int = 252) -> float:
    """Annualized standard deviation of daily returns."""
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) < 2:
        return 0.0
    return float(daily_ret.std() * np.sqrt(trading_days))


def sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = annualized_return(equity, trading_days)
    ann_vol = annualized_volatility(equity, trading_days)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, trading_days: int = 252) -> float:
    """Annualized return / abs(max drawdown)."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return annualized_return(equity, trading_days) / mdd


def sortino_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Sortino ratio (penalizes only downside deviation)."""
    daily_ret = equity.pct_change().dropna()
    excess = daily_ret - risk_free_rate / trading_days
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std() * np.sqrt(trading_days)
    ann_ret = annualized_return(equity, trading_days)
    if downside_std == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_std


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of closed trades with positive P&L."""
    sells = trades[trades["action"] == "SELL"] if "action" in trades.columns else trades
    if sells.empty:
        return 0.0
    return (sells["pnl"] > 0).mean()


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit / gross loss."""
    sells = trades[trades["action"] == "SELL"] if "action" in trades.columns else trades
    if sells.empty:
        return 0.0
    gross_profit = sells.loc[sells["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(sells.loc[sells["pnl"] <= 0, "pnl"].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def avg_hold_days(trades: pd.DataFrame) -> float:
    """Average holding period in calendar days (requires entry_date column)."""
    if "entry_date" not in trades.columns or "date" not in trades.columns:
        return float("nan")
    sells = trades[trades["action"] == "SELL"].copy()
    if sells.empty:
        return float("nan")
    sells["hold_days"] = (
        pd.to_datetime(sells["date"]) - pd.to_datetime(sells["entry_date"])
    ).dt.days
    return sells["hold_days"].mean()


def compute_all(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """Compute all metrics and return as a dict."""
    metrics = {
        "total_return_pct": round(total_return(equity) * 100, 2),
        "annualized_return_pct": round(annualized_return(equity) * 100, 2),
        "annualized_vol_pct": round(annualized_volatility(equity) * 100, 2),
        "sharpe": round(sharpe_ratio(equity, risk_free_rate), 3),
        "sortino": round(sortino_ratio(equity, risk_free_rate), 3),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "calmar": round(calmar_ratio(equity), 3),
    }

    if trades is not None and not trades.empty:
        metrics["win_rate_pct"] = round(win_rate(trades) * 100, 1)
        metrics["profit_factor"] = round(profit_factor(trades), 2)
        n_trades = len(trades[trades["action"] == "SELL"]) if "action" in trades.columns else len(trades)
        metrics["n_trades"] = n_trades

    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(equity.index, method="ffill").dropna()
        eq = equity.reindex(bm.index).dropna()
        if len(eq) >= 2:
            metrics["benchmark_return_pct"] = round(total_return(bm) * 100, 2)
            metrics["alpha_pct"] = round(
                (total_return(eq) - total_return(bm)) * 100, 2
            )
            # Information ratio
            daily_alpha = eq.pct_change() - bm.pct_change()
            daily_alpha = daily_alpha.dropna()
            if daily_alpha.std() > 0:
                metrics["information_ratio"] = round(
                    daily_alpha.mean() / daily_alpha.std() * np.sqrt(252), 3
                )

    return metrics
