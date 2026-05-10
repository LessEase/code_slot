"""Event-driven backtest engine.

Simulates a rebalancing strategy driven by model probability scores:

  Every `rebalance_days` trading days:
    1. Score all stocks with the trained LightGBM model
    2. Select top-N by predicted probability
    3. Close positions not in the new selection (if hold period ≥ min_hold)
    4. Open new positions for newly selected stocks

The engine uses a lightweight in-memory portfolio (no SQLite) for speed.
Results include an equity curve, per-trade log, and full metrics report.

No look-ahead bias: on day t the model only sees features up to day t.
The next-day open is used as execution price (realistic fill assumption).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich import box

from ml_pipeline._lgbm import lgb
from ml_pipeline.features.engineer import FEATURE_COLS
from ml_pipeline.backtest.metrics import compute_all
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)
console = Console()


# ── Lightweight in-memory portfolio ───────────────────────────────────────

@dataclass
class _Position:
    symbol: str
    market: str
    shares: float
    entry_price: float
    entry_date: date
    cost: float         # including commission


@dataclass
class _BacktestPortfolio:
    initial_cash: float
    commission_rate: float = 0.0003
    us_commission_per_share: float = 0.005
    usd_cny: float = 7.2

    cash: float = field(init=False)
    positions: Dict[str, _Position] = field(default_factory=dict, init=False)
    trades: List[dict] = field(default_factory=list, init=False)
    equity_curve: List[Tuple[date, float]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.cash = self.initial_cash

    def _commission(self, market: str, shares: float, price: float) -> float:
        if market == "A":
            return max(shares * price * self.commission_rate, 5.0)
        usd_fee = max(shares * self.us_commission_per_share, 1.0)
        return usd_fee * self.usd_cny

    def _price_cny(self, market: str, price: float) -> float:
        return price * self.usd_cny if market == "US" else price

    def buy(self, symbol: str, market: str, price: float,
            trade_date: date, score: float, max_pos_pct: float) -> bool:
        if symbol in self.positions:
            return False
        equity = self.total_equity({})
        target_value = equity * max_pos_pct
        available = min(target_value, self.cash * 0.95)
        price_cny = self._price_cny(market, price)
        if price_cny <= 0 or available < price_cny:
            return False
        shares = available / price_cny
        if market == "A":
            shares = max(int(shares / 100) * 100, 100)
        else:
            shares = max(int(shares), 1)
        cost = shares * price_cny
        commission = self._commission(market, shares, price_cny)
        total_cost = cost + commission
        if total_cost > self.cash:
            return False
        self.cash -= total_cost
        self.positions[symbol] = _Position(
            symbol=symbol, market=market, shares=shares,
            entry_price=price_cny, entry_date=trade_date, cost=total_cost,
        )
        self.trades.append({
            "date": trade_date, "symbol": symbol, "market": market,
            "action": "BUY", "shares": shares, "price": price_cny,
            "commission": commission, "pnl": None, "score": score,
            "entry_date": None,
        })
        return True

    def sell(self, symbol: str, price: float,
             trade_date: date, reason: str) -> bool:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return False
        price_cny = self._price_cny(pos.market, price)
        proceeds = pos.shares * price_cny
        commission = self._commission(pos.market, pos.shares, price_cny)
        net = proceeds - commission
        pnl = net - pos.cost
        self.cash += net
        self.trades.append({
            "date": trade_date, "symbol": symbol, "market": pos.market,
            "action": "SELL", "shares": pos.shares, "price": price_cny,
            "commission": commission, "pnl": pnl, "score": None,
            "entry_date": pos.entry_date, "reason": reason,
        })
        return True

    def total_equity(self, prices: Dict[str, float]) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            p = prices.get(sym, pos.entry_price)
            p_cny = self._price_cny(pos.market, p)
            equity += pos.shares * p_cny
        return equity

    def snapshot(self, dt: date, prices: Dict[str, float]) -> None:
        self.equity_curve.append((dt, self.total_equity(prices)))

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        dates, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.to_datetime(dates), name="equity")

    def trade_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)


# ── Backtest engine ────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Parameters
    ----------
    cfg         : full config dict
    model       : trained LightGBM booster
    meta        : model metadata dict (contains feature_names)
    feature_data: {symbol: feature_df} with DatetimeIndex
    market_map  : {symbol: 'A'|'US'}
    """

    def __init__(
        self,
        cfg: dict,
        model: lgb.Booster,
        meta: dict,
        feature_data: Dict[str, pd.DataFrame],
        market_map: Dict[str, str],
    ):
        self.cfg = cfg
        self.model = model
        self.feature_names = meta["feature_names"]
        self.feature_data = feature_data
        self.market_map = market_map
        self.rebalance_days = cfg["ml_pipeline"]["rebalance_days"]
        self.top_n = cfg["ml_pipeline"]["top_n"]
        self.min_hold_days = cfg["ml_pipeline"].get("min_hold_days", 2)
        self.max_pos_pct = cfg["portfolio"]["position_size_pct"]
        self.initial_capital = cfg["portfolio"]["initial_capital"]
        self.usd_cny = cfg["portfolio"]["usd_cny_rate"]

    def _all_trading_dates(self, start: str, end: str) -> pd.DatetimeIndex:
        """Union of all dates present in feature_data within [start, end]."""
        all_dates = set()
        for df in self.feature_data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(d for d in all_dates
                       if pd.Timestamp(start) <= d <= pd.Timestamp(end))
        return pd.DatetimeIndex(dates)

    def _score_stocks(self, dt: pd.Timestamp) -> pd.Series:
        """Return {symbol: prob} for all stocks with data on day dt."""
        rows = {}
        for sym, df in self.feature_data.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            avail = [c for c in self.feature_names if c in row.index]
            if len(avail) < len(self.feature_names) * 0.7:
                continue
            x = pd.DataFrame([row[avail].reindex(self.feature_names).fillna(0)])
            prob = float(self.model.predict(x)[0])
            rows[sym] = prob
        return pd.Series(rows).sort_values(ascending=False)

    def _get_price(self, symbol: str, dt: pd.Timestamp) -> Optional[float]:
        """Return close price of symbol on day dt (or nearest prior day)."""
        df = self.feature_data.get(symbol)
        if df is None:
            return None
        # Use next available date if dt not a trading day for this stock
        available = df.index[df.index >= dt]
        if available.empty:
            return None
        row = df.loc[available[0]]
        return float(row["close"]) if "close" in row.index else None

    def run(
        self,
        start: str,
        end: str,
        benchmark_series: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Run full backtest from `start` to `end`.

        Returns dict with:
          equity_curve, trade_log, metrics, period_metrics
        """
        portfolio = _BacktestPortfolio(
            initial_cash=self.initial_capital,
            usd_cny=self.usd_cny,
        )
        trading_dates = self._all_trading_dates(start, end)
        if len(trading_dates) == 0:
            raise ValueError(f"No trading dates found between {start} and {end}")

        log.info(
            f"Backtest {start} → {end}  "
            f"({len(trading_dates)} trading days, "
            f"rebalance every {self.rebalance_days}d, top-{self.top_n})"
        )

        rebalance_counter = 0
        for i, dt in enumerate(trading_dates):
            # Collect current prices for equity snapshot
            prices = {
                sym: self._get_price(sym, dt)
                for sym in portfolio.positions
            }
            prices = {k: v for k, v in prices.items() if v is not None}
            portfolio.snapshot(dt.date(), prices)

            # Rebalance on schedule
            if i % self.rebalance_days != 0:
                continue
            rebalance_counter += 1

            scores = self._score_stocks(dt)
            if scores.empty:
                continue
            top_symbols = set(scores.head(self.top_n).index.tolist())

            # Close positions not in top selection (respecting min hold)
            for sym in list(portfolio.positions.keys()):
                if sym not in top_symbols:
                    pos = portfolio.positions[sym]
                    hold = (dt.date() - pos.entry_date).days
                    if hold >= self.min_hold_days:
                        price = self._get_price(sym, dt)
                        if price is not None:
                            portfolio.sell(sym, price, dt.date(), reason="rebalance")

            # Open new positions
            for sym in scores.head(self.top_n).index:
                if sym not in portfolio.positions:
                    price = self._get_price(sym, dt)
                    mkt = self.market_map.get(sym, "A")
                    if price is not None:
                        portfolio.buy(
                            sym, mkt, price, dt.date(),
                            score=float(scores[sym]),
                            max_pos_pct=self.max_pos_pct,
                        )

        # Close all remaining positions on last day
        last_dt = trading_dates[-1]
        for sym in list(portfolio.positions.keys()):
            price = self._get_price(sym, last_dt)
            if price is not None:
                portfolio.sell(sym, price, last_dt.date(), reason="end_of_backtest")

        equity = portfolio.equity_series()
        trades = portfolio.trade_df()
        metrics = compute_all(equity, trades, benchmark_series)
        metrics["rebalances"] = rebalance_counter

        log.info(
            f"Backtest complete: "
            f"return={metrics['total_return_pct']:+.1f}%  "
            f"sharpe={metrics['sharpe']:.2f}  "
            f"maxDD={metrics['max_drawdown_pct']:.1f}%  "
            f"win_rate={metrics.get('win_rate_pct', 0):.1f}%  "
            f"trades={metrics.get('n_trades', 0)}"
        )
        return {
            "equity_curve": equity,
            "trade_log": trades,
            "metrics": metrics,
        }

    def print_report(self, result: Dict, market: str) -> None:
        """Print a rich summary table of backtest results."""
        m = result["metrics"]
        console.rule(f"[bold cyan]Backtest Report — {market} Market")

        # Metrics table
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", justify="right", width=15)

        def _color(v, positive_good=True):
            ok = v >= 0 if positive_good else v <= 0
            return f"[green]{v}[/]" if ok else f"[red]{v}[/]"

        rows = [
            ("Total Return", f"{_color(m['total_return_pct'])}%"),
            ("Annualized Return", f"{_color(m['annualized_return_pct'])}%"),
            ("Annualized Volatility", f"{m['annualized_vol_pct']:.2f}%"),
            ("Sharpe Ratio", _color(m['sharpe'])),
            ("Sortino Ratio", _color(m['sortino'])),
            ("Max Drawdown", f"{_color(m['max_drawdown_pct'], positive_good=False)}%"),
            ("Calmar Ratio", _color(m['calmar'])),
            ("Win Rate", f"{m.get('win_rate_pct', '-'):.1f}%"),
            ("Profit Factor", str(m.get('profit_factor', '-'))),
            ("Total Trades", str(m.get('n_trades', '-'))),
            ("Rebalances", str(m.get('rebalances', '-'))),
        ]
        if "benchmark_return_pct" in m:
            rows.append(("Benchmark Return", f"{m['benchmark_return_pct']:+.2f}%"))
            rows.append(("Alpha vs Benchmark", f"{_color(m['alpha_pct'])}%"))
        if "information_ratio" in m:
            rows.append(("Information Ratio", _color(m['information_ratio'])))

        for label, value in rows:
            table.add_row(label, value)
        console.print(table)

        # Save equity curve CSV
        equity = result["equity_curve"]
        out_dir = Path("data/backtest")
        out_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(out_dir / f"equity_{market}.csv")
        result["trade_log"].to_csv(out_dir / f"trades_{market}.csv", index=False)
        console.print(f"\nResults saved → [dim]{out_dir}[/]")
