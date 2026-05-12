"""Event-driven backtest engine.

Simulates a rebalancing strategy driven by model probability scores:

  On signal day t (every `rebalance_days` trading days):
    1. Score every stock with the trained LightGBM model using features
       known at the close of day t.
    2. Select top-N by predicted probability — this becomes the *pending*
       target portfolio.

  On execution day t + `execution_lag_days`:
    3. Sell positions not in the pending target (subject to A-share T+1
       and min-hold constraints, skipping limit-down names that can't fill).
    4. Buy positions newly in the target (skipping A-share limit-up names).
    5. Fills are at that day's close, adjusted by `slippage_bps`.

Realism guards (the previous version did none of these):
  - Signal/execution lag prevents using the same close that produced the
    signal as the fill price.
  - Per-fill slippage in basis points.
  - A-share daily price-limit filter (no fills when ret_1d ≥ +limit on
    the buy side or ≤ -limit on the sell side).
  - A-share T+1: a position bought today can be sold no earlier than the
    next trading day.

For honest numbers the deployment model must have been trained with a
holdout, and the backtest start date should be after that train cutoff.
``pipeline.step_backtest`` enforces this automatically.
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
from ml_pipeline.data.industry import UNKNOWN_INDUSTRY
from stock_trading.data.fetcher import classify_a_share_board
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
    slippage_bps: float = 0.0      # one-sided, in basis points (10 = 0.10%)

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

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.slippage_bps / 10000.0
        return price * (1.0 + slip) if side == "BUY" else price * (1.0 - slip)

    def buy(self, symbol: str, market: str, price: float,
            trade_date: date, score: float, max_pos_pct: float) -> bool:
        if symbol in self.positions:
            return False
        fill_price = self._apply_slippage(price, "BUY")
        equity = self.total_equity({})
        target_value = equity * max_pos_pct
        available = min(target_value, self.cash * 0.95)
        price_cny = self._price_cny(market, fill_price)
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
        fill_price = self._apply_slippage(price, "SELL")
        price_cny = self._price_cny(pos.market, fill_price)
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
            p = prices.get(sym)
            if p is None:
                # entry_price is already in CNY; do NOT re-apply the USDCNY rate.
                equity += pos.shares * pos.entry_price
            else:
                equity += pos.shares * self._price_cny(pos.market, p)
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
        industry_map: Optional[Dict[str, str]] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.feature_names = meta["feature_names"]
        self.feature_data = feature_data
        self.market_map = market_map
        self.industry_map = industry_map or {}
        ml = cfg["ml_pipeline"]
        self.rebalance_days = ml["rebalance_days"]
        self.top_n = ml["top_n"]
        self.min_hold_days = ml.get("min_hold_days", 2)
        # Match training-time neutralization. If a model was trained with
        # neutralize_by_industry=true we MUST apply the same transform at
        # inference, otherwise feature distributions diverge sharply.
        self.neutralize = bool(ml.get("neutralize_by_industry", False))
        self.execution_lag_days = max(1, int(ml.get("execution_lag_days", 1)))
        self.slippage_bps = float(ml.get("slippage_bps", 0.0))

        # Board-aware price-limit table: ChiNext/STAR are 20%, ST is 5%.
        # Falls back to legacy scalar `a_share_price_limit` when the new dict
        # is absent (back-compat with older configs).
        limit_cfg = ml.get("a_share_price_limits")
        if isinstance(limit_cfg, dict):
            self.a_share_price_limits = {k: float(v) for k, v in limit_cfg.items()}
        else:
            scalar = float(ml.get("a_share_price_limit", 0.095))
            self.a_share_price_limits = {"DEFAULT": scalar}
        self._a_limit_log_ret = {
            k: float(np.log1p(v)) for k, v in self.a_share_price_limits.items()
        }

        # Point-in-time liquidity gate for A-share universe (avoids picking
        # stocks too thin for the assumed fill model).
        self.min_avg_turnover = float(ml.get("min_avg_turnover_cny", 0.0) or 0.0)

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
        """Return {symbol: prob} for all *liquid* stocks with data on day dt.

        When ``neutralize_by_industry`` is enabled, features for the day's
        cross-section are z-scored within each (industry) bucket before
        prediction — matching the transformation applied at training time.
        """
        # Step 1: collect feature rows for every eligible stock on dt
        symbols: List[str] = []
        rows: List[pd.Series] = []
        for sym, df in self.feature_data.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            if (
                self.min_avg_turnover > 0
                and self.market_map.get(sym) == "A"
                and "amount_20d_avg" in row.index
            ):
                amt = row.get("amount_20d_avg")
                if pd.isna(amt) or float(amt) < self.min_avg_turnover:
                    continue
            avail_count = sum(1 for c in self.feature_names if c in row.index)
            if avail_count < len(self.feature_names) * 0.7:
                continue
            symbols.append(sym)
            rows.append(row[self.feature_names].reindex(self.feature_names))

        if not symbols:
            return pd.Series(dtype=float)

        X = pd.DataFrame(rows, index=symbols).astype(float)

        # Step 2: industry z-score (cross-section on this single date)
        if self.neutralize:
            industries = pd.Series(
                [self.industry_map.get(s.zfill(6), UNKNOWN_INDUSTRY) for s in symbols],
                index=symbols,
            )
            grouped = X.groupby(industries.values, sort=False)
            means = grouped.transform("mean")
            stds = grouped.transform("std").replace(0, np.nan)
            sizes = industries.groupby(industries.values).transform("size")
            # Keep z-scores only where the industry group is non-degenerate;
            # otherwise zero (the model sees "no industry signal").
            keep = pd.Series(sizes.values >= 5, index=X.index)
            X = ((X - means) / stds).where(keep, 0.0).fillna(0.0).clip(-5, 5)
        else:
            X = X.fillna(0.0)

        # Step 3: batch predict
        probs = self.model.predict(X)
        return pd.Series(probs, index=symbols).sort_values(ascending=False)

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

    def _hits_price_limit(self, symbol: str, dt: pd.Timestamp, side: str) -> bool:
        """Return True if symbol is at A-share daily limit on day dt and so
        cannot be filled on the requested side. US tickers are never blocked.

        Approximation: we only have end-of-day data, so we treat ``ret_1d``
        (log return today vs. yesterday's close) >= +limit as "limit-up"
        and <= -limit as "limit-down". This filters out the bulk of
        unfillable A-share names. The threshold is board-specific:
        SH/SZ main 10%, ChiNext/STAR 20%, ST 5%.
        """
        if self.market_map.get(symbol) != "A":
            return False
        df = self.feature_data.get(symbol)
        if df is None or dt not in df.index:
            return False
        ret = df.loc[dt].get("ret_1d")
        if ret is None or pd.isna(ret):
            return False
        ret = float(ret)
        board = classify_a_share_board(symbol)
        limit_log = self._a_limit_log_ret.get(
            board, self._a_limit_log_ret.get("DEFAULT", np.log1p(0.095))
        )
        if side == "BUY":
            return ret >= limit_log
        return ret <= -limit_log

    def _can_sell_today(self, pos: _Position, dt: pd.Timestamp) -> bool:
        """A-share T+1: buy day cannot also be sell day. Plus min_hold_days."""
        days_held = (dt.date() - pos.entry_date).days
        if days_held < self.min_hold_days:
            return False
        if pos.market == "A" and pos.entry_date >= dt.date():
            return False
        return True

    def _universe_buy_hold_return(self, trading_dates: pd.DatetimeIndex) -> Tuple[float, int]:
        """Equal-weighted buy-and-hold return of the entire universe over
        the backtest window. Diagnostic only — answers "how much of the
        strategy's return is just beta from a universe that was filtered
        for survivors?" Returns (mean_return, n_symbols).
        """
        if len(trading_dates) < 2:
            return 0.0, 0
        start_dt, end_dt = trading_dates[0], trading_dates[-1]
        rets = []
        for sym, df in self.feature_data.items():
            window = df.loc[(df.index >= start_dt) & (df.index <= end_dt), "close"]
            if len(window) < 2 or window.iloc[0] <= 0:
                continue
            rets.append(float(window.iloc[-1] / window.iloc[0] - 1.0))
        if not rets:
            return 0.0, 0
        return float(np.mean(rets)), len(rets)

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
            slippage_bps=self.slippage_bps,
        )
        trading_dates = self._all_trading_dates(start, end)
        if len(trading_dates) == 0:
            raise ValueError(f"No trading dates found between {start} and {end}")

        log.info(
            f"Backtest {start} → {end}  "
            f"({len(trading_dates)} trading days, "
            f"rebalance every {self.rebalance_days}d, top-{self.top_n}, "
            f"exec_lag={self.execution_lag_days}d, slip={self.slippage_bps:.0f}bps)"
        )

        rebalance_counter = 0
        # Pending signal generated on a prior signal day, executed `lag` days later.
        # Tuple: (target_symbols, scores, signal_index_in_trading_dates)
        pending: Optional[Tuple[set, pd.Series, int]] = None
        skipped_limit_up = 0
        skipped_limit_down = 0
        # Diagnostics: collect the mean predicted probability of the top-N
        # picks across all rebalances, to spot suspiciously confident models.
        topn_prob_means: List[float] = []

        for i, dt in enumerate(trading_dates):
            # ── 1) Execute any pending order at TODAY's close ─────────────
            if pending is not None and (i - pending[2]) >= self.execution_lag_days:
                target_symbols, scores, _ = pending
                # Sells first (frees cash for buys)
                for sym in list(portfolio.positions.keys()):
                    if sym in target_symbols:
                        continue
                    pos = portfolio.positions[sym]
                    if not self._can_sell_today(pos, dt):
                        continue
                    if self._hits_price_limit(sym, dt, "SELL"):
                        skipped_limit_down += 1
                        continue
                    price = self._get_price(sym, dt)
                    if price is not None:
                        portfolio.sell(sym, price, dt.date(), reason="rebalance")

                # Then buys
                for sym in scores.head(self.top_n).index:
                    if sym in portfolio.positions:
                        continue
                    if self._hits_price_limit(sym, dt, "BUY"):
                        skipped_limit_up += 1
                        continue
                    price = self._get_price(sym, dt)
                    if price is None:
                        continue
                    portfolio.buy(
                        sym, self.market_map.get(sym, "A"), price, dt.date(),
                        score=float(scores[sym]),
                        max_pos_pct=self.max_pos_pct,
                    )
                pending = None

            # ── 2) Mark-to-market snapshot at today's close (post-trade) ──
            prices = {
                sym: self._get_price(sym, dt)
                for sym in portfolio.positions
            }
            prices = {k: v for k, v in prices.items() if v is not None}
            portfolio.snapshot(dt.date(), prices)

            # ── 3) Generate signal on schedule (executes on i + lag) ──────
            if i % self.rebalance_days != 0:
                continue
            # Don't bother generating a signal we won't have time to execute.
            if i + self.execution_lag_days >= len(trading_dates):
                continue
            scores = self._score_stocks(dt)
            if scores.empty:
                continue
            rebalance_counter += 1
            topn_prob_means.append(float(scores.head(self.top_n).mean()))
            pending = (
                set(scores.head(self.top_n).index.tolist()),
                scores,
                i,
            )

        # Close all remaining positions on the last day at its close (no lag,
        # this is a forced liquidation at end of backtest, not a strategy fill).
        last_dt = trading_dates[-1]
        for sym in list(portfolio.positions.keys()):
            price = self._get_price(sym, last_dt)
            if price is not None:
                portfolio.sell(sym, price, last_dt.date(), reason="end_of_backtest")

        if skipped_limit_up or skipped_limit_down:
            log.info(
                f"Skipped fills due to A-share price limits: "
                f"{skipped_limit_up} limit-up (buy), {skipped_limit_down} limit-down (sell)"
            )

        equity = portfolio.equity_series()
        trades = portfolio.trade_df()
        metrics = compute_all(equity, trades, benchmark_series)
        metrics["rebalances"] = rebalance_counter

        # Diagnostic baselines: how much of the return is just survivors-beta?
        bh_return, bh_n = self._universe_buy_hold_return(trading_dates)
        metrics["universe_buy_hold_pct"] = round(bh_return * 100, 2)
        metrics["universe_size"] = bh_n
        metrics["alpha_vs_universe_pct"] = round(
            (metrics["total_return_pct"] / 100 - bh_return) * 100, 2
        )
        if topn_prob_means:
            metrics["topn_prob_mean"] = round(float(np.mean(topn_prob_means)), 4)

        log.info(
            f"Backtest complete: "
            f"return={metrics['total_return_pct']:+.1f}%  "
            f"sharpe={metrics['sharpe']:.2f}  "
            f"maxDD={metrics['max_drawdown_pct']:.1f}%  "
            f"win_rate={metrics.get('win_rate_pct', 0):.1f}%  "
            f"trades={metrics.get('n_trades', 0)}"
        )
        log.info(
            f"Sanity baselines: "
            f"universe equal-weight B&H={metrics['universe_buy_hold_pct']:+.1f}% "
            f"({bh_n} symbols), α_vs_universe={metrics['alpha_vs_universe_pct']:+.1f}%"
        )
        if "topn_prob_mean" in metrics:
            log.info(
                f"Model confidence: mean top-{self.top_n} predicted prob = "
                f"{metrics['topn_prob_mean']:.3f}"
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
        if "universe_buy_hold_pct" in m:
            rows.append((
                f"Universe B&H (eq-weight, n={m.get('universe_size','-')})",
                f"{m['universe_buy_hold_pct']:+.2f}%",
            ))
            rows.append(("α vs Universe B&H", f"{_color(m['alpha_vs_universe_pct'])}%"))
        if "topn_prob_mean" in m:
            rows.append((f"Mean top-N prob (diag.)", f"{m['topn_prob_mean']:.3f}"))

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
