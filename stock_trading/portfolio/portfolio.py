"""Portfolio management with SQLite persistence.

Tracks cash, open positions, closed trades, and running P&L.
All monetary values are stored in CNY (USD positions are converted).
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import (Column, Float, Integer, String, Date, DateTime,
                        Text, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True)
    market = Column(String(4), nullable=False)   # "A" or "US"
    shares = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)   # CNY-equivalent
    entry_date = Column(Date, nullable=False)
    peak_price = Column(Float, nullable=False)    # highest close since entry
    cost_basis = Column(Float, nullable=False)    # total cost including commission


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    market = Column(String(4), nullable=False)
    action = Column(String(4), nullable=False)   # "BUY" or "SELL"
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)         # execution price (CNY-equivalent)
    commission = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)            # realized P&L for SELL trades
    reason = Column(String(50), nullable=True)    # "signal" | "stop_loss" | "trailing" | "timeout"
    trade_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    extra = Column(Text, nullable=True)           # JSON blob for extra info


class PortfolioState(Base):
    __tablename__ = "portfolio_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Portfolio class
# ---------------------------------------------------------------------------

class Portfolio:
    DB_PATH = "data/portfolio.db"

    def __init__(self, cfg: dict):
        self.cfg = cfg["portfolio"]
        self.usd_cny = cfg["portfolio"]["usd_cny_rate"]
        self.commission_rate = self.cfg["commission_rate"]
        self.us_commission_per_share = self.cfg["us_commission_per_share"]
        self.min_commission = self.cfg["min_commission"]

        db_path = Path(self.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self._init_cash(self.cfg["initial_capital"])

    # ── Persistence helpers ──────────────────────────────────────────────

    def _get_state(self, key: str, default=None):
        with self.Session() as s:
            obj = s.query(PortfolioState).filter_by(key=key).first()
            if obj is None:
                return default
            return json.loads(obj.value)

    def _set_state(self, key: str, value) -> None:
        with self.Session() as s:
            obj = s.query(PortfolioState).filter_by(key=key).first()
            if obj is None:
                obj = PortfolioState(key=key, value=json.dumps(value))
                s.add(obj)
            else:
                obj.value = json.dumps(value)
            s.commit()

    def _init_cash(self, initial: float) -> None:
        if self._get_state("cash") is None:
            self._set_state("cash", initial)
            log.info(f"Portfolio initialized with ¥{initial:,.0f}")

    # ── Cash management ──────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._get_state("cash", 0.0)

    @cash.setter
    def cash(self, value: float) -> None:
        self._set_state("cash", value)

    @property
    def initial_capital(self) -> float:
        return self.cfg["initial_capital"]

    # ── Position management ───────────────────────────────────────────────

    def get_positions(self) -> List[Position]:
        with self.Session() as s:
            return s.query(Position).all()

    def get_position(self, symbol: str) -> Optional[Position]:
        with self.Session() as s:
            return s.query(Position).filter_by(symbol=symbol).first()

    def held_symbols(self) -> set:
        with self.Session() as s:
            return {p.symbol for p in s.query(Position).all()}

    # ── Commission calculation ────────────────────────────────────────────

    def _calc_commission(self, market: str, shares: float, price: float) -> float:
        if market == "A":
            commission = shares * price * self.commission_rate
            return max(commission, 5.0)   # A-share minimum ¥5
        else:
            # US stock: per-share commission, converted to CNY
            commission_usd = max(shares * self.us_commission_per_share, self.min_commission)
            return commission_usd * self.usd_cny

    def _price_cny(self, market: str, price: float) -> float:
        return price * self.usd_cny if market == "US" else price

    # ── Buy / Sell ────────────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        market: str,
        price: float,
        trade_date: date,
        score: float = 0.0,
        extra: dict = None,
    ) -> bool:
        """
        Execute a simulated buy order.

        Position size = position_size_pct × total_equity, capped by cash.
        Returns True on success.
        """
        if symbol in self.held_symbols():
            log.debug(f"Already holding {symbol}, skipping buy")
            return False

        positions = self.get_positions()
        if len(positions) >= self.cfg["max_positions"]:
            log.warning(f"Max positions ({self.cfg['max_positions']}) reached, cannot buy {symbol}")
            return False

        equity = self.total_equity({})  # approximate without current prices
        target_value = equity * self.cfg["position_size_pct"]
        available = min(target_value, self.cash * 0.95)  # keep 5% buffer

        price_cny = self._price_cny(market, price)
        shares = available / price_cny
        if market == "A":
            shares = max(int(shares / 100) * 100, 100)   # A-share: round to lot of 100
        else:
            shares = max(int(shares), 1)

        cost = shares * price_cny
        commission = self._calc_commission(market, shares, price_cny)
        total_cost = cost + commission

        if total_cost > self.cash:
            log.warning(f"Insufficient cash for {symbol}: need ¥{total_cost:,.0f}, have ¥{self.cash:,.0f}")
            return False

        with self.Session() as s:
            pos = Position(
                symbol=symbol,
                market=market,
                shares=shares,
                entry_price=price_cny,
                entry_date=trade_date,
                peak_price=price_cny,
                cost_basis=total_cost,
            )
            trade = Trade(
                symbol=symbol,
                market=market,
                action="BUY",
                shares=shares,
                price=price_cny,
                commission=commission,
                trade_date=trade_date,
                extra=json.dumps({"score": score, **(extra or {})}),
            )
            s.add(pos)
            s.add(trade)
            s.commit()

        self.cash -= total_cost
        log.info(
            f"BUY  {symbol:8s} ({market}) "
            f"{shares:>8.0f} shares @ ¥{price_cny:>10.2f}  "
            f"cost=¥{total_cost:>12,.0f}  cash_left=¥{self.cash:>12,.0f}"
        )
        return True

    def sell(
        self,
        symbol: str,
        price: float,
        trade_date: date,
        reason: str = "signal",
    ) -> bool:
        """
        Execute a simulated sell order for a held position.
        Returns True on success.
        """
        with self.Session() as s:
            pos = s.query(Position).filter_by(symbol=symbol).first()
            if pos is None:
                log.warning(f"Cannot sell {symbol}: no position found")
                return False

            price_cny = self._price_cny(pos.market, price)
            proceeds = pos.shares * price_cny
            commission = self._calc_commission(pos.market, pos.shares, price_cny)
            net_proceeds = proceeds - commission
            pnl = net_proceeds - pos.cost_basis

            trade = Trade(
                symbol=symbol,
                market=pos.market,
                action="SELL",
                shares=pos.shares,
                price=price_cny,
                commission=commission,
                pnl=pnl,
                reason=reason,
                trade_date=trade_date,
            )
            s.add(trade)
            s.delete(pos)
            s.commit()

        self.cash += net_proceeds
        pnl_pct = pnl / (net_proceeds - pnl) * 100 if (net_proceeds - pnl) != 0 else 0
        log.info(
            f"SELL {symbol:8s} ({reason:10s}) "
            f"@ ¥{price_cny:>10.2f}  "
            f"PnL=¥{pnl:>+10,.0f} ({pnl_pct:+.1f}%)  "
            f"cash=¥{self.cash:>12,.0f}"
        )
        return True

    def update_peak_price(self, symbol: str, current_price: float) -> None:
        """Update peak_price for trailing stop tracking."""
        with self.Session() as s:
            pos = s.query(Position).filter_by(symbol=symbol).first()
            if pos is None:
                return
            market = pos.market
            price_cny = self._price_cny(market, current_price)
            if price_cny > pos.peak_price:
                pos.peak_price = price_cny
                s.commit()

    # ── Reporting ─────────────────────────────────────────────────────────

    def total_equity(self, current_prices: Dict[str, float]) -> float:
        """Compute total equity = cash + market value of all positions."""
        equity = self.cash
        with self.Session() as s:
            for pos in s.query(Position).all():
                price = current_prices.get(pos.symbol, pos.entry_price)
                price_cny = self._price_cny(pos.market, price)
                equity += pos.shares * price_cny
        return equity

    def summary(self, current_prices: Dict[str, float]) -> dict:
        equity = self.total_equity(current_prices)
        initial = self.initial_capital
        total_return = (equity - initial) / initial * 100

        with self.Session() as s:
            positions = s.query(Position).all()
            trades = s.query(Trade).filter_by(action="SELL").all()

        wins = [t for t in trades if (t.pnl or 0) > 0]
        losses = [t for t in trades if (t.pnl or 0) <= 0]
        total_pnl = sum(t.pnl or 0 for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        return {
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "initial_capital": initial,
            "total_return_pct": round(total_return, 2),
            "total_realized_pnl": round(total_pnl, 2),
            "open_positions": len(positions),
            "total_trades": len(trades),
            "win_rate_pct": round(win_rate, 1),
        }

    def get_trade_history(self) -> pd.DataFrame:
        with self.Session() as s:
            trades = s.query(Trade).order_by(Trade.trade_date).all()
            return pd.DataFrame([{
                "date": t.trade_date,
                "symbol": t.symbol,
                "market": t.market,
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "commission": t.commission,
                "pnl": t.pnl,
                "reason": t.reason,
            } for t in trades])
