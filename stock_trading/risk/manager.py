"""Risk management: stop-loss, trailing stop, and timeout exit rules.

Checks each open position against the current price and returns a list
of (symbol, reason) pairs that should be sold.
"""

from datetime import date
from typing import Dict, List, Tuple

from stock_trading.portfolio.portfolio import Portfolio, Position
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


class RiskManager:
    def __init__(self, cfg: dict):
        self.stop_loss_pct = cfg["risk"]["stop_loss_pct"]
        self.trailing_stop_pct = cfg["risk"]["trailing_stop_pct"]
        self.max_hold_days = cfg["risk"]["max_hold_days"]
        self.max_portfolio_drawdown = cfg["risk"]["max_portfolio_drawdown"]

    def check_positions(
        self,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
        today: date,
    ) -> List[Tuple[str, str]]:
        """
        Evaluate all open positions and return (symbol, reason) pairs to sell.

        Priority: stop_loss > trailing_stop > timeout
        """
        to_sell: List[Tuple[str, str]] = []

        positions = portfolio.get_positions()
        for pos in positions:
            price = current_prices.get(pos.symbol)
            if price is None:
                log.debug(f"No price data for {pos.symbol}, skipping risk check")
                continue

            # Convert US price to CNY for comparison
            price_cny = price * portfolio.usd_cny if pos.market == "US" else price

            # Update peak price first
            portfolio.update_peak_price(pos.symbol, price)

            # Re-fetch position to get updated peak_price
            updated_pos = portfolio.get_position(pos.symbol)
            if updated_pos is None:
                continue

            reason = self._evaluate(updated_pos, price_cny, today)
            if reason:
                to_sell.append((pos.symbol, reason))

        return to_sell

    def _evaluate(self, pos: Position, current_price_cny: float, today: date) -> str:
        """Return reason string if position should be sold, else empty string."""
        entry = pos.entry_price
        peak = pos.peak_price

        # ── Fixed stop-loss ────────────────────────────────────────────────
        loss_pct = (current_price_cny - entry) / entry
        if loss_pct <= -self.stop_loss_pct:
            log.warning(
                f"STOP-LOSS triggered: {pos.symbol}  "
                f"entry=¥{entry:.2f}  current=¥{current_price_cny:.2f}  "
                f"loss={loss_pct:.1%}"
            )
            return "stop_loss"

        # ── Trailing stop (drawdown from peak) ─────────────────────────────
        drawdown_from_peak = (current_price_cny - peak) / peak
        if drawdown_from_peak <= -self.trailing_stop_pct:
            log.warning(
                f"TRAILING-STOP triggered: {pos.symbol}  "
                f"peak=¥{peak:.2f}  current=¥{current_price_cny:.2f}  "
                f"drawdown={drawdown_from_peak:.1%}"
            )
            return "trailing_stop"

        # ── Max hold days ──────────────────────────────────────────────────
        hold_days = (today - pos.entry_date).days
        if hold_days >= self.max_hold_days:
            log.info(
                f"TIMEOUT: {pos.symbol} held for {hold_days} days "
                f"(max={self.max_hold_days})"
            )
            return "timeout"

        return ""

    def check_portfolio_drawdown(
        self,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
    ) -> bool:
        """
        Return True if overall portfolio drawdown exceeds the configured limit.
        When True, the scheduler should pause new position entries.
        """
        equity = portfolio.total_equity(current_prices)
        initial = portfolio.initial_capital
        drawdown = (equity - initial) / initial
        if drawdown <= -self.max_portfolio_drawdown:
            log.warning(
                f"Portfolio drawdown {drawdown:.1%} exceeds limit "
                f"{-self.max_portfolio_drawdown:.1%}. Halting new entries."
            )
            return True
        return False
