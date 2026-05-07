"""Paper-trading simulator.

Orchestrates one full daily cycle:
  1. Fetch latest prices for held positions → update peaks, check risk rules.
  2. Execute sell orders for triggered positions.
  3. Scan full universe, compute scores.
  4. Buy top-N candidates (if portfolio drawdown limit not breached).
  5. Print a rich summary table.
"""

from datetime import date
from typing import Dict

from rich.console import Console
from rich.table import Table
from rich import box

from stock_trading.data.fetcher import DataFetcher
from stock_trading.features.indicators import compute_indicators
from stock_trading.models.scorer import StockScorer
from stock_trading.portfolio.portfolio import Portfolio
from stock_trading.risk.manager import RiskManager
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)
console = Console()


class TradingSimulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fetcher = DataFetcher(cfg)
        self.scorer = StockScorer(cfg)
        self.portfolio = Portfolio(cfg)
        self.risk = RiskManager(cfg)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_latest_price(self, df) -> float:
        return float(df["close"].iloc[-1])

    def _enrich(self, raw_data: Dict) -> Dict:
        """Compute indicators for all fetched stocks; drop failures."""
        enriched = {}
        for sym, df in raw_data.items():
            try:
                enriched[sym] = compute_indicators(df, self.cfg["indicators"])
            except Exception as e:
                log.debug(f"Indicator computation failed for {sym}: {e}")
        return enriched

    # ── Daily cycle ──────────────────────────────────────────────────────────

    def run_daily_cycle(self, today: date = None) -> None:
        today = today or date.today()
        log.info(f"═══ Daily cycle starting: {today} ═══")

        # ── Step 1: Get universe & fetch data ──────────────────────────────
        universe = self.fetcher.get_universe()
        raw_data = self.fetcher.fetch_all(universe)
        market_map = {sym: mkt for sym, mkt in universe.items() if sym in raw_data}

        # Current prices (latest close for each symbol)
        current_prices = {sym: self._get_latest_price(df) for sym, df in raw_data.items()}

        # ── Step 2: Risk checks → sell triggered positions ─────────────────
        held = self.portfolio.held_symbols()
        log.info(f"Open positions: {len(held)}")

        to_sell = self.risk.check_positions(self.portfolio, current_prices, today)
        for symbol, reason in to_sell:
            price = current_prices.get(symbol)
            if price is not None:
                self.portfolio.sell(symbol, price, today, reason=reason)

        # ── Step 3: Compute indicators ─────────────────────────────────────
        enriched = self._enrich(raw_data)
        log.info(f"Enriched {len(enriched)} stocks with indicators")

        # ── Step 4: Check portfolio drawdown before buying ─────────────────
        halt_new_entries = self.risk.check_portfolio_drawdown(self.portfolio, current_prices)

        if not halt_new_entries:
            # ── Step 5: Score & select top-N ──────────────────────────────
            held_now = self.portfolio.held_symbols()
            top_stocks = self.scorer.select_top_n(enriched, exclude=held_now)
            log.info(f"Top-{self.cfg['scoring']['top_n']} candidates selected: "
                     f"{[s for s, _, _ in top_stocks]}")

            # ── Step 6: Buy ────────────────────────────────────────────────
            for symbol, score, info in top_stocks:
                price = current_prices.get(symbol)
                mkt = market_map.get(symbol, "A")
                if price is not None:
                    self.portfolio.buy(
                        symbol=symbol,
                        market=mkt,
                        price=price,
                        trade_date=today,
                        score=score,
                        extra=info,
                    )
        else:
            log.warning("New entries halted due to portfolio drawdown limit")

        # ── Step 7: Print summary ──────────────────────────────────────────
        self._print_summary(current_prices, enriched)

    # ── Reporting ────────────────────────────────────────────────────────────

    def _print_summary(self, current_prices: Dict, enriched: Dict) -> None:
        summary = self.portfolio.summary(current_prices)

        console.rule("[bold cyan]Portfolio Summary")
        console.print(
            f"  Cash:       [green]¥{summary['cash']:>15,.0f}[/]\n"
            f"  Equity:     [green]¥{summary['equity']:>15,.0f}[/]\n"
            f"  Return:     [{'green' if summary['total_return_pct'] >= 0 else 'red'}]"
            f"{summary['total_return_pct']:>+.2f}%[/]\n"
            f"  Realized P&L: [{'green' if summary['total_realized_pnl'] >= 0 else 'red'}]"
            f"¥{summary['total_realized_pnl']:>+,.0f}[/]\n"
            f"  Open Positions: {summary['open_positions']}\n"
            f"  Total Trades:   {summary['total_trades']}  "
            f"Win Rate: {summary['win_rate_pct']:.1f}%"
        )

        # Positions table
        positions = self.portfolio.get_positions()
        if positions:
            table = Table(title="Open Positions", box=box.ROUNDED)
            table.add_column("Symbol", style="cyan")
            table.add_column("Market")
            table.add_column("Shares", justify="right")
            table.add_column("Entry ¥", justify="right")
            table.add_column("Current ¥", justify="right")
            table.add_column("PnL%", justify="right")
            table.add_column("RSI", justify="right")
            table.add_column("Score", justify="right")

            for pos in positions:
                curr = current_prices.get(pos.symbol, pos.entry_price)
                curr_cny = curr * self.portfolio.usd_cny if pos.market == "US" else curr
                pnl_pct = (curr_cny - pos.entry_price) / pos.entry_price * 100
                color = "green" if pnl_pct >= 0 else "red"

                df = enriched.get(pos.symbol)
                rsi = f"{df.iloc[-1]['rsi']:.0f}" if df is not None and "rsi" in df.columns else "-"
                score = f"{self.scorer.score_stock(df):.0f}" if df is not None else "-"

                table.add_row(
                    pos.symbol,
                    pos.market,
                    f"{pos.shares:.0f}",
                    f"¥{pos.entry_price:,.2f}",
                    f"¥{curr_cny:,.2f}",
                    f"[{color}]{pnl_pct:+.1f}%[/]",
                    rsi,
                    score,
                )
            console.print(table)

    def print_top_scores(self, market_filter: str = None, top: int = 20) -> None:
        """Print top-scored stocks without executing any trades (dry-run view)."""
        universe = self.fetcher.get_universe()
        if market_filter:
            universe = {s: m for s, m in universe.items() if m == market_filter}

        raw_data = self.fetcher.fetch_all(universe)
        enriched = self._enrich(raw_data)
        ranked = self.scorer.rank_stocks(enriched)[:top]

        table = Table(title=f"Top {top} Stocks by Score", box=box.ROUNDED)
        table.add_column("#", justify="right")
        table.add_column("Symbol", style="cyan")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("Trend", justify="right")
        table.add_column("Momentum", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Volatility", justify="right")
        table.add_column("Close", justify="right")
        table.add_column("RSI", justify="right")

        for rank, (sym, score, info) in enumerate(ranked, 1):
            table.add_row(
                str(rank),
                sym,
                f"{score:.1f}",
                f"{info.get('trend', '-'):.1f}" if isinstance(info.get('trend'), float) else "-",
                f"{info.get('momentum', '-'):.1f}" if isinstance(info.get('momentum'), float) else "-",
                f"{info.get('volume', '-'):.1f}" if isinstance(info.get('volume'), float) else "-",
                f"{info.get('volatility', '-'):.1f}" if isinstance(info.get('volatility'), float) else "-",
                f"{info.get('close', 0):.2f}",
                f"{info.get('rsi', float('nan')):.0f}" if not __import__('math').isnan(info.get('rsi', float('nan'))) else "-",
            )
        console.print(table)
