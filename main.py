"""Entry point for the stock trading automation system.

Trading commands
----------------
  python main.py                  # Start the daily scheduler (daemon mode)
  python main.py --run-now        # Run one full cycle immediately and exit
  python main.py --scan-only      # Print ML-scored stocks without trading
  python main.py --market A       # Filter to A-shares (with --scan-only)
  python main.py --market US      # Filter to US stocks (with --scan-only)
  python main.py --summary        # Print portfolio summary and exit
  python main.py --history        # Print trade history and exit

ML pipeline commands
--------------------
  python main.py --pipeline       # Run full ML pipeline (all 5 steps)
  python main.py --collect        # Step 1: Download historical data
  python main.py --features       # Step 2: Build feature matrices
  python main.py --samples        # Step 3: Generate training samples
  python main.py --train          # Step 4: Train LightGBM models
  python main.py --backtest       # Step 5: Run strategy backtest
  python main.py --backtest --market US   # Backtest US market only
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich import box

from stock_trading.trading.simulator import TradingSimulator
from stock_trading.scheduler.job import start_scheduler
from stock_trading.portfolio.portfolio import Portfolio
from stock_trading.utils.logger import get_logger
from ml_pipeline.pipeline import (
    step_collect, step_features, step_samples,
    step_train, step_backtest, run_full_pipeline,
)

log = get_logger("main")
console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_run_now(cfg: dict) -> None:
    sim = TradingSimulator(cfg)
    sim.run_daily_cycle(today=date.today())


def cmd_scan_only(cfg: dict, market: str = None) -> None:
    sim = TradingSimulator(cfg)
    sim.print_top_scores(market_filter=market, top=20)


def cmd_summary(cfg: dict) -> None:
    portfolio = Portfolio(cfg)
    summary = portfolio.summary({})
    console.rule("[bold cyan]Portfolio Summary")
    console.print(f"""
  Initial Capital:  ¥{summary['initial_capital']:>15,.0f}
  Current Cash:     ¥{summary['cash']:>15,.0f}
  Total Equity:     ¥{summary['equity']:>15,.0f}
  Total Return:     [{'green' if summary['total_return_pct'] >= 0 else 'red'}]{summary['total_return_pct']:>+.2f}%[/]
  Realized P&L:     [{'green' if summary['total_realized_pnl'] >= 0 else 'red'}]¥{summary['total_realized_pnl']:>+,.0f}[/]
  Open Positions:   {summary['open_positions']}
  Total Trades:     {summary['total_trades']}
  Win Rate:         {summary['win_rate_pct']:.1f}%
""")


def cmd_history(cfg: dict) -> None:
    portfolio = Portfolio(cfg)
    df = portfolio.get_trade_history()
    if df.empty:
        console.print("[yellow]No trade history yet.[/]")
        return

    table = Table(title="Trade History", box=box.ROUNDED)
    for col in ["date", "symbol", "market", "action", "shares", "price", "commission", "pnl", "reason"]:
        table.add_column(col.capitalize(), justify="right" if col in {"shares", "price", "commission", "pnl"} else "left")

    for _, row in df.iterrows():
        pnl_str = f"{row['pnl']:+,.0f}" if row["pnl"] is not None and not __import__("math").isnan(row["pnl"] or float("nan")) else "-"
        color = "green" if (row["pnl"] or 0) >= 0 else "red"
        table.add_row(
            str(row["date"]),
            row["symbol"],
            row["market"] or "-",
            f"[{'green' if row['action'] == 'BUY' else 'red'}]{row['action']}[/]",
            f"{row['shares']:.0f}",
            f"¥{row['price']:,.2f}",
            f"¥{row['commission']:,.2f}",
            f"[{color}]¥{pnl_str}[/]",
            row["reason"] or "-",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock Trading Automation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    # ── Trading commands ───────────────────────────────────────────────────
    parser.add_argument("--run-now", action="store_true", help="Run one full trading cycle")
    parser.add_argument("--scan-only", action="store_true", help="Show ML scores without trading")
    parser.add_argument("--market", choices=["A", "US"], help="Market filter")
    parser.add_argument("--summary", action="store_true", help="Print portfolio summary")
    parser.add_argument("--history", action="store_true", help="Print trade history")

    # ── ML pipeline commands ───────────────────────────────────────────────
    parser.add_argument("--pipeline", action="store_true", help="Run full ML pipeline (all steps)")
    parser.add_argument("--collect", action="store_true", help="Step 1: Download historical data")
    parser.add_argument("--features", action="store_true", help="Step 2: Build feature matrices")
    parser.add_argument("--samples", action="store_true", help="Step 3: Generate training samples")
    parser.add_argument("--train", action="store_true", help="Step 4: Train LightGBM models")
    parser.add_argument("--backtest", action="store_true", help="Step 5: Run strategy backtest")

    args = parser.parse_args()

    if not Path(args.config).exists():
        log.error(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)

    # ── ML pipeline steps ──────────────────────────────────────────────────
    if args.pipeline:
        run_full_pipeline(cfg)
    elif args.collect:
        step_collect(cfg)
    elif args.features:
        step_features(cfg)
    elif args.samples:
        step_samples(cfg)
    elif args.train:
        step_train(cfg)
    elif args.backtest:
        step_backtest(cfg, market=args.market)

    # ── Trading commands ───────────────────────────────────────────────────
    elif args.summary:
        cmd_summary(cfg)
    elif args.history:
        cmd_history(cfg)
    elif args.run_now:
        cmd_run_now(cfg)
    elif args.scan_only:
        cmd_scan_only(cfg, market=args.market)
    else:
        # Default: start the scheduler daemon
        start_scheduler(cfg)


if __name__ == "__main__":
    main()
