"""APScheduler-based daily job scheduler.

Runs the trading cycle automatically according to cron expressions
defined in config.yaml. Runs A-share and US-stock cycles at different times.
"""

from datetime import date

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from stock_trading.trading.simulator import TradingSimulator
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


def run_a_share_cycle(simulator: TradingSimulator) -> None:
    log.info("Scheduled A-share cycle triggered")
    simulator.run_daily_cycle(today=date.today())


def run_us_stock_cycle(simulator: TradingSimulator) -> None:
    log.info("Scheduled US-stock cycle triggered")
    simulator.run_daily_cycle(today=date.today())


def start_scheduler(cfg: dict) -> None:
    """Start the blocking scheduler. Blocks until interrupted."""
    simulator = TradingSimulator(cfg)
    scheduler_cfg = cfg["scheduler"]
    tz = scheduler_cfg.get("timezone", "Asia/Shanghai")

    scheduler = BlockingScheduler(timezone=tz)

    # A-share job (weekdays 16:00 CST)
    scheduler.add_job(
        run_a_share_cycle,
        CronTrigger.from_crontab(scheduler_cfg["a_share_cron"], timezone=tz),
        args=[simulator],
        id="a_share_daily",
        name="A-share daily cycle",
        misfire_grace_time=300,
    )

    # US stock job (weekdays 22:00 CST = ~09:00 EST)
    scheduler.add_job(
        run_us_stock_cycle,
        CronTrigger.from_crontab(scheduler_cfg["us_stock_cron"], timezone=tz),
        args=[simulator],
        id="us_stock_daily",
        name="US-stock daily cycle",
        misfire_grace_time=300,
    )

    log.info(
        f"Scheduler started (TZ={tz})\n"
        f"  A-share  → {scheduler_cfg['a_share_cron']}\n"
        f"  US-stock → {scheduler_cfg['us_stock_cron']}"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")
