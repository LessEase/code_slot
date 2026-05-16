"""Historical data collection for ML pipeline.

Downloads multi-year OHLCV history for the full stock universe plus
market index data (沪深300, S&P500). Results are stored as Parquet files
under data/history/{A,US,INDEX}/.

This module is run once (or periodically refreshed) before feature
engineering begins.  It is intentionally separate from the live-data
fetcher used by the trading simulator so that backtest data can be
accumulated independently.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from stock_trading.data.fetcher import (
    _akshare_call_with_retry,
    fetch_a_index_history_sina,
    fetch_a_share_history_sina,
    get_a_share_universe,
    get_us_universe,
    resolve_a_share_data_source,
)
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

# ── Index symbols ──────────────────────────────────────────────────────────
A_INDEX_SYMBOL = "000300"   # 沪深300
US_INDEX_SYMBOL = "^GSPC"   # S&P 500


# ── Storage helpers ────────────────────────────────────────────────────────

def _history_path(base_dir: str, market: str, symbol: str) -> Path:
    p = Path(base_dir) / market
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{symbol.replace('/', '_')}.parquet"


def _save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow")


def _load(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


# ── A-share history ────────────────────────────────────────────────────────

def _fetch_a_share_history_eastmoney(
    symbol: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    import akshare as ak

    df = _akshare_call_with_retry(
        f"stock_zh_a_hist({symbol})",
        ak.stock_zh_a_hist,
        symbol=symbol, period="daily",
        start_date=start.replace("-", ""), end_date=end.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    out = df[keep].astype(float)
    if "amount" not in out.columns:
        out["amount"] = out["volume"] * out["close"]
    return out


def fetch_a_share_history(
    symbol: str,
    start: str,
    end: str,
    source: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a single A-share (前复权).

    `source` is one of {"eastmoney", "sina"}; when None it's pulled from the
    cached resolver verdict (see resolve_a_share_data_source).
    """
    if source is None:
        # Resolver was already called by the collector; this picks up the cache.
        source = resolve_a_share_data_source({})
    if source == "sina":
        return fetch_a_share_history_sina(symbol, start, end)
    return _fetch_a_share_history_eastmoney(symbol, start, end)


def fetch_a_index_history(
    start: str,
    end: str,
    source: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch 沪深300 index daily OHLCV; routes to Eastmoney or Sina."""
    if source is None:
        source = resolve_a_share_data_source({})
    if source == "sina":
        return fetch_a_index_history_sina(start, end, index_code=A_INDEX_SYMBOL)

    import akshare as ak

    df = _akshare_call_with_retry(
        f"index_zh_a_hist({A_INDEX_SYMBOL})",
        ak.index_zh_a_hist,
        symbol=A_INDEX_SYMBOL, period="daily",
        start_date=start.replace("-", ""), end_date=end.replace("-", ""),
    )
    if df is None or df.empty:
        return None
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ── US-stock history ───────────────────────────────────────────────────────

def fetch_us_history(
    symbol: str,
    start: str,
    end: str,
    attempts: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a US stock via yfinance, with light retries.

    Yahoo intermittently returns empty frames or HTTP 429 under bulk load;
    a couple of jittered retries recover most of those.
    """
    for i in range(1, attempts + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.rename(columns=str.lower)[
                    ["open", "high", "low", "close", "volume"]
                ]
                return df.sort_index()
        except Exception as e:
            log.debug(f"US history attempt {i}/{attempts} failed for {symbol}: {e}")
        if i < attempts:
            time.sleep(1.5 * i + random.random())
    return None


def fetch_us_index_history(start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch S&P 500 daily OHLCV."""
    return fetch_us_history(US_INDEX_SYMBOL, start, end)


# ── Main collector class ───────────────────────────────────────────────────

class HistoricalCollector:
    """Downloads and caches multi-year OHLCV history for the full universe."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_dir = cfg["ml_pipeline"]["history_dir"]
        self.years = cfg["ml_pipeline"]["history_years"]
        self.uni_cfg = cfg["universe"]

    def _date_range(self) -> Tuple[str, str]:
        end = datetime.today()
        start = end - timedelta(days=int(self.years * 365.25) + 60)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # ── Universe helpers ───────────────────────────────────────────────────

    def _a_share_symbols(self) -> List[str]:
        return get_a_share_universe(self.uni_cfg)

    def _us_symbols(self) -> List[str]:
        return get_us_universe(self.uni_cfg)

    # ── Collection methods ─────────────────────────────────────────────────

    def collect_indices(self) -> None:
        """Download index data (沪深300 and S&P500)."""
        start, end = self._date_range()
        source = resolve_a_share_data_source(self.uni_cfg)

        log.info("Collecting 沪深300 index history …")
        df = fetch_a_index_history(start, end, source=source)
        if df is not None:
            _save(df, _history_path(self.base_dir, "INDEX", "CSI300"))
            log.info(f"  CSI300: {len(df)} rows  ({df.index[0].date()} – {df.index[-1].date()})")
        else:
            log.warning("  CSI300: fetch failed")

        log.info("Collecting S&P 500 index history …")
        df = fetch_us_index_history(start, end)
        if df is not None:
            _save(df, _history_path(self.base_dir, "INDEX", "SP500"))
            log.info(f"  SP500:  {len(df)} rows  ({df.index[0].date()} – {df.index[-1].date()})")
        else:
            log.warning("  SP500: fetch failed")

    def collect_a_shares(self) -> Dict[str, int]:
        """Download A-share history (parallel); returns {symbol: row_count}."""
        symbols = self._a_share_symbols()
        start, end = self._date_range()
        workers = max(1, int(self.uni_cfg.get("fetch_workers", 8)))
        source = resolve_a_share_data_source(self.uni_cfg)
        log.info(
            f"Collecting A-share history for {len(symbols)} stocks "
            f"({start} → {end}, workers={workers}, source={source}) …"
        )

        # Separate already-fresh files (today's) from those that need fetching.
        to_fetch: List[str] = []
        results: Dict[str, int] = {}
        today = datetime.today().date()
        for sym in symbols:
            path = _history_path(self.base_dir, "A", sym)
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
                if mtime >= today:
                    existing = _load(path)
                    results[sym] = len(existing) if existing is not None else 0
                    continue
            to_fetch.append(sym)

        if not to_fetch:
            log.info(f"A-share collection: all {len(symbols)} symbols already fresh today")
            return results

        log.info(f"  {len(results)} cached, {len(to_fetch)} to download")

        # Per-worker pacing: small randomized sleep before each request so the
        # N workers don't fire in perfectly synchronized bursts (which is what
        # makes Eastmoney RST connections en masse). 150–400ms per worker keeps
        # the aggregate rate well under the throttling threshold.
        pace_min = float(self.uni_cfg.get("fetch_pace_min_sec", 0.15))
        pace_max = float(self.uni_cfg.get("fetch_pace_max_sec", 0.40))

        def _job(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            time.sleep(pace_min + random.random() * max(0.0, pace_max - pace_min))
            return sym, fetch_a_share_history(sym, start, end, source=source)

        failed: List[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_job, s): s for s in to_fetch}
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    _, df = fut.result()
                except Exception as e:
                    log.debug(f"A-share fetch raised for {sym}: {e}")
                    df = None
                if df is not None and len(df) >= 120:
                    try:
                        _save(df, _history_path(self.base_dir, "A", sym))
                        results[sym] = len(df)
                    except Exception as e:
                        # A bad single DataFrame (duplicate columns, unsupported
                        # dtype, etc.) must not kill the whole 5000-stock run.
                        log.warning(f"A-share save failed for {sym}: {e}")
                        failed.append(sym)
                else:
                    failed.append(sym)
                if done % 100 == 0 or done == len(to_fetch):
                    log.info(
                        f"  A-share: {done}/{len(to_fetch)} fetched "
                        f"({len(results)}/{len(symbols)} valid overall)"
                    )

        # Second pass: retry failures sequentially with longer spacing. Many
        # of these are healthy tickers that got RST during the parallel burst;
        # giving them a single calm shot recovers most of them.
        if failed:
            log.info(
                f"  Retrying {len(failed)} failed symbols sequentially "
                f"(this can take a while) …"
            )
            recovered = 0
            for i, sym in enumerate(failed, 1):
                time.sleep(0.5 + random.random() * 0.5)
                try:
                    df = fetch_a_share_history(sym, start, end, source=source)
                except Exception as e:
                    log.debug(f"A-share retry raised for {sym}: {e}")
                    df = None
                if df is not None and len(df) >= 120:
                    try:
                        _save(df, _history_path(self.base_dir, "A", sym))
                        results[sym] = len(df)
                        recovered += 1
                    except Exception as e:
                        log.warning(f"A-share retry save failed for {sym}: {e}")
                if i % 50 == 0 or i == len(failed):
                    log.info(
                        f"  A-share retry: {i}/{len(failed)} attempted "
                        f"({recovered} recovered)"
                    )

        log.info(f"A-share collection complete: {len(results)}/{len(symbols)} valid")
        return results

    def collect_us_stocks(self) -> Dict[str, int]:
        """Download US stock history (parallel); returns {symbol: row_count}."""
        symbols = self._us_symbols()
        start, end = self._date_range()
        workers = max(1, int(self.uni_cfg.get("us_fetch_workers", 8)))
        log.info(
            f"Collecting US stock history for {len(symbols)} stocks "
            f"(workers={workers}) …"
        )

        to_fetch: List[str] = []
        results: Dict[str, int] = {}
        today = datetime.today().date()
        for sym in symbols:
            path = _history_path(self.base_dir, "US", sym)
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
                if mtime >= today:
                    existing = _load(path)
                    results[sym] = len(existing) if existing is not None else 0
                    continue
            to_fetch.append(sym)

        if not to_fetch:
            log.info(f"US collection: all {len(symbols)} symbols already fresh today")
            return results

        log.info(f"  {len(results)} cached, {len(to_fetch)} to download")

        pace_min = float(self.uni_cfg.get("fetch_pace_min_sec", 0.15))
        pace_max = float(self.uni_cfg.get("fetch_pace_max_sec", 0.40))

        def _job(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            time.sleep(pace_min + random.random() * max(0.0, pace_max - pace_min))
            return sym, fetch_us_history(sym, start, end)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_job, s): s for s in to_fetch}
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    _, df = fut.result()
                except Exception as e:
                    log.debug(f"US fetch raised for {sym}: {e}")
                    df = None
                if df is not None and len(df) >= 120:
                    try:
                        _save(df, _history_path(self.base_dir, "US", sym))
                        results[sym] = len(df)
                    except Exception as e:
                        log.warning(f"US save failed for {sym}: {e}")
                if done % 200 == 0 or done == len(to_fetch):
                    log.info(
                        f"  US: {done}/{len(to_fetch)} fetched "
                        f"({len(results)}/{len(symbols)} valid overall)"
                    )

        log.info(f"US stock collection complete: {len(results)}/{len(symbols)} valid")
        return results

    def collect_all(self) -> None:
        """Run full collection: indices + all markets."""
        self.collect_indices()
        self.collect_a_shares()
        self.collect_us_stocks()

    # ── Data loading ───────────────────────────────────────────────────────

    def load_ohlcv(self, symbol: str, market: str) -> Optional[pd.DataFrame]:
        path = _history_path(self.base_dir, market, symbol)
        return _load(path)

    def load_index(self, market: str) -> Optional[pd.DataFrame]:
        key = "CSI300" if market == "A" else "SP500"
        path = _history_path(self.base_dir, "INDEX", key)
        return _load(path)

    def available_symbols(self, market: str) -> List[str]:
        """Return symbols that have downloaded history files."""
        market_dir = Path(self.base_dir) / market
        if not market_dir.exists():
            return []
        return [p.stem for p in market_dir.glob("*.parquet")]
