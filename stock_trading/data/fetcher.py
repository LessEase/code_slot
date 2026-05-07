"""Data fetching layer: AKShare (A-shares) + yfinance (US stocks).

Data is cached locally as Parquet files to avoid repeated network calls.
Cache validity is controlled by config.data.cache_ttl_hours.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, key: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    safe = key.replace("/", "_").replace(".", "_")
    return Path(cache_dir) / f"{safe}.parquet"


def _is_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_hours * 3600


def _save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow")


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# A-share helpers (AKShare)
# ---------------------------------------------------------------------------

def get_a_share_stock_list(index_code: str = "000300", limit: int = 100) -> List[str]:
    """Return constituent stock codes for the given A-share index."""
    import akshare as ak

    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
        codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
        return codes[:limit]
    except Exception as e:
        log.warning(f"Failed to fetch A-share index {index_code}: {e}, falling back to empty list")
        return []


def fetch_a_share_ohlcv(
    symbol: str,
    lookback_days: int,
    cache_dir: str,
    ttl_hours: float,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a single A-share stock (eastmoney via AKShare)."""
    import akshare as ak

    path = _cache_path(cache_dir, f"a_{symbol}")
    if _is_fresh(path, ttl_hours):
        return _load(path)

    end = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - timedelta(days=lookback_days + 30)).strftime("%Y%m%d")

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",  # 前复权
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df.tail(lookback_days)

        _save(df, path)
        return df
    except Exception as e:
        log.debug(f"A-share fetch failed for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# US-stock helpers (yfinance)
# ---------------------------------------------------------------------------

def fetch_us_ohlcv(
    symbol: str,
    lookback_days: int,
    cache_dir: str,
    ttl_hours: float,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a US stock via yfinance."""
    path = _cache_path(cache_dir, f"us_{symbol}")
    if _is_fresh(path, ttl_hours):
        return _load(path)

    start = (datetime.today() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, auto_adjust=True)
        if df is None or df.empty:
            return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.tail(lookback_days)

        _save(df, path)
        return df
    except Exception as e:
        log.debug(f"US fetch failed for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DataFetcher:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.uni_cfg = cfg["universe"]

    def get_universe(self) -> Dict[str, str]:
        """Return {symbol: market} for the full scanning universe."""
        universe: Dict[str, str] = {}

        # A-shares
        a_codes = get_a_share_stock_list(
            index_code=self.uni_cfg.get("a_share_index", "000300"),
            limit=self.uni_cfg.get("a_share_limit", 100),
        )
        for code in a_codes:
            universe[code] = "A"

        # US stocks
        if self.uni_cfg.get("us_market", True):
            us_list = self.uni_cfg.get("us_stock_list") or []
            for sym in us_list:
                universe[sym] = "US"

        log.info(f"Universe: {len(universe)} stocks "
                 f"(A={sum(1 for v in universe.values() if v=='A')}, "
                 f"US={sum(1 for v in universe.values() if v=='US')})")
        return universe

    def fetch(self, symbol: str, market: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single symbol; returns None on failure."""
        lookback = self.data_cfg["lookback_days"]
        cache_dir = self.data_cfg["cache_dir"]
        ttl = self.data_cfg["cache_ttl_hours"]

        if market == "A":
            return fetch_a_share_ohlcv(symbol, lookback, cache_dir, ttl)
        elif market == "US":
            return fetch_us_ohlcv(symbol, lookback, cache_dir, ttl)
        return None

    def fetch_all(self, universe: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for all symbols; skips failures silently."""
        results: Dict[str, pd.DataFrame] = {}
        total = len(universe)
        for i, (sym, mkt) in enumerate(universe.items(), 1):
            df = self.fetch(sym, mkt)
            if df is not None and len(df) >= 60:
                results[sym] = df
            if i % 20 == 0:
                log.info(f"Fetched {i}/{total} stocks ({len(results)} valid)")
        log.info(f"Fetch complete: {len(results)}/{total} stocks with sufficient data")
        return results
