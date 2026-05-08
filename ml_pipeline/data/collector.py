"""Historical data collection for ML pipeline.

Downloads multi-year OHLCV history for the full stock universe plus
market index data (沪深300, S&P500). Results are stored as Parquet files
under data/history/{A,US,INDEX}/.

This module is run once (or periodically refreshed) before feature
engineering begins.  It is intentionally separate from the live-data
fetcher used by the trading simulator so that backtest data can be
accumulated independently.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

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

def fetch_a_share_history(
    symbol: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a single A-share (前复权) via AKShare."""
    import akshare as ak
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
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
    except Exception as e:
        log.debug(f"A-share history failed for {symbol}: {e}")
        return None


def fetch_a_index_history(
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Fetch 沪深300 index daily OHLCV via AKShare."""
    import akshare as ak
    try:
        df = ak.index_zh_a_hist(
            symbol=A_INDEX_SYMBOL,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
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
    except Exception as e:
        log.warning(f"A-share index fetch failed: {e}")
        return None


# ── US-stock history ───────────────────────────────────────────────────────

def fetch_us_history(
    symbol: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a US stock via yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        return df.sort_index()
    except Exception as e:
        log.debug(f"US history failed for {symbol}: {e}")
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
        import akshare as ak
        index_code = self.uni_cfg.get("a_share_index", "000300")
        limit = self.uni_cfg.get("a_share_limit", 100)
        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
            return df["成分券代码"].astype(str).str.zfill(6).tolist()[:limit]
        except Exception as e:
            log.warning(f"Could not fetch A-share universe: {e}")
            return []

    def _us_symbols(self) -> List[str]:
        return self.uni_cfg.get("us_stock_list") or []

    # ── Collection methods ─────────────────────────────────────────────────

    def collect_indices(self) -> None:
        """Download index data (沪深300 and S&P500)."""
        start, end = self._date_range()

        log.info("Collecting 沪深300 index history …")
        df = fetch_a_index_history(start, end)
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
        """Download A-share history; returns {symbol: row_count}."""
        symbols = self._a_share_symbols()
        start, end = self._date_range()
        log.info(f"Collecting A-share history for {len(symbols)} stocks ({start} → {end}) …")

        results: Dict[str, int] = {}
        for i, sym in enumerate(symbols, 1):
            path = _history_path(self.base_dir, "A", sym)
            # Skip if file exists and was updated today
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
                if mtime >= datetime.today().date():
                    existing = _load(path)
                    results[sym] = len(existing) if existing is not None else 0
                    continue

            df = fetch_a_share_history(sym, start, end)
            if df is not None and len(df) >= 120:
                _save(df, path)
                results[sym] = len(df)
            time.sleep(0.05)   # polite rate-limiting for AKShare

            if i % 20 == 0:
                log.info(f"  A-share: {i}/{len(symbols)} done ({len(results)} valid)")

        log.info(f"A-share collection complete: {len(results)}/{len(symbols)} valid")
        return results

    def collect_us_stocks(self) -> Dict[str, int]:
        """Download US stock history; returns {symbol: row_count}."""
        symbols = self._us_symbols()
        start, end = self._date_range()
        log.info(f"Collecting US stock history for {len(symbols)} stocks …")

        results: Dict[str, int] = {}
        for i, sym in enumerate(symbols, 1):
            path = _history_path(self.base_dir, "US", sym)
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
                if mtime >= datetime.today().date():
                    existing = _load(path)
                    results[sym] = len(existing) if existing is not None else 0
                    continue

            df = fetch_us_history(sym, start, end)
            if df is not None and len(df) >= 120:
                _save(df, path)
                results[sym] = len(df)

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
