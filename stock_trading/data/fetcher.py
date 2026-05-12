"""Data fetching layer: AKShare (A-shares) + yfinance (US stocks).

Data is cached locally as Parquet files to avoid repeated network calls.
Cache validity is controlled by config.data.cache_ttl_hours.
"""

import os
import random
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

# A-share board classification by 6-digit code prefix.
#   SH_MAIN  : Shanghai main board       (600/601/603/605)
#   SZ_MAIN  : Shenzhen main board       (000/001/002/003)
#   ChiNext  : Growth Enterprise (创业板)  (300/301)        — 20% daily limit
#   STAR     : Sci-Tech (科创板)          (688/689)        — 20% daily limit
#   BSE_MAIN : Beijing Stock Exchange    (4xxxxx / 8xxxxx) — 30% daily limit
def classify_a_share_board(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("600", "601", "603", "605")):
        return "SH_MAIN"
    if code.startswith(("000", "001", "002", "003")):
        return "SZ_MAIN"
    if code.startswith(("300", "301")):
        return "ChiNext"
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("4", "8")):
        return "BSE_MAIN"
    return "OTHER"


_CONNECTION_RESET_SIGS = (
    "RemoteDisconnected",
    "ConnectionResetError",
    "Connection aborted",
    "Connection reset",
    "ProtocolError",
    "ReadTimeoutError",
    "Read timed out",
    "ConnectTimeout",
    "Max retries exceeded",
)


def _is_connection_reset(err: BaseException) -> bool:
    """True if the exception looks like upstream throttling / connection reset,
    rather than a real parse error or empty response."""
    s = repr(err)
    return any(sig in s for sig in _CONNECTION_RESET_SIGS)


def _akshare_call_with_retry(
    label: str, fn, *args, attempts: int = 5, base_delay: float = 3.0, **kwargs
):
    """Call an AKShare function with exponential backoff + jitter.

    Eastmoney aggressively closes TCP connections under load
    (RemoteDisconnected / Connection aborted). Recovery requires
    (a) waiting long enough for the per-IP throttle window to reopen,
    (b) staggering retries across concurrent workers via jitter, and
    (c) backing off harder on connection-reset errors than on parse
    errors that won't be fixed by waiting.
    """
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i >= attempts:
                break
            if _is_connection_reset(e):
                delay = base_delay * (2 ** (i - 1))
            else:
                delay = base_delay * (1.5 ** (i - 1))
            delay *= 0.7 + random.random() * 0.6  # ±30% jitter
            log.debug(f"{label} attempt {i}/{attempts} failed: {e}; retry in {delay:.1f}s")
            time.sleep(delay)
    log.warning(f"{label} failed after {attempts} attempts: {last_err}")
    return None


def _pick_code_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort: find the column that holds 6-digit stock codes."""
    for c in ("code", "代码", "symbol", "Symbol", "证券代码"):
        if c in df.columns:
            return c
    # Heuristic: first column whose first non-null value is a 6-digit string
    for c in df.columns:
        try:
            sample = str(df[c].dropna().iloc[0])
            if sample.isdigit() and len(sample) == 6:
                return c
        except Exception:
            continue
    return None


def _fetch_full_a_share_roster() -> Optional[pd.DataFrame]:
    """Try multiple AKShare endpoints to get the full A-share roster.

    The primary endpoint (stock_info_a_code_name) hits one Eastmoney URL;
    if that's blocked / reset / rate-limited, we fall back to the spot-quote
    endpoint (different URL), then to per-exchange listings (SH+SZ separately).
    """
    import akshare as ak

    # Endpoint 1: dedicated code-name list (smallest payload)
    df = _akshare_call_with_retry("stock_info_a_code_name", ak.stock_info_a_code_name)
    if df is not None and not df.empty:
        return df

    # Endpoint 2: spot quote (different URL, heavier but more reliable)
    log.info("Falling back to stock_zh_a_spot_em() for universe roster …")
    df = _akshare_call_with_retry("stock_zh_a_spot_em", ak.stock_zh_a_spot_em)
    if df is not None and not df.empty:
        return df

    # Endpoint 3: per-exchange listings (last resort, may need multiple calls)
    log.info("Falling back to stock_info_sh_name_code + stock_info_sz_name_code …")
    parts = []
    sh = _akshare_call_with_retry(
        "stock_info_sh_name_code",
        ak.stock_info_sh_name_code, symbol="主板A股",
    )
    if sh is not None and not sh.empty:
        parts.append(sh)
    sh_star = _akshare_call_with_retry(
        "stock_info_sh_name_code(科创板)",
        ak.stock_info_sh_name_code, symbol="科创板",
    )
    if sh_star is not None and not sh_star.empty:
        parts.append(sh_star)
    sz = _akshare_call_with_retry(
        "stock_info_sz_name_code",
        ak.stock_info_sz_name_code, symbol="A股列表",
    )
    if sz is not None and not sz.empty:
        parts.append(sz)
    if parts:
        return pd.concat(parts, ignore_index=True)

    return None


def get_a_share_universe(uni_cfg: dict) -> List[str]:
    """Return the A-share universe per config.

    Two modes:
      "all"   – pull the full A-share roster via AKShare and filter by board
      "index" – pull constituents of the given index (legacy behavior)

    The "all" path tries three AKShare endpoints in order to survive
    transient connection resets from any single Eastmoney URL.
    """
    import akshare as ak

    mode = uni_cfg.get("a_share_universe", "index")
    limit = int(uni_cfg.get("a_share_limit", 0) or 0)

    if mode == "all":
        boards = set(uni_cfg.get("a_share_boards", ["SH_MAIN", "SZ_MAIN", "ChiNext", "STAR"]))
        df = _fetch_full_a_share_roster()
        if df is None or df.empty:
            log.warning(
                "All A-share universe endpoints failed; universe is empty. "
                "Check network / proxy / `pip install -U akshare`."
            )
            return []

        code_col = _pick_code_column(df)
        if code_col is None:
            log.warning(f"Couldn't locate a code column in roster; columns={list(df.columns)}")
            return []

        codes = df[code_col].astype(str).str.zfill(6).tolist()
        codes = [c for c in codes if classify_a_share_board(c) in boards]
        codes = sorted(set(codes))   # dedupe (per-exchange fallback can overlap)
        if limit > 0:
            codes = codes[:limit]
        log.info(
            f"A-share universe (all, boards={sorted(boards)}, "
            f"source_col='{code_col}'): {len(codes)} symbols"
        )
        return codes

    # index mode (legacy)
    index_code = uni_cfg.get("a_share_index", "000300")
    df = _akshare_call_with_retry(
        f"index_stock_cons_csindex({index_code})",
        ak.index_stock_cons_csindex, symbol=index_code,
    )
    if df is None or df.empty:
        return []
    codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
    if limit > 0:
        codes = codes[:limit]
    return codes


def get_a_share_stock_list(index_code: str = "000300", limit: int = 100) -> List[str]:
    """Backward-compatible: legacy index-based selector."""
    return get_a_share_universe({
        "a_share_universe": "index",
        "a_share_index": index_code,
        "a_share_limit": limit,
    })


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

    df = _akshare_call_with_retry(
        f"stock_zh_a_hist({symbol})",
        ak.stock_zh_a_hist,
        symbol=symbol, period="daily",
        start_date=start, end_date=end, adjust="qfq",
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
        "成交额": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    df = df[keep].astype(float)
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df["close"]
    df = df.tail(lookback_days)

    _save(df, path)
    return df


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
        a_codes = get_a_share_universe(self.uni_cfg)
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
