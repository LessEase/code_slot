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


# ── A-share data-source selection (Eastmoney vs Sina) ──────────────────────
#
# AKShare's default A-share endpoints (`stock_zh_a_hist`, `index_zh_a_hist`)
# hit Eastmoney (push2his.eastmoney.com). On networks where Eastmoney is
# blocked (e.g. some corporate egress firewalls) every request gets RST'd
# and the entire collector hangs through its retries.
#
# Sina hosts a mirror dataset (`stock_zh_a_daily`, `stock_zh_index_daily`)
# served from finance.sina.com.cn — different physical link, usually
# reachable when Eastmoney isn't. We auto-detect at start of run and pin
# the choice for the rest of the process.

def _sina_stock_symbol(code: str) -> str:
    """Convert 6-digit A-share code → sina-prefixed symbol (sh600000 / sz000001)."""
    code = str(code).zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{code}"
    return f"sz{code}"


def _eastmoney_reachable(timeout: float = 2.0) -> bool:
    """One-shot HEAD/GET probe to Eastmoney's kline API.

    Returns True only if a small canary request returns 2xx within `timeout`.
    Any connection-level failure (RST, timeout, DNS) returns False.
    """
    import socket
    import urllib.error
    import urllib.request

    url = (
        "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        "?secid=1.000300&klt=101&fqt=1&beg=20240101&end=20240110"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 400
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
        return False


_DATA_SOURCE_CACHE: Optional[str] = None


def resolve_a_share_data_source(uni_cfg: dict) -> str:
    """Decide which backend to use for A-share OHLCV ('eastmoney' or 'sina').

    Behavior is controlled by `universe.a_share_data_source`:
      "auto"       — probe Eastmoney once; on failure pin Sina for the run (default)
      "eastmoney"  — force Eastmoney (legacy / when probe is unreliable)
      "sina"       — force Sina (when you know Eastmoney is blocked)

    The verdict is cached process-wide so the probe runs at most once.
    """
    global _DATA_SOURCE_CACHE
    if _DATA_SOURCE_CACHE is not None:
        return _DATA_SOURCE_CACHE

    pref = (uni_cfg.get("a_share_data_source") or "auto").lower()
    if pref in ("eastmoney", "sina"):
        _DATA_SOURCE_CACHE = pref
        log.info(f"A-share data source: {pref} (explicit)")
        return pref

    log.info("Probing Eastmoney reachability …")
    if _eastmoney_reachable():
        _DATA_SOURCE_CACHE = "eastmoney"
        log.info("A-share data source: eastmoney (probe succeeded)")
    else:
        _DATA_SOURCE_CACHE = "sina"
        log.warning(
            "Eastmoney probe failed; using Sina backend for entire run "
            "(set universe.a_share_data_source='eastmoney' to override)"
        )
    return _DATA_SOURCE_CACHE


def fetch_a_share_history_sina(
    symbol: str, start: str, end: str
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for one A-share via Sina (stock_zh_a_daily, qfq-adjusted).

    Returns a DataFrame indexed by date with columns
    [open, high, low, close, volume, amount] — same shape as the Eastmoney path.
    """
    import akshare as ak

    sina_sym = _sina_stock_symbol(symbol)
    # AKShare accepts both "YYYY-MM-DD" and "YYYYMMDD"; normalize to YYYYMMDD.
    start_norm = start.replace("-", "")
    end_norm = end.replace("-", "")

    df = _akshare_call_with_retry(
        f"stock_zh_a_daily({sina_sym})",
        ak.stock_zh_a_daily,
        symbol=sina_sym, adjust="qfq",
        start_date=start_norm, end_date=end_norm,
    )
    if df is None or df.empty:
        return None

    # AKShare's stock_zh_a_daily schema varies by version:
    #   newer: date, open, high, low, close, volume, amount, outstanding_share, turnover
    #          (amount = CNY traded, turnover = 换手率)
    #   older: date, open, high, low, close, volume, outstanding_share, turnover
    #          (turnover = CNY traded; no `amount` column)
    # Only rename turnover→amount when an `amount` column isn't already present,
    # otherwise we end up with two `amount` columns and parquet write blows up.
    if "amount" not in df.columns and "turnover" in df.columns:
        df = df.rename(columns={"turnover": "amount"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    out = df[keep].astype(float)
    if "amount" not in out.columns:
        out["amount"] = out["volume"] * out["close"]
    return out


def fetch_a_index_history_sina(
    start: str, end: str, index_code: str = "000300"
) -> Optional[pd.DataFrame]:
    """Fetch an A-share index daily OHLCV via Sina (stock_zh_index_daily).

    Sina returns full history regardless of date params, so we filter
    client-side. Currently only CSI300 (sh000300) is wired in; other CSI
    indices follow the same `sh000XXX` pattern.
    """
    import akshare as ak

    code = str(index_code).zfill(6)
    sina_sym = f"sh{code}" if code.startswith("000") else _sina_stock_symbol(code)

    df = _akshare_call_with_retry(
        f"stock_zh_index_daily({sina_sym})",
        ak.stock_zh_index_daily,
        symbol=sina_sym,
    )
    if df is None or df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].astype(float)


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


# ---------------------------------------------------------------------------
# US universe (NASDAQ Trader symbol directory)
# ---------------------------------------------------------------------------
#
# NASDAQ Trader publishes the full, official symbol roster as two pipe-
# delimited files (free, no auth):
#   nasdaqlisted.txt — every NASDAQ-listed security
#   otherlisted.txt  — NYSE, NYSE American, NYSE Arca, etc.
# We download both, drop test issues / warrants / rights / units / preferred
# (keeping common stock + ETFs), and cache the result on disk.

_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _fetch_nasdaq_trader_file(url: str, label: str) -> Optional[pd.DataFrame]:
    """Download and parse a pipe-delimited NASDAQ Trader symbol file."""
    import io
    import urllib.request

    def _do():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
        # The final line is a "File Creation Time:" footer — strip it.
        lines = [
            ln for ln in raw.splitlines()
            if ln and not ln.startswith("File Creation Time")
        ]
        return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)

    return _akshare_call_with_retry(label, _do, attempts=4, base_delay=3.0)


def _us_symbol_for_yfinance(sym: str) -> str:
    """NASDAQ Trader denotes share classes with '.'; yfinance expects '-'
    (e.g. BRK.B → BRK-B)."""
    return sym.strip().upper().replace(".", "-")


def _is_excluded_us_security(name: str) -> bool:
    """True for warrants / rights / units / preferred / notes — i.e. anything
    that isn't a common share or an ETF."""
    n = (name or "").lower()
    if any(k in n for k in (
        "warrant", "depositary", "debenture", "preferred",
        "convertible note", "% note", "subordinated",
    )):
        return True
    # token-level check so "Wright"/"United" don't false-positive on right/unit
    tokens = set(n.replace("-", " ").replace(",", " ").replace(".", " ").split())
    return bool(tokens & {"right", "rights", "unit", "units"})


def get_us_universe(uni_cfg: dict) -> List[str]:
    """Return the US stock universe per config.

    Modes (universe.us_universe):
      "all"  — full NASDAQ + NYSE/AMEX roster from NASDAQ Trader files
      "list" — the explicit us_stock_list (legacy behavior)

    In "all" mode the roster is cached on disk for us_universe_cache_days
    so we don't re-download every run. On fetch failure we fall back to
    us_stock_list so a transient network error doesn't empty the universe.
    """
    mode = (uni_cfg.get("us_universe") or "list").lower()
    fallback = list(uni_cfg.get("us_stock_list") or [])
    if mode != "all":
        return fallback

    include_etf = bool(uni_cfg.get("us_include_etf", True))
    limit = int(uni_cfg.get("us_stock_limit", 0) or 0)
    cache_days = float(uni_cfg.get("us_universe_cache_days", 7))
    cache_dir = uni_cfg.get("us_universe_cache_dir", "data/history")
    cache_path = Path(cache_dir) / "US_universe.txt"

    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
        if age_days < cache_days:
            syms = [s.strip() for s in cache_path.read_text().splitlines() if s.strip()]
            if syms:
                log.info(f"US universe (all): {len(syms)} symbols (cached)")
                return syms[:limit] if limit > 0 else syms

    frames: List[pd.DataFrame] = []
    nd = _fetch_nasdaq_trader_file(_NASDAQ_LISTED_URL, "nasdaqlisted.txt")
    if nd is not None and not nd.empty:
        nd = nd.rename(columns=lambda c: c.strip())
        frames.append(pd.DataFrame({
            "sym": nd["Symbol"].astype(str),
            "name": nd["Security Name"].astype(str),
            "test": nd.get("Test Issue", "N").astype(str),
            "etf": nd.get("ETF", "N").astype(str),
        }))
    ot = _fetch_nasdaq_trader_file(_OTHER_LISTED_URL, "otherlisted.txt")
    if ot is not None and not ot.empty:
        ot = ot.rename(columns=lambda c: c.strip())
        sym_col = "ACT Symbol" if "ACT Symbol" in ot.columns else "NASDAQ Symbol"
        frames.append(pd.DataFrame({
            "sym": ot[sym_col].astype(str),
            "name": ot["Security Name"].astype(str),
            "test": ot.get("Test Issue", "N").astype(str),
            "etf": ot.get("ETF", "N").astype(str),
        }))

    if not frames:
        log.warning(
            "US universe: NASDAQ Trader fetch failed; "
            f"falling back to us_stock_list ({len(fallback)} symbols)"
        )
        return fallback

    roster = pd.concat(frames, ignore_index=True)
    roster = roster[roster["test"].str.upper().str.strip() != "Y"]
    if not include_etf:
        roster = roster[roster["etf"].str.upper().str.strip() != "Y"]
    roster = roster[~roster["name"].apply(_is_excluded_us_security)]

    out: List[str] = []
    seen = set()
    for raw in roster["sym"].tolist():
        s = (raw or "").strip()
        if not s or any(ch in s for ch in "$^+ "):  # preferred/warrant notation
            continue
        y = _us_symbol_for_yfinance(s)
        if y and y not in seen:
            seen.add(y)
            out.append(y)
    out.sort()

    if out:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(out))
    log.info(f"US universe (all, include_etf={include_etf}): {len(out)} symbols")
    return out[:limit] if limit > 0 else out


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
            for sym in get_us_universe(self.uni_cfg):
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
