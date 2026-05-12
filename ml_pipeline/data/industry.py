"""A-share industry classification (东方财富 board industries).

Used by the sample generator and backtest engine to compute within-industry
cross-sectional z-scores ("industry neutralization"), so the model learns
*relative* signal within a peer group instead of betting on hot industries.

The mapping is fetched once via AKShare and cached as a parquet file:
  data/history/INDUSTRY/em_industry_map.parquet
A refresh is triggered automatically if the cache is older than 30 days.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

CACHE_FILENAME = "em_industry_map.parquet"
DEFAULT_CACHE_TTL_DAYS = 30
UNKNOWN_INDUSTRY = "UNKNOWN"


def _cache_path(base_dir: str) -> Path:
    p = Path(base_dir) / "INDUSTRY"
    p.mkdir(parents=True, exist_ok=True)
    return p / CACHE_FILENAME


def _is_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_days * 86400


def fetch_industry_map() -> Dict[str, str]:
    """Pull {6-digit-code: industry-name} for the full A-share market.

    Uses Eastmoney board industries (~80 industries, finer than SW1's 28).
    Slow first call (~30 industries × ~1s each); silent best-effort.
    """
    import akshare as ak

    try:
        boards = ak.stock_board_industry_name_em()
    except Exception as e:
        log.warning(f"stock_board_industry_name_em failed: {e}")
        return {}

    name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
    industries = boards[name_col].astype(str).tolist()
    log.info(f"Fetching constituents for {len(industries)} EM industries …")

    mapping: Dict[str, str] = {}
    for i, industry in enumerate(industries, 1):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception as e:
            log.debug(f"  skip industry {industry}: {e}")
            continue
        code_col = "代码" if "代码" in cons.columns else cons.columns[1]
        for code in cons[code_col].astype(str):
            mapping[code.zfill(6)] = industry
        if i % 10 == 0:
            log.info(f"  industries fetched: {i}/{len(industries)} "
                     f"({len(mapping)} unique stocks so far)")

    log.info(f"Industry map: {len(mapping)} stocks classified across "
             f"{len(set(mapping.values()))} industries")
    return mapping


def load_industry_map(
    base_dir: str = "data/history",
    ttl_days: int = DEFAULT_CACHE_TTL_DAYS,
    refresh: bool = False,
) -> Dict[str, str]:
    """Load the industry map from cache, refreshing if missing/stale."""
    path = _cache_path(base_dir)

    if not refresh and _is_fresh(path, ttl_days):
        df = pd.read_parquet(path, engine="pyarrow")
        return dict(zip(df["symbol"].astype(str), df["industry"].astype(str)))

    mapping = fetch_industry_map()
    if not mapping:
        if path.exists():
            log.warning("Industry refresh failed; falling back to stale cache")
            df = pd.read_parquet(path, engine="pyarrow")
            return dict(zip(df["symbol"].astype(str), df["industry"].astype(str)))
        return {}

    df = pd.DataFrame(
        {"symbol": list(mapping.keys()), "industry": list(mapping.values())}
    )
    df.to_parquet(path, engine="pyarrow")
    log.info(f"Industry map cached → {path}")
    return mapping


def assign_industry(
    symbols, mapping: Optional[Dict[str, str]] = None
) -> pd.Series:
    """Return industry labels aligned to ``symbols``; unknown → UNKNOWN."""
    if mapping is None:
        mapping = {}
    return pd.Series(
        [mapping.get(str(s).zfill(6), UNKNOWN_INDUSTRY) for s in symbols],
        index=pd.Index(symbols),
        name="industry",
    )
