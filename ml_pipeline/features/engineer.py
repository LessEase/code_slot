"""Feature engineering for the ML pipeline — pure pandas/numpy implementation.

No external TA library required. All indicators are computed from scratch.
"""

import numpy as np
import pandas as pd
from typing import Optional

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "price_ma5", "price_ma10", "price_ma20", "price_ma60",
    "ma5_ma20", "ma10_ma60",
    "ema_ratio",
    "macd_norm", "macd_hist_norm",
    "rsi",
    "bb_pct", "bb_width",
    "atr_pct",
    "rvol_5d", "rvol_20d",
    "vol_ratio_1d", "vol_ratio_5d", "signed_vol",
    "high52w_ratio", "low52w_ratio",
    "mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d",
    "alpha_1d", "alpha_5d",
    "mkt_rvol_20d", "beta_20d",
]


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 1.0) -> pd.Series:
    return a.div(b.replace(0, np.nan)).fillna(fill)


def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]

    # Auxiliary liquidity column (not a model feature, but kept on the feature
    # matrix so downstream stages can apply a point-in-time turnover filter
    # without re-loading raw OHLCV). Falls back to volume*close if exchange-
    # provided 成交额 is missing (yfinance for US, older A-share parquet).
    if "amount" in df.columns:
        amt = df["amount"]
    else:
        amt = df["volume"] * df["close"]
    out["amount_20d_avg"] = amt.rolling(20, min_periods=10).mean()

    log_ret = np.log(df["close"] / df["close"].shift(1))
    out["ret_1d"] = log_ret
    out["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
    out["ret_5d"] = np.log(df["close"] / df["close"].shift(5))
    out["ret_10d"] = np.log(df["close"] / df["close"].shift(10))
    out["ret_20d"] = np.log(df["close"] / df["close"].shift(20))

    for p in [5, 10, 20, 60]:
        ma = _sma(df["close"], p)
        out[f"price_ma{p}"] = _safe_div(df["close"], ma) - 1.0

    out["ma5_ma20"] = _safe_div(_sma(df["close"], 5), _sma(df["close"], 20)) - 1.0
    out["ma10_ma60"] = _safe_div(_sma(df["close"], 10), _sma(df["close"], 60)) - 1.0

    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    out["ema_ratio"] = _safe_div(ema12, ema26) - 1.0

    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    atr = _atr(df["high"], df["low"], df["close"], 14)
    atr_safe = atr.replace(0, np.nan)
    out["macd_norm"] = macd_line / atr_safe
    out["macd_hist_norm"] = macd_hist / atr_safe

    out["rsi"] = _rsi(df["close"], 14) / 100.0

    bb_mid = _sma(df["close"], 20)
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_range = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_pct"] = (df["close"] - bb_lower) / band_range
    out["bb_width"] = band_range / bb_mid.replace(0, np.nan)

    out["atr_pct"] = atr / df["close"].replace(0, np.nan)

    out["rvol_5d"] = log_ret.rolling(5).std()
    out["rvol_20d"] = log_ret.rolling(20).std()

    vol_ma20 = _sma(df["volume"], 20).replace(0, np.nan)
    vol_ma5 = _sma(df["volume"], 5).replace(0, np.nan)
    out["vol_ratio_1d"] = df["volume"] / vol_ma20
    out["vol_ratio_5d"] = vol_ma5 / vol_ma20
    out["signed_vol"] = out["vol_ratio_1d"] * np.sign(log_ret.fillna(0))

    roll252 = df["close"].rolling(252, min_periods=60)
    out["high52w_ratio"] = df["close"] / roll252.max().replace(0, np.nan)
    out["low52w_ratio"] = df["close"] / roll252.min().replace(0, np.nan)

    return out


def compute_market_features(index_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=index_df.index)
    log_ret = np.log(index_df["close"] / index_df["close"].shift(1))
    out["mkt_ret_1d"] = log_ret
    out["mkt_ret_5d"] = np.log(index_df["close"] / index_df["close"].shift(5))
    out["mkt_ret_20d"] = np.log(index_df["close"] / index_df["close"].shift(20))
    out["mkt_rvol_20d"] = log_ret.rolling(20).std()
    return out


def merge_market_features(stock_feat: pd.DataFrame, market_feat: pd.DataFrame) -> pd.DataFrame:
    market_aligned = market_feat.reindex(stock_feat.index, method="ffill")
    merged = pd.concat([stock_feat, market_aligned], axis=1)

    if "ret_1d" in merged.columns and "mkt_ret_1d" in merged.columns:
        merged["alpha_1d"] = merged["ret_1d"] - merged["mkt_ret_1d"]
    if "ret_5d" in merged.columns and "mkt_ret_5d" in merged.columns:
        merged["alpha_5d"] = merged["ret_5d"] - merged["mkt_ret_5d"]

    if "ret_1d" in merged.columns and "mkt_ret_1d" in merged.columns:
        r = merged["ret_1d"]
        m = merged["mkt_ret_1d"]
        cov = r.rolling(20).cov(m)
        var = m.rolling(20).var().replace(0, np.nan)
        merged["beta_20d"] = cov / var

    return merged


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    index_ohlcv: Optional[pd.DataFrame],
) -> pd.DataFrame:
    feat = compute_stock_features(ohlcv)

    if index_ohlcv is not None and not index_ohlcv.empty:
        mkt = compute_market_features(index_ohlcv)
        feat = merge_market_features(feat, mkt)
    else:
        for col in ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d",
                    "mkt_rvol_20d", "alpha_1d", "alpha_5d", "beta_20d"]:
            feat[col] = 0.0

    key_cols = ["ret_20d", "rsi", "bb_pct", "atr_pct", "rvol_20d",
                "mkt_ret_20d", "beta_20d"]
    available_key = [c for c in key_cols if c in feat.columns]
    feat = feat.dropna(subset=available_key)

    ratio_cols = [c for c in FEATURE_COLS if c in feat.columns
                  and c not in ("rsi", "bb_pct", "atr_pct", "rvol_5d", "rvol_20d",
                                "mkt_rvol_20d", "high52w_ratio", "low52w_ratio")]
    feat[ratio_cols] = feat[ratio_cols].clip(-10, 10)

    return feat
