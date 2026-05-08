"""Feature engineering for the ML pipeline.

Produces a per-stock feature DataFrame where each row is a trading day and
each column is a model input feature.  All features are strictly
backward-looking so that no future information leaks into the training set.

Feature groups
──────────────
  price_return   – 1/3/5/10/20-day log-returns
  ma_ratio       – price vs. rolling SMA (5/10/20/60), SMA cross-ratios
  ema_ratio       – EMA12/EMA26 ratio, MACD-family normalized by ATR
  rsi            – RSI(14)
  bbands         – Bollinger %B, normalized bandwidth
  atr            – ATR/price (normalized volatility)
  realized_vol   – 5-day and 20-day rolling std of returns
  volume         – today/20d avg volume, 5d/20d avg volume, vol×return sign
  price_level    – distance to 52-week high/low
  market         – index 1/5/20-day returns, index vol, alpha vs. index,
                   rolling 20-day beta

The last column 'close' is kept for label computation downstream
(it is dropped before model training).
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

# All feature column names (excluding 'close' which is metadata)
FEATURE_COLS = [
    # returns
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # MA ratios
    "price_ma5", "price_ma10", "price_ma20", "price_ma60",
    "ma5_ma20", "ma10_ma60",
    # EMA / MACD
    "ema_ratio",
    "macd_norm", "macd_hist_norm",
    # RSI
    "rsi",
    # Bollinger Bands
    "bb_pct", "bb_width",
    # ATR
    "atr_pct",
    # Realized vol
    "rvol_5d", "rvol_20d",
    # Volume
    "vol_ratio_1d", "vol_ratio_5d", "signed_vol",
    # Price level
    "high52w_ratio", "low52w_ratio",
    # Market / index features
    "mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d",
    "alpha_1d", "alpha_5d",
    "mkt_rvol_20d", "beta_20d",
]


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 1.0) -> pd.Series:
    return a.div(b.replace(0, np.nan)).fillna(fill)


def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all stock-level features.  Returns enriched DataFrame."""
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]

    # ── Returns ────────────────────────────────────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    out["ret_1d"] = log_ret
    out["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
    out["ret_5d"] = np.log(df["close"] / df["close"].shift(5))
    out["ret_10d"] = np.log(df["close"] / df["close"].shift(10))
    out["ret_20d"] = np.log(df["close"] / df["close"].shift(20))

    # ── MA ratios ──────────────────────────────────────────────────────────
    for p in [5, 10, 20, 60]:
        ma = ta.sma(df["close"], length=p)
        out[f"price_ma{p}"] = _safe_div(df["close"], ma) - 1.0

    out["ma5_ma20"] = _safe_div(ta.sma(df["close"], 5), ta.sma(df["close"], 20)) - 1.0
    out["ma10_ma60"] = _safe_div(ta.sma(df["close"], 10), ta.sma(df["close"], 60)) - 1.0

    # ── EMA / MACD ─────────────────────────────────────────────────────────
    ema12 = ta.ema(df["close"], length=12)
    ema26 = ta.ema(df["close"], length=26)
    out["ema_ratio"] = _safe_div(ema12, ema26) - 1.0

    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    atr_safe = atr.replace(0, np.nan)
    if macd_df is not None:
        out["macd_norm"] = macd_df.iloc[:, 0] / atr_safe       # MACD / ATR
        out["macd_hist_norm"] = macd_df.iloc[:, 1] / atr_safe  # histogram / ATR

    # ── RSI ────────────────────────────────────────────────────────────────
    out["rsi"] = ta.rsi(df["close"], length=14) / 100.0         # scale to [0,1]

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        upper, mid, lower = bb.iloc[:, 2], bb.iloc[:, 1], bb.iloc[:, 0]
        band_range = (upper - lower).replace(0, np.nan)
        out["bb_pct"] = (df["close"] - lower) / band_range
        out["bb_width"] = band_range / mid.replace(0, np.nan)

    # ── ATR (normalized) ──────────────────────────────────────────────────
    out["atr_pct"] = atr / df["close"].replace(0, np.nan)

    # ── Realized volatility ────────────────────────────────────────────────
    out["rvol_5d"] = log_ret.rolling(5).std()
    out["rvol_20d"] = log_ret.rolling(20).std()

    # ── Volume ────────────────────────────────────────────────────────────
    vol_ma20 = ta.sma(df["volume"], length=20).replace(0, np.nan)
    vol_ma5 = ta.sma(df["volume"], length=5).replace(0, np.nan)
    out["vol_ratio_1d"] = df["volume"] / vol_ma20
    out["vol_ratio_5d"] = vol_ma5 / vol_ma20
    # signed volume: positive if price went up, negative if down
    out["signed_vol"] = out["vol_ratio_1d"] * np.sign(log_ret.fillna(0))

    # ── Price level (52-week high/low) ─────────────────────────────────────
    roll252 = df["close"].rolling(252, min_periods=60)
    out["high52w_ratio"] = df["close"] / roll252.max().replace(0, np.nan)
    out["low52w_ratio"] = df["close"] / roll252.min().replace(0, np.nan)

    return out


def compute_market_features(index_df: pd.DataFrame, prefix: str = "mkt") -> pd.DataFrame:
    """Compute market (index) features.  Returns DataFrame indexed by date."""
    out = pd.DataFrame(index=index_df.index)
    log_ret = np.log(index_df["close"] / index_df["close"].shift(1))
    out[f"{prefix}_ret_1d"] = log_ret
    out[f"{prefix}_ret_5d"] = np.log(index_df["close"] / index_df["close"].shift(5))
    out[f"{prefix}_ret_20d"] = np.log(index_df["close"] / index_df["close"].shift(20))
    out[f"{prefix}_rvol_20d"] = log_ret.rolling(20).std()
    return out


def merge_market_features(
    stock_feat: pd.DataFrame,
    market_feat: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join market features onto stock features by date."""
    # Align on date index; forward-fill to handle different trading calendars
    market_aligned = market_feat.reindex(stock_feat.index, method="ffill")
    merged = pd.concat([stock_feat, market_aligned], axis=1)

    # Alpha = stock return - market return
    if "ret_1d" in merged.columns and "mkt_ret_1d" in merged.columns:
        merged["alpha_1d"] = merged["ret_1d"] - merged["mkt_ret_1d"]
    if "ret_5d" in merged.columns and "mkt_ret_5d" in merged.columns:
        merged["alpha_5d"] = merged["ret_5d"] - merged["mkt_ret_5d"]

    # Rolling 20-day beta vs. market
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
    """
    Full feature engineering for one stock.

    Parameters
    ----------
    ohlcv       : OHLCV DataFrame for the stock
    index_ohlcv : OHLCV for the benchmark index (same market)

    Returns
    -------
    DataFrame with FEATURE_COLS + 'close', index = trading dates.
    Rows with any NaN in essential features are dropped.
    """
    feat = compute_stock_features(ohlcv)

    if index_ohlcv is not None and not index_ohlcv.empty:
        mkt = compute_market_features(index_ohlcv)
        feat = merge_market_features(feat, mkt)
    else:
        # Fill market columns with zero so shape is consistent
        for col in ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d",
                    "mkt_rvol_20d", "alpha_1d", "alpha_5d", "beta_20d"]:
            feat[col] = 0.0

    # Drop rows where key features are missing (warm-up period)
    key_cols = ["ret_20d", "rsi", "bb_pct", "atr_pct", "rvol_20d",
                "mkt_ret_20d", "beta_20d"]
    available_key = [c for c in key_cols if c in feat.columns]
    feat = feat.dropna(subset=available_key)

    # Clip extreme values to [-10, 10] for ratio features
    ratio_cols = [c for c in FEATURE_COLS if c in feat.columns
                  and c not in ("rsi", "bb_pct", "atr_pct", "rvol_5d", "rvol_20d",
                                "mkt_rvol_20d", "high52w_ratio", "low52w_ratio")]
    feat[ratio_cols] = feat[ratio_cols].clip(-10, 10)

    return feat
