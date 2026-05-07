"""Technical indicator computation using pandas-ta.

All indicators are added as extra columns to the OHLCV DataFrame.
Returns the enriched DataFrame (original is not mutated).
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute all technical indicators and append them as columns.

    Parameters
    ----------
    df  : OHLCV DataFrame with columns [open, high, low, close, volume]
    cfg : indicators section from config.yaml

    Returns
    -------
    DataFrame with additional indicator columns; rows with NaN are dropped.
    """
    df = df.copy()

    # ── Moving Averages ────────────────────────────────────────────────────
    for p in cfg.get("ma_periods", [5, 10, 20, 60]):
        df[f"ma{p}"] = ta.sma(df["close"], length=p)

    for p in cfg.get("ema_periods", [12, 26]):
        df[f"ema{p}"] = ta.ema(df["close"], length=p)

    # ── MACD ────────────────────────────────────────────────────────────────
    macd_cfg = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    macd_df = ta.macd(
        df["close"],
        fast=macd_cfg["fast"],
        slow=macd_cfg["slow"],
        signal=macd_cfg["signal"],
    )
    if macd_df is not None:
        df["macd"] = macd_df.iloc[:, 0]        # MACD line
        df["macd_signal"] = macd_df.iloc[:, 2]  # Signal line
        df["macd_hist"] = macd_df.iloc[:, 1]   # Histogram

    # ── RSI ─────────────────────────────────────────────────────────────────
    rsi_period = cfg.get("rsi_period", 14)
    df["rsi"] = ta.rsi(df["close"], length=rsi_period)

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bb_cfg = cfg.get("bbands", {"period": 20, "std": 2.0})
    bb_df = ta.bbands(df["close"], length=bb_cfg["period"], std=bb_cfg["std"])
    if bb_df is not None:
        df["bb_upper"] = bb_df.iloc[:, 2]
        df["bb_mid"] = bb_df.iloc[:, 1]
        df["bb_lower"] = bb_df.iloc[:, 0]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── ATR ─────────────────────────────────────────────────────────────────
    atr_period = cfg.get("atr_period", 14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
    df["atr_pct"] = df["atr"] / df["close"]  # normalized ATR

    # ── Volume indicators ────────────────────────────────────────────────────
    vol_ma_period = cfg.get("volume_ma_period", 20)
    df["vol_ma"] = ta.sma(df["volume"], length=vol_ma_period)
    df["vol_ratio"] = df["volume"] / df["vol_ma"]

    # Price change
    df["pct_chg"] = df["close"].pct_change()

    # Up-day volume ratio (volume on up days vs down days)
    df["is_up"] = (df["close"] > df["close"].shift(1)).astype(float)
    df["up_vol_ratio"] = df["vol_ratio"] * df["is_up"]

    # Drop rows where essential indicators are NaN
    essential = ["ma5", "ma20", "ema12", "ema26", "macd", "rsi",
                 "bb_width", "atr", "vol_ratio"]
    df = df.dropna(subset=essential)

    return df
