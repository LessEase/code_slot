"""Technical indicator computation — pure pandas/numpy, no external TA library."""

import numpy as np
import pandas as pd

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    for p in cfg.get("ma_periods", [5, 10, 20, 60]):
        df[f"ma{p}"] = _sma(df["close"], p)

    for p in cfg.get("ema_periods", [12, 26]):
        df[f"ema{p}"] = _ema(df["close"], p)

    macd_cfg = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    ema_fast = _ema(df["close"], macd_cfg["fast"])
    ema_slow = _ema(df["close"], macd_cfg["slow"])
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, macd_cfg["signal"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_line - macd_signal

    df["rsi"] = _rsi(df["close"], cfg.get("rsi_period", 14))

    bb_cfg = cfg.get("bbands", {"period": 20, "std": 2.0})
    bb_mid = _sma(df["close"], bb_cfg["period"])
    bb_std = df["close"].rolling(bb_cfg["period"]).std()
    df["bb_upper"] = bb_mid + bb_cfg["std"] * bb_std
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_mid - bb_cfg["std"] * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    atr_period = cfg.get("atr_period", 14)
    df["atr"] = _atr(df["high"], df["low"], df["close"], atr_period)
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

    vol_ma_period = cfg.get("volume_ma_period", 20)
    df["vol_ma"] = _sma(df["volume"], vol_ma_period)
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    df["pct_chg"] = df["close"].pct_change()
    df["is_up"] = (df["close"] > df["close"].shift(1)).astype(float)
    df["up_vol_ratio"] = df["vol_ratio"] * df["is_up"]

    essential = ["ma5", "ma20", "ema12", "ema26", "macd", "rsi", "bb_width", "atr", "vol_ratio"]
    df = df.dropna(subset=essential)
    return df
