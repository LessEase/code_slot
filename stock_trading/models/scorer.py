"""Multi-factor scoring model for stock selection.

Each factor produces a score in [0, 100]. The final score is a weighted
average of four sub-scores:

  trend      – MA alignment + price vs MA
  momentum   – MACD + RSI positioning
  volume     – volume-price confirmation
  volatility – Bollinger Band squeeze + normalized ATR

Stocks are ranked by final score; the top-N are selected as buy candidates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-score functions (each returns float in [0, 100])
# ---------------------------------------------------------------------------

def _trend_score(row: pd.Series) -> float:
    """Score based on MA alignment and price position."""
    score = 0.0

    # MA alignment: ma5 > ma10 > ma20 > ma60 = fully bullish
    pairs = [("ma5", "ma10"), ("ma10", "ma20"), ("ma20", "ma60")]
    available = [(a, b) for a, b in pairs if a in row.index and b in row.index]
    if available:
        aligned = sum(row[a] > row[b] for a, b in available)
        score += 40 * (aligned / len(available))

    # Price above ma20
    if "ma20" in row.index and row["close"] > row["ma20"]:
        score += 20

    # Price above ma60
    if "ma60" in row.index and row["close"] > row["ma60"]:
        score += 15

    # EMA12 > EMA26 (golden cross zone)
    if "ema12" in row.index and "ema26" in row.index:
        if row["ema12"] > row["ema26"]:
            score += 25

    return min(score, 100.0)


def _momentum_score(row: pd.Series) -> float:
    """Score based on MACD and RSI."""
    score = 0.0

    # MACD: line above signal → bullish momentum
    if "macd" in row.index and "macd_signal" in row.index:
        if row["macd"] > row["macd_signal"]:
            score += 30
        # MACD histogram positive and growing
        if "macd_hist" in row.index and row["macd_hist"] > 0:
            score += 10

    # RSI: optimal zone 45-65 (trending up, not overbought)
    if "rsi" in row.index:
        rsi = row["rsi"]
        if pd.isna(rsi):
            pass
        elif 45 <= rsi <= 65:
            score += 40          # ideal momentum zone
        elif 35 <= rsi < 45:
            score += 25          # recovering from oversold
        elif 65 < rsi <= 75:
            score += 20          # strong but watch for reversal
        elif rsi < 35:
            score += 15          # oversold bounce potential
        else:
            score += 0           # overbought (rsi > 75)

    # MACD above zero line
    if "macd" in row.index and row["macd"] > 0:
        score += 20

    return min(score, 100.0)


def _volume_score(row: pd.Series) -> float:
    """Score based on volume-price confirmation."""
    score = 0.0

    if "vol_ratio" not in row.index:
        return 50.0

    vol_ratio = row["vol_ratio"]
    if pd.isna(vol_ratio):
        return 50.0

    # Volume expansion
    if vol_ratio >= 2.0:
        score += 40
    elif vol_ratio >= 1.5:
        score += 30
    elif vol_ratio >= 1.2:
        score += 20
    elif vol_ratio >= 0.8:
        score += 10

    # Up-day volume bonus (volume on up days)
    if "up_vol_ratio" in row.index and not pd.isna(row["up_vol_ratio"]):
        if row["up_vol_ratio"] >= 1.5 and row.get("pct_chg", 0) > 0:
            score += 30         # volume confirms price rise
        elif row["up_vol_ratio"] >= 1.0 and row.get("pct_chg", 0) > 0:
            score += 15

    # Price change on above-avg volume
    pct = row.get("pct_chg", 0)
    if not pd.isna(pct) and pct > 0.02 and vol_ratio > 1.2:
        score += 30             # strong up day with volume

    return min(score, 100.0)


def _volatility_score(row: pd.Series) -> float:
    """Score based on Bollinger Bands and ATR.

    Strategy: prefer stocks in a low-volatility squeeze (potential breakout)
    that are near or above the mid-band.
    """
    score = 0.0

    # BB position: price above mid-band = bullish
    if "bb_pct" in row.index and not pd.isna(row["bb_pct"]):
        bb_pct = row["bb_pct"]
        if 0.5 <= bb_pct <= 0.85:
            score += 40     # between mid and upper — healthy trend
        elif bb_pct > 0.85:
            score += 20     # approaching upper band (overbought risk)
        elif 0.3 <= bb_pct < 0.5:
            score += 30     # near mid-band, consolidating

    # BB squeeze: low width relative to recent (breakout potential)
    if "bb_width" in row.index and not pd.isna(row["bb_width"]):
        bw = row["bb_width"]
        if bw < 0.05:
            score += 40     # very tight squeeze → imminent breakout
        elif bw < 0.08:
            score += 25
        elif bw < 0.12:
            score += 10

    # ATR: moderate ATR preferred (not too volatile, not stagnant)
    if "atr_pct" in row.index and not pd.isna(row["atr_pct"]):
        atr_pct = row["atr_pct"]
        if 0.01 <= atr_pct <= 0.03:
            score += 20     # healthy volatility
        elif atr_pct < 0.01:
            score += 5      # too quiet
        else:
            score += 10     # volatile but not disqualifying

    return min(score, 100.0)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class StockScorer:
    def __init__(self, cfg: dict):
        self.weights = cfg["scoring"]["weights"]
        self.top_n = cfg["scoring"]["top_n"]

    def score_stock(self, df: pd.DataFrame) -> float:
        """Compute composite score for a single stock using its latest row."""
        if df.empty:
            return 0.0
        row = df.iloc[-1]

        t = _trend_score(row)
        m = _momentum_score(row)
        v = _volume_score(row)
        vl = _volatility_score(row)

        composite = (
            self.weights["trend"] * t
            + self.weights["momentum"] * m
            + self.weights["volume"] * v
            + self.weights["volatility"] * vl
        )
        return round(composite, 2)

    def rank_stocks(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> List[Tuple[str, float, dict]]:
        """
        Rank all stocks by composite score.

        Returns list of (symbol, score, sub_scores) sorted descending.
        """
        records = []
        for sym, df in data.items():
            if df.empty:
                continue
            row = df.iloc[-1]
            t = _trend_score(row)
            m = _momentum_score(row)
            v = _volume_score(row)
            vl = _volatility_score(row)
            composite = (
                self.weights["trend"] * t
                + self.weights["momentum"] * m
                + self.weights["volume"] * v
                + self.weights["volatility"] * vl
            )
            records.append((sym, round(composite, 2), {
                "trend": round(t, 1),
                "momentum": round(m, 1),
                "volume": round(v, 1),
                "volatility": round(vl, 1),
                "close": row["close"],
                "rsi": round(row.get("rsi", float("nan")), 1),
                "macd": round(row.get("macd", float("nan")), 4),
            }))

        records.sort(key=lambda x: x[1], reverse=True)
        return records

    def select_top_n(
        self,
        data: Dict[str, pd.DataFrame],
        exclude: set = None,
    ) -> List[Tuple[str, float, dict]]:
        """Return top-N stocks, optionally excluding symbols already held."""
        exclude = exclude or set()
        ranked = self.rank_stocks(data)
        filtered = [(s, sc, info) for s, sc, info in ranked if s not in exclude]
        return filtered[: self.top_n]
