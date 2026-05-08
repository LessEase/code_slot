"""Training sample generation with strict no-look-ahead guarantee.

For every (symbol, date=t) pair we produce:
  X  =  feature vector computed from data up to and including day t
  y  =  1  if  close(t + forward_days) / close(t) - 1  >  threshold
         0  otherwise

The label computation shifts the close series forward, so all label
information lives strictly in the future relative to the feature date.

Output
──────
  samples.parquet   – columns: FEATURE_COLS + ['symbol', 'market', 'label']
                       index = (date, symbol) MultiIndex
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ml_pipeline.features.engineer import FEATURE_COLS
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

SAMPLE_FILE = "data/ml/samples_{market}.parquet"


def make_labels(
    feat: pd.DataFrame,
    forward_days: int,
    threshold: float,
) -> pd.Series:
    """
    Create binary label: 1 if forward return exceeds threshold.

    The label at row t uses close(t + forward_days), so it must be
    *dropped* for any date where that future price is unavailable.
    """
    close = feat["close"]
    future_close = close.shift(-forward_days)           # look-forward shift
    fwd_return = future_close / close - 1.0
    label = (fwd_return > threshold).astype(int)
    # Rows where future_close is NaN have an invalid label → mark as NaN
    label = label.where(future_close.notna())
    return label


def build_sample_df(
    feat: pd.DataFrame,
    symbol: str,
    market: str,
    forward_days: int,
    threshold: float,
) -> Optional[pd.DataFrame]:
    """
    Build (X, y) rows for a single stock.

    Returns a DataFrame with FEATURE_COLS + ['symbol', 'market', 'label'],
    indexed by date.  Rows with NaN label (last forward_days rows) are dropped.
    """
    if feat.empty:
        return None

    available = [c for c in FEATURE_COLS if c in feat.columns]
    if len(available) < len(FEATURE_COLS) * 0.7:   # require ≥70% of features
        return None

    label = make_labels(feat, forward_days, threshold)
    df = feat[available].copy()
    df["label"] = label
    df["symbol"] = symbol
    df["market"] = market

    # Drop rows with NaN label or any NaN in feature columns
    df = df.dropna(subset=["label"] + available)
    df["label"] = df["label"].astype(int)

    return df if not df.empty else None


class SampleGenerator:
    def __init__(self, cfg: dict):
        ml = cfg["ml_pipeline"]
        self.forward_days = ml["forward_days"]
        self.threshold = ml["return_threshold"]
        self.output_dir = Path("data/ml")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        feature_data: Dict[str, pd.DataFrame],   # {symbol: feature_df}
        market: str,
    ) -> pd.DataFrame:
        """
        Generate samples for all stocks in `feature_data`.

        Returns the combined sample DataFrame and also persists it to disk.
        """
        parts: List[pd.DataFrame] = []
        skipped = 0
        for sym, feat in feature_data.items():
            df = build_sample_df(feat, sym, market, self.forward_days, self.threshold)
            if df is not None:
                parts.append(df)
            else:
                skipped += 1

        if not parts:
            log.warning(f"No samples generated for market={market}")
            return pd.DataFrame()

        samples = pd.concat(parts, axis=0).sort_index()
        n_pos = (samples["label"] == 1).sum()
        n_neg = (samples["label"] == 0).sum()
        pos_rate = n_pos / len(samples) * 100

        log.info(
            f"Samples [{market}]: {len(samples):,} rows  "
            f"pos={n_pos:,} ({pos_rate:.1f}%)  neg={n_neg:,}  "
            f"skipped={skipped} stocks"
        )

        path = self.output_dir / f"samples_{market}.parquet"
        samples.to_parquet(path, engine="pyarrow")
        log.info(f"Saved samples → {path}")
        return samples

    def load(self, market: str) -> pd.DataFrame:
        path = self.output_dir / f"samples_{market}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No samples file for market={market}. Run generate first.")
        return pd.read_parquet(path, engine="pyarrow")

    # ── Train / test split ────────────────────────────────────────────────

    @staticmethod
    def temporal_split(
        samples: pd.DataFrame,
        test_start: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Hard temporal split: everything before test_start is train.

        Never uses random shuffling to prevent look-ahead bias.
        """
        samples = samples.copy()
        # Ensure the index is a DatetimeIndex or has a usable date column
        if not isinstance(samples.index, pd.DatetimeIndex):
            if "date" in samples.columns:
                samples = samples.set_index("date")
            else:
                raise ValueError("samples must have DatetimeIndex or 'date' column")

        cut = pd.Timestamp(test_start)
        train = samples[samples.index < cut]
        test = samples[samples.index >= cut]
        log.info(
            f"Temporal split at {test_start}: "
            f"train={len(train):,}  test={len(test):,}"
        )
        return train, test

    @staticmethod
    def walk_forward_splits(
        samples: pd.DataFrame,
        train_years: float,
        test_months: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """
        Generate expanding-window walk-forward splits.

        Yields (train_df, test_df, period_label) for each fold.
        The train window expands; the test window slides forward.
        """
        if not isinstance(samples.index, pd.DatetimeIndex):
            if "date" in samples.columns:
                samples = samples.set_index("date")

        dates = samples.index.unique().sort_values()
        start = dates[0]
        end = dates[-1]

        splits = []
        test_start = start + pd.DateOffset(years=int(train_years),
                                           months=int((train_years % 1) * 12))

        while test_start < end:
            test_end = test_start + pd.DateOffset(months=test_months)
            train = samples[samples.index < test_start]
            test = samples[(samples.index >= test_start) & (samples.index < test_end)]

            if len(train) >= 500 and len(test) >= 50:
                label = f"{test_start.strftime('%Y-%m')}~{min(test_end, end).strftime('%Y-%m')}"
                splits.append((train, test, label))

            test_start = test_end

        log.info(f"Walk-forward splits: {len(splits)} folds")
        return splits
