"""Training sample generation with strict no-look-ahead guarantee.

Two label modes are supported via ``ml_pipeline.label_mode``:

  "absolute"               – y = 1 if  fwd_return  > return_threshold
  "cross_section_quantile" – y = 1 if  fwd_return  is in the top
                             (1 - cross_section_quantile) of the *universe*
                             on the same date

Cross-sectional mode is strongly preferred for stock-picking: it removes the
beta dependence (in a bull market most "absolute" labels are 1 and the model
loses discriminative power) and forces the model to learn *relative* signal.

Optional industry neutralization (``ml_pipeline.neutralize_by_industry``)
replaces every feature value with its z-score within the (date, industry)
bucket. The model therefore learns within-industry relative signal rather
than betting on hot industries — directly targeting the "α vs Universe B&H
is large negative" symptom we saw in the long-only top-N backtest.

All transformations are point-in-time: forward returns are computed via a
strictly negative shift (so labels live in the future relative to features)
and z-scores use only same-day cross-section information.

Output
──────
  samples.parquet   – columns: FEATURE_COLS + ['symbol', 'market', 'label',
                       'fwd_return', 'industry'],  index = date
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ml_pipeline.features.engineer import FEATURE_COLS
from ml_pipeline.data.industry import UNKNOWN_INDUSTRY
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)

SAMPLE_FILE = "data/ml/samples_{market}.parquet"
LABEL_ABSOLUTE = "absolute"
LABEL_CROSS_SECTION_QUANTILE = "cross_section_quantile"


# ── Per-stock raw row builder ──────────────────────────────────────────────

def _build_stock_rows(
    feat: pd.DataFrame,
    symbol: str,
    market: str,
    forward_days: int,
    min_avg_turnover: float,
) -> Optional[pd.DataFrame]:
    """Build per-date rows for a single stock with forward return.

    Returns DataFrame indexed by date with columns:
      FEATURE_COLS (available) + ['symbol', 'market', 'fwd_return']
    No labelling and no neutralization happens here — those are panel-level
    operations that need cross-sectional context.
    """
    if feat.empty:
        return None

    available = [c for c in FEATURE_COLS if c in feat.columns]
    if len(available) < len(FEATURE_COLS) * 0.7:
        return None

    close = feat["close"]
    future_close = close.shift(-forward_days)
    fwd_return = future_close / close - 1.0          # NaN for last `forward_days` rows

    df = feat[available].copy()
    df["fwd_return"] = fwd_return
    df["symbol"] = symbol
    df["market"] = market

    # Liquidity gate (A-share only)
    if min_avg_turnover > 0 and market == "A" and "amount_20d_avg" in feat.columns:
        liquid = feat["amount_20d_avg"].reindex(df.index)
        df = df[liquid.fillna(0.0) >= min_avg_turnover]

    # Drop rows with NaN label *target* or any NaN feature
    df = df.dropna(subset=["fwd_return"] + available)

    return df if not df.empty else None


# ── Panel transformations (label + neutralization) ─────────────────────────

def _label_absolute(samples: pd.DataFrame, threshold: float) -> pd.Series:
    return (samples["fwd_return"] > threshold).astype(int)


def _label_cross_section_quantile(
    samples: pd.DataFrame, quantile: float
) -> pd.Series:
    """Per-date label: 1 iff fwd_return is in the top (1-q) of the universe.

    Uses ``rank(pct=True)`` so the threshold is the *percentile rank* within
    each cross-section, which gives a near-constant positive rate of (1-q)
    regardless of market regime (no more "everyone wins" in a bull market).
    """
    if not isinstance(samples.index, pd.DatetimeIndex):
        raise ValueError("samples must be indexed by date for cross-sectional label")
    # rank(pct=True) gives values in (0, 1]; >= q matches top (1-q) fraction.
    pct_rank = samples.groupby(level=0)["fwd_return"].rank(pct=True, method="average")
    return (pct_rank >= quantile).astype(int)


def neutralize_by_industry(
    samples: pd.DataFrame,
    feature_cols: List[str],
    min_group_size: int = 5,
) -> pd.DataFrame:
    """Replace each feature with its z-score within (date, industry) bucket.

    Groups smaller than ``min_group_size`` (e.g. UNKNOWN industry, or a
    sparse industry on a given day) are passed through with z = 0 to avoid
    degenerate normalization.

    Returns a new DataFrame; does not mutate `samples`.
    """
    out = samples.copy()
    if "industry" not in out.columns:
        log.warning("neutralize_by_industry: no 'industry' column, skipping")
        return out

    # Use the date index + industry as a grouping key
    if isinstance(out.index, pd.DatetimeIndex):
        date_key = out.index
    else:
        raise ValueError("samples must be DatetimeIndex-ed for industry neutralization")

    grouper = pd.MultiIndex.from_arrays(
        [date_key, out["industry"].values], names=["__date", "__industry"]
    )

    # For groups too small to z-score meaningfully, zero them out post-hoc
    sizes = out.groupby(grouper).size()
    valid_groups = sizes[sizes >= min_group_size].index

    grouped = out[feature_cols].groupby(grouper, sort=False)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    z = (out[feature_cols] - means) / stds

    # Zero out tiny groups (z would be NaN due to single-row std)
    is_valid_group = pd.MultiIndex.from_arrays(
        [date_key, out["industry"].values]
    ).isin(valid_groups)
    z = z.where(pd.Series(is_valid_group, index=out.index), 0.0)
    z = z.fillna(0.0).clip(-5, 5)

    out[feature_cols] = z
    return out


# ── Generator class ────────────────────────────────────────────────────────

class SampleGenerator:
    def __init__(self, cfg: dict):
        ml = cfg["ml_pipeline"]
        self.forward_days = int(ml["forward_days"])
        self.threshold = float(ml.get("return_threshold", 0.0))
        self.min_avg_turnover = float(ml.get("min_avg_turnover_cny", 0.0) or 0.0)
        self.label_mode = str(ml.get("label_mode", LABEL_ABSOLUTE))
        self.cross_section_quantile = float(ml.get("cross_section_quantile", 0.90))
        self.neutralize = bool(ml.get("neutralize_by_industry", False))
        self.output_dir = Path("data/ml")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        feature_data: Dict[str, pd.DataFrame],   # {symbol: feature_df}
        market: str,
        industry_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Build panel of labelled samples for ``market``.

        Stages:
          1. Per-stock: assemble features + forward_return rows
          2. Panel concat
          3. Attach industry (UNKNOWN if missing)
          4. Apply label (absolute or cross-section quantile)
          5. Optionally industry-neutralize features (z-score per date×industry)
        """
        parts: List[pd.DataFrame] = []
        skipped = 0
        for sym, feat in feature_data.items():
            df = _build_stock_rows(
                feat, sym, market, self.forward_days, self.min_avg_turnover,
            )
            if df is not None:
                parts.append(df)
            else:
                skipped += 1

        if not parts:
            log.warning(f"No samples generated for market={market}")
            return pd.DataFrame()

        samples = pd.concat(parts, axis=0).sort_index()

        # Industry tag — only A-share has classification here; other markets
        # collapse to a single bucket so neutralization (if enabled) is a no-op.
        if market == "A" and industry_map:
            samples["industry"] = (
                samples["symbol"].map(industry_map).fillna(UNKNOWN_INDUSTRY)
            )
        else:
            samples["industry"] = UNKNOWN_INDUSTRY

        # Label
        if self.label_mode == LABEL_CROSS_SECTION_QUANTILE:
            samples["label"] = _label_cross_section_quantile(
                samples, self.cross_section_quantile
            )
            label_desc = (
                f"cross_section_quantile (top {(1 - self.cross_section_quantile):.0%})"
            )
        else:
            samples["label"] = _label_absolute(samples, self.threshold)
            label_desc = f"absolute (fwd > {self.threshold:.2%})"

        # Industry neutralization (z-score features within date × industry)
        if self.neutralize and market == "A":
            available_features = [c for c in FEATURE_COLS if c in samples.columns]
            samples = neutralize_by_industry(samples, available_features)

        n_pos = int((samples["label"] == 1).sum())
        n_neg = int((samples["label"] == 0).sum())
        pos_rate = n_pos / len(samples) * 100 if len(samples) else 0.0
        n_industries = samples["industry"].nunique()
        log.info(
            f"Samples [{market}]: {len(samples):,} rows  "
            f"pos={n_pos:,} ({pos_rate:.1f}%)  neg={n_neg:,}  "
            f"skipped={skipped} stocks  label={label_desc}  "
            f"industries={n_industries}  "
            f"neutralized={'yes' if self.neutralize and market == 'A' else 'no'}"
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
        """Hard temporal split; never random-shuffles to prevent look-ahead bias."""
        samples = samples.copy()
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
        """Expanding-window walk-forward splits."""
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
