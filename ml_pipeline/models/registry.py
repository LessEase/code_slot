"""Model versioning and persistence.

Models are stored under data/models/{market}/{timestamp}/.
  model.lgb         – LightGBM booster binary
  meta.json         – feature names, config snapshot, training metrics
  feature_importance.csv

The 'latest' symlink always points to the most recently trained model.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import pandas as pd

from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


class ModelRegistry:
    def __init__(self, base_dir: str = "data/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _market_dir(self, market: str) -> Path:
        d = self.base_dir / market
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Save ──────────────────────────────────────────────────────────────

    def save(
        self,
        model: lgb.Booster,
        market: str,
        feature_names: List[str],
        fold_results: List[Dict],
        feature_importance: Optional[pd.DataFrame] = None,
        cfg_snapshot: Optional[dict] = None,
    ) -> Path:
        """Persist a trained model and its metadata."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self._market_dir(market) / ts
        model_dir.mkdir(parents=True, exist_ok=True)

        # LightGBM booster
        model_path = model_dir / "model.lgb"
        model.save_model(str(model_path))

        # Metadata
        meta = {
            "market": market,
            "timestamp": ts,
            "feature_names": feature_names,
            "n_trees": model.num_trees(),
            "fold_results": fold_results,
            "config": cfg_snapshot or {},
        }
        with open(model_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Feature importance
        if feature_importance is not None:
            feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

        # Update 'latest' pointer
        latest_link = self._market_dir(market) / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ts)

        log.info(f"Model saved → {model_dir}")
        return model_dir

    # ── Load ──────────────────────────────────────────────────────────────

    def load_latest(self, market: str) -> Tuple[lgb.Booster, Dict]:
        """Load the most recently saved model for a market."""
        latest_link = self._market_dir(market) / "latest"
        if not latest_link.exists():
            raise FileNotFoundError(
                f"No trained model found for market={market}. "
                f"Run 'python main.py --train' first."
            )
        model_dir = latest_link.resolve()
        return self._load_from(model_dir)

    def load_version(self, market: str, timestamp: str) -> Tuple[lgb.Booster, Dict]:
        """Load a specific model version by timestamp string."""
        model_dir = self._market_dir(market) / timestamp
        if not model_dir.exists():
            raise FileNotFoundError(f"Model version {timestamp} not found for {market}")
        return self._load_from(model_dir)

    def _load_from(self, model_dir: Path) -> Tuple[lgb.Booster, Dict]:
        model = lgb.Booster(model_file=str(model_dir / "model.lgb"))
        with open(model_dir / "meta.json") as f:
            meta = json.load(f)
        log.info(f"Loaded model from {model_dir}  (trees={meta['n_trees']})")
        return model, meta

    # ── Listing ───────────────────────────────────────────────────────────

    def list_versions(self, market: str) -> List[str]:
        """Return all saved timestamps for a market, newest first."""
        market_dir = self._market_dir(market)
        versions = sorted(
            [p.name for p in market_dir.iterdir()
             if p.is_dir() and p.name != "latest"],
            reverse=True,
        )
        return versions

    def latest_meta(self, market: str) -> Optional[Dict]:
        """Return metadata of the latest model without loading the booster."""
        try:
            latest_link = self._market_dir(market) / "latest"
            if not latest_link.exists():
                return None
            model_dir = latest_link.resolve()
            with open(model_dir / "meta.json") as f:
                return json.load(f)
        except Exception:
            return None
