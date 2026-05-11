"""Online predictor — replaces the rule-based StockScorer for live trading.

Loads the latest trained LightGBM model from the registry and scores
fresh OHLCV data fetched by the live DataFetcher.  The interface is
intentionally compatible with the existing TradingSimulator so that
swapping rule-based for ML-based scoring requires minimal changes.

Usage
─────
    predictor = MLPredictor(cfg)
    scores = predictor.score_universe(raw_ohlcv_dict)  # {sym: prob}
    top = predictor.select_top_n(scores, exclude=held_symbols)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_pipeline._lgbm import lgb
from ml_pipeline.data.collector import HistoricalCollector
from ml_pipeline.features.engineer import build_feature_matrix, FEATURE_COLS
from ml_pipeline.models.registry import ModelRegistry
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


class MLPredictor:
    """
    Scores stocks using a trained LightGBM model.

    Parameters
    ----------
    cfg : full config dict
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.registry = ModelRegistry(cfg["ml_pipeline"].get("model_dir", "data/models"))
        self.top_n = cfg["ml_pipeline"]["top_n"]

        # Cache loaded models per market to avoid repeated disk reads
        self._models: Dict[str, Tuple[lgb.Booster, List[str]]] = {}

        # Index data cache
        self._index_cache: Dict[str, Optional[pd.DataFrame]] = {}

    def _get_model(self, market: str) -> Optional[Tuple[lgb.Booster, List[str]]]:
        """Lazy-load model for a given market."""
        if market not in self._models:
            try:
                model, meta = self.registry.load_latest(market)
                self._models[market] = (model, meta["feature_names"])
                log.info(f"Loaded {market} model: {meta['n_trees']} trees")
            except FileNotFoundError:
                log.warning(
                    f"No trained model for market={market}. "
                    f"Falling back to rule-based scoring."
                )
                return None
        return self._models[market]

    def _get_index(self, market: str) -> Optional[pd.DataFrame]:
        """Load market index OHLCV (cached per process)."""
        if market not in self._index_cache:
            collector = HistoricalCollector(self.cfg)
            self._index_cache[market] = collector.load_index(market)
        return self._index_cache[market]

    def _score_one(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        market: str,
        model: lgb.Booster,
        feature_names: List[str],
    ) -> Optional[float]:
        """
        Compute model probability for a single stock.

        Uses only the latest row of features (no look-ahead).
        Returns None if feature computation fails.
        """
        try:
            index_df = self._get_index(market)
            feat = build_feature_matrix(ohlcv, index_df)
            if feat.empty:
                return None

            last_row = feat.iloc[-1]
            avail = [c for c in feature_names if c in last_row.index]
            if len(avail) < len(feature_names) * 0.6:
                return None

            x = pd.DataFrame([last_row[avail].reindex(feature_names).fillna(0)])
            prob = float(model.predict(x)[0])
            return prob
        except Exception as e:
            log.debug(f"Score failed for {symbol}: {e}")
            return None

    def score_universe(
        self,
        raw_ohlcv: Dict[str, pd.DataFrame],
        market_map: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Score all stocks and return {symbol: probability}.

        Stocks whose market has no trained model are skipped.
        """
        scores: Dict[str, float] = {}
        model_cache: Dict[str, Optional[Tuple]] = {}

        for sym, df in raw_ohlcv.items():
            market = market_map.get(sym, "A")
            if market not in model_cache:
                model_cache[market] = self._get_model(market)
            result = model_cache[market]
            if result is None:
                continue
            model, feature_names = result
            prob = self._score_one(sym, df, market, model, feature_names)
            if prob is not None:
                scores[sym] = round(prob, 4)

        log.info(f"Scored {len(scores)}/{len(raw_ohlcv)} stocks")
        return scores

    def select_top_n(
        self,
        scores: Dict[str, float],
        exclude: Optional[set] = None,
        n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return [(symbol, probability)] sorted by probability descending.

        Parameters
        ----------
        scores  : {symbol: probability} from score_universe()
        exclude : symbols to skip (e.g., already held)
        n       : override top-N (defaults to config value)
        """
        exclude = exclude or set()
        n = n or self.top_n
        filtered = [(s, p) for s, p in scores.items() if s not in exclude]
        filtered.sort(key=lambda x: -x[1])
        return filtered[:n]

    def reload_models(self) -> None:
        """Force reload of all cached models (call after retraining)."""
        self._models.clear()
        self._index_cache.clear()
        log.info("Model cache cleared — will reload on next scoring call")
