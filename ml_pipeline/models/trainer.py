"""LightGBM model trainer with walk-forward cross-validation.

Training philosophy
───────────────────
- Binary classification: predict P(stock rises > threshold in N days)
- Walk-forward folds (expanding train window, sliding test window)
- Each fold produces one trained model; the *final* model is trained on
  the complete dataset and used for live scoring
- Feature importance is tracked across folds

No data leakage: feature computation and label generation are done
upstream; this module only consumes the (X, y) samples.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ml_pipeline._lgbm import lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    log_loss, accuracy_score,
)
from sklearn.preprocessing import label_binarize

from ml_pipeline.features.engineer import FEATURE_COLS
from ml_pipeline.samples.generator import SampleGenerator
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)


def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract (X, y) from a sample DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available], df["label"].astype(int)


def _lgbm_params(cfg: dict) -> dict:
    p = cfg["ml_pipeline"]["lgbm"]
    return {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": -1,
        "learning_rate": p.get("learning_rate", 0.05),
        "num_leaves": p.get("num_leaves", 31),
        "max_depth": p.get("max_depth", 6),
        "min_child_samples": p.get("min_child_samples", 50),
        "subsample": p.get("subsample", 0.8),
        "subsample_freq": 1,
        "colsample_bytree": p.get("colsample_bytree", 0.8),
        "reg_alpha": p.get("reg_alpha", 0.1),
        "reg_lambda": p.get("reg_lambda", 0.1),
    }


def _eval_fold(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute evaluation metrics for a single fold."""
    prob = model.predict(X_test)
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "auc": round(roc_auc_score(y_test, prob), 4),
        "ap": round(average_precision_score(y_test, prob), 4),
        "logloss": round(log_loss(y_test, prob), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "pos_rate_true": round(y_test.mean(), 4),
        "pos_rate_pred": round(pred.mean(), 4),
        "n_test": len(y_test),
    }
    return metrics


class ModelTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lgbm_params = _lgbm_params(cfg)
        self.n_estimators = cfg["ml_pipeline"]["lgbm"].get("n_estimators", 500)
        self.early_stop = cfg["ml_pipeline"]["lgbm"].get("early_stopping_rounds", 50)
        self.train_years = cfg["ml_pipeline"]["train_years"]
        self.test_months = cfg["ml_pipeline"]["test_months"]

    # ── Single-fold training ───────────────────────────────────────────────

    def _train_one(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> lgb.Booster:
        """Train one LightGBM model with optional early stopping."""
        dtrain = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.log_evaluation(period=-1)]   # suppress per-iter logs

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            callbacks.append(lgb.early_stopping(self.early_stop, verbose=False))
            model = lgb.train(
                self.lgbm_params,
                dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=[dval],
                callbacks=callbacks,
            )
        else:
            model = lgb.train(
                self.lgbm_params,
                dtrain,
                num_boost_round=self.n_estimators,
                callbacks=callbacks,
            )
        return model

    # ── Walk-forward cross-validation ─────────────────────────────────────

    def walk_forward_cv(
        self,
        samples: pd.DataFrame,
    ) -> Tuple[List[Dict], lgb.Booster]:
        """
        Walk-forward cross-validation.

        Returns
        -------
        fold_results : list of per-fold metric dicts
        final_model  : model trained on ALL data (for deployment)
        """
        splits = SampleGenerator.walk_forward_splits(
            samples, self.train_years, self.test_months
        )
        fold_results = []

        for i, (train_df, test_df, label) in enumerate(splits, 1):
            X_tr, y_tr = _xy(train_df)
            X_te, y_te = _xy(test_df)

            # Use last 20% of train as internal validation for early stopping
            val_cut = int(len(X_tr) * 0.8)
            X_val_inner = X_tr.iloc[val_cut:]
            y_val_inner = y_tr.iloc[val_cut:]
            X_tr_inner = X_tr.iloc[:val_cut]
            y_tr_inner = y_tr.iloc[:val_cut]

            model = self._train_one(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner)
            metrics = _eval_fold(model, X_te, y_te)
            metrics["period"] = label
            metrics["n_train"] = len(X_tr)
            metrics["n_trees"] = model.num_trees()
            fold_results.append(metrics)

            log.info(
                f"Fold {i}/{len(splits)} [{label}]  "
                f"AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
                f"Acc={metrics['accuracy']:.4f}  "
                f"trees={metrics['n_trees']}  "
                f"n_train={metrics['n_train']:,}  n_test={metrics['n_test']:,}"
            )

        # Final model: train on all data (no early stopping)
        log.info("Training final model on full dataset …")
        X_all, y_all = _xy(samples)
        final_model = self._train_one(X_all, y_all)
        log.info(f"Final model: {final_model.num_trees()} trees, "
                 f"{len(X_all):,} training samples")

        self._log_feature_importance(final_model, X_all.columns.tolist())
        return fold_results, final_model

    # ── Feature importance ─────────────────────────────────────────────────

    def _log_feature_importance(
        self,
        model: lgb.Booster,
        feature_names: List[str],
    ) -> None:
        importance = model.feature_importance(importance_type="gain")
        fi = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        log.info("Top-15 features by gain:")
        for name, score in fi[:15]:
            bar = "█" * int(score / max(fi[0][1], 1e-9) * 20)
            log.info(f"  {name:25s} {score:>10.1f}  {bar}")

    def get_feature_importance(
        self,
        model: lgb.Booster,
        feature_names: List[str],
    ) -> pd.DataFrame:
        gain = model.feature_importance(importance_type="gain")
        split = model.feature_importance(importance_type="split")
        df = pd.DataFrame({
            "feature": feature_names,
            "gain": gain,
            "split": split,
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        df["gain_pct"] = df["gain"] / df["gain"].sum() * 100
        return df
