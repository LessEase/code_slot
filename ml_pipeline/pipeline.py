"""Full ML pipeline orchestrator.

Steps (can be run individually or all at once):

  collect   → Download historical OHLCV + index data
  features  → Build feature matrices per stock
  samples   → Generate (X, y) training samples with forward-return labels
  train     → Walk-forward LightGBM training + final model
  backtest  → Simulate strategy on held-out historical period
  all       → Run collect → features → samples → train → backtest in sequence

Each step caches its output so subsequent runs only redo what changed.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from ml_pipeline.data.collector import HistoricalCollector
from ml_pipeline.features.engineer import build_feature_matrix, FEATURE_COLS
from ml_pipeline.samples.generator import SampleGenerator
from ml_pipeline.models.trainer import ModelTrainer
from ml_pipeline.models.registry import ModelRegistry
from ml_pipeline.backtest.engine import BacktestEngine
from stock_trading.utils.logger import get_logger

log = get_logger(__name__)
console = Console()

FEATURE_STORE_DIR = Path("data/features")
MARKETS = ["A", "US"]


# ── Step helpers ───────────────────────────────────────────────────────────

def step_collect(cfg: dict) -> None:
    """Download full historical OHLCV for universe + indices."""
    log.info("═══ Step 1/5: Data Collection ═══")
    collector = HistoricalCollector(cfg)
    collector.collect_all()


def step_features(cfg: dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Build feature matrices for all stocks.

    Returns {market: {symbol: feature_df}}
    Also persists per-stock features to FEATURE_STORE_DIR.
    """
    log.info("═══ Step 2/5: Feature Engineering ═══")
    collector = HistoricalCollector(cfg)
    result: Dict[str, Dict[str, pd.DataFrame]] = {}

    for market in MARKETS:
        symbols = collector.available_symbols(market)
        if not symbols:
            log.warning(f"No history data for market={market}, skipping")
            continue

        index_df = collector.load_index(market)
        market_features: Dict[str, pd.DataFrame] = {}
        out_dir = FEATURE_STORE_DIR / market
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Building features for {len(symbols)} {market} stocks …")
        skipped = 0
        for sym in symbols:
            feat_path = out_dir / f"{sym}.parquet"
            ohlcv = collector.load_ohlcv(sym, market)
            if ohlcv is None or len(ohlcv) < 80:
                skipped += 1
                continue
            try:
                feat = build_feature_matrix(ohlcv, index_df)
                if len(feat) >= 60:
                    market_features[sym] = feat
                    feat.to_parquet(feat_path, engine="pyarrow")
            except Exception as e:
                log.debug(f"Feature build failed for {sym}: {e}")
                skipped += 1

        log.info(f"{market}: {len(market_features)} stocks with features "
                 f"({skipped} skipped)")
        result[market] = market_features

    return result


def _load_features(market: str) -> Dict[str, pd.DataFrame]:
    """Load pre-computed feature matrices from disk."""
    feat_dir = FEATURE_STORE_DIR / market
    if not feat_dir.exists():
        return {}
    result = {}
    for path in feat_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(path, engine="pyarrow")
            if not df.empty:
                result[path.stem] = df
        except Exception:
            pass
    return result


def step_samples(cfg: dict) -> Dict[str, pd.DataFrame]:
    """Generate labeled training samples from feature matrices."""
    log.info("═══ Step 3/5: Sample Generation ═══")
    generator = SampleGenerator(cfg)
    result = {}
    for market in MARKETS:
        features = _load_features(market)
        if not features:
            log.warning(f"No features for {market}, run --features first")
            continue
        samples = generator.generate(features, market)
        if not samples.empty:
            result[market] = samples
    return result


def step_train(cfg: dict) -> None:
    """Train models with walk-forward CV and save to registry.

    The deployment model is trained only on samples up to
    ``max_date - backtest_holdout_months``; the tail is reserved as a
    strictly out-of-sample window for ``step_backtest``. This is the
    single most important change for honest backtest numbers.
    """
    log.info("═══ Step 4/5: Model Training ═══")
    generator = SampleGenerator(cfg)
    trainer = ModelTrainer(cfg)
    registry = ModelRegistry(cfg["ml_pipeline"].get("model_dir", "data/models"))
    holdout_months = int(cfg["ml_pipeline"].get("backtest_holdout_months", 12))

    for market in MARKETS:
        try:
            samples = generator.load(market)
        except FileNotFoundError:
            log.warning(f"No samples for {market}, skipping training")
            continue

        if len(samples) < 500:
            log.warning(f"Too few samples for {market} ({len(samples)}), skipping")
            continue

        if not isinstance(samples.index, pd.DatetimeIndex):
            samples = samples.set_index("date")

        # The label at row t uses close[t + forward_days]. To ensure NO data
        # from the OOS window leaks into training labels, we shift the cutoff
        # back by an extra forward_days padding (× 2 for calendar→trading
        # day buffer + weekend rollover).
        forward_days = int(cfg["ml_pipeline"]["forward_days"])
        label_pad = pd.Timedelta(days=forward_days * 2)
        train_end = samples.index.max() - pd.DateOffset(months=holdout_months) - label_pad

        log.info(
            f"Training {market} model on {len(samples):,} samples "
            f"(train ≤ {train_end.date()}, OOS holdout ≈ last {holdout_months}mo, "
            f"label_pad={forward_days*2}d) …"
        )
        fold_results, final_model, effective_train_end = trainer.walk_forward_cv(
            samples, train_end_date=train_end,
        )

        available_features = [c for c in FEATURE_COLS if c in samples.columns]
        fi = trainer.get_feature_importance(final_model, available_features)

        registry.save(
            model=final_model,
            market=market,
            feature_names=available_features,
            fold_results=fold_results,
            feature_importance=fi,
            train_end_date=effective_train_end.strftime("%Y-%m-%d"),
            cfg_snapshot={
                "forward_days": cfg["ml_pipeline"]["forward_days"],
                "return_threshold": cfg["ml_pipeline"]["return_threshold"],
                "train_years": cfg["ml_pipeline"]["train_years"],
                "test_months": cfg["ml_pipeline"]["test_months"],
                "backtest_holdout_months": holdout_months,
            },
        )
        _print_fold_summary(fold_results, market)


def _print_fold_summary(fold_results, market: str) -> None:
    table = Table(
        title=f"Walk-Forward CV Results — {market}",
        box=box.ROUNDED,
    )
    table.add_column("Period", style="dim")
    table.add_column("AUC", justify="right", style="cyan")
    table.add_column("AP", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Pos Rate", justify="right")
    table.add_column("N Train", justify="right")
    table.add_column("N Test", justify="right")
    table.add_column("Trees", justify="right")

    aucs = []
    for r in fold_results:
        aucs.append(r["auc"])
        auc_color = "green" if r["auc"] >= 0.55 else "red"
        table.add_row(
            r["period"],
            f"[{auc_color}]{r['auc']:.4f}[/]",
            f"{r['ap']:.4f}",
            f"{r['accuracy']:.4f}",
            f"{r['pos_rate_true']:.2%}",
            f"{r['n_train']:,}",
            f"{r['n_test']:,}",
            str(r.get("n_trees", "-")),
        )

    avg_auc = sum(aucs) / len(aucs) if aucs else 0
    table.add_row(
        "[bold]Average[/]",
        f"[bold cyan]{avg_auc:.4f}[/bold cyan]",
        "", "", "", "", "", "",
    )
    console.print(table)


def step_backtest(cfg: dict, market: Optional[str] = None) -> None:
    """Run backtest using trained model on historical data."""
    log.info("═══ Step 5/5: Backtest ═══")
    log.warning(
        "⚠ Survivorship bias: the universe is built from *currently-listed* tickers — "
        "names that delisted before today are absent, so backtest returns may be biased "
        "upward (especially for A-share full-market mode where ~5% of historical names "
        "are missing). Compare 'α vs Universe B&H' (printed below) against pure model α "
        "to gauge how much is alpha vs survivors-beta."
    )
    registry = ModelRegistry(cfg["ml_pipeline"].get("model_dir", "data/models"))
    collector = HistoricalCollector(cfg)

    markets_to_test = [market] if market else MARKETS

    for mkt in markets_to_test:
        try:
            model, meta = registry.load_latest(mkt)
        except FileNotFoundError:
            log.warning(f"No trained model for {mkt}, run --train first")
            continue

        features = _load_features(mkt)
        if not features:
            log.warning(f"No feature data for {mkt}")
            continue

        # Backtest strictly OUT of the training window. The model meta records
        # the last sample date the deployment model saw; we start the backtest
        # one day after it. Falling back to the legacy "recent N months" only
        # for older models that pre-date this change.
        last_feature_date = max(df.index.max() for df in features.values())
        end = pd.Timestamp(last_feature_date).strftime("%Y-%m-%d")
        train_end_str = meta.get("train_end_date")
        if train_end_str:
            start = (pd.Timestamp(train_end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            log.info(f"OOS backtest window for {mkt}: {start} → {end} "
                     f"(model train_end={train_end_str})")
        else:
            test_months = cfg["ml_pipeline"]["test_months"] * 4
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - timedelta(days=test_months * 30)).strftime("%Y-%m-%d")
            log.warning(
                f"Model meta has no train_end_date — falling back to legacy "
                f"window {start} → {end}. Re-run --train to get an honest OOS backtest."
            )

        market_map = {sym: mkt for sym in features}

        # Load benchmark
        index_df = collector.load_index(mkt)
        benchmark = None
        if index_df is not None and not index_df.empty:
            benchmark = index_df["close"].rename("benchmark")
            benchmark.index = pd.to_datetime(benchmark.index)

        engine = BacktestEngine(cfg, model, meta, features, market_map)
        result = engine.run(start, end, benchmark)
        engine.print_report(result, mkt)


def run_full_pipeline(cfg: dict) -> None:
    """Run all 5 steps in sequence."""
    step_collect(cfg)
    step_features(cfg)
    step_samples(cfg)
    step_train(cfg)
    step_backtest(cfg)
