#!/usr/bin/env python3
"""End-to-end IPCA training pipeline using real (or cached) Binance data.

This script orchestrates four steps:

1. Collect OHLCV candles from the Binance public dataset (or user supplied
   parquet files when offline).
2. Derive daily returns and basic characteristics required by the
   :class:`CryptoIPCAModel`.
3. Fit the Instrumented PCA model using the production preprocessor.
4. Persist the trained artefacts (model, gamma loadings, factor time-series)
   and, optionally, the prepared panel that was used for training.

Example
-------
.. code-block:: bash

    PYTHONPATH=. ./crypto-dlsa-env/bin/python scripts/run_real_ipca_training.py \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT \
        --start 2024-01-01 --end 2024-06-30 \
        --output-dir models/ipca_20240630

When network access is unavailable, pass one or more parquet files that follow
the schema ``timestamp,symbol,open,high,low,close,volume`` via
``--input-parquet`` to reuse previously downloaded data.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel
from crypto_dlsa_bot.ml.residual_calculator import ResidualCalculator
from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT,XRPUSDT,DOTUSDT,MATICUSDT",
        help="Comma separated Binance tickers (default: top majors including BTC/ETH/BNB/ADA/SOL/XRP/DOT/MATIC)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Kline timeframe such as 1d or 4h (default: 1d)",
    )
    parser.add_argument(
        "--market-type",
        type=str,
        choices=["spot", "futures"],
        default="spot",
        help="Binance market to query (default: spot)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        nargs="+",
        help="Optional parquet files to reuse instead of downloading",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=3,
        help="Number of latent factors (default: 3)",
    )
    parser.add_argument(
        "--intercept",
        action="store_true",
        help="Include intercept when fitting the IPCA model",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Elastic-net regularisation strength (default: 0.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum IPCA iterations (default: 1000)",
    )
    parser.add_argument(
        "--iter-tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance for IPCA (default: 1e-5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/ipca_latest"),
        help="Directory to store model artefacts (default: models/ipca_latest)",
    )
    parser.add_argument(
        "--save-returns",
        action="store_true",
        help="Persist the prepared returns panel as parquet",
    )
    parser.add_argument(
        "--oos-backtest",
        action="store_true",
        help="Run rolling out-of-sample backtest using predictOOS",
    )
    parser.add_argument(
        "--oos-warmup",
        type=int,
        default=60,
        help="Warm-up periods (days) before starting OOS backtest (default: 60)",
    )
    parser.add_argument(
        "--oos-train-window",
        type=int,
        default=None,
        help="When set, limit each OOS training to the last N days up to prediction date (default: use all history)",
    )
    parser.add_argument(
        "--oos-refit-frequency",
        type=int,
        default=1,
        help="Refit model every k prediction steps during OOS; when k>1, reuse last fitted model between refits (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _parse_timestamps(symbols: str, start: str, end: str) -> tuple[List[str], datetime, datetime]:
    tickers = [sym.strip().upper() for sym in symbols.split(",") if sym.strip()]
    if not tickers:
        raise ValueError("At least one symbol must be provided")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if start_dt >= end_dt:
        raise ValueError("Start date must be strictly earlier than end date")
    return tickers, start_dt, end_dt


def load_from_parquet(paths: Iterable[Path], symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            logger.warning("Parquet file %s does not exist", path)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("Failed to read %s: %s", path, exc)
            continue

        required_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            logger.warning("Parquet file %s missing required columns; skipping", path)
            continue

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df[(df["symbol"].isin(symbols)) & (df["timestamp"] >= start) & (df["timestamp"] <= end)]
        if df.empty:
            logger.info("No rows retained from %s after filtering", path)
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"])
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    return combined


def collect_ohlcv(args: argparse.Namespace) -> pd.DataFrame:
    symbols, start, end = _parse_timestamps(args.symbols, args.start, args.end)

    if args.input_parquet:
        df = load_from_parquet(args.input_parquet, symbols, start, end)
        if not df.empty:
            logger.info(
                "Loaded %d rows from parquet spanning %s -> %s",
                len(df),
                df["timestamp"].min(),
                df["timestamp"].max(),
            )
            return df
        logger.warning("Parquet inputs yielded no data; attempting live collection")

    collector = BinanceDataCollector(
        data_dir=str(Path("data/raw/binance_cached")),
        use_public_data=True,
    )
    logger.info("Collecting OHLCV for %s", ",".join(symbols))
    df = collector.collect_ohlcv(
        symbols=symbols,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
        market_type=args.market_type,
    )
    if df.empty:
        raise RuntimeError("Collector returned no OHLCV data; provide parquet input via --input-parquet")

    logger.info(
        "Collected %d rows across %d symbols (%s -> %s)",
        len(df),
        df["symbol"].nunique(),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    return df


def prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])
    df["return"] = df.groupby("symbol")["close"].pct_change()
    df = df.dropna(subset=["return"])
    # guard against obvious bad ticks
    df = df[df["return"].abs() < 0.5]
    logger.info(
        "Prepared %d return observations across %d assets",
        len(df),
        df["symbol"].nunique(),
    )
    return df[["symbol", "date", "return", "close", "volume"]]


def instantiate_model(args: argparse.Namespace) -> CryptoIPCAModel:
    return CryptoIPCAModel(
        n_factors=max(1, args.n_factors),
        intercept=args.intercept,
        max_iter=args.max_iter,
        iter_tol=args.iter_tol,
        alpha=args.alpha,
    )


def fit_model(panel: pd.DataFrame, args: argparse.Namespace) -> CryptoIPCAModel:
    model = instantiate_model(args)
    model.fit(panel)
    logger.info(
        "Model fitted: %d factors, gamma %s, factors %s",
        model.n_factors,
        model.gamma.shape if model.gamma is not None else None,
        model.factors.shape if model.factors is not None else None,
    )
    return model


def _to_serializable(value):
    """Convert numpy/pandas scalars to JSON serialisable objects."""

    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    return value


def analyse_market_structure(panel: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    """Analyse BTC dominance and BTC-vs-alt correlations."""

    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, float] = {}

    if panel.empty:
        return results

    panel = panel.copy()
    panel['date'] = pd.to_datetime(panel['date'])

    # --- BTC dominance (proxy using close * volume) ---
    panel['market_cap_proxy'] = panel['close'] * panel['volume']
    market_caps = panel.pivot_table(
        index='date',
        columns='symbol',
        values='market_cap_proxy',
        aggfunc='sum'
    ).sort_index()

    btc_symbol = next((col for col in market_caps.columns if col.startswith('BTC')), None)
    if btc_symbol is None:
        return results

    total_caps = market_caps.sum(axis=1)
    btc_caps = market_caps[btc_symbol]
    dominance = (btc_caps / total_caps).replace([np.inf, -np.inf], np.nan).dropna()

    dominance_path = output_dir / 'btc_dominance.csv'
    dominance.to_csv(dominance_path, header=['btc_dominance'])

    if not dominance.empty:
        results['btc_dominance_mean'] = float(dominance.mean())
        results['btc_dominance_median'] = float(dominance.median())
        results['btc_dominance_last'] = float(dominance.iloc[-1])
        results['btc_dominance_change'] = float(dominance.iloc[-1] - dominance.iloc[0])

    # --- BTC vs Alt correlations ---
    returns_pivot = panel.pivot_table(
        index='date',
        columns='symbol',
        values='return',
        aggfunc='mean'
    ).sort_index()

    if btc_symbol not in returns_pivot.columns:
        return results

    alt_symbols = [sym for sym in returns_pivot.columns if sym != btc_symbol]
    if not alt_symbols:
        return results

    btc_returns = returns_pivot[btc_symbol]
    alt_mean_returns = returns_pivot[alt_symbols].mean(axis=1)
    combined = pd.DataFrame({
        'btc_return': btc_returns,
        'alt_return': alt_mean_returns,
    }).dropna()

    if combined.empty:
        return results

    overall_corr = combined['btc_return'].corr(combined['alt_return'])
    results['btc_alt_correlation'] = float(overall_corr)

    rolling_window = min(14, max(5, len(combined) // 4))
    rolling_corr = combined['btc_return'].rolling(rolling_window, min_periods=5).corr(combined['alt_return'])
    rolling_corr = rolling_corr.dropna()

    corr_path = output_dir / 'btc_alt_rolling_correlation.csv'
    if not rolling_corr.empty:
        rolling_corr.to_csv(corr_path, header=['rolling_corr'])
        results['btc_alt_correlation_recent'] = float(rolling_corr.iloc[-1])
        results['btc_alt_correlation_min'] = float(rolling_corr.min())
        results['btc_alt_correlation_max'] = float(rolling_corr.max())

    # Alt-season heuristic: dominance trending down and correlation weakening
    if results.get('btc_dominance_change') is not None and results.get('btc_alt_correlation_recent') is not None:
        dominance_down = results['btc_dominance_change'] < 0
        corr_soft = results['btc_alt_correlation_recent'] < 0.5
        results['alt_season_signal'] = bool(dominance_down and corr_soft)

    structure_json = output_dir / 'market_structure_metrics.json'
    with structure_json.open('w') as fh:
        json.dump(_to_serializable(results), fh, indent=2)

    return results


def generate_symbol_residual_plots(residuals: pd.DataFrame, output_dir: Path) -> None:
    """Generate per-symbol residual plots and summary statistics."""

    if residuals.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = residuals.copy()
    residuals['date'] = pd.to_datetime(residuals['date'])

    stats_records = []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for symbol, group in residuals.groupby('symbol'):
        group = group.sort_values('date')
        stats_records.append({
            'symbol': symbol,
            'observations': len(group),
            'mean_residual': float(group['residual'].mean()),
            'std_residual': float(group['residual'].std(ddof=1)),
            'min_residual': float(group['residual'].min()),
            'max_residual': float(group['residual'].max()),
            'skewness': float(group['residual'].skew()),
            'kurtosis': float(group['residual'].kurtosis()),
        })

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        axes[0].plot(group['date'], group['residual'], color='steelblue', alpha=0.8)
        axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0].set_title(f'{symbol} Residual Time Series')
        axes[0].set_ylabel('Residual')
        axes[0].grid(alpha=0.3)

        axes[1].hist(group['residual'], bins=40, color='darkorange', alpha=0.7)
        axes[1].axvline(group['residual'].mean(), color='black', linestyle='--', linewidth=1,
                        label=f"Mean={group['residual'].mean():.4f}")
        axes[1].set_title(f'{symbol} Residual Distribution')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / f'{symbol}_residuals.png'
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    if stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_df.to_csv(output_dir / 'residual_stats_by_symbol_detailed.csv', index=False)

    logger.info("Saved per-symbol residual plots to %s", output_dir)


def simulate_oos_backtest(
    panel: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, float]:
    """Perform rolling one-step-ahead OOS backtest using predictOOS."""

    # Ensure enough warm-up: at least 5 and at least train window if specified
    train_window = args.oos_train_window if getattr(args, 'oos_train_window', None) else None
    warmup = max(args.oos_warmup, 5, (train_window or 0))
    dates = sorted(panel['date'].unique())
    if len(dates) <= warmup + 1:
        logger.warning("Not enough history for OOS backtest (need > warmup + 1 dates)")
        return {}

    prediction_records: List[pd.DataFrame] = []
    last_model: Optional[CryptoIPCAModel] = None
    last_refit_idx: Optional[int] = None
    refit_k = max(1, int(getattr(args, 'oos_refit_frequency', 1)))

    for idx in range(warmup, len(dates) - 1):
        predict_date = dates[idx]
        target_date = dates[idx + 1]

        # Select training window: all history up to predict_date, or last N days if specified
        if train_window and train_window > 0:
            cutoff_idx = max(0, idx - train_window + 1)  # inclusive of predict_date
            start_date = dates[cutoff_idx]
            train_panel = panel[(panel['date'] >= start_date) & (panel['date'] <= predict_date)]
        else:
            train_panel = panel[panel['date'] <= predict_date]
        today_panel = panel[panel['date'] == predict_date]
        next_panel = panel[panel['date'] == target_date]

        if train_panel.empty or today_panel.empty or next_panel.empty:
            continue

        # Refit according to frequency; reuse last model in between
        need_refit = (last_model is None) or (last_refit_idx is None) or ((idx - last_refit_idx) >= refit_k)
        if need_refit:
            model_iter = instantiate_model(args)
            model_iter.fit(train_panel)
            last_model = model_iter
            last_refit_idx = idx
        else:
            model_iter = last_model  # type: ignore[assignment]

        pred_df = model_iter.predict_oos(today_panel)
        pred_df['prediction_date'] = predict_date
        pred_df['target_date'] = target_date

        actual_df = next_panel[['symbol', 'return']].rename(columns={'return': 'actual_ret'})
        merged = pred_df.merge(actual_df, on='symbol', how='left')
        prediction_records.append(merged)

    if not prediction_records:
        logger.warning("OOS backtest produced no predictions")
        return {}

    eval_df = pd.concat(prediction_records, ignore_index=True)
    eval_df = eval_df.dropna(subset=['actual_ret'])
    if eval_df.empty:
        logger.warning("No valid OOS evaluation rows after merging actual returns")
        return {}

    eval_df['error'] = eval_df['predicted_ret_t_plus_1'] - eval_df['actual_ret']
    eval_df['abs_error'] = eval_df['error'].abs()
    eval_df['squared_error'] = eval_df['error'] ** 2
    eval_df['hit'] = (np.sign(eval_df['predicted_ret_t_plus_1']) == np.sign(eval_df['actual_ret'])).astype(float)

    metrics: Dict[str, Any] = {
        'oos_observations': float(len(eval_df)),
        'oos_mse': float(eval_df['squared_error'].mean()),
        'oos_mae': float(eval_df['abs_error'].mean()),
        'oos_bias': float(eval_df['error'].mean()),
        'oos_hit_ratio': float(eval_df['hit'].mean()),
    }

    corr = eval_df['predicted_ret_t_plus_1'].corr(eval_df['actual_ret'])
    if not np.isnan(corr):
        metrics['oos_correlation'] = float(corr)

    output_dir.mkdir(parents=True, exist_ok=True)
    oos_path = output_dir / 'oos_predictions.csv'
    eval_df.to_csv(oos_path, index=False)
    logger.info("Saved OOS predictions to %s", oos_path)

    return metrics


def evaluate_model(
    model: CryptoIPCAModel,
    panel: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Compute in-sample diagnostics and persist them alongside the model."""

    _, residuals = model.transform(panel)

    try:
        r2_score = float(  # type: ignore[arg-type]
            model.ipca_model.score(  # type: ignore[attr-defined]
                model.ipca_model.X,  # type: ignore[attr-defined]
                model.ipca_model.y,  # type: ignore[attr-defined]
                model.ipca_model.indices,  # type: ignore[attr-defined]
                data_type='panel'
            )
        )
    except Exception:  # pragma: no cover - defensive fallback
        logger.warning("Failed to obtain R² via InstrumentedPCA.score; falling back to NaN")
        r2_score = float('nan')
    model.r2_pred = r2_score

    metrics = {
        'r2_panel': r2_score,
        'mean_residual': float(residuals['residual'].mean()),
        'std_residual': float(residuals['residual'].std(ddof=1)),
        'rmse_residual': float(np.sqrt(np.mean(np.square(residuals['residual'])))),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'evaluation_metrics.json'
    residual_stats_path = output_dir / 'residual_stats_by_symbol.csv'
    residuals_path = output_dir / 'residual_timeseries.parquet'
    diagnostics_img = output_dir / 'residual_diagnostics.png'
    diagnostics_json = output_dir / 'residual_diagnostics.json'
    validation_json = output_dir / 'residual_validation.json'

    residuals.groupby('symbol')['residual'].agg(['mean', 'std']).to_csv(residual_stats_path)
    residuals.to_parquet(residuals_path, index=False)

    calculator = ResidualCalculator()
    fig, diagnostics = calculator.plot_residual_diagnostics(
        residuals,
        save_path=str(diagnostics_img),
    )

    try:
        import matplotlib.pyplot as plt  # Local import to avoid mandatory dependency during CLI parsing

        plt.close(fig)
    except Exception:  # pragma: no cover - optional cleanup
        pass

    validation: Dict[str, Any] = calculator.validate_residuals(residuals)

    diagnostics_serialisable = _to_serializable(diagnostics)
    validation_serialisable = _to_serializable(validation)

    with diagnostics_json.open('w') as fh:
        json.dump(diagnostics_serialisable, fh, indent=2)

    with validation_json.open('w') as fh:
        json.dump(validation_serialisable, fh, indent=2)

    metrics['residual_quality_score'] = float(validation['quality_score'])
    metrics['residual_quality_passed'] = bool(validation['passed'])

    # Market structure analysis (BTC dominance & BTC-vs-alt correlation)
    market_structure = analyse_market_structure(panel, output_dir)
    for key, value in market_structure.items():
        metrics[key] = value

    # Per-symbol residual visualisation
    symbol_plot_dir = output_dir / 'residuals_by_symbol'
    generate_symbol_residual_plots(residuals, symbol_plot_dir)

    if args.oos_backtest:
        oos_metrics = simulate_oos_backtest(panel, args, output_dir)
        metrics.update(oos_metrics)

    with metrics_path.open('w') as fh:
        json.dump(_to_serializable(metrics), fh, indent=2)

    logger.info("Saved evaluation metrics to %s", metrics_path)
    logger.info("Saved residual summary to %s", residual_stats_path)
    logger.info("Saved residual time-series to %s", residuals_path)
    logger.info("Saved residual diagnostics plot to %s", diagnostics_img)
    logger.info("Saved residual diagnostics summary to %s", diagnostics_json)
    logger.info("Saved residual validation report to %s", validation_json)

    return metrics


def persist_outputs(
    model: CryptoIPCAModel,
    panel: pd.DataFrame,
    output_dir: Path,
    save_returns: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "crypto_ipca_model.joblib"
    gamma_path = output_dir / "crypto_ipca_gamma.csv"
    factors_path = output_dir / "crypto_ipca_factors.csv"

    model.save_model(str(model_path))
    logger.info("Saved model artefact to %s", model_path)

    if model.gamma is not None:
        model.gamma.to_csv(gamma_path)
        logger.info("Saved factor loadings to %s", gamma_path)

    if model.factors is not None:
        model.factors.to_csv(factors_path, index=True)
        logger.info("Saved factor time-series to %s", factors_path)

    if save_returns:
        returns_path = output_dir / "training_panel.parquet"
        panel.to_parquet(returns_path, index=False)
        logger.info("Saved training panel to %s", returns_path)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        raw_df = collect_ohlcv(args)
        panel = prepare_returns(raw_df)
        if panel.empty:
            raise RuntimeError("Prepared panel is empty after return calculation")
        model = fit_model(panel, args)
        metrics = evaluate_model(model, panel, args.output_dir, args)
        logger.info("In-sample diagnostics: %s", metrics)
        persist_outputs(model, panel, args.output_dir, args.save_returns)
        logger.info("Latest factor observations:\n%s", model.factors.tail() if model.factors is not None else "<missing>")
        return 0
    except Exception:
        logger.exception("IPCA training pipeline failed")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
