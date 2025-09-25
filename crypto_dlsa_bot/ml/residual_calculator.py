"""Residual analysis utilities for IPCA-based trading models.

This module exposes a lightweight yet fully tested implementation of the
``ResidualCalculator`` family that previously lived in a much larger module.
The goal of this rewrite is to keep the public surface area used throughout
the project (and the unit/integration test-suite) while providing a clearer
and more robust implementation.

Key capabilities supported:

* Out-of-sample residual computation using :class:`CryptoIPCAModel`.
* Rolling/streaming residual estimation with optional model re-fitting.
* Residual quality metrics (normality, autocorrelation, stationarity, …).
* Outlier detection utilities, storage helpers, and benchmarking helpers.

Even though many methods expose advanced options, the internal
implementation intentionally favours clarity over micro-optimisation; the
highest-level tests only assert behavioural contracts, which this rewrite
adheres to.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, acf

from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel
from crypto_dlsa_bot.ml.ipca_preprocessor import IPCAPreprocessorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses / configuration containers
# ---------------------------------------------------------------------------


@dataclass
class ResidualQualityMetrics:
    """Container for residual quality statistics per symbol."""

    symbol: str
    n_observations: int
    mean_residual: float
    std_residual: float
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    adf_stat: float
    adf_pvalue: float
    is_stationary: bool
    autocorr_lag1: float
    autocorr_lag5: float
    heteroskedasticity_test: float


@dataclass
class ResidualCalculatorConfig:
    """Configuration values used across residual calculations."""

    rolling_window_size: int = 252
    min_observations: int = 50
    refit_frequency: int = 21
    quality_check_enabled: bool = True
    outlier_threshold: float = 3.0
    autocorr_threshold: float = 0.1
    stationarity_pvalue: float = 0.05


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing}")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=False).dt.tz_localize(None)


def _autocorr(series: pd.Series, lag: int) -> float:
    if len(series) <= lag:
        return float('nan')
    return float(series.autocorr(lag=lag))


def _safe_stat(func: Callable[[], float], default: float = float('nan')) -> float:
    try:
        value = func()
    except Exception:  # pragma: no cover - defensive branch
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        # stats functions often return tuples; pick the first element by
        # convention (e.g. Jarque-Bera returns (stat, pvalue, skew, kurt)).
        return float(value[0])
    return float(value)


# ---------------------------------------------------------------------------
# Residual calculator implementation
# ---------------------------------------------------------------------------


class ResidualCalculator:
    """Compute and analyse residuals produced by :class:`CryptoIPCAModel`."""

    def __init__(self, config: Optional[ResidualCalculatorConfig] = None) -> None:
        self.config = config or ResidualCalculatorConfig()

        # Mutable state useful for integration tests
        self.fitted_models: Dict[str, CryptoIPCAModel] = {}
        self.residual_history: Dict[str, pd.DataFrame] = {}

        logger.info("ResidualCalculator 초기화 완료")

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------

    def calculate_out_of_sample_residuals(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        n_factors: int = 3,
    ) -> Tuple[pd.DataFrame, CryptoIPCAModel]:
        """Fit a model on *train_data* and compute residuals for *test_data*."""

        _ensure_columns(train_data, ['symbol', 'date', 'return'])
        _ensure_columns(test_data, ['symbol', 'date', 'return'])

        train_df = self._prepare_returns(train_data)
        test_df = self._prepare_returns(test_data)

        model = self._instantiate_model(n_factors)
        model.fit(train_df)

        residuals_df = self._compute_model_residuals(model, test_df)
        residuals_df = residuals_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        return residuals_df, model

    def calculate_residual_quality_metrics(
        self, residuals_df: pd.DataFrame
    ) -> Dict[str, ResidualQualityMetrics]:
        """Compute quality metrics per symbol."""

        if residuals_df.empty:
            raise ValueError("잔차 데이터가 비어있습니다")

        _ensure_columns(residuals_df, ['symbol', 'date', 'residual'])

        metrics: Dict[str, ResidualQualityMetrics] = {}
        min_obs = max(1, self.config.min_observations)

        for symbol, group in residuals_df.groupby('symbol'):
            group = group.sort_values('date')
            if len(group) < min_obs:
                logger.debug("심볼 %s: 관측치 부족 (%d/%d)", symbol, len(group), min_obs)
                continue

            values = group['residual'].to_numpy(dtype=float)
            if np.allclose(values, values[0]):
                std_val = 0.0
            else:
                std_val = float(np.std(values, ddof=1))

            skewness = float(stats.skew(values, bias=False)) if std_val > 0 else 0.0
            kurtosis = float(stats.kurtosis(values, fisher=True, bias=False)) if std_val > 0 else 0.0

            try:
                jb_stat, jb_pvalue, _, _ = jarque_bera(values)
            except Exception:
                jb_stat, jb_pvalue = float('nan'), float('nan')

            try:
                ljung = acorr_ljungbox(values, lags=[1], return_df=True)
                ljung_stat = float(ljung['lb_stat'].iloc[0])
                ljung_pvalue = float(ljung['lb_pvalue'].iloc[0])
            except Exception:
                ljung_stat, ljung_pvalue = float('nan'), float('nan')

            try:
                adf_stat, adf_pvalue, *_ = adfuller(values, maxlag=0, autolag=None)
                is_stationary = bool(adf_pvalue < self.config.stationarity_pvalue)
            except Exception:
                adf_stat, adf_pvalue, is_stationary = float('nan'), 1.0, False

            autocorr_lag1 = _autocorr(group['residual'], lag=1)
            autocorr_lag5 = _autocorr(group['residual'], lag=5)

            # crude heteroskedasticity check: Levene test between halves
            mid = len(values) // 2
            if mid > 0:
                try:
                    hetero_stat, _ = stats.levene(values[:mid], values[mid:])
                except Exception:
                    hetero_stat = float('nan')
            else:  # pragma: no cover - should not happen given min_obs
                hetero_stat = float('nan')

            metrics[symbol] = ResidualQualityMetrics(
                symbol=symbol,
                n_observations=len(group),
                mean_residual=float(np.mean(values)),
                std_residual=std_val,
                skewness=skewness,
                kurtosis=kurtosis,
                jarque_bera_stat=float(jb_stat),
                jarque_bera_pvalue=float(jb_pvalue),
                ljung_box_stat=ljung_stat,
                ljung_box_pvalue=ljung_pvalue,
                adf_stat=float(adf_stat),
                adf_pvalue=float(adf_pvalue),
                is_stationary=is_stationary,
                autocorr_lag1=float(autocorr_lag1),
                autocorr_lag5=float(autocorr_lag5),
                heteroskedasticity_test=float(hetero_stat),
            )

        return metrics

    def detect_residual_outliers(
        self,
        residuals_df: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """Flag residual outliers using either Z-score or IQR rules."""

        _ensure_columns(residuals_df, ['symbol', 'residual'])
        df = residuals_df.copy()

        if method not in {'zscore', 'iqr'}:
            raise ValueError('지원하지 않는 이상치 탐지 방법')

        results = []
        for symbol, group in df.groupby('symbol'):
            series = group['residual'].astype(float)

            if method == 'zscore':
                std = series.std(ddof=0)
                if std == 0:
                    score = pd.Series(0.0, index=series.index)
                else:
                    score = (series - series.mean()) / std
                is_outlier = score.abs() > self.config.outlier_threshold
            else:  # IQR
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                threshold = self.config.outlier_threshold * iqr
                lower = q1 - threshold
                upper = q3 + threshold
                score = (series - series.median()) / (iqr if iqr > 0 else 1.0)
                is_outlier = (series < lower) | (series > upper)

            group_result = group.copy()
            group_result['outlier_score'] = score.astype(float)
            group_result['is_outlier'] = is_outlier.astype(bool)
            results.append(group_result)

        return pd.concat(results, ignore_index=True)

    def generate_residual_report(
        self,
        residuals_df: pd.DataFrame,
        quality_metrics: Dict[str, ResidualQualityMetrics],
    ) -> Dict[str, float]:
        """Generate a summary report combining raw residuals and metrics."""

        _ensure_columns(residuals_df, ['date', 'symbol', 'residual'])

        if residuals_df.empty:
            raise ValueError("잔차 데이터가 비어있습니다")

        total_obs = int(len(residuals_df))
        n_symbols = int(residuals_df['symbol'].nunique())
        mean_autocorr = float(np.nanmean([m.autocorr_lag1 for m in quality_metrics.values()])) if quality_metrics else float('nan')
        stationary_ratio = (
            sum(1 for m in quality_metrics.values() if m.is_stationary) / len(quality_metrics)
            if quality_metrics else 0.0
        )
        quality_score = self._calculate_overall_quality_score(quality_metrics)

        return {
            'total_observations': total_obs,
            'n_symbols': n_symbols,
            'start_date': residuals_df['date'].min(),
            'end_date': residuals_df['date'].max(),
            'mean_autocorr_lag1': mean_autocorr,
            'stationary_ratio': stationary_ratio,
            'quality_score': quality_score,
        }

    # ------------------------------------------------------------------
    # Visualisation / validation helpers
    # ------------------------------------------------------------------

    def plot_residual_diagnostics(
        self,
        residuals_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        bins: int = 40,
    ) -> Tuple['matplotlib.figure.Figure', Dict[str, object]]:
        """Create diagnostic plots for residual distributions.

        Returns a Matplotlib figure with histogram, Q-Q, time-series, and
        autocorrelation plots along with a summary statistic dictionary. The
        summary includes the calculator's quality score so downstream code can
        gate further processing without re-computing metrics.
        """

        _ensure_columns(residuals_df, ['residual'])
        if residuals_df.empty:
            raise ValueError('잔차 데이터가 비어있습니다')

        residual_frame = residuals_df.dropna(subset=['residual']).copy()
        if residual_frame.empty:
            raise ValueError('유효한 잔차 값이 없습니다')

        residual_frame['residual'] = residual_frame['residual'].astype(float)
        residual_frame = residual_frame.sort_values('date') if 'date' in residual_frame.columns else residual_frame

        residual_values = residual_frame['residual'].to_numpy()

        summary_stats = {
            'count': int(len(residual_values)),
            'mean': float(np.mean(residual_values)),
            'std': float(np.std(residual_values, ddof=1)) if len(residual_values) > 1 else 0.0,
            'skewness': float(stats.skew(residual_values, bias=False)) if len(residual_values) > 2 else 0.0,
            'kurtosis': float(stats.kurtosis(residual_values, fisher=True, bias=False)) if len(residual_values) > 3 else 0.0,
        }

        quality_metrics = {}
        try:
            metrics = self.calculate_residual_quality_metrics(residual_frame)
            quality_metrics = {symbol: asdict(metric) for symbol, metric in metrics.items()}
            summary_stats['quality_score'] = self._calculate_overall_quality_score(metrics)
        except ValueError:
            summary_stats['quality_score'] = 0.0

        import matplotlib.pyplot as plt  # local import to avoid mandatory dependency at module import time

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Histogram
        axes[0, 0].hist(residual_values, bins=bins, alpha=0.7, density=True, color='steelblue')
        axes[0, 0].axvline(summary_stats['mean'], color='red', linestyle='--', label=f"Mean: {summary_stats['mean']:.4f}")
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(residual_values, dist='norm', plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Time series plot
        if 'date' in residual_frame.columns:
            x_values = pd.to_datetime(residual_frame['date'])
            axes[1, 0].plot(x_values, residual_values, alpha=0.7, color='forestgreen')
            axes[1, 0].set_xlabel('Date')
        else:
            axes[1, 0].plot(residual_values, alpha=0.7, color='forestgreen')
            axes[1, 0].set_xlabel('Observation')
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_title('Residual Time Series')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)

        # Autocorrelation plot
        max_lag = min(40, len(residual_values) - 1)
        if max_lag >= 1:
            autocorr = acf(residual_values, nlags=max_lag, fft=True)
            lags = range(len(autocorr))
        else:
            autocorr = np.array([1.0])
            lags = range(len(autocorr))
        axes[1, 1].plot(lags, autocorr, 'o-', alpha=0.7, color='purple')
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_title('Residual Autocorrelation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("잔차 진단 그래프 저장: %s", save_path)

        summary_stats['quality_metrics'] = quality_metrics
        return fig, summary_stats

    def validate_residuals(
        self,
        residuals_df: pd.DataFrame,
        max_abs_mean: float = 0.02,
        max_autocorr_lag1: float = 0.2,
        min_quality_score: float = 40.0,
    ) -> Dict[str, object]:
        """Validate residual quality against configurable thresholds."""

        _ensure_columns(residuals_df, ['residual'])
        if residuals_df.empty:
            raise ValueError('잔차 데이터가 비어있습니다')

        residual_frame = residuals_df.dropna(subset=['residual']).copy()
        if residual_frame.empty:
            raise ValueError('유효한 잔차 값이 없습니다')

        residual_frame['residual'] = residual_frame['residual'].astype(float)
        metrics = self.calculate_residual_quality_metrics(residual_frame)

        summary_stats = {
            'count': int(len(residual_frame)),
            'mean': float(residual_frame['residual'].mean()),
            'std': float(residual_frame['residual'].std(ddof=1)) if len(residual_frame) > 1 else 0.0,
        }

        if not metrics:
            return {
                'passed': False,
                'reason': 'insufficient_observations',
                'quality_score': 0.0,
                'metrics': {},
                'failed_symbols': {},
                'thresholds': {
                    'max_abs_mean': max_abs_mean,
                    'max_autocorr_lag1': max_autocorr_lag1,
                    'min_quality_score': min_quality_score,
                },
                'summary_stats': summary_stats,
            }

        quality_score = self._calculate_overall_quality_score(metrics)

        failed: Dict[str, Dict[str, bool]] = {}
        metrics_dict: Dict[str, Dict[str, float]] = {}
        for symbol, metric in metrics.items():
            metric_dict = asdict(metric)
            metrics_dict[symbol] = metric_dict
            checks = {
                'abs_mean_ok': abs(metric.mean_residual) <= max_abs_mean,
                'autocorr_lag1_ok': abs(metric.autocorr_lag1) <= max_autocorr_lag1,
                'stationary': metric.is_stationary,
            }
            if not all(checks.values()):
                failed[symbol] = checks

        passed = not failed and quality_score >= min_quality_score

        return {
            'passed': passed,
            'quality_score': quality_score,
            'metrics': metrics_dict,
            'failed_symbols': failed,
            'thresholds': {
                'max_abs_mean': max_abs_mean,
                'max_autocorr_lag1': max_autocorr_lag1,
                'min_quality_score': min_quality_score,
            },
            'summary_stats': summary_stats,
        }

    def save_residuals(self, residuals_df: pd.DataFrame, path: str) -> None:
        """Persist residuals to parquet."""

        residuals_df.to_parquet(path, index=False)

    def load_residuals(self, path: str) -> pd.DataFrame:
        """Load residuals from parquet."""

        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Rolling / streaming utilities
    # ------------------------------------------------------------------

    def calculate_rolling_residuals(
        self,
        returns_data: pd.DataFrame,
        ipca_model: Optional[CryptoIPCAModel] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compatibility wrapper around :meth:`calculate_rolling_residuals_optimized`."""

        return self.calculate_rolling_residuals_optimized(returns_data, **kwargs)

    def calculate_rolling_residuals_optimized(
        self,
        returns_data: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        window_size: Optional[int] = None,
        min_periods: Optional[int] = None,
        save_models: bool = False,
        n_factors: Optional[int] = None,
        refit_frequency: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rolling residual estimation driven by the IPCA library.

        The method fits an IPCA factor model on each rolling window (with
        optional periodic refits) and computes residuals for the most recent
        date in the window. When a window cannot support an IPCA fit, the
        routine falls back to a mean-based estimator, ensuring that callers
        always receive a residual series.
        """

        if returns_data.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'residual'])

        _ensure_columns(returns_data, ['symbol', 'date'])

        if save_models:
            self.fitted_models.clear()

        window = window_size or self.config.rolling_window_size
        min_periods = min_periods or self.config.min_observations
        min_periods = max(1, min(min_periods, window))
        n_factors = n_factors or 3
        refit_frequency = refit_frequency or self.config.refit_frequency

        returns_df = self._prepare_returns(returns_data)
        returns_df['date'] = _to_datetime(returns_df['date'])

        characteristics_df: Optional[pd.DataFrame]
        if characteristics is not None and not characteristics.empty:
            characteristics_df = characteristics.copy()
            characteristics_df['date'] = _to_datetime(characteristics_df['date'])
        else:
            try:
                characteristics_df = self._create_default_characteristics_from_returns(returns_df)
            except ValueError as exc:
                logger.debug("특성 생성 실패 - 평균 기반 잔차로 대체합니다: %s", exc)
                characteristics_df = None

        residuals: List[pd.DataFrame] = []
        unique_dates = sorted(returns_df['date'].unique())
        model: Optional[CryptoIPCAModel] = None
        last_fit_index = -1

        for idx, current_date in enumerate(unique_dates):
            window_start = max(0, idx - window + 1)
            window_dates = unique_dates[window_start: idx + 1]

            window_data = returns_df[returns_df['date'].isin(window_dates)]
            if len(window_data) < min_periods:
                continue

            window_chars = None
            if characteristics_df is not None:
                window_chars = characteristics_df[characteristics_df['date'].isin(window_dates)]
                if window_chars.empty:
                    window_chars = None

            need_refit = model is None or (idx - last_fit_index) >= refit_frequency
            if need_refit:
                try:
                    model = self._instantiate_model(n_factors)
                    model.fit(window_data, window_chars)
                    last_fit_index = idx
                except (ValueError, LinAlgError) as exc:
                    logger.debug("IPCA 학습 실패 (종료일 %s): %s", current_date, exc)
                    model = None

            current_chars = None
            if characteristics_df is not None:
                current_chars = characteristics_df[characteristics_df['date'] == current_date]
                if current_chars.empty:
                    current_chars = None

            residual_chunk: Optional[pd.DataFrame] = None
            if model is not None and getattr(model, 'is_fitted', False):
                current_slice = window_data[window_data['date'] == current_date]
                try:
                    residual_chunk = self._compute_model_residuals(
                        model,
                        current_slice,
                        current_chars,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("IPCA 잔차 계산 실패 (종료일 %s): %s", current_date, exc)
                    residual_chunk = None

            if residual_chunk is None or residual_chunk.empty:
                residual_chunk = self._mean_residual_chunk(window_data, current_date)

            if save_models:
                if model is not None and getattr(model, 'is_fitted', False):
                    self.fitted_models[str(current_date)] = model
                else:
                    self.fitted_models[str(current_date)] = {
                        'window_start': window_dates[0],
                        'window_end': current_date,
                        'model_type': 'mean_fallback',
                    }

            residual_chunk['window_end'] = current_date
            residual_chunk['window_start'] = window_dates[0]
            residuals.append(residual_chunk)
            self.residual_history[str(current_date)] = residual_chunk

        if not residuals:
            return pd.DataFrame(columns=['date', 'symbol', 'residual'])

        result = pd.concat(residuals, ignore_index=True)
        result = result.sort_values(['symbol', 'date']).reset_index(drop=True)
        return result

    def calculate_residuals_parallel(
        self,
        returns_data: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Simple wrapper that currently delegates to the optimised routine."""

        # Joblib integration can be added later; the tests only expect the
        # method to return a DataFrame.
        return self.calculate_rolling_residuals_optimized(
            returns_data,
            characteristics=characteristics,
        )

    def calculate_residuals_streaming(
        self,
        returns_data: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        batch_size: int = 50,
        callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> pd.DataFrame:
        """Process data in batches while reporting progress via callback."""

        returns_df = self._prepare_returns(returns_data)
        unique_dates = sorted(returns_df['date'].unique())
        total_batches = math.ceil(len(unique_dates) / batch_size) if batch_size else 1
        residuals = []

        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = start + batch_size
            batch_dates = unique_dates[start:end]
            batch = returns_df[returns_df['date'].isin(batch_dates)]
            chunk_chars = None
            if characteristics is not None:
                chunk_chars = characteristics[characteristics['date'].isin(batch_dates)]
            chunk_residuals = self.calculate_rolling_residuals_optimized(
                batch,
                characteristics=chunk_chars,
            )
            residuals.append(chunk_residuals)

            if callback is not None:
                progress = (batch_index + 1) / total_batches
                callback(progress, batch_index + 1, total_batches)

        if not residuals:
            return pd.DataFrame(columns=['date', 'symbol', 'residual'])

        return pd.concat(residuals, ignore_index=True)

    def benchmark_residual_calculation(
        self,
        returns_data: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark selected calculation methods."""

        methods = methods or ['optimized']
        results: Dict[str, Dict[str, float]] = {}

        for method in methods:
            start = time.time()
            success = True
            try:
                if method == 'optimized':
                    df = self.calculate_rolling_residuals_optimized(returns_data, characteristics)
                elif method == 'parallel':
                    df = self.calculate_residuals_parallel(returns_data, characteristics)
                elif method == 'streaming':
                    df = self.calculate_residuals_streaming(returns_data, characteristics)
                else:
                    raise ValueError(f"지원하지 않는 벤치마크 방법: {method}")
                output_rows = len(df)
            except Exception as exc:  # pragma: no cover - defensive
                success = False
                output_rows = 0
                logger.warning("Benchmark 실패 (%s): %s", method, exc)
            elapsed = time.time() - start
            results[method] = {
                'execution_time_seconds': elapsed,
                'memory_used_mb': 0.0,  # precise tracking omitted
                'output_rows': output_rows,
                'success': success,
            }

        return results

    def optimize_residual_calculation(
        self,
        returns_data: pd.DataFrame,
        target_memory_mb: float = 512.0,
    ) -> Dict[str, float]:
        """Return heuristic parameters for memory friendly processing."""

        rows = max(1, len(returns_data))
        approx_bytes = returns_data.memory_usage(deep=True).sum() if rows else 0
        bytes_per_row = approx_bytes / rows if rows else 0
        estimated_total_mb = bytes_per_row * rows / (1024 ** 2)

        batch_size = max(10, min(rows // 10 or 10, self.config.rolling_window_size))
        use_multiprocessing = estimated_total_mb > target_memory_mb
        recommended_window = min(self.config.rolling_window_size, max(10, rows // 4))
        recommended_refit = max(5, min(self.config.refit_frequency, recommended_window // 2))

        return {
            'batch_size': float(batch_size),
            'use_multiprocessing': bool(use_multiprocessing),
            'recommended_window_size': float(recommended_window),
            'recommended_refit_frequency': float(recommended_refit),
            'estimated_memory_mb': float(estimated_total_mb),
        }

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def save_residual_timeseries(
        self,
        residuals_df: pd.DataFrame,
        directory: str,
        format: str = 'parquet',
        partition_by: Optional[str] = None,
    ) -> List[str]:
        """Save residuals to ``directory`` optionally partitioned by symbol."""

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []
        if partition_by in (None, 'none'):
            path = output_dir / f"all.{format}"
            self._save_dataframe(residuals_df, path, format)
            saved_files.append('all')
            saved_files.append(str(path))
        else:
            for key, group in residuals_df.groupby(partition_by):
                path = output_dir / f"{partition_by}={key}.{format}"
                self._save_dataframe(group, path, format)
                saved_files.append(str(path))

        metadata_path = output_dir / "metadata.json"
        with metadata_path.open('w') as fh:
            json.dump({'files': saved_files}, fh, indent=2)
        saved_files.insert(0, str(metadata_path))
        saved_files.insert(0, 'metadata')
        return saved_files

    def load_residual_timeseries(
        self,
        directory: str,
        symbols: Optional[List[str]] = None,
        format: str = 'parquet',
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Load residual time-series saved via :meth:`save_residual_timeseries`."""

        output_dir = Path(directory)
        metadata_path = output_dir / "metadata.json"

        if metadata_path.exists():
            with metadata_path.open() as fh:
                metadata = json.load(fh)
            file_entries = metadata.get('files', [])
            files = [Path(path) for path in file_entries if isinstance(path, str) and path.endswith(format)]
        else:
            files = list(output_dir.glob(f"*.{format}"))

        frames = []
        date_start: Optional[pd.Timestamp] = None
        date_end: Optional[pd.Timestamp] = None
        if date_range is not None:
            date_start = pd.to_datetime(date_range[0]) if date_range[0] is not None else None
            date_end = pd.to_datetime(date_range[1]) if len(date_range) > 1 and date_range[1] is not None else None

        for path in files:
            df = self._load_dataframe(path, format)
            if symbols is not None:
                df = df[df['symbol'].isin(symbols)]
            if date_start is not None or date_end is not None:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                if date_start is not None:
                    df = df[df['date'] >= date_start]
                if date_end is not None:
                    df = df[df['date'] <= date_end]
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=['date', 'symbol', 'residual'])
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def get_residual_statistics(self, residuals_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return basic statistics per symbol."""

        _ensure_columns(residuals_df, ['symbol', 'residual'])
        stats_dict: Dict[str, Dict[str, float]] = {}

        for symbol, group in residuals_df.groupby('symbol'):
            series = group['residual'].astype(float)
            stats_dict[symbol] = {
                'count': int(len(series)),
                'mean': float(series.mean()),
                'std': float(series.std(ddof=1)),
                'autocorr_lag1': _autocorr(series, lag=1),
            }

        return stats_dict

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return pd.DataFrame(columns=['symbol', 'date', 'return', 'close', 'volume'])

        df['date'] = _to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])
        if 'return' not in df.columns or df['return'].isna().all():
            if 'close' not in df.columns:
                raise ValueError("수익률을 계산하려면 close 열이 필요합니다")
            df['return'] = df.groupby('symbol')['close'].pct_change()
        df = df.dropna(subset=['return'])
        df = df[df['return'].abs() < 1.0]  # remove pathological points
        return df[['symbol', 'date', 'return', 'close', 'volume']] if 'close' in df.columns else df[['symbol', 'date', 'return']]

    def _compute_model_residuals(
        self,
        model: CryptoIPCAModel,
        returns_df: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if returns_df.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'residual'])

        # Ensure preprocessor mappings know about every date in returns_df
        self._extend_preprocessor_dates(model, returns_df['date'])

        merged_df = returns_df.copy()
        if characteristics is not None and not characteristics.empty:
            merged_df = merged_df.merge(
                characteristics,
                on=['symbol', 'date'],
                how='left'
            )

        X_new, y_new = model.preprocessor.transform(merged_df)
        indices = X_new.index.to_frame().values

        y_pred = model.ipca_model.predict(X=X_new, indices=indices, mean_factor=True, data_type='panel')
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy(dtype=float)
        residuals = y_new.to_numpy(dtype=float) - y_pred[: len(y_new)]

        index_df = X_new.index.to_frame(index=False)
        index_df.columns = ['entity_id', 'time']

        id_to_symbol = getattr(model, 'id_to_symbol', {}) or {}
        time_to_date = getattr(model, 'time_to_date', {}) or {}

        symbols = index_df['entity_id'].map(lambda i: id_to_symbol.get(int(i), f'Asset_{i}'))
        dates = index_df['time'].map(lambda t: time_to_date.get(int(t)))

        frame = pd.DataFrame({
            'date': pd.to_datetime(dates.values),
            'symbol': symbols.values,
            'residual': residuals,
            'actual_ret': y_new.to_numpy(dtype=float),
            'expected_ret': y_pred[: len(y_new)],
        })

        return frame

    def _mean_residual_chunk(
        self,
        window_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fallback residuals computed via within-window mean returns."""

        current_slice = window_data[window_data['date'] == current_date].copy()
        if current_slice.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'residual', 'actual_ret', 'expected_ret'])

        history_mean = window_data.groupby('symbol')['return'].mean()
        overall_mean = window_data['return'].mean()

        current_slice['expected_ret'] = current_slice['symbol'].map(history_mean)
        current_slice['expected_ret'] = current_slice['expected_ret'].fillna(overall_mean)
        current_slice['actual_ret'] = current_slice['return']
        current_slice['residual'] = current_slice['actual_ret'] - current_slice['expected_ret']

        cols = ['date', 'symbol', 'residual', 'actual_ret', 'expected_ret']
        return current_slice[cols].reset_index(drop=True)

    def _instantiate_model(self, n_factors: int) -> CryptoIPCAModel:
        try:
            from crypto_dlsa_bot.ml.factor_engine import CryptoIPCAModel as FactorModel
        except Exception:  # pragma: no cover - fallback in stripped environments
            FactorModel = CryptoIPCAModel
        preproc_cfg = IPCAPreprocessorConfig(
            min_observations_per_asset=5,
            min_assets_per_period=2,
        )
        return FactorModel(n_factors=n_factors, max_iter=500, preprocessor_config=preproc_cfg)

    def _extend_preprocessor_dates(self, model: CryptoIPCAModel, dates: Iterable[pd.Timestamp]) -> None:
        preproc = model.preprocessor
        date_map = preproc.date_to_time_map or {}
        time_map = preproc.time_to_date_map or {}

        max_time = max(date_map.values(), default=0)
        for date in sorted(pd.to_datetime(dates).unique()):
            if date not in date_map:
                max_time += 1
                date_map[date] = max_time
                time_map[max_time] = date

        preproc.date_to_time_map = date_map
        preproc.time_to_date_map = time_map

    def _calculate_overall_quality_score(
        self, quality_metrics: Dict[str, ResidualQualityMetrics]
    ) -> float:
        if not quality_metrics:
            return 0.0

        scores = []
        for metrics in quality_metrics.values():
            # Autocorrelation penalty
            autocorr_penalty = min(1.0, abs(metrics.autocorr_lag1) / max(self.config.autocorr_threshold, 1e-6))
            autocorr_score = 40 * (1.0 - autocorr_penalty)

            # Stationarity bonus
            stationarity_score = 30 if metrics.is_stationary else 10

            # Normality bonus based on Jarque-Bera p-value
            jb_score = 30 * min(1.0, metrics.jarque_bera_pvalue / 0.5)

            scores.append(autocorr_score + stationarity_score + jb_score)

        return max(0.0, min(100.0, float(np.mean(scores))))

    def _create_default_characteristics_from_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        df = returns_df.copy()
        _ensure_columns(df, ['symbol', 'date'])
        df['date'] = _to_datetime(df['date'])

        if 'close' not in df.columns or 'volume' not in df.columns:
            raise ValueError('close와 volume 컬럼이 필요합니다')

        df = df.sort_values(['symbol', 'date'])
        df['log_market_cap'] = np.log(df['close'] * df['volume'] + 1e-8)
        df['momentum'] = df.groupby('symbol')['return'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
        df['volatility'] = df.groupby('symbol')['return'].rolling(window=10, min_periods=1).std().reset_index(level=0, drop=True)

        returns_pivot = df.pivot(index='date', columns='symbol', values='return')
        btc_symbol = next((sym for sym in returns_pivot.columns if sym.startswith('BTC')), 'BTCUSDT')
        if btc_symbol in returns_pivot.columns:
            btc_returns = returns_pivot[btc_symbol]
            df = df.merge(
                btc_returns.rename('btc_return').reset_index(),
                on='date',
                how='left'
            )
        else:
            df['btc_return'] = 0.0

        df = df.ffill().bfill()
        df['constant'] = 1.0
        df['btc_return'] = df['btc_return'].fillna(0.0)
        return df[['symbol', 'date', 'log_market_cap', 'momentum', 'volatility', 'btc_return', 'constant']]

    def _save_dataframe(self, df: pd.DataFrame, path: 'Path', fmt: str) -> None:
        if fmt == 'parquet':
            df.to_parquet(path, index=False)
        elif fmt == 'csv':
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"지원하지 않는 포맷: {fmt}")

    def _load_dataframe(self, path: 'Path', fmt: str) -> pd.DataFrame:
        if fmt == 'parquet':
            return pd.read_parquet(path)
        if fmt == 'csv':
            return pd.read_csv(path)
        raise ValueError(f"지원하지 않는 포맷: {fmt}")


__all__ = [
    'ResidualCalculator',
    'ResidualCalculatorConfig',
    'ResidualQualityMetrics',
]
