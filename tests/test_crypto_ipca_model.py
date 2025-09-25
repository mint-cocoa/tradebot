"""Pytest suite for CryptoIPCAModel.

These tests use synthetic panel data to verify that the model can train,
transform, and surface diagnostics without relying on external datasets.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel


@pytest.fixture(scope="module")
def sample_returns() -> pd.DataFrame:
    """Generate synthetic crypto-style return data with basic features."""
    rng = np.random.default_rng(42)
    symbols = ["BTC", "ETH", "SOL", "ADA"]
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    rows = []
    for symbol in symbols:
        price = 100 + rng.normal(scale=5.0)
        volume = 1e7 + rng.normal(scale=5e5)
        for date in dates:
            daily_return = rng.normal(0.0, 0.015)
            price = max(price * (1.0 + daily_return), 1e-3)
            volume = max(volume + rng.normal(scale=2e5), 1e5)
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "return": daily_return,
                    "close": price,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype(float)
    return df


def test_fit_sets_expected_attributes(sample_returns: pd.DataFrame) -> None:
    model = CryptoIPCAModel(n_factors=3, max_iter=200)
    model.fit(sample_returns)

    assert model.is_fitted is True
    assert model.gamma is not None
    assert model.factors is not None
    assert model.gamma.shape[1] == model.n_factors
    assert model.factors.shape[1] == model.n_factors
    assert len(model.asset_names) == sample_returns["symbol"].nunique()


def test_transform_returns_factor_exposures_and_residuals(sample_returns: pd.DataFrame) -> None:
    model = CryptoIPCAModel(n_factors=2, max_iter=150)
    model.fit(sample_returns)

    exposures, residuals = model.transform(sample_returns.copy())

    assert exposures.shape[0] == residuals.shape[0] == len(sample_returns)
    assert set(['date', 'symbol', 'id', 'time']).issubset(exposures.columns)
    factor_cols = [col for col in exposures.columns if col.startswith('Factor_')]
    assert factor_cols == [f"Factor_{i+1}" for i in range(model.n_factors)]

    assert set(['date', 'symbol', 'id', 'time', 'actual_ret', 'expected_ret', 'residual']).issubset(residuals.columns)
    np.testing.assert_allclose(
        residuals['actual_ret'] - residuals['expected_ret'],
        residuals['residual']
    )


def test_transform_requires_fit(sample_returns: pd.DataFrame) -> None:
    model = CryptoIPCAModel(n_factors=1)
    with pytest.raises(ValueError):
        model.transform(sample_returns)


def test_save_and_load_roundtrip(sample_returns: pd.DataFrame, tmp_path) -> None:
    model = CryptoIPCAModel(n_factors=2, max_iter=150)
    model.fit(sample_returns)

    model_path = tmp_path / "ipca_model.pkl"
    model.save_model(str(model_path))

    reloaded = CryptoIPCAModel()
    reloaded.load_model(str(model_path))

    pd.testing.assert_frame_equal(reloaded.gamma, model.gamma)
    pd.testing.assert_frame_equal(reloaded.get_factor_timeseries(), model.get_factor_timeseries())
    assert reloaded.n_factors == model.n_factors


def test_diagnostics_report_basic_counts(sample_returns: pd.DataFrame) -> None:
    model = CryptoIPCAModel(n_factors=2, max_iter=150)
    model.fit(sample_returns)

    diagnostics = model.get_model_diagnostics()

    assert diagnostics["n_factors"] == model.n_factors
    assert diagnostics["n_assets"] == sample_returns["symbol"].nunique()
    assert diagnostics["n_periods"] > 0
    assert diagnostics["gamma_shape"][1] == model.n_factors
