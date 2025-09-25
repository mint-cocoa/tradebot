#!/usr/bin/env python3
"""Download (or reuse) Binance OHLCV data and train the CryptoIPCAModel."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
import sys

import pandas as pd
from numpy.linalg import LinAlgError

from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector
from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel

logger = logging.getLogger(__name__)

_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT",
        help="Comma separated list of Binance symbols (default: BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Kline timeframe to use (default: 1d)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date YYYY-MM-DD used for filtering (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-02-29",
        help="End date YYYY-MM-DD used for filtering (default: 2024-02-29)",
    )
    parser.add_argument(
        "--market-type",
        type=str,
        default="spot",
        choices=["spot", "futures"],
        help="Binance market type (default: spot)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/binance"),
        help="Directory to cache raw downloads if collector succeeds",
    )
    parser.add_argument(
        "--local-public-data-root",
        type=Path,
        default=Path("binance-public-data"),
        help="Root of the binance-public-data repository for offline reuse",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=3,
        help="Number of factors for IPCA (default: 3)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum IPCA fitting iterations (default: 500)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/crypto_ipca_model.joblib"),
        help="Path to persist the fitted model (default: models/crypto_ipca_model.joblib)",
    )
    parser.add_argument(
        "--gamma-out",
        type=Path,
        default=Path("models/crypto_ipca_gamma.csv"),
        help="Optional CSV path for factor loadings",
    )
    parser.add_argument(
        "--factors-out",
        type=Path,
        default=Path("models/crypto_ipca_factors.csv"),
        help="Optional CSV path for factor time series",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        nargs="+",
        help="One or more pre-downloaded Parquet files to use instead of live collection",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def attempt_collector_download(
    symbols: Iterable[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    market_type: str,
    data_dir: Path,
) -> pd.DataFrame:
    try:
        collector = BinanceDataCollector(
            data_dir=str(data_dir),
            use_public_data=True,
            max_workers=4,
        )
        logger.info("Collecting OHLCV via BinanceDataCollector...")
        return collector.collect_ohlcv(
            symbols=list(symbols),
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            market_type=market_type,
        )
    except Exception:
        logger.exception("Collector download failed; falling back to local repository if available")
        return pd.DataFrame()


def iter_local_archives(base: Path, symbol: str, timeframe: str) -> Iterable[Path]:
    monthly_pattern = f"**/spot/monthly/klines/{symbol}/{timeframe}/**/*.zip"
    daily_pattern = f"**/spot/daily/klines/{symbol}/{timeframe}/**/*.zip"
    seen = set()
    for pattern in (monthly_pattern, daily_pattern):
        for archive in base.glob(pattern):
            if archive not in seen:
                seen.add(archive)
                yield archive


def read_kline_archive(archive: Path, symbol: str) -> pd.DataFrame:
    from zipfile import ZipFile

    with ZipFile(archive) as zf:
        csv_candidates = [name for name in zf.namelist() if name.endswith(".csv")]
        if not csv_candidates:
            logger.debug("No CSV payload in %s", archive)
            return pd.DataFrame()
        with zf.open(csv_candidates[0]) as handle:
            df = pd.read_csv(handle, header=None, names=_COLUMNS)
    df["symbol"] = symbol
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def load_from_local_repository(
    base: Path,
    symbols: Iterable[str],
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if not base.exists():
        logger.warning("Local public data root %s does not exist", base)
        return pd.DataFrame()

    for symbol in symbols:
        logger.info("Scanning local archives for %s", symbol)
        symbol_frames: List[pd.DataFrame] = []
        for archive in iter_local_archives(base, symbol, timeframe):
            df = read_kline_archive(archive, symbol)
            if df.empty:
                continue
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
            if df.empty:
                continue
            symbol_frames.append(df)
        if symbol_frames:
            symbol_df = pd.concat(symbol_frames, ignore_index=True)
            symbol_df = symbol_df.sort_values("timestamp")
            symbol_df = symbol_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
            frames.append(symbol_df)
        else:
            logger.warning("No local data found for %s", symbol)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def collect_real_data(args: argparse.Namespace) -> pd.DataFrame:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    if start >= end:
        raise ValueError("start date must be earlier than end date")

    if args.input_parquet:
        df = load_from_parquet_files(
            parquet_paths=args.input_parquet,
            symbols=symbols,
            start=start,
            end=end,
        )
        if not df.empty:
            logger.info(
                "Loaded %d rows from local parquet sources spanning %s -> %s",
                len(df),
                df["timestamp"].min(),
                df["timestamp"].max(),
            )
            return df
        logger.warning("Provided parquet files produced no usable data; falling back to downloads")

    df = attempt_collector_download(
        symbols=symbols,
        timeframe=args.timeframe,
        start=start,
        end=end,
        market_type=args.market_type,
        data_dir=args.data_dir,
    )

    if df.empty:
        logger.info("Collector returned no data; attempting to reuse local binance-public-data dumps")
        df = load_from_local_repository(
            base=args.local_public_data_root,
            symbols=symbols,
            timeframe=args.timeframe,
            start=start,
            end=end,
        )

    if df.empty:
        raise RuntimeError("Unable to source OHLCV data from collector or local repository")

    logger.info(
        "Collected %d OHLCV rows for %d symbols spanning %s -> %s",
        len(df),
        df["symbol"].nunique(),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    return df


def load_from_parquet_files(
    parquet_paths: List[Path],
    symbols: List[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for path in parquet_paths:
        if not path.exists():
            logger.warning("Parquet file %s does not exist", path)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            continue

        if "timestamp" not in df.columns:
            logger.warning("Parquet file %s missing 'timestamp' column; skipping", path)
            continue

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)

        if symbols:
            df = df[df["symbol"].isin(symbols)]

        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        if df.empty:
            logger.info("No data in %s after filtering by symbols/date", path)
            continue

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"])
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    return combined


def prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])
    df["return"] = df.groupby("symbol")["close"].pct_change()
    df = df.dropna(subset=["return"])
    df = df[df["return"].abs() < 0.5]
    logger.info(
        "Prepared returns panel: %d observations across %d assets",
        len(df),
        df["symbol"].nunique(),
    )
    return df[["symbol", "date", "return", "close", "volume"]]


def fit_ipca_model(returns_df: pd.DataFrame, args: argparse.Namespace) -> CryptoIPCAModel:
    n_factors = max(1, args.n_factors)
    last_error: Optional[Exception] = None

    while n_factors >= 1:
        try:
            model = CryptoIPCAModel(
                n_factors=n_factors,
                max_iter=args.max_iter,
            )
            model.fit(returns_df)
            if n_factors != args.n_factors:
                logger.warning("Reduced factor count to %d due to numerical issues", n_factors)
            logger.info("Model fitted: gamma %s, factors %s", model.gamma.shape, model.factors.shape)
            return model
        except (LinAlgError, ValueError) as err:
            last_error = err
            logger.warning("IPCA fit failed with %d factors: %s", n_factors, err)
            n_factors -= 1

    raise RuntimeError("IPCA fitting failed for all factor counts") from last_error


def persist_outputs(model: CryptoIPCAModel, args: argparse.Namespace) -> None:
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_out))
    logger.info("Saved model to %s", args.model_out)

    if model.gamma is not None:
        args.gamma_out.parent.mkdir(parents=True, exist_ok=True)
        model.gamma.to_csv(args.gamma_out)
        logger.info("Saved factor loadings to %s", args.gamma_out)

    if model.factors is not None:
        args.factors_out.parent.mkdir(parents=True, exist_ok=True)
        model.factors.to_csv(args.factors_out, index=False)
        logger.info("Saved factor time-series to %s", args.factors_out)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        raw_df = collect_real_data(args)
        returns_df = prepare_returns(raw_df)
        if returns_df.empty:
            raise RuntimeError("Prepared returns panel is empty")
        model = fit_ipca_model(returns_df, args)
        persist_outputs(model, args)

        logger.info("Top of gamma matrix:\n%s", model.gamma.head())
        logger.info("Latest factor observations:\n%s", model.factors.tail())
        return 0
    except Exception:
        logger.exception("IPCA training pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
