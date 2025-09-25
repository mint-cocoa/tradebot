#!/usr/bin/env python3
"""
잔차 계산 성능 벤치마크 스크립트

다양한 데이터 크기와 설정에서 잔차 계산 성능을 측정합니다.
"""

import sys
import os
import time
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto_dlsa_bot.ml.residual_calculator import ResidualCalculator, ResidualCalculatorConfig
from crypto_dlsa_bot.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger(__name__)


def _compute_cross_sectional_ic(panel: pd.DataFrame) -> Tuple[float, float, int]:
    """
    날짜별 단면 상관(Information Coefficient)을 계산하고 평균/표준편차/유효일수를 반환.
    Pearson 상관을 기본으로 사용합니다.
    """
    required = {'date', 'symbol', 'actual_ret', 'expected_ret'}
    if not required.issubset(panel.columns):
        return 0.0, 0.0, 0

    ics = []
    for d, g in panel.groupby('date'):
        g = g.dropna(subset=['actual_ret', 'expected_ret'])
        if g['symbol'].nunique() < 2:
            continue
        x = g['expected_ret'].astype(float)
        y = g['actual_ret'].astype(float)
        # 방어 로직: 상수열 방지
        if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
            continue
        ic = x.corr(y)  # Pearson
        if pd.notna(ic):
            ics.append(ic)

    if not ics:
        return 0.0, 0.0, 0
    arr = np.asarray(ics, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), int(len(arr))


def _compute_oos_error_metrics(panel: pd.DataFrame) -> dict:
    """OOS 예측오차 지표(MSE/MAE/R2_OS)를 계산."""
    required = {'actual_ret', 'expected_ret'}
    if not required.issubset(panel.columns) or panel.empty:
        return {'mse': None, 'mae': None, 'r2_os': None}

    df = panel.dropna(subset=['actual_ret', 'expected_ret']).copy()
    if df.empty:
        return {'mse': None, 'mae': None, 'r2_os': None}
    err = (df['actual_ret'] - df['expected_ret']).astype(float)
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    # R2_OS = 1 - SSE/SST (SST는 실제값에서 전체 평균을 뺀 제곱합)
    sse = float(np.sum(np.square(err)))
    y = df['actual_ret'].astype(float)
    sst = float(np.sum(np.square(y - y.mean())))
    r2_os = 1.0 - sse / sst if sst > 0 else None
    return {'mse': mse, 'mae': mae, 'r2_os': r2_os}


def _backtest_long_short(panel: pd.DataFrame, q: float = 0.2) -> dict:
    """
    expected_ret를 신호로 날짜별 상위 q, 하위 q 포트폴리오를 구성해 롱숏 수익률을 산출.
    Sharpe(비연율화), hit rate(양의 수익률 비율), 기간 수를 반환.
    """
    required = {'date', 'symbol', 'actual_ret', 'expected_ret'}
    if not required.issubset(panel.columns):
        return {'ls_mean': None, 'ls_std': None, 'ls_sharpe': None, 'hit_rate': None, 'n_days': 0}

    daily_ls = []
    for d, g in panel.groupby('date'):
        g = g.dropna(subset=['actual_ret', 'expected_ret'])
        if g['symbol'].nunique() < 3:
            continue
        k = max(1, int(len(g) * q))
        g = g.sort_values('expected_ret')
        bottom = g.head(k)
        top = g.tail(k)
        if bottom.empty or top.empty:
            continue
        # equal-weight
        long_ret = float(top['actual_ret'].mean())
        short_ret = float(bottom['actual_ret'].mean())
        daily_ls.append(long_ret - short_ret)

    if not daily_ls:
        return {'ls_mean': None, 'ls_std': None, 'ls_sharpe': None, 'hit_rate': None, 'n_days': 0}

    arr = np.asarray(daily_ls, dtype=float)
    ls_mean = float(arr.mean())
    ls_std = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
    ls_sharpe = float(ls_mean / ls_std) if ls_std > 0 else None
    hit_rate = float(np.mean(arr > 0))
    return {'ls_mean': ls_mean, 'ls_std': ls_std, 'ls_sharpe': ls_sharpe, 'hit_rate': hit_rate, 'n_days': int(len(arr))}


def _find_latest_parquet(base_dir: Path, timeframe: str) -> Optional[Path]:
    """지정된 시간프레임의 최신 멀티 심볼 Parquet 파일 경로를 찾습니다."""
    dir_path = base_dir / timeframe / "multi_symbol"
    if not dir_path.exists():
        return None
    files = list(dir_path.glob("*.parquet"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_real_ohlcv(
    symbols: List[str],
    timeframe: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_base_dir: str = "data/processed/ohlcv"
) -> pd.DataFrame:
    """
    처리된 실측 차트 데이터를 Parquet에서 로드합니다.

    Args:
        symbols: 대상 심볼 목록 (빈 리스트면 전체)
        timeframe: '1d', '4h', '1h' 등
        start_date: 시작일 (None이면 파일 전체 범위)
        end_date: 종료일 (None이면 파일 전체 범위)
        data_base_dir: 저장소 기본 경로

    Returns:
        OHLCV DataFrame [timestamp, symbol, open, high, low, close, volume, ...]
    """
    base_dir = Path(data_base_dir)
    latest = _find_latest_parquet(base_dir, timeframe)
    if not latest or not latest.exists():
        raise FileNotFoundError(
            f"실측 데이터 파일을 찾을 수 없습니다: {base_dir}/{timeframe}/multi_symbol/*.parquet"
        )

    df = pd.read_parquet(latest)
    if df.empty:
        raise ValueError("로드된 데이터가 비어있습니다")

    if 'timestamp' not in df.columns:
        raise ValueError("Parquet 파일에 'timestamp' 컬럼이 필요합니다")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if symbols:
        df = df[df['symbol'].isin([s.upper() for s in symbols])]

    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    # 필수 컬럼 보정
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Parquet에 '{col}' 컬럼이 없습니다")
    if 'volume' not in df.columns:
        df['volume'] = 0.0

    df = df.sort_values(['symbol', 'timestamp']).drop_duplicates(['symbol', 'timestamp'])
    return df


def prepare_returns_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV에서 일자별 수익률 패널을 생성합니다."""
    if ohlcv.empty:
        return pd.DataFrame(columns=['symbol', 'date', 'return', 'close', 'volume'])

    df = ohlcv.copy()
    # 일자 정규화 (timeframe이 일봉이 아니어도 일 단위로 정규화해 일별 패널 구성)
    df['date'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.sort_values(['symbol', 'date'])

    # 수익률 계산
    df['return'] = df.groupby('symbol')['close'].pct_change()
    df = df.dropna(subset=['return'])
    # 이상치 필터 (명백한 오류 제거)
    df = df[df['return'].abs() < 1.0]

    # 필요 컬럼만 반환
    return df[['symbol', 'date', 'return', 'close', 'volume']]


def create_characteristics_data(returns_data: pd.DataFrame) -> pd.DataFrame:
    """
    특성 데이터 생성
    
    Args:
        returns_data: 수익률 데이터
        
    Returns:
        특성 데이터
    """
    logger.info("특성 데이터 생성 중...")
    
    characteristics = returns_data[['symbol', 'date', 'close', 'volume']].copy()
    
    # 기본 특성들 (market_cap이 없는 실측 데이터 고려)
    if 'market_cap' not in returns_data.columns:
        market_cap = returns_data['close'] * returns_data['volume']
    else:
        market_cap = returns_data['market_cap']
    characteristics['log_market_cap'] = np.log(market_cap.replace({0: np.nan}).fillna(0) + 1e-8)
    characteristics['log_volume'] = np.log(returns_data['volume'].replace({0: np.nan}).fillna(0) + 1e-8)
    
    # 기술적 지표 계산
    returns_pivot = returns_data.pivot(index='date', columns='symbol', values='return')
    
    # 모멘텀 (여러 기간)
    for window in [5, 21, 63]:
        momentum = returns_pivot.rolling(window).sum().stack().reset_index()
        momentum.columns = ['date', 'symbol', f'momentum_{window}d']
        characteristics = characteristics.merge(momentum, on=['symbol', 'date'], how='left')
    
    # 변동성 (여러 기간)
    for window in [5, 21, 63]:
        volatility = returns_pivot.rolling(window).std().stack().reset_index()
        volatility.columns = ['date', 'symbol', f'volatility_{window}d']
        characteristics = characteristics.merge(volatility, on=['symbol', 'date'], how='left')
    
    # 결측치 처리
    characteristics = characteristics.drop(columns=['close'], errors='ignore')
    characteristics = characteristics.fillna(0)
    
    logger.info(f"특성 데이터 생성 완료: {characteristics.shape}")
    return characteristics


def run_single_benchmark(
    method: str,
    returns_data: pd.DataFrame,
    characteristics_data: Optional[pd.DataFrame],
    config: ResidualCalculatorConfig
) -> dict:
    """
    단일 벤치마크 실행
    
    Args:
        method: 계산 방법
        returns_data: 수익률 데이터
        characteristics_data: 특성 데이터
        config: 설정
        
    Returns:
        벤치마크 결과
    """
    logger.info(f"벤치마크 실행: {method}")

    calculator = ResidualCalculator(config)

    # 메모리 사용량 측정 시작 (psutil이 없으면 0으로 대체)
    process = psutil.Process() if psutil else None
    memory_before = (process.memory_info().rss / 1024 / 1024) if process else 0.0  # MB

    # 시간 측정 시작
    start_time = time.time()

    try:
        if method == 'standard':
            result = calculator.calculate_rolling_residuals(
                returns_data, save_models=False
            )
        elif method == 'optimized':
            result = calculator.calculate_rolling_residuals_optimized(
                returns_data, characteristics_data, save_models=False
            )
        elif method == 'parallel':
            result = calculator.calculate_residuals_parallel(
                returns_data, characteristics_data, n_jobs=2
            )
        elif method == 'streaming':
            result = calculator.calculate_residuals_streaming(
                returns_data, characteristics_data, batch_size=100
            )
        else:
            raise ValueError(f"알 수 없는 방법: {method}")

        # 시간 측정 종료
        end_time = time.time()
        execution_time = end_time - start_time

        # 메모리 사용량 측정 종료
        memory_after = (process.memory_info().rss / 1024 / 1024) if process else memory_before  # MB
        memory_used = (memory_after - memory_before) if process else 0.0

        # 모델 품질 및 백테스트 평가 (실데이터 기준)
        model_metrics = {}
        backtest_metrics = {}
        if len(result) > 0:
            quality_metrics = calculator.calculate_residual_quality_metrics(result)
            quality_score = calculator._calculate_overall_quality_score(quality_metrics)
            # 잔차 통계
            residual_stats = calculator.get_residual_statistics(result)
            avg_autocorr = np.mean([
                abs(stats['autocorr_lag1']) for stats in residual_stats.values()
                if not np.isnan(stats['autocorr_lag1'])
            ]) if residual_stats else 0
            # 예측 품질 (expected_ret vs actual_ret)
            ic_mean, ic_std, ic_days = _compute_cross_sectional_ic(result)
            oos_errors = _compute_oos_error_metrics(result)
            model_metrics = {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_days': ic_days,
                **oos_errors,
            }
            # 단순 롱숏 백테스트
            backtest_metrics = _backtest_long_short(result, q=0.2)
        else:
            quality_score = 0
            avg_autocorr = 0

        return {
            'method': method,
            'execution_time_seconds': execution_time,
            'memory_used_mb': memory_used,
            'output_rows': len(result),
            'throughput_rows_per_second': len(result) / execution_time if execution_time > 0 else 0,
            'quality_score': quality_score,
            'avg_autocorr': avg_autocorr,
            'success': True,
            'error': None
            , 'model_metrics': model_metrics
            , 'backtest': backtest_metrics
        }

    except Exception as e:
        logger.error(f"{method} 실패: {e}")
        return {
            'method': method,
            'execution_time_seconds': 0,
            'memory_used_mb': 0,
            'output_rows': 0,
            'throughput_rows_per_second': 0,
            'quality_score': 0,
            'avg_autocorr': 0,
            'success': False,
            'error': str(e)
        }


def run_scalability_benchmark(
    methods: list,
    base_returns: pd.DataFrame,
    base_characteristics: Optional[pd.DataFrame] = None,
    max_symbols: int = 50,
    max_days: int = 500,
):
    """
    확장성 벤치마크 실행
    
    Args:
        methods: 테스트할 방법들
        max_symbols: 최대 심볼 수
        max_days: 최대 일수
    """
    logger.info("확장성 벤치마크 시작")
    
    # 사용 가능한 범위 내에서 테스트 케이스 구성
    available_symbols = base_returns['symbol'].nunique()
    available_days = base_returns['date'].nunique()
    target_cases = [
        (min(10, available_symbols), min(100, available_days)),
        (min(20, available_symbols), min(200, available_days)),
        (min(30, available_symbols), min(300, available_days)),
    ]
    if max_symbols >= 50 and max_days >= 500 and available_symbols >= 50 and available_days >= 500:
        target_cases.append((50, 500))
    
    results = []
    
    config = ResidualCalculatorConfig(
        rolling_window_size=60,
        min_observations=20,
        refit_frequency=10
    )
    
    # 심볼은 거래대금(= close*volume) 합계 상위로 선택, 날짜는 최신부터 최근 n_days만 사용
    symbol_liquidity = (
        base_returns.assign(liq=base_returns['close'] * base_returns['volume'])
        .groupby('symbol')['liq']
        .sum()
        .sort_values(ascending=False)
    )

    sorted_dates = sorted(base_returns['date'].unique())

    for n_symbols, n_days in target_cases:
        if n_symbols < 2 or n_days < 10:
            continue

        logger.info(f"테스트 케이스: {n_symbols}개 심볼, {n_days}일")

        top_symbols = list(symbol_liquidity.index[:n_symbols])
        sel_dates = sorted_dates[-n_days:]

        returns_data = base_returns[
            base_returns['symbol'].isin(top_symbols) & base_returns['date'].isin(sel_dates)
        ].copy()
        characteristics_data = None
        if base_characteristics is not None and not base_characteristics.empty:
            characteristics_data = base_characteristics[
                base_characteristics['symbol'].isin(top_symbols) & base_characteristics['date'].isin(sel_dates)
            ].copy()
        
        case_results = {
            'n_symbols': n_symbols,
            'n_days': n_days,
            'total_observations': len(returns_data),
            'methods': {}
        }
        
        # 각 방법별 테스트
        for method in methods:
            try:
                result = run_single_benchmark(method, returns_data, characteristics_data, config)
                case_results['methods'][method] = result
                logger.info(
                    f"  {method}: {result['execution_time_seconds']:.2f}초, "
                    f"{result['throughput_rows_per_second']:.1f} 행/초"
                )
            except Exception as e:
                logger.error(f"  {method} 실패: {e}")
                case_results['methods'][method] = {'success': False, 'error': str(e)}

        results.append(case_results)
    
    return results


def run_configuration_benchmark(base_data: pd.DataFrame, base_characteristics: Optional[pd.DataFrame]):
    """
    설정별 성능 벤치마크
    
    Args:
        base_data: 기본 데이터
        base_characteristics: 기본 특성 데이터
    """
    logger.info("설정별 성능 벤치마크 시작")
    
    # 테스트할 설정들
    configs = [
        ResidualCalculatorConfig(rolling_window_size=30, min_observations=10, refit_frequency=5),
        ResidualCalculatorConfig(rolling_window_size=60, min_observations=20, refit_frequency=10),
        ResidualCalculatorConfig(rolling_window_size=120, min_observations=40, refit_frequency=20),
        ResidualCalculatorConfig(rolling_window_size=252, min_observations=60, refit_frequency=21),
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        logger.info(
            f"설정 {i+1}: 윈도우={config.rolling_window_size}, "
            f"최소관측={config.min_observations}, 재학습={config.refit_frequency}"
        )

        result = run_single_benchmark('optimized', base_data, base_characteristics, config)
        result['config'] = {
            'rolling_window_size': config.rolling_window_size,
            'min_observations': config.min_observations,
            'refit_frequency': config.refit_frequency
        }

        results.append(result)

        logger.info(
            f"  결과: {result['execution_time_seconds']:.2f}초, "
            f"품질점수={result['quality_score']:.1f}"
        )
    
    return results


def save_benchmark_results(results: dict, output_file: str):
    """
    벤치마크 결과 저장
    
    Args:
        results: 벤치마크 결과
        output_file: 출력 파일 경로
    """
    # 결과에 메타데이터 추가
    cpu_count = psutil.cpu_count() if psutil else None
    mem_total_gb = (psutil.virtual_memory().total / (1024**3)) if psutil else None
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': cpu_count,
            'memory_gb': mem_total_gb,
            'python_version': sys.version,
            'platform': sys.platform
        }
    }
    
    # JSON으로 저장
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"벤치마크 결과 저장: {output_file}")


def print_summary(results: dict):
    """
    벤치마크 결과 요약 출력
    
    Args:
        results: 벤치마크 결과
    """
    print("\n" + "="*80)
    print("잔차 계산 성능 벤치마크 결과 요약")
    print("="*80)
    
    if 'scalability' in results:
        print("\n📊 확장성 벤치마크:")
        for case in results['scalability']:
            print(f"\n  데이터 크기: {case['n_symbols']}개 심볼 × {case['n_days']}일 "
                  f"({case['total_observations']:,} 관측치)")
            
            for method, result in case['methods'].items():
                if result['success']:
                    print(f"    {method:12}: {result['execution_time_seconds']:6.2f}초 "
                          f"({result['throughput_rows_per_second']:8.1f} 행/초)")
                    # 모델 성능 지표 요약
                    mm = result.get('model_metrics') or {}
                    bt = result.get('backtest') or {}
                    ic_mean = mm.get('ic_mean')
                    ic_std = mm.get('ic_std')
                    r2_os = mm.get('r2_os')
                    mse = mm.get('mse')
                    mae = mm.get('mae')
                    ls_sharpe = bt.get('ls_sharpe')
                    hit_rate = bt.get('hit_rate')
                    # 가독성 있는 출력 (None 허용)
                    def _fmt(x, fmt):
                        return (fmt % x) if isinstance(x, (int, float)) and x is not None else str(x)
                    print(
                        f"      · IC(mean/std)={_fmt(ic_mean, '%.3f')}/{_fmt(ic_std, '%.3f')}  "
                        f"R2_OS={_fmt(r2_os, '%.3f')}  MSE/MAE={_fmt(mse, '%.6f')}/{_fmt(mae, '%.6f')}  "
                        f"LS Sharpe={_fmt(ls_sharpe, '%.2f')}  Hit%={_fmt(hit_rate, '%.1f')}"
                    )
                else:
                    print(f"    {method:12}: 실패 - {result.get('error', 'Unknown error')}")
    
    if 'configuration' in results:
        print("\n⚙️  설정별 성능:")
        for result in results['configuration']:
            if result['success']:
                config = result['config']
                print(f"    윈도우={config['rolling_window_size']:3d}, "
                      f"최소={config['min_observations']:2d}, "
                      f"재학습={config['refit_frequency']:2d}: "
                      f"{result['execution_time_seconds']:6.2f}초 "
                      f"(품질={result['quality_score']:5.1f})")
                # 모델 성능 지표 요약
                mm = result.get('model_metrics') or {}
                bt = result.get('backtest') or {}
                def _fmt(x, fmt):
                    return (fmt % x) if isinstance(x, (int, float)) and x is not None else str(x)
                print(
                    f"      · IC(mean/std)={_fmt(mm.get('ic_mean'), '%.3f')}/{_fmt(mm.get('ic_std'), '%.3f')}  "
                    f"R2_OS={_fmt(mm.get('r2_os'), '%.3f')}  MSE/MAE={_fmt(mm.get('mse'), '%.6f')}/{_fmt(mm.get('mae'), '%.6f')}  "
                    f"LS Sharpe={_fmt(bt.get('ls_sharpe'), '%.2f')}  Hit%={_fmt(bt.get('hit_rate'), '%.1f')}"
                )
    
    print("\n" + "="*80)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='잔차 계산 성능 벤치마크 (실측 차트 데이터 기반)')
    parser.add_argument('--methods', nargs='+', 
                       choices=['standard', 'optimized', 'parallel', 'streaming'],
                       default=['optimized', 'streaming'],
                       help='테스트할 방법들')
    parser.add_argument('--max-symbols', type=int, default=30,
                       help='확장성 테스트 시 최대 심볼 수')
    parser.add_argument('--max-days', type=int, default=300,
                       help='확장성 테스트 시 최대 일수')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='결과 출력 파일')
    parser.add_argument('--skip-scalability', action='store_true',
                       help='확장성 테스트 건너뛰기')
    parser.add_argument('--skip-config', action='store_true',
                       help='설정 테스트 건너뛰기')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT,BNBUSDT',
                       help='벤치마크 대상 심볼 (콤마로 구분)')
    parser.add_argument('--timeframe', type=str, default='1d',
                       help='데이터 시간 프레임 (예: 1d, 4h, 1h)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='시작일 YYYY-MM-DD (미지정 시 전체)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='종료일 YYYY-MM-DD (미지정 시 전체)')
    parser.add_argument('--data-dir', type=str, default='data/processed/ohlcv',
                       help='처리된 데이터가 저장된 기본 경로')
    
    args = parser.parse_args()
    
    logger.info("잔차 계산 성능 벤치마크 시작 (실측 데이터)")
    logger.info(f"테스트 방법: {args.methods}")
    logger.info(f"확장성 상한: {args.max_symbols}개 심볼, {args.max_days}일")
    logger.info(f"대상 심볼: {args.symbols}")
    logger.info(f"시간프레임: {args.timeframe}")
    
    all_results = {}
    
    try:
        # 실측 데이터 로드
        symbols = [s.strip().upper() for s in (args.symbols or '').split(',') if s.strip()]
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

        ohlcv = load_real_ohlcv(symbols, args.timeframe, start_dt, end_dt, data_base_dir=args.data_dir)
        logger.info(f"로딩 완료: {len(ohlcv):,} 레코드, 심볼 {ohlcv['symbol'].nunique()}개, 범위 {ohlcv['timestamp'].min()} ~ {ohlcv['timestamp'].max()}")

        returns_panel = prepare_returns_from_ohlcv(ohlcv)
        characteristics_df = create_characteristics_data(returns_panel)

        # 확장성 벤치마크
        if not args.skip_scalability:
            scalability_results = run_scalability_benchmark(
                args.methods,
                returns_panel,
                characteristics_df,
                max_symbols=args.max_symbols,
                max_days=args.max_days,
            )
            all_results['scalability'] = scalability_results

        # 설정별 벤치마크 (동일 실측 데이터 기반)
        if not args.skip_config:
            config_results = run_configuration_benchmark(returns_panel, characteristics_df)
            all_results['configuration'] = config_results
        
        # 결과 저장
        save_benchmark_results(all_results, args.output)
        
        # 요약 출력
        print_summary(all_results)
        
        logger.info("벤치마크 완료")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"벤치마크 실행 중 오류: {e}")
        raise


if __name__ == "__main__":
    main()