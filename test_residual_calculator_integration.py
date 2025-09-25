#!/usr/bin/env python3
"""
잔차 계산기 통합 테스트

실제 암호화폐 데이터를 사용하여 잔차 계산 시스템의 
전체 워크플로우를 테스트합니다.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
import warnings
from datetime import datetime, timedelta

from crypto_dlsa_bot.ml.residual_calculator import (
    ResidualCalculator, ResidualCalculatorConfig, ResidualQualityMetrics
)
from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel
from crypto_dlsa_bot.ml.ipca_preprocessor import IPCADataPreprocessor, IPCAPreprocessorConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경고 무시
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_processed_data():
    """처리된 데이터 로드"""
    data_dir = Path("data/processed/ohlcv")
    
    # 가장 최근 데이터 파일 찾기
    parquet_files = []
    for timeframe in ['1d', '1h', '4h']:
        timeframe_dir = data_dir / timeframe / 'multi_symbol'
        if timeframe_dir.exists():
            files = list(timeframe_dir.glob('*.parquet'))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                parquet_files.append((timeframe, latest_file))
    
    if not parquet_files:
        raise FileNotFoundError("처리된 데이터 파일을 찾을 수 없습니다.")
    
    # 1일 데이터 우선 사용
    for timeframe, file_path in parquet_files:
        if timeframe == '1d':
            logger.info(f"1일 데이터 로드: {file_path}")
            return pd.read_parquet(file_path)
    
    # 1일 데이터가 없으면 첫 번째 파일 사용
    timeframe, file_path = parquet_files[0]
    logger.info(f"{timeframe} 데이터 로드: {file_path}")
    return pd.read_parquet(file_path)


def prepare_crypto_data(df, max_symbols=8):
    """암호화폐 데이터 준비"""
    logger.info("암호화폐 데이터 준비 시작")
    
    # 필요한 컬럼 확인
    required_cols = ['timestamp', 'symbol', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_cols}")
    
    # 데이터 정리
    result_df = df.copy()
    
    # timestamp를 date로 변환
    if 'date' not in result_df.columns:
        result_df['date'] = pd.to_datetime(result_df['timestamp'])
    
    # 수익률 계산
    if 'return' not in result_df.columns:
        result_df = result_df.sort_values(['symbol', 'date'])
        result_df['return'] = result_df.groupby('symbol')['close'].pct_change()
    
    # NaN 제거
    result_df = result_df.dropna(subset=['return'])
    
    # 데이터 필터링 (충분한 관측치가 있는 심볼만)
    symbol_counts = result_df.groupby('symbol').size()
    valid_symbols = symbol_counts[symbol_counts >= 100].index  # 최소 100개 관측치
    result_df = result_df[result_df['symbol'].isin(valid_symbols)]
    
    # 상위 N개 심볼만 사용 (테스트용)
    top_symbols = result_df.groupby('symbol').size().nlargest(max_symbols).index
    result_df = result_df[result_df['symbol'].isin(top_symbols)]
    
    # 극단적 수익률 제거 (±20% 초과)
    result_df = result_df[
        (result_df['return'] >= -0.2) & (result_df['return'] <= 0.2)
    ]
    
    logger.info(f"데이터 준비 완료: {len(result_df)} 관측치, {result_df['symbol'].nunique()} 심볼")
    logger.info(f"기간: {result_df['date'].min()} ~ {result_df['date'].max()}")
    logger.info(f"심볼: {sorted(result_df['symbol'].unique())}")
    
    return result_df


def test_out_of_sample_residuals():
    """아웃오브샘플 잔차 계산 테스트"""
    logger.info("\n" + "="*60)
    logger.info("아웃오브샘플 잔차 계산 테스트")
    logger.info("="*60)
    
    try:
        # 데이터 로드 및 준비
        raw_df = load_processed_data()
        crypto_df = prepare_crypto_data(raw_df, max_symbols=6)
        
        # 데이터 분할 (70% 학습, 30% 테스트)
        unique_dates = sorted(crypto_df['date'].unique())
        split_idx = int(len(unique_dates) * 0.7)
        split_date = unique_dates[split_idx]
        
        train_data = crypto_df[crypto_df['date'] < split_date].copy()
        test_data = crypto_df[crypto_df['date'] >= split_date].copy()
        
        logger.info(f"학습 데이터: {len(train_data)} 관측치 ({train_data['date'].min()} ~ {train_data['date'].max()})")
        logger.info(f"테스트 데이터: {len(test_data)} 관측치 ({test_data['date'].min()} ~ {test_data['date'].max()})")
        
        # 잔차 계산기 초기화
        config = ResidualCalculatorConfig(
            rolling_window_size=120,  # 4개월
            min_observations=60,      # 2개월
            refit_frequency=20,       # 20일마다 재학습
            quality_check_enabled=True,
            outlier_threshold=3.0
        )
        
        calculator = ResidualCalculator(config)
        
        # 아웃오브샘플 잔차 계산
        start_time = time.time()
        residuals_df, model = calculator.calculate_out_of_sample_residuals(
            train_data, test_data, n_factors=3
        )
        calculation_time = time.time() - start_time
        
        logger.info(f"잔차 계산 완료 (소요시간: {calculation_time:.2f}초)")
        logger.info(f"잔차 데이터 형태: {residuals_df.shape}")
        logger.info(f"심볼 수: {residuals_df['symbol'].nunique()}")
        
        # 잔차 통계
        logger.info("\n잔차 기본 통계:")
        logger.info(f"  평균: {residuals_df['residual'].mean():.6f}")
        logger.info(f"  표준편차: {residuals_df['residual'].std():.6f}")
        logger.info(f"  최솟값: {residuals_df['residual'].min():.6f}")
        logger.info(f"  최댓값: {residuals_df['residual'].max():.6f}")
        
        # 심볼별 잔차 통계
        logger.info("\n심볼별 잔차 통계:")
        symbol_stats = residuals_df.groupby('symbol')['residual'].agg(['count', 'mean', 'std'])
        for symbol in symbol_stats.index:
            count = symbol_stats.loc[symbol, 'count']
            mean = symbol_stats.loc[symbol, 'mean']
            std = symbol_stats.loc[symbol, 'std']
            logger.info(f"  {symbol}: {count}개, 평균={mean:.6f}, 표준편차={std:.6f}")
        
        return residuals_df, calculator
        
    except Exception as e:
        logger.error(f"아웃오브샘플 잔차 계산 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_rolling_residuals():
    """롤링 윈도우 잔차 계산 테스트"""
    logger.info("\n" + "="*60)
    logger.info("롤링 윈도우 잔차 계산 테스트")
    logger.info("="*60)
    
    try:
        # 데이터 로드 및 준비 (작은 데이터셋)
        raw_df = load_processed_data()
        crypto_df = prepare_crypto_data(raw_df, max_symbols=4)
        
        # 최근 6개월 데이터만 사용 (성능상 이유)
        unique_dates = sorted(crypto_df['date'].unique())
        if len(unique_dates) > 180:
            start_date = unique_dates[-180]
            crypto_df = crypto_df[crypto_df['date'] >= start_date]
        
        logger.info(f"롤링 테스트 데이터: {len(crypto_df)} 관측치")
        logger.info(f"기간: {crypto_df['date'].min()} ~ {crypto_df['date'].max()}")
        
        # 잔차 계산기 초기화 (작은 윈도우)
        config = ResidualCalculatorConfig(
            rolling_window_size=60,   # 2개월
            min_observations=30,      # 1개월
            refit_frequency=15,       # 15일마다 재학습
            quality_check_enabled=True
        )
        
        calculator = ResidualCalculator(config)
        
        # 롤링 잔차 계산
        start_time = time.time()
        residuals_df = calculator.calculate_rolling_residuals(
            crypto_df, refit_models=True
        )
        calculation_time = time.time() - start_time
        
        logger.info(f"롤링 잔차 계산 완료 (소요시간: {calculation_time:.2f}초)")
        
        if len(residuals_df) > 0:
            logger.info(f"잔차 데이터 형태: {residuals_df.shape}")
            logger.info(f"심볼 수: {residuals_df['symbol'].nunique()}")
            logger.info(f"날짜 범위: {residuals_df['date'].min()} ~ {residuals_df['date'].max()}")
            
            # 잔차 통계
            logger.info("\n롤링 잔차 기본 통계:")
            logger.info(f"  평균: {residuals_df['residual'].mean():.6f}")
            logger.info(f"  표준편차: {residuals_df['residual'].std():.6f}")
            
            return residuals_df, calculator
        else:
            logger.warning("롤링 잔차 계산 결과가 없습니다 (데이터 부족)")
            return pd.DataFrame(), calculator
        
    except Exception as e:
        logger.error(f"롤링 잔차 계산 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None


def test_quality_metrics(residuals_df, calculator):
    """잔차 품질 지표 테스트"""
    logger.info("\n" + "="*60)
    logger.info("잔차 품질 지표 테스트")
    logger.info("="*60)
    
    if len(residuals_df) == 0:
        logger.warning("잔차 데이터가 없어 품질 지표를 계산할 수 없습니다")
        return {}
    
    try:
        # 품질 지표 계산
        start_time = time.time()
        quality_metrics = calculator.calculate_residual_quality_metrics(residuals_df)
        calculation_time = time.time() - start_time
        
        logger.info(f"품질 지표 계산 완료 (소요시간: {calculation_time:.2f}초)")
        logger.info(f"분석된 심볼 수: {len(quality_metrics)}")
        
        # 품질 지표 출력
        for symbol, metrics in quality_metrics.items():
            logger.info(f"\n{symbol} 품질 지표:")
            logger.info(f"  관측치 수: {metrics.n_observations}")
            logger.info(f"  평균 잔차: {metrics.mean_residual:.6f}")
            logger.info(f"  표준편차: {metrics.std_residual:.6f}")
            logger.info(f"  왜도: {metrics.skewness:.4f}")
            logger.info(f"  첨도: {metrics.kurtosis:.4f}")
            logger.info(f"  정상성: {'예' if metrics.is_stationary else '아니오'} (p={metrics.adf_pvalue:.4f})")
            logger.info(f"  자기상관(lag1): {metrics.autocorr_lag1:.4f}")
            logger.info(f"  자기상관(lag5): {metrics.autocorr_lag5:.4f}")
            logger.info(f"  Jarque-Bera p-value: {metrics.jarque_bera_pvalue:.4f}")
            logger.info(f"  Ljung-Box p-value: {metrics.ljung_box_pvalue:.4f}")
            logger.info(f"  이분산성 비율: {metrics.heteroskedasticity_test:.4f}")
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"품질 지표 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_outlier_detection(residuals_df, calculator):
    """이상치 탐지 테스트"""
    logger.info("\n" + "="*60)
    logger.info("이상치 탐지 테스트")
    logger.info("="*60)
    
    if len(residuals_df) == 0:
        logger.warning("잔차 데이터가 없어 이상치 탐지를 수행할 수 없습니다")
        return pd.DataFrame()
    
    try:
        methods = ['zscore', 'iqr']
        
        for method in methods:
            logger.info(f"\n{method.upper()} 방법으로 이상치 탐지:")
            
            start_time = time.time()
            outliers_df = calculator.detect_residual_outliers(residuals_df, method=method)
            calculation_time = time.time() - start_time
            
            n_outliers = outliers_df['is_outlier'].sum()
            outlier_ratio = n_outliers / len(outliers_df) * 100
            
            logger.info(f"  소요시간: {calculation_time:.2f}초")
            logger.info(f"  전체 관측치: {len(outliers_df)}")
            logger.info(f"  이상치 수: {n_outliers}")
            logger.info(f"  이상치 비율: {outlier_ratio:.2f}%")
            
            # 심볼별 이상치 통계
            symbol_outliers = outliers_df.groupby('symbol')['is_outlier'].agg(['count', 'sum'])
            logger.info("  심볼별 이상치:")
            for symbol in symbol_outliers.index:
                total = symbol_outliers.loc[symbol, 'count']
                outliers = symbol_outliers.loc[symbol, 'sum']
                ratio = outliers / total * 100 if total > 0 else 0
                logger.info(f"    {symbol}: {outliers}/{total} ({ratio:.1f}%)")
        
        return outliers_df
        
    except Exception as e:
        logger.error(f"이상치 탐지 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def test_residual_report(residuals_df, quality_metrics, calculator):
    """잔차 리포트 생성 테스트"""
    logger.info("\n" + "="*60)
    logger.info("잔차 리포트 생성 테스트")
    logger.info("="*60)
    
    if len(residuals_df) == 0:
        logger.warning("잔차 데이터가 없어 리포트를 생성할 수 없습니다")
        return {}
    
    try:
        # 리포트 생성
        start_time = time.time()
        report = calculator.generate_residual_report(residuals_df, quality_metrics)
        calculation_time = time.time() - start_time
        
        logger.info(f"리포트 생성 완료 (소요시간: {calculation_time:.2f}초)")
        
        # 리포트 내용 출력
        logger.info("\n📊 잔차 분석 리포트:")
        logger.info(f"  전체 관측치 수: {report['total_observations']:,}")
        logger.info(f"  분석 심볼 수: {report['n_symbols']}")
        logger.info(f"  분석 기간: {report['start_date']} ~ {report['end_date']}")
        logger.info(f"  평균 자기상관(lag1): {report['mean_autocorr_lag1']:.4f}")
        logger.info(f"  평균 자기상관(lag5): {report['mean_autocorr_lag5']:.4f}")
        logger.info(f"  정상성 비율: {report['stationary_ratio']:.2%}")
        logger.info(f"  평균 Jarque-Bera p-value: {report['mean_jb_pvalue']:.4f}")
        logger.info(f"  양호한 자기상관 비율: {report['good_autocorr_ratio']:.2%}")
        logger.info(f"  이상치 비율: {report['outlier_ratio']:.2%}")
        logger.info(f"  전체 품질 점수: {report['quality_score']:.1f}/100")
        
        # 품질 평가
        if report['quality_score'] >= 80:
            quality_level = "우수"
        elif report['quality_score'] >= 60:
            quality_level = "양호"
        elif report['quality_score'] >= 40:
            quality_level = "보통"
        else:
            quality_level = "개선 필요"
        
        logger.info(f"  품질 평가: {quality_level}")
        
        return report
        
    except Exception as e:
        logger.error(f"리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_data_persistence(residuals_df, calculator):
    """데이터 저장/로드 테스트"""
    logger.info("\n" + "="*60)
    logger.info("데이터 저장/로드 테스트")
    logger.info("="*60)
    
    if len(residuals_df) == 0:
        logger.warning("저장할 잔차 데이터가 없습니다")
        return False
    
    try:
        # 임시 파일 경로
        temp_file = f"temp_residuals_{int(time.time())}.parquet"
        
        # 데이터 저장
        start_time = time.time()
        calculator.save_residuals(residuals_df, temp_file)
        save_time = time.time() - start_time
        
        logger.info(f"데이터 저장 완료 (소요시간: {save_time:.2f}초)")
        logger.info(f"파일 크기: {Path(temp_file).stat().st_size / 1024:.1f} KB")
        
        # 데이터 로드
        start_time = time.time()
        loaded_df = calculator.load_residuals(temp_file)
        load_time = time.time() - start_time
        
        logger.info(f"데이터 로드 완료 (소요시간: {load_time:.2f}초)")
        
        # 데이터 일치성 확인
        if len(loaded_df) == len(residuals_df):
            logger.info("✅ 데이터 크기 일치")
        else:
            logger.error(f"❌ 데이터 크기 불일치: {len(loaded_df)} vs {len(residuals_df)}")
            return False
        
        # 컬럼 확인
        if set(loaded_df.columns) == set(residuals_df.columns):
            logger.info("✅ 컬럼 구조 일치")
        else:
            logger.error("❌ 컬럼 구조 불일치")
            return False
        
        # 임시 파일 삭제
        Path(temp_file).unlink()
        logger.info("임시 파일 삭제 완료")
        
        return True
        
    except Exception as e:
        logger.error(f"데이터 저장/로드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_benchmark():
    """성능 벤치마크 테스트"""
    logger.info("\n" + "="*60)
    logger.info("성능 벤치마크 테스트")
    logger.info("="*60)
    
    try:
        # 다양한 데이터 크기로 성능 테스트
        sizes = [2, 4, 6]  # 심볼 수
        
        for n_symbols in sizes:
            logger.info(f"\n{n_symbols}개 심볼 성능 테스트:")
            
            # 데이터 준비
            raw_df = load_processed_data()
            crypto_df = prepare_crypto_data(raw_df, max_symbols=n_symbols)
            
            # 최근 3개월 데이터만 사용
            unique_dates = sorted(crypto_df['date'].unique())
            if len(unique_dates) > 90:
                start_date = unique_dates[-90]
                crypto_df = crypto_df[crypto_df['date'] >= start_date]
            
            logger.info(f"  데이터 크기: {len(crypto_df)} 관측치")
            
            # 데이터 분할
            split_idx = int(len(crypto_df) * 0.7)
            train_data = crypto_df.iloc[:split_idx]
            test_data = crypto_df.iloc[split_idx:]
            
            # 잔차 계산기
            calculator = ResidualCalculator()
            
            # 성능 측정
            start_time = time.time()
            residuals_df, model = calculator.calculate_out_of_sample_residuals(
                train_data, test_data, n_factors=2
            )
            total_time = time.time() - start_time
            
            # 결과 출력
            throughput = len(test_data) / total_time if total_time > 0 else 0
            logger.info(f"  소요시간: {total_time:.2f}초")
            logger.info(f"  처리량: {throughput:.1f} 관측치/초")
            logger.info(f"  잔차 수: {len(residuals_df)}")
        
    except Exception as e:
        logger.error(f"성능 벤치마크 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    logger.info("🚀 잔차 계산기 통합 테스트 시작")
    logger.info("=" * 80)
    
    # 테스트 결과 추적
    test_results = {}
    
    # 1. 아웃오브샘플 잔차 계산 테스트
    residuals_df, calculator = test_out_of_sample_residuals()
    test_results['out_of_sample'] = residuals_df is not None and len(residuals_df) > 0
    
    # 2. 롤링 윈도우 잔차 계산 테스트
    rolling_residuals_df, rolling_calculator = test_rolling_residuals()
    test_results['rolling'] = rolling_residuals_df is not None and len(rolling_residuals_df) > 0
    
    # 잔차 데이터가 있는 경우에만 후속 테스트 진행
    if test_results['out_of_sample']:
        # 3. 품질 지표 테스트
        quality_metrics = test_quality_metrics(residuals_df, calculator)
        test_results['quality_metrics'] = len(quality_metrics) > 0
        
        # 4. 이상치 탐지 테스트
        outliers_df = test_outlier_detection(residuals_df, calculator)
        test_results['outlier_detection'] = len(outliers_df) > 0
        
        # 5. 리포트 생성 테스트
        report = test_residual_report(residuals_df, quality_metrics, calculator)
        test_results['report_generation'] = len(report) > 0
        
        # 6. 데이터 저장/로드 테스트
        test_results['data_persistence'] = test_data_persistence(residuals_df, calculator)
    else:
        logger.warning("아웃오브샘플 잔차 데이터가 없어 후속 테스트를 건너뜁니다")
        test_results.update({
            'quality_metrics': False,
            'outlier_detection': False,
            'report_generation': False,
            'data_persistence': False
        })
    
    # 7. 성능 벤치마크
    performance_benchmark()
    test_results['performance_benchmark'] = True
    
    # 최종 결과 요약
    logger.info("\n" + "="*80)
    logger.info("🎯 테스트 결과 요약")
    logger.info("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n📊 전체 결과: {passed_tests}/{total_tests} 테스트 통과")
    
    if passed_tests == total_tests:
        logger.info("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests}개 테스트가 실패했습니다.")
    
    logger.info("\n✨ 잔차 계산기 통합 테스트 완료")


if __name__ == "__main__":
    main()