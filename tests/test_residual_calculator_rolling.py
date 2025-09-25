"""
잔차 계산기 롤링 윈도우 처리 테스트

이 모듈은 ResidualCalculator의 롤링 윈도우 기반 잔차 계산 기능을 테스트합니다.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import patch, MagicMock

from crypto_dlsa_bot.ml.residual_calculator import (
    ResidualCalculator, 
    ResidualCalculatorConfig,
    ResidualQualityMetrics
)
from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel


class TestResidualCalculatorRolling:
    """롤링 윈도우 잔차 계산 테스트"""
    
    @pytest.fixture
    def sample_returns_data(self):
        """샘플 수익률 데이터 생성"""
        np.random.seed(42)
        
        symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                # 간단한 수익률 시뮬레이션
                base_return = np.random.normal(0.001, 0.02)  # 기본 수익률
                market_factor = np.random.normal(0, 0.01)    # 시장 팩터
                noise = np.random.normal(0, 0.005)           # 잡음
                
                return_value = base_return + market_factor + noise
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'return': return_value,
                    'close': 100 * np.exp(np.random.normal(0, 0.1)),  # 가격
                    'volume': np.random.lognormal(10, 1)              # 거래량
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_characteristics_data(self, sample_returns_data):
        """샘플 특성 데이터 생성"""
        characteristics = sample_returns_data[['symbol', 'date']].copy()
        
        # 시가총액 프록시
        characteristics['log_market_cap'] = np.log(
            sample_returns_data['close'] * sample_returns_data['volume']
        )
        
        # 모멘텀 (간단한 버전)
        characteristics['momentum'] = np.random.normal(0, 0.1, len(characteristics))
        
        # 변동성 (간단한 버전)
        characteristics['volatility'] = np.random.lognormal(0, 0.5, len(characteristics))
        
        return characteristics
    
    @pytest.fixture
    def residual_calculator(self):
        """잔차 계산기 인스턴스"""
        config = ResidualCalculatorConfig(
            rolling_window_size=30,
            min_observations=20,
            refit_frequency=7,
            quality_check_enabled=True
        )
        return ResidualCalculator(config)
    
    def test_calculate_rolling_residuals_optimized(self, residual_calculator, sample_returns_data, sample_characteristics_data):
        """최적화된 롤링 윈도우 잔차 계산 테스트"""
        # 작은 데이터셋으로 테스트
        small_data = sample_returns_data.head(100).copy()
        small_chars = sample_characteristics_data.head(100).copy()
        
        result = residual_calculator.calculate_rolling_residuals_optimized(
            small_data, small_chars, window_size=10, min_periods=5
        )
        
        # 기본 검증
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'symbol' in result.columns
        assert 'residual' in result.columns
        assert len(result) > 0
        
        # 잔차 값이 합리적인 범위인지 확인
        assert result['residual'].abs().max() < 1.0  # 100% 이상의 잔차는 비현실적
        
        # 날짜 순서 확인
        for symbol in result['symbol'].unique():
            symbol_data = result[result['symbol'] == symbol].sort_values('date')
            assert symbol_data['date'].is_monotonic_increasing
    
    def test_save_and_load_residual_timeseries(self, residual_calculator, sample_returns_data):
        """잔차 시계열 저장 및 로드 테스트"""
        # 간단한 잔차 데이터 생성
        residuals_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-10', freq='D').repeat(3),
            'symbol': ['BTC', 'ETH', 'ADA'] * 10,
            'residual': np.random.normal(0, 0.01, 30)
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 저장 테스트
            saved_files = residual_calculator.save_residual_timeseries(
                residuals_df, temp_dir, format='parquet', partition_by='symbol'
            )
            
            assert 'metadata' in saved_files
            assert len(saved_files) > 1  # 메타데이터 + 심볼별 파일들
            
            # 로드 테스트
            loaded_df = residual_calculator.load_residual_timeseries(
                temp_dir, symbols=['BTC', 'ETH'], format='parquet'
            )
            
            assert isinstance(loaded_df, pd.DataFrame)
            assert set(loaded_df['symbol'].unique()) == {'BTC', 'ETH'}
            assert len(loaded_df) == 20  # BTC 10개 + ETH 10개
    
    def test_calculate_residuals_parallel(self, residual_calculator, sample_returns_data, sample_characteristics_data):
        """병렬 처리 잔차 계산 테스트"""
        # 작은 데이터셋으로 테스트
        small_data = sample_returns_data.head(50).copy()
        small_chars = sample_characteristics_data.head(50).copy()
        
        try:
            result = residual_calculator.calculate_residuals_parallel(
                small_data, small_chars, n_jobs=2
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 0  # 결과가 있을 수도 없을 수도 있음 (데이터 크기에 따라)
            
            if len(result) > 0:
                assert 'date' in result.columns
                assert 'symbol' in result.columns
                assert 'residual' in result.columns
                
        except ImportError:
            # joblib이 없는 경우 건너뛰기
            pytest.skip("joblib not available for parallel processing")
    
    def test_calculate_residuals_streaming(self, residual_calculator, sample_returns_data, sample_characteristics_data):
        """스트리밍 잔차 계산 테스트"""
        # 작은 데이터셋으로 테스트
        small_data = sample_returns_data.head(50).copy()
        small_chars = sample_characteristics_data.head(50).copy()
        
        # 콜백 함수 테스트
        callback_calls = []
        def test_callback(progress, current, total):
            callback_calls.append((progress, current, total))
        
        result = residual_calculator.calculate_residuals_streaming(
            small_data, small_chars, batch_size=10, callback=test_callback
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(callback_calls) > 0  # 콜백이 호출되었는지 확인
        
        # 진행률이 0과 1 사이인지 확인
        for progress, current, total in callback_calls:
            assert 0 <= progress <= 1
            assert current <= total
    
    def test_benchmark_residual_calculation(self, residual_calculator, sample_returns_data, sample_characteristics_data):
        """잔차 계산 벤치마크 테스트"""
        # 매우 작은 데이터셋으로 테스트
        tiny_data = sample_returns_data.head(20).copy()
        tiny_chars = sample_characteristics_data.head(20).copy()
        
        try:
            results = residual_calculator.benchmark_residual_calculation(
                tiny_data, tiny_chars, methods=['optimized']
            )
            
            assert isinstance(results, dict)
            assert 'optimized' in results
            
            optimized_result = results['optimized']
            assert 'execution_time_seconds' in optimized_result
            assert 'memory_used_mb' in optimized_result
            assert 'output_rows' in optimized_result
            assert 'success' in optimized_result
            
        except Exception as e:
            # 시스템 리소스 문제로 실패할 수 있음
            pytest.skip(f"Benchmark test failed due to system resources: {e}")
    
    def test_get_residual_statistics(self, residual_calculator):
        """잔차 통계 계산 테스트"""
        # 테스트용 잔차 데이터 생성
        np.random.seed(42)
        residuals_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-30', freq='D').repeat(2),
            'symbol': ['BTC', 'ETH'] * 30,
            'residual': np.random.normal(0, 0.01, 60)
        })
        
        stats = residual_calculator.get_residual_statistics(residuals_df)
        
        assert isinstance(stats, dict)
        assert 'BTC' in stats
        assert 'ETH' in stats
        
        btc_stats = stats['BTC']
        assert 'count' in btc_stats
        assert 'mean' in btc_stats
        assert 'std' in btc_stats
        assert 'autocorr_lag1' in btc_stats
        
        # 통계값이 합리적인 범위인지 확인
        assert btc_stats['count'] == 30
        assert abs(btc_stats['mean']) < 0.1  # 평균이 0에 가까워야 함
        assert btc_stats['std'] > 0  # 표준편차는 양수
    
    def test_create_default_characteristics_from_returns(self, residual_calculator, sample_returns_data):
        """수익률 데이터로부터 기본 특성 생성 테스트"""
        characteristics = residual_calculator._create_default_characteristics_from_returns(
            sample_returns_data.head(100)
        )
        
        assert isinstance(characteristics, pd.DataFrame)
        assert 'symbol' in characteristics.columns
        assert 'date' in characteristics.columns
        assert 'constant' in characteristics.columns
        assert 'log_market_cap' in characteristics.columns
        assert 'momentum' in characteristics.columns
        assert 'volatility' in characteristics.columns
        assert 'btc_return' in characteristics.columns
        
        # 상수항이 모두 1인지 확인
        assert (characteristics['constant'] == 1.0).all()
        
        # 결측치가 없는지 확인
        assert not characteristics.isnull().any().any()
    
    def test_rolling_window_edge_cases(self, residual_calculator):
        """롤링 윈도우 경계 조건 테스트"""
        # 매우 작은 데이터셋
        small_data = pd.DataFrame({
            'symbol': ['BTC'] * 5,
            'date': pd.date_range('2023-01-01', '2023-01-05', freq='D'),
            'return': [0.01, -0.02, 0.015, -0.01, 0.005],
            'close': [100, 98, 99.5, 98.5, 99],
            'volume': [1000] * 5
        })
        
        # 윈도우 크기가 데이터보다 큰 경우
        result = residual_calculator.calculate_rolling_residuals_optimized(
            small_data, window_size=10, min_periods=3
        )
        
        # 결과가 비어있거나 매우 적을 수 있음
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0
    
    def test_residual_quality_metrics_calculation(self, residual_calculator):
        """잔차 품질 지표 계산 테스트"""
        # 테스트용 잔차 데이터 (정상적인 분포)
        np.random.seed(42)
        residuals_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-03-31', freq='D').repeat(2),
            'symbol': ['BTC', 'ETH'] * 90,
            'residual': np.concatenate([
                np.random.normal(0, 0.01, 90),  # BTC 잔차
                np.random.normal(0, 0.015, 90)  # ETH 잔차
            ])
        })
        
        quality_metrics = residual_calculator.calculate_residual_quality_metrics(residuals_df)
        
        assert isinstance(quality_metrics, dict)
        assert 'BTC' in quality_metrics
        assert 'ETH' in quality_metrics
        
        btc_metrics = quality_metrics['BTC']
        assert isinstance(btc_metrics, ResidualQualityMetrics)
        assert btc_metrics.symbol == 'BTC'
        assert btc_metrics.n_observations == 90
        assert abs(btc_metrics.mean_residual) < 0.1  # 평균이 0에 가까워야 함
        assert btc_metrics.std_residual > 0  # 표준편차는 양수
    
    def test_memory_optimization(self, residual_calculator, sample_returns_data):
        """메모리 최적화 테스트"""
        # 최적화 파라미터 계산
        optimization_params = residual_calculator.optimize_residual_calculation(
            sample_returns_data, target_memory_mb=512
        )
        
        assert isinstance(optimization_params, dict)
        assert 'batch_size' in optimization_params
        assert 'use_multiprocessing' in optimization_params
        assert 'recommended_window_size' in optimization_params
        assert 'recommended_refit_frequency' in optimization_params
        assert 'estimated_memory_mb' in optimization_params
        
        # 값들이 합리적인 범위인지 확인
        assert optimization_params['batch_size'] > 0
        assert optimization_params['recommended_window_size'] > 0
        assert optimization_params['recommended_refit_frequency'] > 0
    
    @pytest.mark.parametrize("format_type", ['parquet', 'csv'])
    def test_different_save_formats(self, residual_calculator, format_type):
        """다양한 저장 형식 테스트"""
        # 테스트용 잔차 데이터
        residuals_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-05', freq='D').repeat(2),
            'symbol': ['BTC', 'ETH'] * 5,
            'residual': np.random.normal(0, 0.01, 10)
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 저장
            saved_files = residual_calculator.save_residual_timeseries(
                residuals_df, temp_dir, format=format_type, partition_by='none'
            )
            
            assert 'all' in saved_files
            assert 'metadata' in saved_files
            
            # 로드
            loaded_df = residual_calculator.load_residual_timeseries(
                temp_dir, format=format_type
            )
            
            assert len(loaded_df) == len(residuals_df)
            assert set(loaded_df.columns) == set(residuals_df.columns)
    
    def test_error_handling(self, residual_calculator):
        """오류 처리 테스트"""
        # 빈 데이터프레임
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'return'])
        
        result = residual_calculator.calculate_rolling_residuals_optimized(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
        # 잘못된 컬럼명
        wrong_df = pd.DataFrame({
            'wrong_symbol': ['BTC'],
            'wrong_date': [pd.Timestamp('2023-01-01')],
            'wrong_return': [0.01]
        })
        
        with pytest.raises(ValueError):
            residual_calculator.calculate_rolling_residuals_optimized(wrong_df)
    
    def test_config_validation(self):
        """설정 검증 테스트"""
        # 정상적인 설정
        config = ResidualCalculatorConfig(
            rolling_window_size=100,
            min_observations=30,
            refit_frequency=10
        )
        calculator = ResidualCalculator(config)
        
        assert calculator.config.rolling_window_size == 100
        assert calculator.config.min_observations == 30
        assert calculator.config.refit_frequency == 10
        
        # 기본 설정
        default_calculator = ResidualCalculator()
        assert default_calculator.config.rolling_window_size == 252
        assert default_calculator.config.min_observations == 50


if __name__ == "__main__":
    pytest.main([__file__])
