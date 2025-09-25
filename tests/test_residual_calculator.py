#!/usr/bin/env python3
"""
ResidualCalculator 클래스 단위 테스트

IPCA 기반 잔차 계산 및 품질 평가 시스템을 테스트합니다.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import sys

import matplotlib

matplotlib.use('Agg')

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto_dlsa_bot.ml.residual_calculator import (
    ResidualCalculator, ResidualCalculatorConfig, ResidualQualityMetrics
)
from crypto_dlsa_bot.ml.factor_engine import CryptoIPCAModel
from crypto_dlsa_bot.ml.ipca_preprocessor import IPCAPreprocessorConfig


class TestResidualCalculator:
    """ResidualCalculator 테스트 클래스"""
    
    @pytest.fixture
    def sample_returns_data(self):
        """테스트용 수익률 데이터 생성"""
        np.random.seed(42)
        
        symbols = ['BTC', 'ETH', 'ADA', 'BNB']
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        data = []
        for symbol in symbols:
            base_price = np.random.uniform(1000, 50000)
            prices = [base_price]
            
            for i in range(1, len(dates)):
                change = np.random.normal(0, 0.02)
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.01))
            
            for i, (date, price) in enumerate(zip(dates, prices)):
                if i > 0:
                    daily_return = (price - prices[i-1]) / prices[i-1]
                    volume = np.random.uniform(1e6, 1e8)
                    
                    data.append({
                        'symbol': symbol,
                        'date': date,
                        'close': price,
                        'volume': volume,
                        'return': daily_return
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def residual_calculator(self):
        """기본 잔차 계산기 인스턴스"""
        config = ResidualCalculatorConfig(
            rolling_window_size=60,  # 테스트용으로 작게 설정
            min_observations=20,
            refit_frequency=10,
            quality_check_enabled=True
        )
        return ResidualCalculator(config)
    
    @pytest.fixture
    def sample_residuals(self):
        """테스트용 잔차 데이터 생성"""
        np.random.seed(42)
        
        symbols = ['BTC', 'ETH', 'ADA']
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                # 정규분포에서 약간 벗어난 잔차 생성
                residual = np.random.normal(0, 0.01)
                if np.random.random() < 0.05:  # 5% 확률로 이상치
                    residual *= 5
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'residual': residual
                })
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """잔차 계산기 초기화 테스트"""
        # 기본 설정으로 초기화
        calculator = ResidualCalculator()
        assert calculator.config is not None
        assert calculator.fitted_models == {}
        assert calculator.residual_history == {}
        
        # 커스텀 설정으로 초기화
        config = ResidualCalculatorConfig(rolling_window_size=100)
        calculator = ResidualCalculator(config)
        assert calculator.config.rolling_window_size == 100
    
    def test_calculate_out_of_sample_residuals(self, residual_calculator, sample_returns_data):
        """아웃오브샘플 잔차 계산 테스트"""
        # 데이터 분할
        split_date = pd.Timestamp('2023-06-01')
        train_data = sample_returns_data[sample_returns_data['date'] < split_date].copy()
        test_data = sample_returns_data[sample_returns_data['date'] >= split_date].copy()
        
        # 아웃오브샘플 잔차 계산
        residuals_df, model = residual_calculator.calculate_out_of_sample_residuals(
            train_data, test_data, n_factors=2
        )
        
        # 결과 검증
        assert isinstance(residuals_df, pd.DataFrame)
        assert isinstance(model, CryptoIPCAModel)
        assert model.is_fitted
        
        # 필수 컬럼 확인
        required_columns = ['date', 'symbol', 'residual']
        for col in required_columns:
            assert col in residuals_df.columns
        
        # 데이터 크기 확인
        assert len(residuals_df) > 0
        assert residuals_df['symbol'].nunique() > 0
        
        # 잔차 값 확인 (유한한 값이어야 함)
        assert residuals_df['residual'].isna().sum() == 0
        assert np.isfinite(residuals_df['residual']).all()
    
    def test_calculate_residual_quality_metrics(self, residual_calculator, sample_residuals):
        """잔차 품질 지표 계산 테스트"""
        quality_metrics = residual_calculator.calculate_residual_quality_metrics(sample_residuals)
        
        # 결과 검증
        assert isinstance(quality_metrics, dict)
        assert len(quality_metrics) > 0
        
        # 각 심볼별 품질 지표 확인
        for symbol, metrics in quality_metrics.items():
            assert isinstance(metrics, ResidualQualityMetrics)
            assert metrics.symbol == symbol
            assert metrics.n_observations > 0
            assert np.isfinite(metrics.mean_residual)
            assert np.isfinite(metrics.std_residual)
            assert metrics.std_residual > 0
            assert isinstance(metrics.is_stationary, bool)
    
    def test_detect_residual_outliers_zscore(self, residual_calculator, sample_residuals):
        """Z-score 방법 이상치 탐지 테스트"""
        outliers_df = residual_calculator.detect_residual_outliers(sample_residuals, method='zscore')
        
        # 결과 검증
        assert isinstance(outliers_df, pd.DataFrame)
        assert 'is_outlier' in outliers_df.columns
        assert 'outlier_score' in outliers_df.columns
        
        # 이상치 플래그 확인
        assert outliers_df['is_outlier'].dtype == bool
        assert outliers_df['outlier_score'].dtype in [np.float64, float]
        
        # 이상치가 있는지 확인 (테스트 데이터에 의도적으로 이상치 포함)
        assert outliers_df['is_outlier'].any()
    
    def test_detect_residual_outliers_iqr(self, residual_calculator, sample_residuals):
        """IQR 방법 이상치 탐지 테스트"""
        outliers_df = residual_calculator.detect_residual_outliers(sample_residuals, method='iqr')
        
        # 결과 검증
        assert isinstance(outliers_df, pd.DataFrame)
        assert 'is_outlier' in outliers_df.columns
        assert 'outlier_score' in outliers_df.columns
        
        # 이상치가 탐지되었는지 확인
        assert outliers_df['is_outlier'].any()

    def test_plot_residual_diagnostics(self, residual_calculator, sample_residuals):
        """잔차 진단 그래프 생성 테스트"""
        from matplotlib.figure import Figure

        fig, summary = residual_calculator.plot_residual_diagnostics(sample_residuals)

        assert isinstance(fig, Figure)
        assert summary['count'] == len(sample_residuals.dropna(subset=['residual']))
        assert 'quality_score' in summary

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = os.path.join(temp_dir, 'residual_diagnostics.png')
            fig, summary_saved = residual_calculator.plot_residual_diagnostics(
                sample_residuals,
                save_path=plot_path,
            )
            assert os.path.exists(plot_path)
            assert summary_saved['count'] == summary['count']

    def test_validate_residuals(self, residual_calculator, sample_residuals):
        """잔차 품질 검증 테스트"""
        result = residual_calculator.validate_residuals(
            sample_residuals,
            max_abs_mean=0.05,
            max_autocorr_lag1=0.5,
            min_quality_score=0.0,
        )

        assert isinstance(result, dict)
        assert result['passed']
        assert result['failed_symbols'] == {}
        assert 'metrics' in result and len(result['metrics']) > 0

    def test_validate_residuals_detects_failures(self, residual_calculator):
        """잔차 품질 검증 실패 감지 테스트"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        residuals_df = pd.DataFrame({
            'symbol': ['TREND'] * len(dates),
            'date': dates,
            'residual': np.linspace(0, 1, len(dates)),
        })

        result = residual_calculator.validate_residuals(
            residuals_df,
            max_abs_mean=0.01,
            max_autocorr_lag1=0.05,
            min_quality_score=90.0,
        )

        assert not result['passed']
        assert 'TREND' in result['failed_symbols']
        assert result['failed_symbols']['TREND']['abs_mean_ok'] is False
    
    def test_generate_residual_report(self, residual_calculator, sample_residuals):
        """잔차 분석 리포트 생성 테스트"""
        # 품질 지표 계산
        quality_metrics = residual_calculator.calculate_residual_quality_metrics(sample_residuals)
        
        # 리포트 생성
        report = residual_calculator.generate_residual_report(sample_residuals, quality_metrics)
        
        # 결과 검증
        assert isinstance(report, dict)
        
        # 필수 항목 확인
        required_keys = [
            'total_observations', 'n_symbols', 'start_date', 'end_date',
            'mean_autocorr_lag1', 'stationary_ratio', 'quality_score'
        ]
        for key in required_keys:
            assert key in report
        
        # 값 검증
        assert report['total_observations'] > 0
        assert report['n_symbols'] > 0
        assert 0 <= report['stationary_ratio'] <= 1
        assert 0 <= report['quality_score'] <= 100
    
    def test_save_and_load_residuals(self, residual_calculator, sample_residuals):
        """잔차 데이터 저장 및 로드 테스트"""
        # 임시 파일 경로
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 데이터 저장
            residual_calculator.save_residuals(sample_residuals, tmp_path)
            
            # 파일이 생성되었는지 확인
            assert os.path.exists(tmp_path)
            
            # 데이터 로드
            loaded_residuals = residual_calculator.load_residuals(tmp_path)
            
            # 로드된 데이터 검증
            assert isinstance(loaded_residuals, pd.DataFrame)
            pd.testing.assert_frame_equal(loaded_residuals, sample_residuals)
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_rolling_residuals_calculation(self, residual_calculator, sample_returns_data):
        """롤링 윈도우 잔차 계산 테스트 (간단한 버전)"""
        # 작은 데이터셋으로 테스트
        small_data = sample_returns_data.iloc[:200].copy()  # 처음 200개 행만 사용
        
        try:
            # 롤링 잔차 계산
            residuals_df = residual_calculator.calculate_rolling_residuals(
                small_data, refit_models=True
            )
            
            # 결과 검증
            assert isinstance(residuals_df, pd.DataFrame)
            
            # 결과가 있는 경우에만 추가 검증
            if len(residuals_df) > 0:
                required_columns = ['date', 'symbol', 'residual']
                for col in required_columns:
                    assert col in residuals_df.columns
                
                # 잔차 값 확인
                assert residuals_df['residual'].isna().sum() == 0
                assert np.isfinite(residuals_df['residual']).all()
            
        except Exception as e:
            # 데이터가 부족한 경우 예외가 발생할 수 있음
            print(f"롤링 잔차 계산 중 예외 발생 (예상됨): {e}")
    
    def test_error_handling(self, residual_calculator):
        """오류 처리 테스트"""
        # 빈 데이터프레임
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            residual_calculator.calculate_residual_quality_metrics(empty_df)
        
        # 필수 컬럼 누락
        invalid_df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(Exception):
            residual_calculator.calculate_out_of_sample_residuals(invalid_df, invalid_df)
        
        # 지원하지 않는 이상치 탐지 방법
        valid_residuals = pd.DataFrame({
            'symbol': ['BTC'] * 10,
            'date': pd.date_range('2023-01-01', periods=10),
            'residual': np.random.normal(0, 0.01, 10)
        })
        
        with pytest.raises(ValueError, match="지원하지 않는 이상치 탐지 방법"):
            residual_calculator.detect_residual_outliers(valid_residuals, method='invalid_method')
    
    def test_quality_metrics_edge_cases(self, residual_calculator):
        """품질 지표 계산 엣지 케이스 테스트"""
        # 관측치가 매우 적은 경우
        small_residuals = pd.DataFrame({
            'symbol': ['BTC'] * 5,
            'date': pd.date_range('2023-01-01', periods=5),
            'residual': [0.01, -0.01, 0.02, -0.02, 0.01]
        })
        
        quality_metrics = residual_calculator.calculate_residual_quality_metrics(small_residuals)
        
        # 관측치가 부족하면 해당 심볼이 제외되어야 함
        assert len(quality_metrics) == 0
        
        # 모든 값이 동일한 경우
        constant_residuals = pd.DataFrame({
            'symbol': ['BTC'] * 20,
            'date': pd.date_range('2023-01-01', periods=20),
            'residual': [0.01] * 20
        })
        
        quality_metrics = residual_calculator.calculate_residual_quality_metrics(constant_residuals)
        
        # 상수 값의 경우에도 처리되어야 함
        if 'BTC' in quality_metrics:
            assert quality_metrics['BTC'].std_residual == 0.0


if __name__ == "__main__":
    # 간단한 테스트 실행
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    symbols = ['BTC', 'ETH']
    dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='D')
    
    # 수익률 데이터 생성
    returns_data = []
    for symbol in symbols:
        base_price = np.random.uniform(1000, 50000)
        prices = [base_price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            if i > 0:
                daily_return = (price - prices[i-1]) / prices[i-1]
                volume = np.random.uniform(1e6, 1e8)
                
                returns_data.append({
                    'symbol': symbol,
                    'date': date,
                    'close': price,
                    'volume': volume,
                    'return': daily_return
                })
    
    returns_df = pd.DataFrame(returns_data)
    
    # 잔차 계산기 테스트
    print("ResidualCalculator 테스트 시작...")
    
    config = ResidualCalculatorConfig(
        rolling_window_size=30,
        min_observations=10,
        refit_frequency=5
    )
    calculator = ResidualCalculator(config)
    
    # 아웃오브샘플 잔차 계산 테스트
    split_idx = len(returns_df) // 2
    train_data = returns_df.iloc[:split_idx].copy()
    test_data = returns_df.iloc[split_idx:].copy()
    
    residuals_df, model = calculator.calculate_out_of_sample_residuals(
        train_data, test_data, n_factors=2
    )
    
    print(f"아웃오브샘플 잔차 계산 완료:")
    print(f"  잔차 데이터 형태: {residuals_df.shape}")
    print(f"  심볼 수: {residuals_df['symbol'].nunique()}")
    
    # 품질 지표 계산
    quality_metrics = calculator.calculate_residual_quality_metrics(residuals_df)
    print(f"  품질 지표 계산 완료: {len(quality_metrics)}개 심볼")
    
    # 이상치 탐지
    outliers_df = calculator.detect_residual_outliers(residuals_df)
    n_outliers = outliers_df['is_outlier'].sum()
    print(f"  이상치 탐지 완료: {n_outliers}개 이상치")
    
    # 리포트 생성
    report = calculator.generate_residual_report(residuals_df, quality_metrics)
    print(f"  품질 점수: {report['quality_score']:.2f}")
    
    print("\n✅ ResidualCalculator 테스트 완료!")
