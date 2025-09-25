"""
잔차 계산기 통합 테스트

실제 데이터와 유사한 환경에서 전체 워크플로우를 테스트합니다.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from crypto_dlsa_bot.ml.residual_calculator import ResidualCalculator, ResidualCalculatorConfig
from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel
from crypto_dlsa_bot.services.data_preprocessor import DataPreprocessor


class TestResidualCalculatorIntegration:
    """잔차 계산기 통합 테스트"""
    
    @pytest.fixture
    def realistic_crypto_data(self):
        """현실적인 암호화폐 데이터 생성"""
        np.random.seed(42)
        
        # 주요 암호화폐들
        symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'SUSHI']
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        
        data = []
        
        # 각 암호화폐별로 시계열 데이터 생성
        for symbol in symbols:
            # 심볼별 특성 설정
            if symbol == 'BTC':
                base_vol = 0.03
                market_beta = 1.0
            elif symbol == 'ETH':
                base_vol = 0.04
                market_beta = 1.2
            else:
                base_vol = np.random.uniform(0.05, 0.08)
                market_beta = np.random.uniform(0.8, 1.5)
            
            # 시장 팩터 (공통)
            market_returns = np.random.normal(0, 0.02, len(dates))
            
            # 심볼별 고유 팩터
            idiosyncratic_returns = np.random.normal(0, base_vol * 0.5, len(dates))
            
            for i, date in enumerate(dates):
                # 수익률 계산
                market_component = market_beta * market_returns[i]
                idiosyncratic_component = idiosyncratic_returns[i]
                noise = np.random.normal(0, base_vol * 0.3)
                
                total_return = market_component + idiosyncratic_component + noise
                
                # 가격 및 거래량 (로그 정규분포)
                base_price = 100 if symbol == 'BTC' else np.random.uniform(1, 50)
                price = base_price * np.exp(np.random.normal(0, 0.1))
                volume = np.random.lognormal(15, 1)  # 큰 거래량
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'return': total_return,
                    'close': price,
                    'volume': volume,
                    'market_cap': price * volume * np.random.uniform(0.1, 1.0),  # 시가총액 프록시
                    'true_market_beta': market_beta  # 실제 베타 (검증용)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def residual_calculator_production(self):
        """프로덕션 환경과 유사한 설정의 잔차 계산기"""
        config = ResidualCalculatorConfig(
            rolling_window_size=252,  # 1년
            min_observations=60,      # 최소 2개월
            refit_frequency=21,       # 월 1회 재학습
            quality_check_enabled=True,
            outlier_threshold=3.0,
            autocorr_threshold=0.1
        )
        return ResidualCalculator(config)
    
    def test_end_to_end_residual_calculation(self, residual_calculator_production, realistic_crypto_data):
        """전체 워크플로우 end-to-end 테스트"""
        # 1. 데이터 전처리
        preprocessor = DataPreprocessor()
        
        # 기본 검증
        cleaned_data = preprocessor.validate_ohlcv_data(realistic_crypto_data)
        assert len(cleaned_data) > 0
        
        # 2. 특성 데이터 생성
        characteristics = self._create_comprehensive_characteristics(cleaned_data)
        
        # 3. 롤링 윈도우 잔차 계산
        residuals_df = residual_calculator_production.calculate_rolling_residuals_optimized(
            cleaned_data, characteristics, 
            window_size=100,  # 테스트용으로 작게 설정
            min_periods=30
        )
        
        # 4. 결과 검증
        assert isinstance(residuals_df, pd.DataFrame)
        assert len(residuals_df) > 0
        assert set(residuals_df.columns) >= {'date', 'symbol', 'residual'}
        
        # 5. 품질 지표 계산
        quality_metrics = residual_calculator_production.calculate_residual_quality_metrics(residuals_df)
        assert len(quality_metrics) > 0
        
        # 6. 통계적 검증
        for symbol, metrics in quality_metrics.items():
            # 잔차 평균이 0에 가까운지 확인
            assert abs(metrics.mean_residual) < 0.05, f"{symbol} 잔차 평균이 너무 큼: {metrics.mean_residual}"
            
            # 자기상관이 낮은지 확인
            assert abs(metrics.autocorr_lag1) < 0.3, f"{symbol} 자기상관이 너무 높음: {metrics.autocorr_lag1}"
    
    def test_performance_with_large_dataset(self, residual_calculator_production):
        """대용량 데이터셋 성능 테스트"""
        # 대용량 데이터 생성 (1년치 일일 데이터, 20개 암호화폐)
        np.random.seed(42)
        
        symbols = [f'CRYPTO_{i:02d}' for i in range(20)]
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        large_data = []
        for symbol in symbols:
            for date in dates:
                large_data.append({
                    'symbol': symbol,
                    'date': date,
                    'return': np.random.normal(0, 0.03),
                    'close': np.random.uniform(10, 1000),
                    'volume': np.random.lognormal(12, 1)
                })
        
        large_df = pd.DataFrame(large_data)
        
        # 성능 측정
        import time
        start_time = time.time()
        
        # 스트리밍 방식으로 처리 (메모리 효율적)
        result = residual_calculator_production.calculate_residuals_streaming(
            large_df, batch_size=100
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 성능 검증
        assert execution_time < 300, f"처리 시간이 너무 오래 걸림: {execution_time}초"
        assert isinstance(result, pd.DataFrame)
        
        # 처리량 계산
        throughput = len(large_df) / execution_time
        print(f"처리량: {throughput:.1f} 행/초")
        assert throughput > 10, "처리량이 너무 낮음"
    
    def test_model_persistence_and_reuse(self, residual_calculator_production, realistic_crypto_data):
        """모델 지속성 및 재사용 테스트"""
        # 첫 번째 기간 데이터
        first_period = realistic_crypto_data[
            realistic_crypto_data['date'] <= pd.Timestamp('2023-06-30')
        ].copy()
        
        # 두 번째 기간 데이터
        second_period = realistic_crypto_data[
            realistic_crypto_data['date'] > pd.Timestamp('2023-06-30')
        ].copy()
        
        # 첫 번째 기간으로 모델 학습
        first_residuals = residual_calculator_production.calculate_rolling_residuals_optimized(
            first_period, window_size=50, save_models=True
        )
        
        # 학습된 모델이 저장되었는지 확인
        assert len(residual_calculator_production.fitted_models) > 0
        
        # 두 번째 기간에서 기존 모델 재사용
        second_residuals = residual_calculator_production.calculate_rolling_residuals_optimized(
            second_period, window_size=50, save_models=False
        )
        
        # 결과 검증
        assert len(first_residuals) > 0
        assert len(second_residuals) > 0
        
        # 모델 재사용으로 인한 일관성 확인
        common_symbols = set(first_residuals['symbol']) & set(second_residuals['symbol'])
        assert len(common_symbols) > 0
    
    def test_data_quality_impact_on_residuals(self, residual_calculator_production):
        """데이터 품질이 잔차에 미치는 영향 테스트"""
        np.random.seed(42)
        
        # 고품질 데이터 생성
        high_quality_data = self._generate_clean_data(n_symbols=5, n_days=100)
        
        # 저품질 데이터 생성 (결측치, 이상치 포함)
        low_quality_data = self._generate_noisy_data(n_symbols=5, n_days=100)
        
        # 각각에 대해 잔차 계산
        high_quality_residuals = residual_calculator_production.calculate_rolling_residuals_optimized(
            high_quality_data, window_size=30, min_periods=15
        )
        
        low_quality_residuals = residual_calculator_production.calculate_rolling_residuals_optimized(
            low_quality_data, window_size=30, min_periods=15
        )
        
        # 품질 지표 비교
        if len(high_quality_residuals) > 0 and len(low_quality_residuals) > 0:
            hq_metrics = residual_calculator_production.calculate_residual_quality_metrics(high_quality_residuals)
            lq_metrics = residual_calculator_production.calculate_residual_quality_metrics(low_quality_residuals)
            
            # 고품질 데이터의 잔차가 더 좋은 품질을 가져야 함
            if hq_metrics and lq_metrics:
                hq_score = residual_calculator_production._calculate_overall_quality_score(hq_metrics)
                lq_score = residual_calculator_production._calculate_overall_quality_score(lq_metrics)
                
                # 고품질 데이터의 점수가 더 높아야 함 (또는 비슷해야 함)
                assert hq_score >= lq_score * 0.8, f"고품질 데이터 점수가 예상보다 낮음: {hq_score} vs {lq_score}"
    
    def test_cross_validation_residuals(self, residual_calculator_production, realistic_crypto_data):
        """교차 검증을 통한 잔차 안정성 테스트"""
        # 데이터를 시간순으로 3개 구간으로 분할
        data_sorted = realistic_crypto_data.sort_values('date')
        n_total = len(data_sorted)
        
        fold1 = data_sorted.iloc[:n_total//3]
        fold2 = data_sorted.iloc[n_total//3:2*n_total//3]
        fold3 = data_sorted.iloc[2*n_total//3:]
        
        folds = [fold1, fold2, fold3]
        fold_results = []
        
        for i, fold in enumerate(folds):
            if len(fold) < 100:  # 최소 데이터 요구사항
                continue
                
            residuals = residual_calculator_production.calculate_rolling_residuals_optimized(
                fold, window_size=30, min_periods=15
            )
            
            if len(residuals) > 0:
                quality_metrics = residual_calculator_production.calculate_residual_quality_metrics(residuals)
                quality_score = residual_calculator_production._calculate_overall_quality_score(quality_metrics)
                fold_results.append(quality_score)
        
        # 교차 검증 결과 안정성 확인
        if len(fold_results) >= 2:
            score_std = np.std(fold_results)
            score_mean = np.mean(fold_results)
            
            # 변동계수가 너무 크지 않아야 함
            cv = score_std / score_mean if score_mean > 0 else float('inf')
            assert cv < 0.5, f"교차 검증 결과가 불안정함: CV={cv}"
    
    def test_residual_storage_and_retrieval_workflow(self, residual_calculator_production, realistic_crypto_data):
        """잔차 저장 및 검색 워크플로우 테스트"""
        # 잔차 계산
        residuals_df = residual_calculator_production.calculate_rolling_residuals_optimized(
            realistic_crypto_data.head(200), window_size=30, min_periods=15
        )
        
        if len(residuals_df) == 0:
            pytest.skip("잔차 계산 결과가 없어 저장 테스트를 건너뜁니다")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 심볼별 파티션으로 저장
            saved_files = residual_calculator_production.save_residual_timeseries(
                residuals_df, temp_dir, format='parquet', partition_by='symbol'
            )
            
            # 2. 특정 심볼만 로드
            symbols_to_load = list(residuals_df['symbol'].unique())[:2]
            loaded_partial = residual_calculator_production.load_residual_timeseries(
                temp_dir, symbols=symbols_to_load, format='parquet'
            )
            
            # 3. 날짜 범위로 필터링하여 로드
            date_range = (
                residuals_df['date'].min(),
                residuals_df['date'].min() + timedelta(days=30)
            )
            loaded_date_filtered = residual_calculator_production.load_residual_timeseries(
                temp_dir, date_range=date_range, format='parquet'
            )
            
            # 검증
            assert len(loaded_partial) > 0
            assert set(loaded_partial['symbol'].unique()) == set(symbols_to_load)
            
            assert len(loaded_date_filtered) > 0
            assert loaded_date_filtered['date'].max() <= date_range[1]
    
    def _create_comprehensive_characteristics(self, returns_data):
        """포괄적인 특성 데이터 생성"""
        characteristics = returns_data[['symbol', 'date']].copy()
        
        # 시가총액 기반 특성
        characteristics['log_market_cap'] = np.log(returns_data['market_cap'] + 1)
        characteristics['market_cap_rank'] = returns_data.groupby('date')['market_cap'].rank(ascending=False)
        
        # 기술적 지표
        returns_pivot = returns_data.pivot(index='date', columns='symbol', values='return')
        
        # 모멘텀 (다양한 기간)
        momentum_5d = returns_pivot.rolling(5).sum().stack().reset_index()
        momentum_21d = returns_pivot.rolling(21).sum().stack().reset_index()
        momentum_5d.columns = ['date', 'symbol', 'momentum_5d']
        momentum_21d.columns = ['date', 'symbol', 'momentum_21d']
        
        # 변동성 (다양한 기간)
        volatility_5d = returns_pivot.rolling(5).std().stack().reset_index()
        volatility_21d = returns_pivot.rolling(21).std().stack().reset_index()
        volatility_5d.columns = ['date', 'symbol', 'volatility_5d']
        volatility_21d.columns = ['date', 'symbol', 'volatility_21d']
        
        # 특성 병합
        for df in [momentum_5d, momentum_21d, volatility_5d, volatility_21d]:
            characteristics = characteristics.merge(df, on=['symbol', 'date'], how='left')
        
        # 결측치 처리
        characteristics = characteristics.fillna(0)
        
        return characteristics
    
    def _generate_clean_data(self, n_symbols=5, n_days=100):
        """깨끗한 테스트 데이터 생성"""
        np.random.seed(42)
        
        symbols = [f'CLEAN_{i}' for i in range(n_symbols)]
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'return': np.random.normal(0, 0.02),
                    'close': np.random.uniform(50, 200),
                    'volume': np.random.lognormal(10, 0.5),
                    'market_cap': np.random.lognormal(15, 1)
                })
        
        return pd.DataFrame(data)
    
    def _generate_noisy_data(self, n_symbols=5, n_days=100):
        """노이즈가 많은 테스트 데이터 생성"""
        np.random.seed(42)
        
        symbols = [f'NOISY_{i}' for i in range(n_symbols)]
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                # 가끔 극단적인 값 추가
                if np.random.random() < 0.05:  # 5% 확률로 이상치
                    return_val = np.random.normal(0, 0.2)  # 큰 변동성
                else:
                    return_val = np.random.normal(0, 0.03)
                
                # 가끔 결측치 시뮬레이션 (NaN 대신 0 사용)
                if np.random.random() < 0.02:  # 2% 확률로 문제 데이터
                    return_val = 0
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'return': return_val,
                    'close': np.random.uniform(10, 500),
                    'volume': np.random.lognormal(8, 2),  # 더 큰 변동성
                    'market_cap': np.random.lognormal(12, 2)
                })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    pytest.main([__file__])