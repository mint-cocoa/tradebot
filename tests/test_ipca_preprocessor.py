#!/usr/bin/env python3
"""
IPCA 데이터 전처리기 단위 테스트

IPCADataPreprocessor 클래스의 기능을 테스트합니다.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto_dlsa_bot.ml.ipca_preprocessor import IPCADataPreprocessor, IPCAPreprocessorConfig


class TestIPCADataPreprocessor:
    """IPCA 데이터 전처리기 테스트 클래스"""
    
    @pytest.fixture
    def sample_crypto_data(self):
        """테스트용 암호화폐 데이터 생성"""
        np.random.seed(42)
        
        symbols = ['BTC', 'ETH', 'ADA', 'BNB', 'SOL']
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        
        data = []
        for symbol in symbols:
            base_price = np.random.uniform(100, 50000)
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
                        'date': date,
                        'symbol': symbol,
                        'close': price,
                        'volume': volume,
                        'return': daily_return
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self):
        """기본 전처리기 인스턴스"""
        config = IPCAPreprocessorConfig(
            min_observations_per_asset=10,
            min_assets_per_period=3,
            standardize_characteristics=True
        )
        return IPCADataPreprocessor(config)
    
    def test_initialization(self):
        """전처리기 초기화 테스트"""
        # 기본 설정으로 초기화
        preprocessor = IPCADataPreprocessor()
        assert preprocessor.config is not None
        assert preprocessor.symbol_to_id_map is None
        assert preprocessor.date_to_time_map is None
        
        # 커스텀 설정으로 초기화
        config = IPCAPreprocessorConfig(min_observations_per_asset=20)
        preprocessor = IPCADataPreprocessor(config)
        assert preprocessor.config.min_observations_per_asset == 20
    
    def test_create_entity_time_mapping(self, preprocessor, sample_crypto_data):
        """엔티티-시간 매핑 생성 테스트"""
        symbol_map, date_map = preprocessor.create_entity_time_mapping(sample_crypto_data)
        
        # 매핑 검증
        assert len(symbol_map) == 5  # BTC, ETH, ADA, BNB, SOL
        assert len(date_map) > 0
        assert all(isinstance(v, int) for v in symbol_map.values())
        assert all(isinstance(v, int) for v in date_map.values())
        
        # 역매핑 검증
        assert preprocessor.id_to_symbol_map is not None
        assert preprocessor.time_to_date_map is not None
        
        # 매핑 일관성 검증
        for symbol, id_val in symbol_map.items():
            assert preprocessor.id_to_symbol_map[id_val] == symbol
    
    def test_extract_characteristics(self, preprocessor, sample_crypto_data):
        """특성 변수 추출 테스트"""
        result_df = preprocessor.extract_characteristics(sample_crypto_data)
        
        # 새로 생성된 특성 변수들 확인
        expected_chars = ['market_cap', 'momentum', 'volatility', 'nvt_ratio']
        for char in expected_chars:
            assert char in result_df.columns
        
        # 시가총액 계산 검증
        assert (result_df['market_cap'] == result_df['close'] * result_df['volume']).all()
        
        # 모멘텀 계산 검증 (NaN 제외)
        momentum_values = result_df['momentum'].dropna()
        assert len(momentum_values) > 0
        assert all(np.isfinite(momentum_values))
    
    def test_handle_missing_values(self, preprocessor, sample_crypto_data):
        """결측치 처리 테스트"""
        # 인위적으로 결측치 생성
        test_df = sample_crypto_data.copy()
        test_df = preprocessor.extract_characteristics(test_df)
        
        # 일부 값을 NaN으로 설정
        test_df.loc[test_df.index[:10], 'market_cap'] = np.nan
        test_df.loc[test_df.index[20:30], 'volume'] = np.nan
        
        # 결측치 처리
        result_df = preprocessor.handle_missing_values(test_df)
        
        # 결측치가 처리되었는지 확인
        for char in preprocessor.config.characteristics_to_use:
            if char in result_df.columns:
                assert not result_df[char].isna().any(), f"{char}에 여전히 결측치가 있습니다"
    
    def test_remove_outliers(self, preprocessor, sample_crypto_data):
        """이상치 제거 테스트"""
        test_df = sample_crypto_data.copy()
        test_df = preprocessor.extract_characteristics(test_df)
        
        # 인위적으로 이상치 생성
        test_df.loc[test_df.index[0], 'market_cap'] = test_df['market_cap'].mean() * 100
        
        original_max = test_df['market_cap'].max()
        result_df = preprocessor.remove_outliers(test_df)
        new_max = result_df['market_cap'].max()
        
        # 이상치가 제거되었는지 확인
        assert new_max < original_max
    
    def test_apply_transformations(self, preprocessor, sample_crypto_data):
        """변수 변환 테스트"""
        test_df = sample_crypto_data.copy()
        test_df = preprocessor.extract_characteristics(test_df)
        
        result_df = preprocessor.apply_transformations(test_df)
        
        # 로그 변환된 변수들 확인
        for var in preprocessor.config.log_transform_vars:
            log_var = f'log_{var}'
            if var in test_df.columns:
                assert log_var in result_df.columns
                
                # 로그 변환 검증 (양수 값에 대해)
                positive_mask = test_df[var] > 0
                if positive_mask.any():
                    expected_log = np.log(test_df.loc[positive_mask, var])
                    actual_log = result_df.loc[positive_mask, log_var]
                    np.testing.assert_array_almost_equal(expected_log, actual_log, decimal=10)
    
    def test_standardize_characteristics(self, preprocessor, sample_crypto_data):
        """특성 변수 표준화 테스트"""
        test_df = sample_crypto_data.copy()
        test_df = preprocessor.extract_characteristics(test_df)
        test_df = preprocessor.apply_transformations(test_df)
        test_df = preprocessor.handle_missing_values(test_df)
        
        result_df = preprocessor.standardize_characteristics(test_df)
        
        # 각 날짜별로 표준화 확인
        for char in preprocessor.config.characteristics_to_use:
            if char in result_df.columns:
                for date in result_df['date'].unique():
                    date_values = result_df[result_df['date'] == date][char]
                    if len(date_values) > 1:
                        # 평균이 0에 가까운지 확인 (부동소수점 오차 고려)
                        assert abs(date_values.mean()) < 1e-10
                        # 표준편차가 1에 가까운지 확인
                        if date_values.std() > 1e-8:
                            assert abs(date_values.std() - 1.0) < 1e-10
    
    def test_filter_data_quality(self, preprocessor, sample_crypto_data):
        """데이터 품질 필터링 테스트"""
        test_df = sample_crypto_data.copy()
        
        original_symbols = test_df['symbol'].nunique()
        original_dates = test_df['date'].nunique()
        
        result_df = preprocessor.filter_data_quality(test_df)
        
        # 필터링 후에도 데이터가 남아있는지 확인
        assert len(result_df) > 0
        assert result_df['symbol'].nunique() <= original_symbols
        assert result_df['date'].nunique() <= original_dates
        
        # 품질 기준 확인
        asset_counts = result_df.groupby('symbol').size()
        assert all(count >= preprocessor.config.min_observations_per_asset for count in asset_counts)
        
        period_counts = result_df.groupby('date')['symbol'].nunique()
        assert all(count >= preprocessor.config.min_assets_per_period for count in period_counts)
    
    def test_convert_to_ipca_format(self, preprocessor, sample_crypto_data):
        """IPCA 형식 변환 테스트"""
        test_df = sample_crypto_data.copy()
        test_df = preprocessor.extract_characteristics(test_df)
        test_df = preprocessor.apply_transformations(test_df)
        test_df = preprocessor.handle_missing_values(test_df)
        
        X, y = preprocessor.convert_to_ipca_format(test_df)
        
        # 형식 검증
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(X.index, pd.MultiIndex)
        assert isinstance(y.index, pd.MultiIndex)
        assert X.index.names == ['entity_id', 'time']
        assert y.index.names == ['entity_id', 'time']
        
        # 데이터 일관성 검증
        assert len(X) == len(y)
        assert len(X) > 0
        
        # 특성 변수 확인
        expected_chars = [char for char in preprocessor.config.characteristics_to_use 
                         if char in test_df.columns or f'log_{char}' in test_df.columns]
        assert len(X.columns) > 0
        
        # NaN 값이 없는지 확인
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_fit_transform_pipeline(self, preprocessor, sample_crypto_data):
        """전체 파이프라인 테스트"""
        X, y = preprocessor.fit_transform(sample_crypto_data)
        
        # 결과 검증
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0
        
        # 매핑이 생성되었는지 확인
        assert preprocessor.symbol_to_id_map is not None
        assert preprocessor.date_to_time_map is not None
        assert preprocessor.fitted_characteristics is not None
        
        # 데이터 품질 확인
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_transform_new_data(self, preprocessor, sample_crypto_data):
        """새 데이터 변환 테스트"""
        # 먼저 fit
        X_train, y_train = preprocessor.fit_transform(sample_crypto_data)
        
        # 새 데이터 생성 (일부 겹치는 심볼과 날짜)
        new_data = sample_crypto_data.iloc[-50:].copy()  # 마지막 50개 행
        
        # 새 데이터 변환
        X_new, y_new = preprocessor.transform(new_data)
        
        # 결과 검증
        assert isinstance(X_new, pd.DataFrame)
        assert isinstance(y_new, pd.Series)
        assert len(X_new) == len(y_new)
        
        # 컬럼이 동일한지 확인
        assert list(X_new.columns) == list(X_train.columns)
    
    def test_mapping_methods(self, preprocessor, sample_crypto_data):
        """매핑 메서드 테스트"""
        # fit 실행
        preprocessor.fit_transform(sample_crypto_data)
        
        # 매핑 반환 테스트
        symbol_map = preprocessor.get_symbol_mapping()
        date_map = preprocessor.get_date_mapping()
        id_to_symbol, time_to_date = preprocessor.get_reverse_mappings()
        
        assert isinstance(symbol_map, dict)
        assert isinstance(date_map, dict)
        assert isinstance(id_to_symbol, dict)
        assert isinstance(time_to_date, dict)
        
        assert len(symbol_map) > 0
        assert len(date_map) > 0
        assert len(id_to_symbol) == len(symbol_map)
        assert len(time_to_date) == len(date_map)
    
    def test_error_handling(self, preprocessor):
        """오류 처리 테스트"""
        # 빈 데이터프레임
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            preprocessor.fit_transform(empty_df)
        
        # 필수 컬럼 누락
        invalid_df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(Exception):
            preprocessor.fit_transform(invalid_df)
        
        # fit 없이 transform 시도
        preprocessor_new = IPCADataPreprocessor()
        sample_df = pd.DataFrame({
            'symbol': ['BTC', 'ETH'],
            'date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')],
            'return': [0.01, -0.02]
        })
        
        with pytest.raises(ValueError, match="전처리기가 아직 fit되지 않았습니다"):
            preprocessor_new.transform(sample_df)


if __name__ == "__main__":
    # 간단한 테스트 실행
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    symbols = ['BTC', 'ETH', 'ADA']
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    
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
                    'date': date,
                    'symbol': symbol,
                    'close': price,
                    'volume': volume,
                    'return': daily_return
                })
    
    df = pd.DataFrame(data)
    
    # 전처리기 테스트
    config = IPCAPreprocessorConfig(
        min_observations_per_asset=5,
        min_assets_per_period=2
    )
    preprocessor = IPCADataPreprocessor(config)
    
    print("전처리 파이프라인 테스트 시작...")
    X, y = preprocessor.fit_transform(df)
    
    print(f"결과:")
    print(f"  X 형태: {X.shape}")
    print(f"  y 형태: {y.shape}")
    print(f"  특성 변수: {X.columns.tolist()}")
    print(f"  심볼 매핑: {preprocessor.get_symbol_mapping()}")
    
    print("\n✅ 전처리기 테스트 완료!")
