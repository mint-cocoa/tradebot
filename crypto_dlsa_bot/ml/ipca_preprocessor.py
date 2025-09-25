"""
IPCA 데이터 전처리 파이프라인

Kelly, Pruitt, Su (2017) IPCA 라이브러리를 위한 암호화폐 데이터 전처리 모듈입니다.
암호화폐 시장 데이터를 IPCA 요구사항에 맞는 패널 데이터 형식으로 변환합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from crypto_dlsa_bot.utils.validation import validate_dataframe, validate_numeric_column

logger = logging.getLogger(__name__)


@dataclass
class IPCAPreprocessorConfig:
    """IPCA 전처리 설정"""
    min_observations_per_asset: int = 30  # 자산별 최소 관측치 수
    min_assets_per_period: int = 5        # 기간별 최소 자산 수
    characteristics_to_use: List[str] = None  # 사용할 특성 변수들
    log_transform_vars: List[str] = None     # 로그 변환할 변수들
    standardize_characteristics: bool = True  # 특성 변수 표준화 여부
    handle_missing_method: str = 'forward_fill'  # 결측치 처리 방법
    outlier_threshold: float = 3.0        # 이상치 임계값 (표준편차 배수)
    
    def __post_init__(self):
        if self.characteristics_to_use is None:
            self.characteristics_to_use = [
                'market_cap', 'volume', 'momentum', 'volatility', 'nvt_ratio', 'btc_return'
            ]
        if self.log_transform_vars is None:
            self.log_transform_vars = ['market_cap', 'volume']


class IPCADataPreprocessor:
    """
    IPCA를 위한 암호화폐 데이터 전처리기
    
    암호화폐 시장 데이터를 Kelly, Pruitt, Su (2017) IPCA 라이브러리가 
    요구하는 패널 데이터 형식으로 변환합니다.
    """
    
    def __init__(self, config: Optional[IPCAPreprocessorConfig] = None):
        """
        전처리기 초기화
        
        Args:
            config: 전처리 설정
        """
        self.config = config or IPCAPreprocessorConfig()
        self.symbol_to_id_map: Optional[Dict[str, int]] = None
        self.date_to_time_map: Optional[Dict[pd.Timestamp, int]] = None
        self.id_to_symbol_map: Optional[Dict[int, str]] = None
        self.time_to_date_map: Optional[Dict[int, pd.Timestamp]] = None
        self.fitted_characteristics: Optional[List[str]] = None
        
        logger.info("IPCA 데이터 전처리기 초기화 완료")
    
    def create_entity_time_mapping(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[pd.Timestamp, int]]:
        """
        엔티티(암호화폐)와 시간 매핑 생성
        
        Args:
            df: 원본 데이터프레임 (symbol, date 컬럼 포함)
            
        Returns:
            tuple: (symbol_to_id_map, date_to_time_map)
        """
        logger.info("엔티티-시간 매핑 생성 시작")
        
        # 심볼을 숫자 ID로 매핑
        unique_symbols = sorted(df['symbol'].unique())
        symbol_to_id = {symbol: i + 1 for i, symbol in enumerate(unique_symbols)}
        
        # 날짜를 숫자 시간 인덱스로 매핑
        unique_dates = sorted(df['date'].unique())
        date_to_time = {date: i + 1 for i, date in enumerate(unique_dates)}
        
        # 역매핑도 생성
        id_to_symbol = {v: k for k, v in symbol_to_id.items()}
        time_to_date = {v: k for k, v in date_to_time.items()}
        
        self.symbol_to_id_map = symbol_to_id
        self.date_to_time_map = date_to_time
        self.id_to_symbol_map = id_to_symbol
        self.time_to_date_map = time_to_date
        
        logger.info(f"매핑 생성 완료: {len(unique_symbols)}개 심볼, {len(unique_dates)}개 날짜")
        return symbol_to_id, date_to_time
    
    def extract_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        암호화폐 특성 변수 추출 및 계산
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            특성 변수가 포함된 데이터프레임
        """
        logger.info("암호화폐 특성 변수 추출 시작")
        
        result_df = df.copy()
        
        # 기본 특성 변수들이 없으면 계산
        if 'market_cap' not in result_df.columns and 'close' in result_df.columns and 'volume' in result_df.columns:
            # 시가총액 프록시 (가격 * 거래량)
            result_df['market_cap'] = result_df['close'] * result_df['volume']
            logger.info("시가총액 프록시 계산 완료")
        
        if 'momentum' not in result_df.columns and 'close' in result_df.columns:
            # 모멘텀 계산 (21일 수익률)
            result_df = result_df.sort_values(['symbol', 'date'])
            result_df['momentum'] = result_df.groupby('symbol')['close'].pct_change(periods=21)
            logger.info("모멘텀 계산 완료")
        
        if 'volatility' not in result_df.columns and 'close' in result_df.columns:
            # 변동성 계산 (30일 롤링 표준편차)
            result_df = result_df.sort_values(['symbol', 'date'])
            returns = result_df.groupby('symbol')['close'].pct_change()
            result_df['volatility'] = returns.groupby(result_df['symbol']).rolling(window=30).std().reset_index(0, drop=True)
            logger.info("변동성 계산 완료")
        
        if 'nvt_ratio' not in result_df.columns:
            # NVT 비율 프록시 (시가총액 / 거래량)
            if 'market_cap' in result_df.columns and 'volume' in result_df.columns:
                result_df['nvt_ratio'] = result_df['market_cap'] / (result_df['volume'] + 1e-8)  # 0으로 나누기 방지
                logger.info("NVT 비율 프록시 계산 완료")

        if 'btc_return' not in result_df.columns and 'return' in result_df.columns:
            logger.info("비트코인 수익률 특성 계산")
            temp_df = result_df.sort_values(['symbol', 'date'])
            returns_pivot = temp_df.pivot(index='date', columns='symbol', values='return')
            btc_symbol = next((sym for sym in returns_pivot.columns if sym.startswith('BTC')), None)
            if btc_symbol is not None:
                btc_returns = returns_pivot[btc_symbol].rename('btc_return').reset_index()
                result_df = result_df.merge(btc_returns, on='date', how='left')
            else:
                result_df['btc_return'] = 0.0
            result_df['btc_return'] = result_df['btc_return'].fillna(0.0)

        logger.info("특성 변수 추출 완료")
        return result_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 처리
        
        Args:
            df: 데이터프레임
            
        Returns:
            결측치가 처리된 데이터프레임
        """
        logger.info(f"결측치 처리 시작 (방법: {self.config.handle_missing_method})")
        
        result_df = df.copy()
        
        if self.config.handle_missing_method == 'forward_fill':
            # 심볼별로 전진 채우기
            result_df = result_df.sort_values(['symbol', 'date'])
            for char in self.config.characteristics_to_use:
                if char in result_df.columns:
                    result_df[char] = result_df.groupby('symbol')[char].ffill()
        
        elif self.config.handle_missing_method == 'interpolate':
            # 심볼별로 선형 보간
            result_df = result_df.sort_values(['symbol', 'date'])
            for char in self.config.characteristics_to_use:
                if char in result_df.columns:
                    result_df[char] = result_df.groupby('symbol')[char].apply(
                        lambda x: x.interpolate(method='linear')
                    ).reset_index(0, drop=True)
        
        elif self.config.handle_missing_method == 'drop':
            # 결측치가 있는 행 제거
            result_df = result_df.dropna(subset=self.config.characteristics_to_use)
        
        # 여전히 결측치가 있으면 후진 채우기
        for char in self.config.characteristics_to_use:
            if char in result_df.columns:
                result_df[char] = result_df.groupby('symbol')[char].bfill()

        # 그래도 결측치가 있으면 0으로 채우기 (존재하는 컬럼만 대상)
        existing_chars = [char for char in self.config.characteristics_to_use if char in result_df.columns]
        for char in existing_chars:
            result_df[char] = result_df[char].fillna(0)

        logger.info("결측치 처리 완료")
        return result_df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 제거
        
        Args:
            df: 데이터프레임
            
        Returns:
            이상치가 제거된 데이터프레임
        """
        logger.info(f"이상치 제거 시작 (임계값: {self.config.outlier_threshold}σ)")
        
        result_df = df.copy()
        
        for char in self.config.characteristics_to_use:
            if char in result_df.columns:
                # 각 시점별로 이상치 탐지 (횡단면 기준)
                for date in result_df['date'].unique():
                    date_mask = result_df['date'] == date
                    values = result_df.loc[date_mask, char]
                    if len(values) > 3:  # 최소 3개 관측치 필요
                        mean_val = values.mean()
                        std_val = values.std()
                        values_array = values.to_numpy(dtype=float)
                        if np.isnan(values_array).all():
                            continue
                        median_val = np.nanmedian(values_array)
                        mad = np.nanmedian(np.abs(values_array - median_val)) if len(values_array) > 0 else 0.0

                        if std_val > 0:
                            sigma_mask = np.abs(values - mean_val) > (self.config.outlier_threshold * std_val)
                        else:
                            sigma_mask = pd.Series(False, index=values.index)

                        if mad > 0:
                            mad_mask = np.abs(values - median_val) > (self.config.outlier_threshold * mad)
                        else:
                            mad_mask = pd.Series(False, index=values.index)

                        outlier_mask = sigma_mask | mad_mask

                        if outlier_mask.any():
                            result_df.loc[date_mask & outlier_mask, char] = median_val
        
        logger.info("이상치 제거 완료")
        return result_df
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        변수 변환 적용 (로그 변환 등)
        
        Args:
            df: 데이터프레임
            
        Returns:
            변환된 데이터프레임
        """
        logger.info("변수 변환 적용 시작")
        
        result_df = df.copy()
        
        # 로그 변환
        for var in self.config.log_transform_vars:
            if var in result_df.columns:
                # 양수 값만 로그 변환
                positive_mask = result_df[var] > 0
                if positive_mask.any():
                    log_var_name = f'log_{var}'
                    result_df[log_var_name] = np.nan
                    result_df.loc[positive_mask, log_var_name] = np.log(result_df.loc[positive_mask, var])
                    
                    # 원본 변수를 로그 변환된 변수로 교체
                    if var in self.config.characteristics_to_use:
                        idx = self.config.characteristics_to_use.index(var)
                        self.config.characteristics_to_use[idx] = log_var_name
                    
                    logger.info(f"{var} -> {log_var_name} 로그 변환 완료")
        
        logger.info("변수 변환 적용 완료")
        return result_df
    
    def standardize_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        특성 변수 표준화 (횡단면 기준)
        
        Args:
            df: 데이터프레임
            
        Returns:
            표준화된 데이터프레임
        """
        if not self.config.standardize_characteristics:
            return df
        
        logger.info("특성 변수 표준화 시작")
        
        result_df = df.copy()
        
        # 각 시점별로 횡단면 표준화
        for char in self.config.characteristics_to_use:
            if char in result_df.columns:
                for date in result_df['date'].unique():
                    date_mask = result_df['date'] == date
                    values = result_df.loc[date_mask, char]
                    
                    if len(values) > 1:
                        mean_val = values.mean()
                        std_val = values.std()
                        
                        if std_val > 1e-8:  # 표준편차가 0이 아닌 경우만
                            result_df.loc[date_mask, char] = (values - mean_val) / std_val
                        else:
                            result_df.loc[date_mask, char] = 0
        
        logger.info("특성 변수 표준화 완료")
        return result_df
    
    def filter_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 품질 필터링
        
        Args:
            df: 데이터프레임
            
        Returns:
            필터링된 데이터프레임
        """
        logger.info("데이터 품질 필터링 시작")
        
        result_df = df.copy()
        initial_count = len(result_df)
        
        # 자산별 최소 관측치 수 확인
        asset_counts = result_df.groupby('symbol').size()
        valid_assets = asset_counts[asset_counts >= self.config.min_observations_per_asset].index
        result_df = result_df[result_df['symbol'].isin(valid_assets)]
        
        logger.info(f"최소 관측치 필터링: {len(asset_counts)} -> {len(valid_assets)} 자산")
        
        # 기간별 최소 자산 수 확인
        period_counts = result_df.groupby('date')['symbol'].nunique()
        valid_periods = period_counts[period_counts >= self.config.min_assets_per_period].index
        result_df = result_df[result_df['date'].isin(valid_periods)]
        
        logger.info(f"최소 자산 수 필터링: {len(period_counts)} -> {len(valid_periods)} 기간")
        
        final_count = len(result_df)
        logger.info(f"데이터 품질 필터링 완료: {initial_count} -> {final_count} 관측치")
        
        return result_df
    
    def convert_to_ipca_format(
        self,
        df: pd.DataFrame,
        returns_column: str = 'return',
        selected_characteristics: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        IPCA 라이브러리 입력 형식으로 변환
        
        Args:
            df: 전처리된 데이터프레임
            returns_column: 수익률 컬럼명
            
        Returns:
            tuple: (X, y) - IPCA 입력 형식
        """
        logger.info("IPCA 형식 변환 시작")
        
        # 매핑이 없으면 생성
        if self.symbol_to_id_map is None or self.date_to_time_map is None:
            self.create_entity_time_mapping(df)
        
        # 엔티티 ID와 시간 인덱스 추가
        result_df = df.copy()
        result_df['entity_id'] = result_df['symbol'].map(self.symbol_to_id_map)
        if self.date_to_time_map is not None:
            result_df['time'] = result_df['date'].map(self.date_to_time_map)
        else:
            # 안전장치: 매핑이 없으면 생성
            self.create_entity_time_mapping(result_df)
            result_df['time'] = result_df['date'].map(self.date_to_time_map)

        # 특성 변수 선택
        if selected_characteristics is not None:
            available_chars = [char for char in selected_characteristics if char in result_df.columns]
        else:
            available_chars = [char for char in self.config.characteristics_to_use if char in result_df.columns]
            self.fitted_characteristics = available_chars

        # X: 특성 데이터 (MultiIndex DataFrame with entity/time)
        index_columns = ['entity_id', 'time']
        X = result_df[index_columns + available_chars].copy()
        X = X.set_index(index_columns).sort_index()

        # y: 수익률 데이터 (Series with MultiIndex)
        y = result_df.set_index(index_columns)[returns_column]

        # NaN 제거
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"IPCA 형식 변환 완료: X {X.shape}, y {y.shape}")
        logger.info(f"사용된 특성 변수: {available_chars}")

        return X, y
    
    def fit_transform(self, df: pd.DataFrame, returns_column: str = 'return') -> Tuple[pd.DataFrame, pd.Series]:
        """
        전체 전처리 파이프라인 실행
        
        Args:
            df: 원본 데이터프레임
            returns_column: 수익률 컬럼명
            
        Returns:
            tuple: (X, y) - IPCA 입력 형식
        """
        logger.info("IPCA 전처리 파이프라인 시작")
        
        # 입력 데이터 검증
        required_columns = ['symbol', 'date', returns_column]
        validate_dataframe(df, required_columns)
        
        # 1. 특성 변수 추출
        processed_df = self.extract_characteristics(df)
        
        # 2. 변수 변환
        processed_df = self.apply_transformations(processed_df)
        
        # 3. 결측치 처리
        processed_df = self.handle_missing_values(processed_df)
        
        # 4. 이상치 제거
        processed_df = self.remove_outliers(processed_df)
        
        # 5. 데이터 품질 필터링
        processed_df = self.filter_data_quality(processed_df)
        
        # 6. 특성 변수 표준화
        processed_df = self.standardize_characteristics(processed_df)
        
        # 7. IPCA 형식 변환
        X, y = self.convert_to_ipca_format(
            processed_df,
            returns_column,
            selected_characteristics=self.fitted_characteristics
        )
        
        logger.info("IPCA 전처리 파이프라인 완료")
        return X, y
    
    def transform(self, df: pd.DataFrame, returns_column: str = 'return') -> Tuple[pd.DataFrame, pd.Series]:
        """
        이미 fit된 전처리기로 새 데이터 변환
        
        Args:
            df: 새로운 데이터프레임
            returns_column: 수익률 컬럼명
            
        Returns:
            tuple: (X, y) - IPCA 입력 형식
        """
        if self.fitted_characteristics is None:
            raise ValueError("전처리기가 아직 fit되지 않았습니다. fit_transform을 먼저 실행하세요.")
        
        logger.info("새 데이터 변환 시작")
        
        # 기존과 동일한 전처리 적용 (표준화 제외)
        processed_df = self.extract_characteristics(df)
        processed_df = self.apply_transformations(processed_df)
        processed_df = self.handle_missing_values(processed_df)
        processed_df = self.remove_outliers(processed_df)
        processed_df = self.filter_data_quality(processed_df)
        processed_df = self.standardize_characteristics(processed_df)

        # 기존 매핑 사용하여 IPCA 형식 변환
        X, y = self.convert_to_ipca_format(
            processed_df,
            returns_column,
            selected_characteristics=self.fitted_characteristics
        )

        logger.info("새 데이터 변환 완료")
        return X, y

    def transform_predict(self, df: pd.DataFrame, returns_column: str = 'return') -> Tuple[pd.DataFrame, pd.Series]:
        """
        예측(실시간/단일 시점) 용 변환: 품질 필터를 완화하여 단일 날짜 패널에서도 X를 생성.

        Note:
            - fitted_characteristics가 설정되어 있어야 합니다.
            - filter_data_quality를 건너뛰어 자산/기간 최소 개수 제약을 적용하지 않습니다.
        """
        if self.fitted_characteristics is None:
            raise ValueError("전처리기가 아직 fit되지 않았습니다. fit_transform을 먼저 실행하세요.")

        logger.info("예측용 데이터 변환 시작 (완화된 품질 필터)")

        processed_df = self.extract_characteristics(df)
        processed_df = self.apply_transformations(processed_df)
        processed_df = self.handle_missing_values(processed_df)
        processed_df = self.remove_outliers(processed_df)
        # 품질 필터링은 생략
        processed_df = self.standardize_characteristics(processed_df)

        X, y = self.convert_to_ipca_format(
            processed_df,
            returns_column,
            selected_characteristics=self.fitted_characteristics
        )

        logger.info("예측용 데이터 변환 완료")
        return X, y
    
    def get_symbol_mapping(self) -> Dict[str, int]:
        """심볼-ID 매핑 반환"""
        return self.symbol_to_id_map.copy() if self.symbol_to_id_map else {}
    
    def get_date_mapping(self) -> Dict[pd.Timestamp, int]:
        """날짜-시간 매핑 반환"""
        return self.date_to_time_map.copy() if self.date_to_time_map else {}
    
    def get_reverse_mappings(self) -> Tuple[Dict[int, str], Dict[int, pd.Timestamp]]:
        """역매핑 반환"""
        return (
            self.id_to_symbol_map.copy() if self.id_to_symbol_map else {},
            self.time_to_date_map.copy() if self.time_to_date_map else {}
        )
