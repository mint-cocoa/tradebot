"""
CryptoIPCAModel - Kelly, Pruitt, Su (2017) IPCA 라이브러리 기반 암호화폐 팩터 모델

이 모듈은 IPCA 라이브러리의 올바른 사용법을 구현합니다:
1. MultiIndex DataFrame (entity, time) 형식의 데이터 사용
2. X (특성 데이터)와 y (수익률)을 분리하여 fit 메서드에 전달
3. get_factors()로 Gamma (팩터 로딩)와 Factors (팩터 시계열) 추출
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import joblib

from ipca import InstrumentedPCA

from crypto_dlsa_bot.ml.ipca_preprocessor import IPCADataPreprocessor, IPCAPreprocessorConfig

logger = logging.getLogger(__name__)


class CryptoIPCAModel:
    """
    Kelly, Pruitt, Su (2017) IPCA 라이브러리 기반 암호화폐 팩터 모델
    
    Instrumented Principal Components Analysis를 사용하여 
    암호화폐 시장의 팩터를 추출하고 잔차를 계산합니다.
    """
    
    def __init__(self, 
                 n_factors: int = 3,
                 intercept: bool = False,
                 max_iter: int = 1000,
                 iter_tol: float = 1e-5,
                 alpha: float = 0.0,
                 l1_ratio: float = 1.0,
                 n_jobs: int = 1,
                 preprocessor_config: Optional[IPCAPreprocessorConfig] = None):
        """
        IPCA 모델 초기화
        
        Args:
            n_factors: 추출할 팩터 수
            intercept: 절편 포함 여부
            max_iter: 최대 반복 횟수
            iter_tol: 수렴 허용 오차
            alpha: 정규화 상수 (0이면 정규화 없음)
            l1_ratio: L1/L2 정규화 비율 (elastic net용)
            n_jobs: 병렬 처리 작업 수
            preprocessor_config: IPCA 전처리 설정 (선택적)
        """
        self.n_factors = n_factors
        self.intercept = intercept
        self.max_iter = max_iter
        self.iter_tol = iter_tol
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_jobs = n_jobs
        if preprocessor_config is None:
            preprocessor_config = IPCAPreprocessorConfig(min_assets_per_period=2)
        self.preprocessor_config = preprocessor_config
        
        # IPCA 모델 초기화
        self.ipca_model = InstrumentedPCA(
            n_factors=n_factors,
            intercept=intercept,
            max_iter=max_iter,
            iter_tol=iter_tol,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_jobs=n_jobs
        )
        
        self.is_fitted = False
        self.gamma = None  # 팩터 로딩 (L x K)
        self.factors = None  # 팩터 시계열 (K x T)
        self.asset_names = None
        self.factor_names = None
        self.characteristic_names = None
        # 호환성/진단용 속성
        self.symbol_mapping: Dict[str, int] = {}
        self.date_mapping: Dict[pd.Timestamp, int] = {}
        self.id_to_symbol: Dict[int, str] = {}
        self.time_to_date: Dict[int, pd.Timestamp] = {}
        self.fitted_characteristics: Optional[List[str]] = None
        self.preprocessor = IPCADataPreprocessor(self.preprocessor_config)
        self.summary = None
        self.r2_pred = None

        logger.info(f"CryptoIPCAModel 초기화: {n_factors}개 팩터, intercept={intercept}")
    
    def fit(self, returns: pd.DataFrame, characteristics: Optional[pd.DataFrame] = None) -> 'CryptoIPCAModel':
        """
        IPCA 모델 학습

        Args:
            returns: 수익률 데이터 (columns: symbol, date, return)
            characteristics: 특성 데이터 (columns: symbol, date, char1, char2, ...)
            
        Returns:
            학습된 IPCA 모델
        """
        logger.info("CryptoIPCAModel 학습 시작")

        # 입력 데이터 검증
        if returns.empty:
            raise ValueError("수익률 데이터가 비어있습니다")

        required_columns = ['symbol', 'date', 'return']
        missing_columns = [col for col in required_columns if col not in returns.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")

        returns_df = returns.copy()
        returns_df['date'] = pd.to_datetime(returns_df['date'])

        merged_data = self._merge_returns_with_characteristics(returns_df, characteristics)

        base_cols = {'symbol', 'date', 'return'}
        feature_cols = [col for col in merged_data.columns if col not in base_cols]
        if not feature_cols:
            logger.info("특성 정보가 없어 기본 특성을 생성합니다.")
            default_chars = self._create_default_characteristics(returns_df)
            merged_data = self._merge_returns_with_characteristics(returns_df, default_chars)

        # IPCA 형식으로 데이터 변환
        X, y = self.preprocessor.fit_transform(merged_data, returns_column='return')
        if X.empty:
            raise ValueError("전처리 결과가 비어 있습니다. 입력 데이터를 확인하세요.")

        # IPCA 모델 학습 (panel 데이터 사용)
        logger.info(f"IPCA 학습 시작: X shape={X.shape}, y shape={y.shape}")
        self.ipca_model.fit(X=X, y=y, data_type='panel')

        # 특성/메타데이터 저장
        self.characteristic_names = list(X.columns)
        self.fitted_characteristics = list(self.preprocessor.fitted_characteristics or self.characteristic_names)
        self.symbol_mapping = self.preprocessor.get_symbol_mapping()
        self.date_mapping = self.preprocessor.get_date_mapping()
        self.id_to_symbol, self.time_to_date = self.preprocessor.get_reverse_mappings()
        self.asset_names = sorted(self.symbol_mapping.keys())

        # 팩터 이름 구성 (절편 포함 여부 감안)
        gamma_arr = np.asarray(self.ipca_model.Gamma)
        factor_count = gamma_arr.shape[1] if gamma_arr.ndim > 1 else 1
        if self.intercept and factor_count == self.n_factors + 1:
            self.factor_names = [f'Factor_{i+1}' for i in range(self.n_factors)] + ['Intercept']
        else:
            self.factor_names = [f'Factor_{i+1}' for i in range(factor_count)]

        # Gamma 행렬 (특성 x 팩터)
        gamma_arr = gamma_arr.reshape(len(self.characteristic_names), factor_count)
        self.gamma = pd.DataFrame(gamma_arr, index=self.characteristic_names, columns=self.factor_names)

        # 팩터 시계열 (시간 x 팩터)
        factors_arr = np.asarray(self.ipca_model.Factors)
        if factors_arr.ndim == 1:
            factors_arr = factors_arr.reshape(1, -1)
        if factors_arr.shape[0] == len(self.factor_names):
            factors_arr = factors_arr.T
        ordered_dates: List[pd.Timestamp]
        if self.time_to_date:
            ordered_dates = [self.time_to_date[key] for key in sorted(self.time_to_date.keys())]
        else:
            ordered_dates = sorted(X.index.get_level_values('date').unique())
        if len(ordered_dates) != factors_arr.shape[0]:
            logger.warning(
                "팩터 시계열 길이(%d)와 날짜 매핑(%d)이 일치하지 않아 숫자 인덱스를 사용합니다.",
                factors_arr.shape[0],
                len(ordered_dates)
            )
            date_index = pd.RangeIndex(factors_arr.shape[0], name='time_step')
        else:
            date_index = pd.Index(ordered_dates, name='date')
        self.factors = pd.DataFrame(
            factors_arr,
            columns=self.factor_names,
            index=date_index
        )

        # 진단 정보 저장
        self.summary = getattr(self.ipca_model, 'summary', None)
        self.r2_pred = getattr(self.ipca_model, 'R2_pred', None)

        self.is_fitted = True

        logger.info("CryptoIPCAModel 학습 완료")
        logger.info(f"  팩터 로딩 Gamma 형태: {self.gamma.shape}")
        logger.info(f"  팩터 시계열 F 형태: {self.factors.shape}")
        logger.info(f"  자산 수: {len(self.asset_names)}")
        logger.info(f"  특성 수: {len(self.characteristic_names)}")

        return self
    
    def transform(self, returns: pd.DataFrame, characteristics: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        수익률을 팩터 노출도와 잔차로 변환

        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터 (선택적)
            
        Returns:
            tuple: (factor_exposures, residuals)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")

        logger.info("CryptoIPCAModel 변환 시작")

        returns_df = returns.copy()
        if 'date' not in returns_df.columns:
            raise ValueError("returns 데이터에 'date' 컬럼이 필요합니다")
        returns_df['date'] = pd.to_datetime(returns_df['date'])

        merged_data = self._merge_returns_with_characteristics(returns_df, characteristics)
        X_new, y_new = self.preprocessor.transform(merged_data, returns_column='return')
        if X_new.empty:
            raise ValueError("변환 결과가 비어 있습니다. 입력 데이터를 확인하세요.")

        predictions = self.ipca_model.predict(X=X_new, data_type='panel')
        if isinstance(predictions, (pd.Series, pd.DataFrame)):
            predictions = np.asarray(predictions).reshape(-1)
        expected_returns = np.asarray(predictions, dtype=float)
        if expected_returns.ndim > 1:
            expected_returns = expected_returns.reshape(-1)

        actual_returns = y_new.to_numpy(dtype=float)
        pred_len = len(expected_returns)
        actual_len = len(actual_returns)
        if pred_len != actual_len:
            logger.warning(
                "예측 벡터 길이(%d)와 실제 길이(%d)가 일치하지 않아 최소 길이에 맞춥니다.",
                pred_len,
                actual_len
            )
            min_len = min(pred_len, actual_len)
            expected_returns = expected_returns[:min_len]
            actual_returns = actual_returns[:min_len]
        residuals = actual_returns - expected_returns

        # 팩터 노출도 계산 (특성 * Gamma)
        if self.gamma is None:
            raise ValueError("모델이 학습되지 않았습니다: Gamma 가 없습니다")

        feature_matrix = X_new.reindex(columns=self.characteristic_names, fill_value=0.0).to_numpy(dtype=float)
        gamma_matrix = self.gamma.loc[self.characteristic_names].to_numpy(dtype=float)
        factor_exposures = feature_matrix @ gamma_matrix

        idx = X_new.index
        entity_ids = idx.get_level_values('entity_id').to_numpy(dtype=int)
        time_ids = idx.get_level_values('time').to_numpy(dtype=int)
        dates = np.array([
            self.time_to_date.get(int(time_id), time_id)
            for time_id in time_ids
        ])
        symbols = np.array([
            self.id_to_symbol.get(int(entity_id), f'Asset_{entity_id}')
            for entity_id in entity_ids
        ])

        target_len = len(actual_returns)
        if factor_exposures.shape[0] != target_len:
            factor_exposures = factor_exposures[:target_len]
        if len(entity_ids) != target_len:
            entity_ids = entity_ids[:target_len]
            time_ids = time_ids[:target_len]
            dates = dates[:target_len]
            symbols = symbols[:target_len]

        factor_exposures_df = pd.DataFrame(factor_exposures, columns=self.factor_names)
        factor_exposures_df.insert(0, 'time', time_ids)
        factor_exposures_df.insert(0, 'id', entity_ids)
        factor_exposures_df.insert(0, 'symbol', symbols)
        factor_exposures_df.insert(0, 'date', dates)

        residuals_df = pd.DataFrame({
            'date': dates,
            'symbol': symbols,
            'id': entity_ids,
            'time': time_ids,
            'actual_ret': actual_returns,
            'expected_ret': expected_returns,
            'residual': residuals
        })

        factor_exposures_df = factor_exposures_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        residuals_df = residuals_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"CryptoIPCAModel 변환 완료: {len(factor_exposures_df)} 관측치")

        return factor_exposures_df, residuals_df
    
    def _create_default_characteristics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        기본 특성 데이터 생성

        Args:
            returns: 수익률 데이터
            
        Returns:
            특성 데이터 DataFrame
        """
        logger.info("기본 특성 데이터 생성 중...")
        
        # 필요한 컬럼 확인
        if 'close' in returns.columns and 'volume' in returns.columns:
            try:
                # 시가총액 프록시 (가격 * 거래량)
                market_cap_proxy = returns['close'] * returns['volume']
                
                # 로그 시가총액
                log_market_cap = np.log(market_cap_proxy + 1e-8)  # 0 방지
                
                # 모멘텀 (21일 수익률)
                returns_pivot = returns.pivot(index='date', columns='symbol', values='return')
                momentum = returns_pivot.rolling(21).sum().stack().reset_index()
                momentum.columns = ['date', 'symbol', 'momentum']
                
                # 변동성 (21일 롤링 표준편차)
                volatility = returns_pivot.rolling(21).std().stack().reset_index()
                volatility.columns = ['date', 'symbol', 'volatility']

                # 특성 데이터 결합
                characteristics = returns[['symbol', 'date']].copy()
                characteristics['log_market_cap'] = log_market_cap
                characteristics['constant'] = 1.0

                # 모멘텀과 변동성 병합
                characteristics = characteristics.merge(momentum, on=['symbol', 'date'], how='left')
                characteristics = characteristics.merge(volatility, on=['symbol', 'date'], how='left')

                # 비트코인 수익률 특성 추가
                btc_symbol = next((sym for sym in returns_pivot.columns if sym.startswith('BTC')), 'BTCUSDT')
                if btc_symbol in returns_pivot.columns:
                    btc_returns = returns_pivot[btc_symbol].rename('btc_return').reset_index()
                    characteristics = characteristics.merge(btc_returns, on='date', how='left')
                else:
                    characteristics['btc_return'] = 0.0

                # 결측치 처리
                characteristics = characteristics.fillna(0)

            except Exception as e:
                logger.warning(f"고급 특성 생성 실패, 기본 특성 사용: {e}")
                # 기본 특성으로 폴백
                characteristics = returns[['symbol', 'date']].copy()
                characteristics['constant'] = 1.0
                characteristics['log_market_cap'] = 0.0
                characteristics['momentum'] = 0.0
                characteristics['volatility'] = 0.0
                characteristics['btc_return'] = 0.0
        else:
            # 최소한의 특성 (상수항)
            characteristics = returns[['symbol', 'date']].copy()
            characteristics['constant'] = 1.0
            characteristics['log_market_cap'] = 0.0
            characteristics['momentum'] = 0.0
            characteristics['volatility'] = 0.0
            characteristics['btc_return'] = 0.0

        logger.info(f"특성 데이터 생성 완료: {characteristics.shape}")
        return characteristics

    def _merge_returns_with_characteristics(
        self,
        returns: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """수익률 데이터와 특성 데이터를 병합하고 전처리 구성 업데이트"""

        merged = returns.copy()
        merged['date'] = pd.to_datetime(merged['date'])

        if characteristics is None or characteristics.empty:
            return merged

        if not {'symbol', 'date'}.issubset(characteristics.columns):
            raise ValueError("characteristics 데이터에 'symbol'과 'date' 컬럼이 필요합니다")

        characteristics = characteristics.copy()
        characteristics['date'] = pd.to_datetime(characteristics['date'])

        feature_cols = [col for col in characteristics.columns if col not in {'symbol', 'date'}]
        if not feature_cols:
            return merged

        # 전처리 설정에 새로운 특성 등록
        for col in feature_cols:
            if col not in self.preprocessor.config.characteristics_to_use:
                self.preprocessor.config.characteristics_to_use.append(col)

        overlap = [col for col in feature_cols if col in merged.columns]
        new_cols = [col for col in feature_cols if col not in merged.columns]

        if new_cols:
            merged = merged.merge(
                characteristics[['symbol', 'date'] + new_cols],
                on=['symbol', 'date'],
                how='left'
            )

        if overlap:
            merged_indexed = merged.set_index(['symbol', 'date'])
            char_indexed = characteristics.set_index(['symbol', 'date'])
            merged_indexed.update(char_indexed[overlap])
            merged = merged_indexed.reset_index()

        return merged
    
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        팩터 로딩 반환
        
        Returns:
            팩터 로딩 DataFrame (특성 x 팩터)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")
        assert self.gamma is not None
        return self.gamma.copy()
    
    def get_factor_timeseries(self) -> pd.DataFrame:
        """
        팩터 시계열 반환
        
        Returns:
            팩터 시계열 DataFrame (시간 x 팩터)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")
        assert self.factors is not None
        return self.factors.copy()
    
    def calculate_residuals(self, returns: pd.DataFrame, characteristics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        잔차 계산
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터 (선택적)
            
        Returns:
            잔차 DataFrame
        """
        _, residuals = self.transform(returns, characteristics)
        return residuals
    
    def predict_returns(self, characteristics: pd.DataFrame) -> pd.DataFrame:
        """
        특성 데이터로부터 수익률 예측

        Args:
            characteristics: 특성 데이터
            
        Returns:
            예측된 수익률
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")

        required_cols = {'symbol', 'date'}
        if not required_cols.issubset(characteristics.columns):
            missing = required_cols - set(characteristics.columns)
            raise ValueError(f"특성 데이터에 필요한 컬럼이 없습니다: {missing}")

        temp_returns = characteristics[['symbol', 'date']].copy()
        temp_returns['return'] = 0.0

        merged = self._merge_returns_with_characteristics(temp_returns, characteristics)
        X_new, _ = self.preprocessor.transform(merged, returns_column='return')
        preds = self.ipca_model.predict(X=X_new, data_type='panel')
        if isinstance(preds, (pd.Series, pd.DataFrame)):
            preds = np.asarray(preds).reshape(-1)

        idx = X_new.index
        entity_ids = idx.get_level_values('entity_id').to_numpy(dtype=int)
        time_ids = idx.get_level_values('time').to_numpy(dtype=int)
        dates = np.array([
            self.time_to_date.get(int(time_id), time_id)
            for time_id in time_ids
        ])
        symbols = [self.id_to_symbol.get(int(eid), f'Asset_{eid}') for eid in entity_ids]

        result_df = pd.DataFrame({
            'date': dates,
            'symbol': symbols,
            'id': entity_ids,
            'time': time_ids,
            'predicted_return': preds[:len(entity_ids)]
        })
        return result_df.sort_values(['symbol', 'date']).reset_index(drop=True)

    def predict_oos(self, today_panel: pd.DataFrame) -> pd.DataFrame:
        """
        워크포워드 한 스텝 OOS 예측 생성

        Args:
            today_panel: 예측 기준일의 패널 데이터 (columns: symbol, date, return, close, volume)

        Returns:
            columns: symbol, predicted_ret_t_plus_1

        Note:
            현재 구현은 '오늘'의 특성으로 같은 날짜의 기대수익(expected_ret)을 계산하여
            이를 다음 날 예측으로 사용합니다. 전처리 파이프라인이 롤링 특성을 포함하므로
            미래 정보가 누출되지 않도록 today_panel은 해당 일자의 정보만 포함해야 합니다.
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")

        # transform을 통해 오늘(예측 기준일)의 기대수익을 얻고 이를 t+1 예측으로 간주
        # 예측 일자 단일 스냅샷은 일반 transform에서 품질 필터로 탈락할 수 있어
        # 전처리기의 완화된 변환 경로를 사용한다.
        X_new, _ = self.preprocessor.transform_predict(today_panel, returns_column='return')
        idx = X_new.index
        entity_ids = idx.get_level_values('entity_id').to_numpy(dtype=int)
        time_ids = idx.get_level_values('time').to_numpy(dtype=int)
        symbols = [self.id_to_symbol.get(int(eid), f'Asset_{eid}') for eid in entity_ids]

        # 특징 행렬과 감마 정렬
        feature_matrix = X_new.reindex(columns=self.characteristic_names, fill_value=0.0).to_numpy(dtype=float)
        gamma_matrix = self.gamma.loc[self.characteristic_names].to_numpy(dtype=float)
        exposures = feature_matrix @ gamma_matrix  # shape: (n_assets, K)

        # 예측 대상 factor_t 선택 (단일 날짜만 존재해야 함)
        unique_times = np.unique(time_ids)
        if len(unique_times) != 1:
            # 혼합된 날짜가 오면 마지막 날짜 사용
            target_time = int(unique_times[-1])
        else:
            target_time = int(unique_times[0])
        target_date = self.time_to_date.get(target_time, None)
        if target_date is None:
            # 매핑 실패 시 마지막 팩터 사용
            factor_vec = self.factors.iloc[-1].to_numpy(dtype=float)
        else:
            try:
                factor_vec = self.factors.loc[target_date].to_numpy(dtype=float)
            except Exception:
                factor_vec = self.factors.iloc[-1].to_numpy(dtype=float)

        preds_arr = exposures @ factor_vec  # shape: (n_assets,)
        preds_df = pd.DataFrame({'symbol': symbols, 'predicted_ret_t_plus_1': preds_arr[:len(symbols)]})
        return preds_df
    
    def get_model_diagnostics(self) -> Dict[str, float]:
        """
        모델 진단 지표 반환
        
        Returns:
            진단 지표 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")
        
        gamma_shape = (0, 0)
        factors_shape = (0, 0)
        n_periods = 0
        if isinstance(self.factors, pd.DataFrame):
            n_periods = int(self.factors.shape[0])
            factors_shape = tuple(self.factors.shape)
        elif hasattr(self.factors, 'shape') and self.factors is not None:
            # numpy-like
            factors_shape = tuple(self.factors.shape)
            if len(self.factors.shape) >= 1:
                n_periods = int(self.factors.shape[0])
        if hasattr(self.gamma, 'shape') and self.gamma is not None:
            gamma_shape = tuple(self.gamma.shape)

        diagnostics = {
            'n_factors': self.n_factors,
            'n_characteristics': len(self.characteristic_names) if self.characteristic_names else 0,
            'n_assets': len(self.asset_names) if self.asset_names else 0,
            'n_periods': n_periods,
            'gamma_shape': gamma_shape,
            'factors_shape': factors_shape,
        }

        # 팩터 로딩의 통계
        if self.gamma is not None:
            gamma_values = self.gamma.to_numpy(dtype=float) if hasattr(self.gamma, 'to_numpy') else np.asarray(self.gamma, dtype=float)
            gamma_stats = {
                'gamma_mean': float(np.mean(gamma_values)),
                'gamma_std': float(np.std(gamma_values)),
                'gamma_min': float(np.min(gamma_values)),
                'gamma_max': float(np.max(gamma_values))
            }
            diagnostics.update(gamma_stats)

        if self.r2_pred is not None:
            diagnostics['r2_pred'] = float(self.r2_pred)

        return diagnostics
    
    def save_model(self, filepath: str) -> None:
        """
        모델 저장
        
        Args:
            filepath: 저장 경로
        """
        if not self.is_fitted:
            raise ValueError("학습되지 않은 모델은 저장할 수 없습니다")
        
        model_data = {
            'ipca_model': self.ipca_model,
            'gamma': self.gamma,
            'factors': self.factors,
            'asset_names': self.asset_names,
            'factor_names': self.factor_names,
            'characteristic_names': self.characteristic_names,
            'n_factors': self.n_factors,
            'intercept': self.intercept,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"CryptoIPCAModel 저장 완료: {filepath}")
    
    def load_model(self, filepath: str) -> 'CryptoIPCAModel':
        """
        모델 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 모델
        """
        model_data = joblib.load(filepath)
        
        self.ipca_model = model_data['ipca_model']
        self.gamma = model_data['gamma']
        self.factors = model_data['factors']
        self.asset_names = model_data['asset_names']
        self.factor_names = model_data['factor_names']
        self.characteristic_names = model_data['characteristic_names']
        self.n_factors = model_data['n_factors']
        self.intercept = model_data['intercept']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"CryptoIPCAModel 로드 완료: {filepath}")
        return self
