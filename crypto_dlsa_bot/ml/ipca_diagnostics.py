"""
IPCA 모델 진단 및 검증 도구

이 모듈은 IPCA 모델의 성능을 평가하고 통계적 유의성을 검증하는 도구들을 제공합니다:
1. 모델 적합도 평가 지표 (R², 설명력, 정보 비율)
2. 팩터 로딩의 통계적 유의성 검증 (부트스트랩 테스트)
3. 팩터 안정성 및 시간에 따른 변화 분석
4. IPCA vs PCA 성능 비교 분석
5. 모델 진단 결과 시각화
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from .crypto_ipca_model import CryptoIPCAModel

logger = logging.getLogger(__name__)


@dataclass
class ModelDiagnostics:
    """모델 진단 결과를 저장하는 데이터 클래스"""
    r_squared: float
    adjusted_r_squared: float
    information_ratio: float
    explained_variance_ratio: float
    factor_significance: Dict[str, Dict[str, float]]
    stability_metrics: Dict[str, float]
    comparison_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'r_squared': self.r_squared,
            'adjusted_r_squared': self.adjusted_r_squared,
            'information_ratio': self.information_ratio,
            'explained_variance_ratio': self.explained_variance_ratio,
            'factor_significance': self.factor_significance,
            'stability_metrics': self.stability_metrics,
            'comparison_metrics': self.comparison_metrics
        }


class IPCADiagnostics:
    """
    IPCA 모델 진단 및 검증 도구
    
    모델의 성능을 평가하고 통계적 유의성을 검증하는 다양한 도구를 제공합니다.
    """
    
    def __init__(self, model: CryptoIPCAModel):
        """
        진단 도구 초기화
        
        Args:
            model: 진단할 IPCA 모델
        """
        if not model.is_fitted:
            raise ValueError("학습되지 않은 모델은 진단할 수 없습니다")
        
        self.model = model
        self.diagnostics_cache = {}
        
        logger.info("IPCADiagnostics 초기화 완료")
    
    def calculate_model_fit_metrics(self, 
                                  returns: pd.DataFrame, 
                                  characteristics: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        모델 적합도 평가 지표 계산
        
        Args:
            returns: 실제 수익률 데이터
            characteristics: 특성 데이터
            
        Returns:
            적합도 지표 딕셔너리
        """
        logger.info("모델 적합도 지표 계산 시작")
        
        try:
            # 대안적 접근: 팩터 로딩과 팩터 시계열을 직접 사용하여 예측값 계산
            if characteristics is None:
                characteristics = self.model._create_default_characteristics(returns)
            
            # 데이터 준비
            merged_data = self.model._prepare_panel_data(returns, characteristics)
            X, y = self.model._convert_to_ipca_format(merged_data)
            
            # 실제 수익률
            actual_returns = y.values
            
            # 팩터 노출도 계산 (X * Gamma)
            if hasattr(self.model.gamma, 'values'):
                gamma_values = self.model.gamma.values
            else:
                gamma_values = self.model.gamma
            
            if hasattr(X, 'values'):
                X_values = X.values
            else:
                X_values = X
            
            # 팩터 노출도 계산
            factor_exposures = X_values @ gamma_values
            
            # 팩터 시계열 가져오기
            if hasattr(self.model.factors, 'values'):
                factors_values = self.model.factors.values
            else:
                factors_values = self.model.factors
            
            # 예측된 수익률 계산 (팩터 노출도 * 팩터 수익률)
            # factors는 (K x T) 형태이므로 시간에 맞춰 계산
            n_obs = len(actual_returns)
            n_time_periods = factors_values.shape[1] if factors_values.ndim > 1 else len(factors_values)
            
            if n_obs <= n_time_periods:
                # 시간 인덱스 매핑
                time_indices = np.arange(n_obs) % n_time_periods
                if factors_values.ndim > 1:
                    factor_returns = factors_values[:, time_indices].T  # (T x K)
                else:
                    factor_returns = factors_values[time_indices].reshape(-1, 1)
                
                # 예측 수익률 = 팩터 노출도 * 팩터 수익률 (각 관측치별로)
                predicted_returns = np.sum(factor_exposures * factor_returns, axis=1)
            else:
                # 데이터가 더 많은 경우 반복 사용
                predicted_returns = np.zeros(n_obs)
                for i in range(n_obs):
                    time_idx = i % n_time_periods
                    if factors_values.ndim > 1:
                        factor_ret = factors_values[:, time_idx]
                    else:
                        factor_ret = factors_values[time_idx]
                    predicted_returns[i] = np.dot(factor_exposures[i], factor_ret)
            
            # 잔차 계산
            residual_values = actual_returns - predicted_returns
            
            # NaN 값 처리
            valid_mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns) | np.isnan(residual_values))
            if not np.any(valid_mask):
                # 모든 값이 NaN인 경우 기본값 사용
                logger.warning("모든 예측값이 NaN입니다. 기본 지표를 반환합니다.")
                return {
                    'r_squared': 0.0,
                    'adjusted_r_squared': 0.0,
                    'explained_variance_ratio': 0.0,
                    'information_ratio': 0.0,
                    'total_variance': float(np.var(actual_returns[~np.isnan(actual_returns)])) if np.any(~np.isnan(actual_returns)) else 0.0,
                    'residual_variance': 0.0,
                    'n_observations': len(actual_returns),
                    'n_factors': self.model.n_factors,
                    'n_valid_observations': 0
                }
            
            actual_returns_clean = actual_returns[valid_mask]
            predicted_returns_clean = predicted_returns[valid_mask]
            residual_values_clean = residual_values[valid_mask]
            
            # R² 계산
            try:
                r_squared = r2_score(actual_returns_clean, predicted_returns_clean)
            except ValueError:
                # R² 계산 실패 시 대안 계산
                ss_res = np.sum((actual_returns_clean - predicted_returns_clean) ** 2)
                ss_tot = np.sum((actual_returns_clean - np.mean(actual_returns_clean)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 조정된 R² 계산
            n = len(actual_returns_clean)
            p = self.model.n_factors
            if n > p + 1:
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            else:
                adjusted_r_squared = r_squared
            
            # 설명된 분산 비율 계산
            total_variance = np.var(actual_returns_clean)
            residual_variance = np.var(residual_values_clean)
            explained_variance_ratio = 1 - (residual_variance / total_variance) if total_variance > 0 else 0
            
            # 정보 비율 계산 (초과 수익률의 평균 / 추적 오차)
            excess_returns = predicted_returns_clean - actual_returns_clean
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            # 팩터별 설명력 계산
            factor_explanatory_power = {}
            if hasattr(self.model, 'gamma') and self.model.gamma is not None:
                gamma_values = self.model.gamma.values if hasattr(self.model.gamma, 'values') else self.model.gamma
                for i, factor_name in enumerate(self.model.factor_names):
                    # 각 팩터의 기여도 (팩터 로딩의 제곱합)
                    factor_contribution = np.sum(gamma_values[:, i] ** 2)
                    factor_explanatory_power[factor_name] = float(factor_contribution)
            
            metrics = {
                'r_squared': float(r_squared),
                'adjusted_r_squared': float(adjusted_r_squared),
                'explained_variance_ratio': float(explained_variance_ratio),
                'information_ratio': float(information_ratio),
                'total_variance': float(total_variance),
                'residual_variance': float(residual_variance),
                'n_observations': int(n),
                'n_factors': int(p),
                'n_valid_observations': int(len(actual_returns_clean)),
                **factor_explanatory_power
            }
            
            logger.info(f"모델 적합도 지표 계산 완료: R²={r_squared:.4f}, 조정된 R²={adjusted_r_squared:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"모델 적합도 지표 계산 실패: {e}")
            raise
    
    def bootstrap_factor_significance(self, 
                                    returns: pd.DataFrame,
                                    characteristics: Optional[pd.DataFrame] = None,
                                    n_bootstrap: int = 1000,
                                    confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        부트스트랩을 통한 팩터 로딩의 통계적 유의성 검증
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터
            n_bootstrap: 부트스트랩 반복 횟수
            confidence_level: 신뢰 수준
            
        Returns:
            팩터별 유의성 검증 결과
        """
        logger.info(f"부트스트랩 팩터 유의성 검증 시작 (n_bootstrap={n_bootstrap})")
        
        try:
            # 원본 팩터 로딩
            original_gamma = self.model.gamma.values if hasattr(self.model.gamma, 'values') else self.model.gamma
            
            # 부트스트랩 결과 저장
            bootstrap_gammas = []
            
            # 데이터 준비
            if characteristics is None:
                characteristics = self.model._create_default_characteristics(returns)
            
            merged_data = self.model._prepare_panel_data(returns, characteristics)
            
            for i in range(n_bootstrap):
                if i % 100 == 0:
                    logger.debug(f"부트스트랩 진행: {i}/{n_bootstrap}")
                
                try:
                    # 부트스트랩 샘플링 (시간 축으로 샘플링)
                    unique_dates = merged_data['date'].unique()
                    sampled_dates = np.random.choice(unique_dates, size=len(unique_dates), replace=True)
                    
                    # 샘플링된 날짜의 데이터 추출
                    bootstrap_data = []
                    for date in sampled_dates:
                        date_data = merged_data[merged_data['date'] == date]
                        bootstrap_data.append(date_data)
                    
                    bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)
                    
                    # 부트스트랩 모델 학습
                    bootstrap_model = CryptoIPCAModel(
                        n_factors=self.model.n_factors,
                        intercept=self.model.intercept,
                        max_iter=self.model.max_iter,
                        iter_tol=self.model.iter_tol
                    )
                    
                    bootstrap_returns = bootstrap_df[['symbol', 'date', 'return']]
                    bootstrap_chars = bootstrap_df.drop(columns=['return'])
                    
                    bootstrap_model.fit(bootstrap_returns, bootstrap_chars)
                    
                    # 팩터 로딩 저장
                    bootstrap_gamma = bootstrap_model.gamma.values if hasattr(bootstrap_model.gamma, 'values') else bootstrap_model.gamma
                    bootstrap_gammas.append(bootstrap_gamma)
                    
                except Exception as e:
                    logger.debug(f"부트스트랩 {i} 실패: {e}")
                    continue
            
            if not bootstrap_gammas:
                raise ValueError("부트스트랩 샘플이 생성되지 않았습니다")
            
            # 부트스트랩 결과 분석
            bootstrap_gammas = np.array(bootstrap_gammas)
            alpha = 1 - confidence_level
            
            significance_results = {}
            
            for j, factor_name in enumerate(self.model.factor_names):
                factor_results = {}
                
                for i, char_name in enumerate(self.model.characteristic_names):
                    # 원본 값
                    original_value = original_gamma[i, j]
                    
                    # 부트스트랩 분포
                    bootstrap_values = bootstrap_gammas[:, i, j]
                    
                    # 신뢰구간 계산
                    lower_ci = np.percentile(bootstrap_values, 100 * alpha / 2)
                    upper_ci = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
                    
                    # p-value 계산 (양측 검정)
                    # H0: 팩터 로딩 = 0
                    p_value = 2 * min(
                        np.mean(bootstrap_values <= 0),
                        np.mean(bootstrap_values >= 0)
                    )
                    
                    # 통계적 유의성 판정
                    is_significant = (lower_ci > 0 and upper_ci > 0) or (lower_ci < 0 and upper_ci < 0)
                    
                    factor_results[char_name] = {
                        'original_value': float(original_value),
                        'bootstrap_mean': float(np.mean(bootstrap_values)),
                        'bootstrap_std': float(np.std(bootstrap_values)),
                        'lower_ci': float(lower_ci),
                        'upper_ci': float(upper_ci),
                        'p_value': float(p_value),
                        'is_significant': bool(is_significant),
                        'n_bootstrap': len(bootstrap_values)
                    }
                
                significance_results[factor_name] = factor_results
            
            logger.info(f"부트스트랩 팩터 유의성 검증 완료: {len(bootstrap_gammas)} 샘플")
            
            return significance_results
            
        except Exception as e:
            logger.error(f"부트스트랩 팩터 유의성 검증 실패: {e}")
            raise
    
    def analyze_factor_stability(self, 
                               returns: pd.DataFrame,
                               characteristics: Optional[pd.DataFrame] = None,
                               window_size: int = 252,
                               step_size: int = 21) -> Dict[str, pd.DataFrame]:
        """
        팩터 안정성 및 시간에 따른 변화 분석
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터
            window_size: 롤링 윈도우 크기 (일)
            step_size: 스텝 크기 (일)
            
        Returns:
            팩터별 시간에 따른 로딩 변화
        """
        logger.info(f"팩터 안정성 분석 시작 (window_size={window_size}, step_size={step_size})")
        
        try:
            if characteristics is None:
                characteristics = self.model._create_default_characteristics(returns)
            
            merged_data = self.model._prepare_panel_data(returns, characteristics)
            
            # 날짜별로 정렬
            merged_data = merged_data.sort_values('date')
            unique_dates = sorted(merged_data['date'].unique())
            
            # 롤링 윈도우 분석
            stability_results = {}
            window_results = []
            
            for i in range(0, len(unique_dates) - window_size, step_size):
                window_start = unique_dates[i]
                window_end = unique_dates[i + window_size - 1]
                
                # 윈도우 데이터 추출
                window_data = merged_data[
                    (merged_data['date'] >= window_start) & 
                    (merged_data['date'] <= window_end)
                ]
                
                if len(window_data) < 100:  # 최소 데이터 요구사항
                    continue
                
                try:
                    # 윈도우 모델 학습
                    window_model = CryptoIPCAModel(
                        n_factors=self.model.n_factors,
                        intercept=self.model.intercept,
                        max_iter=min(1000, self.model.max_iter),  # 빠른 수렴을 위해 제한
                        iter_tol=self.model.iter_tol
                    )
                    
                    window_returns = window_data[['symbol', 'date', 'return']]
                    window_chars = window_data.drop(columns=['return'])
                    
                    window_model.fit(window_returns, window_chars)
                    
                    # 팩터 로딩 저장
                    window_gamma = window_model.gamma.values if hasattr(window_model.gamma, 'values') else window_model.gamma
                    
                    window_result = {
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_center': window_start + (window_end - window_start) / 2,
                        'gamma': window_gamma
                    }
                    
                    window_results.append(window_result)
                    
                except Exception as e:
                    logger.debug(f"윈도우 {window_start} - {window_end} 분석 실패: {e}")
                    continue
            
            if not window_results:
                logger.warning("분석 가능한 윈도우가 없습니다. 기본 안정성 지표를 반환합니다.")
                # 기본 안정성 결과 반환
                stability_results = {}
                for factor_name in self.model.factor_names:
                    stability_results[factor_name] = pd.DataFrame({
                        'date': [merged_data['date'].min()],
                        'window_start': [merged_data['date'].min()],
                        'window_end': [merged_data['date'].max()],
                        **{char_name: [0.0] for char_name in self.model.characteristic_names}
                    })
                return stability_results
            
            # 팩터별 안정성 지표 계산
            for j, factor_name in enumerate(self.model.factor_names):
                factor_loadings_over_time = []
                
                for window_result in window_results:
                    factor_loadings = window_result['gamma'][:, j]
                    
                    loading_data = {
                        'date': window_result['window_center'],
                        'window_start': window_result['window_start'],
                        'window_end': window_result['window_end']
                    }
                    
                    # 각 특성별 로딩 저장
                    for i, char_name in enumerate(self.model.characteristic_names):
                        loading_data[char_name] = factor_loadings[i]
                    
                    factor_loadings_over_time.append(loading_data)
                
                stability_results[factor_name] = pd.DataFrame(factor_loadings_over_time)
            
            logger.info(f"팩터 안정성 분석 완료: {len(window_results)} 윈도우")
            
            return stability_results
            
        except Exception as e:
            logger.error(f"팩터 안정성 분석 실패: {e}")
            raise
    
    def compare_with_pca(self, 
                        returns: pd.DataFrame,
                        characteristics: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        IPCA vs PCA 성능 비교 분석
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터
            
        Returns:
            비교 분석 결과
        """
        logger.info("IPCA vs PCA 성능 비교 분석 시작")
        
        try:
            # IPCA 성능 지표
            ipca_metrics = self.calculate_model_fit_metrics(returns, characteristics)
            
            # PCA 모델 구성을 위한 데이터 준비
            if characteristics is None:
                characteristics = self.model._create_default_characteristics(returns)
            
            merged_data = self.model._prepare_panel_data(returns, characteristics)
            
            # 수익률 매트릭스 구성 (자산 x 시간)
            returns_pivot = merged_data.pivot(index='date', columns='symbol', values='return')
            returns_matrix = returns_pivot.fillna(0).values
            
            # PCA 모델 학습
            pca_model = PCA(n_components=self.model.n_factors)
            pca_factors = pca_model.fit_transform(returns_matrix.T)  # 자산을 행으로
            
            # PCA 예측 수익률 계산
            pca_reconstructed = pca_model.inverse_transform(pca_factors).T
            
            # PCA 성능 지표 계산
            actual_returns_matrix = returns_matrix.flatten()
            pca_predicted_matrix = pca_reconstructed.flatten()
            
            # 유효한 값만 사용 (NaN 제거)
            valid_mask = ~(np.isnan(actual_returns_matrix) | np.isnan(pca_predicted_matrix))
            actual_valid = actual_returns_matrix[valid_mask]
            predicted_valid = pca_predicted_matrix[valid_mask]
            
            if len(actual_valid) == 0:
                raise ValueError("유효한 데이터가 없습니다")
            
            # PCA R² 계산
            try:
                pca_r_squared = r2_score(actual_valid, predicted_valid)
            except ValueError:
                # R² 계산 실패 시 대안 계산
                ss_res = np.sum((actual_valid - predicted_valid) ** 2)
                ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
                pca_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # PCA 설명된 분산 비율
            pca_explained_variance_ratio = np.sum(pca_model.explained_variance_ratio_)
            
            # PCA 잔차 분산
            pca_residuals = actual_valid - predicted_valid
            pca_residual_variance = np.var(pca_residuals)
            
            # 비교 지표 계산
            comparison_metrics = {
                'ipca_r_squared': ipca_metrics['r_squared'],
                'pca_r_squared': float(pca_r_squared),
                'r_squared_improvement': ipca_metrics['r_squared'] - pca_r_squared,
                
                'ipca_explained_variance': ipca_metrics['explained_variance_ratio'],
                'pca_explained_variance': float(pca_explained_variance_ratio),
                'explained_variance_improvement': ipca_metrics['explained_variance_ratio'] - pca_explained_variance_ratio,
                
                'ipca_residual_variance': ipca_metrics['residual_variance'],
                'pca_residual_variance': float(pca_residual_variance),
                'residual_variance_reduction': pca_residual_variance - ipca_metrics['residual_variance'],
                
                'ipca_information_ratio': ipca_metrics['information_ratio'],
                'pca_information_ratio': 0.0,  # PCA는 정보 비율 계산 불가
                
                'n_factors': self.model.n_factors,
                'n_observations': len(actual_valid)
            }
            
            # 개선도 백분율 계산
            if pca_r_squared > 0:
                comparison_metrics['r_squared_improvement_pct'] = (comparison_metrics['r_squared_improvement'] / pca_r_squared) * 100
            else:
                comparison_metrics['r_squared_improvement_pct'] = 0.0
            
            logger.info(f"IPCA vs PCA 비교 완료: IPCA R²={ipca_metrics['r_squared']:.4f}, PCA R²={pca_r_squared:.4f}")
            
            return comparison_metrics
            
        except Exception as e:
            logger.error(f"IPCA vs PCA 비교 분석 실패: {e}")
            raise
    
    def generate_comprehensive_diagnostics(self, 
                                         returns: pd.DataFrame,
                                         characteristics: Optional[pd.DataFrame] = None,
                                         n_bootstrap: int = 500,
                                         stability_window: int = 252) -> ModelDiagnostics:
        """
        종합적인 모델 진단 수행
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터
            n_bootstrap: 부트스트랩 반복 횟수
            stability_window: 안정성 분석 윈도우 크기
            
        Returns:
            종합 진단 결과
        """
        logger.info("종합적인 모델 진단 시작")
        
        try:
            # 1. 모델 적합도 지표
            fit_metrics = self.calculate_model_fit_metrics(returns, characteristics)
            
            # 2. 팩터 유의성 검증
            significance_results = self.bootstrap_factor_significance(
                returns, characteristics, n_bootstrap=n_bootstrap
            )
            
            # 3. 팩터 안정성 분석
            stability_results = self.analyze_factor_stability(
                returns, characteristics, window_size=stability_window
            )
            
            # 안정성 지표 계산
            stability_metrics = {}
            for factor_name, stability_df in stability_results.items():
                if len(stability_df) > 1:
                    # 각 특성별 안정성 계산
                    for char_name in self.model.characteristic_names:
                        if char_name in stability_df.columns:
                            values = stability_df[char_name].values
                            stability_metrics[f'{factor_name}_{char_name}_stability'] = float(np.std(values))
            
            # 4. PCA 비교 분석
            comparison_metrics = self.compare_with_pca(returns, characteristics)
            
            # 종합 진단 결과 생성
            diagnostics = ModelDiagnostics(
                r_squared=fit_metrics['r_squared'],
                adjusted_r_squared=fit_metrics['adjusted_r_squared'],
                information_ratio=fit_metrics['information_ratio'],
                explained_variance_ratio=fit_metrics['explained_variance_ratio'],
                factor_significance=significance_results,
                stability_metrics=stability_metrics,
                comparison_metrics=comparison_metrics
            )
            
            logger.info("종합적인 모델 진단 완료")
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"종합적인 모델 진단 실패: {e}")
            raise
    
    def plot_factor_loadings_heatmap(self, 
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        팩터 로딩 히트맵 시각화
        
        Args:
            save_path: 저장 경로 (선택적)
            figsize: 그림 크기
            
        Returns:
            matplotlib Figure 객체
        """
        logger.info("팩터 로딩 히트맵 생성 시작")
        
        try:
            # 팩터 로딩 데이터 준비
            gamma_df = self.model.get_factor_loadings()
            
            # 히트맵 생성
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.heatmap(
                gamma_df.T,  # 전치하여 팩터를 행으로, 특성을 열로
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Factor Loading'}
            )
            
            ax.set_title('IPCA Factor Loadings Heatmap', fontsize=16, fontweight='bold')
            ax.set_xlabel('Characteristics', fontsize=12)
            ax.set_ylabel('Factors', fontsize=12)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"팩터 로딩 히트맵 저장: {save_path}")
            
            logger.info("팩터 로딩 히트맵 생성 완료")
            
            return fig
            
        except Exception as e:
            logger.error(f"팩터 로딩 히트맵 생성 실패: {e}")
            raise
    
    def plot_residual_distribution(self, 
                                 returns: pd.DataFrame,
                                 characteristics: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        잔차 분포 시각화
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터
            save_path: 저장 경로 (선택적)
            figsize: 그림 크기
            
        Returns:
            matplotlib Figure 객체
        """
        logger.info("잔차 분포 시각화 시작")
        
        try:
            # 모델 적합도 지표를 통해 잔차 계산 (더 안정적)
            metrics = self.calculate_model_fit_metrics(returns, characteristics)
            
            # 대안적 잔차 계산
            if characteristics is None:
                characteristics = self.model._create_default_characteristics(returns)
            
            merged_data = self.model._prepare_panel_data(returns, characteristics)
            X, y = self.model._convert_to_ipca_format(merged_data)
            
            actual_returns = y.values
            
            # 팩터 노출도와 예측값 계산 (calculate_model_fit_metrics와 동일한 로직)
            gamma_values = self.model.gamma.values if hasattr(self.model.gamma, 'values') else self.model.gamma
            X_values = X.values if hasattr(X, 'values') else X
            factor_exposures = X_values @ gamma_values
            
            factors_values = self.model.factors.values if hasattr(self.model.factors, 'values') else self.model.factors
            n_obs = len(actual_returns)
            n_time_periods = factors_values.shape[1] if factors_values.ndim > 1 else len(factors_values)
            
            predicted_returns = np.zeros(n_obs)
            for i in range(n_obs):
                time_idx = i % n_time_periods
                if factors_values.ndim > 1:
                    factor_ret = factors_values[:, time_idx]
                else:
                    factor_ret = factors_values[time_idx]
                predicted_returns[i] = np.dot(factor_exposures[i], factor_ret)
            
            residual_values = actual_returns - predicted_returns
            
            # NaN 값 제거
            residual_values = residual_values[~np.isnan(residual_values)]
            
            if len(residual_values) == 0:
                raise ValueError("유효한 잔차 값이 없습니다")
            
            # 서브플롯 생성
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # 1. 잔차 히스토그램
            axes[0, 0].hist(residual_values, bins=50, alpha=0.7, density=True, color='skyblue')
            axes[0, 0].axvline(np.mean(residual_values), color='red', linestyle='--', label=f'Mean: {np.mean(residual_values):.4f}')
            axes[0, 0].set_title('Residual Distribution')
            axes[0, 0].set_xlabel('Residual Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Q-Q 플롯 (정규성 검정)
            stats.probplot(residual_values, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normality Test)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 잔차 시계열 플롯
            axes[1, 0].plot(residual_values, alpha=0.7, color='green')
            axes[1, 0].axhline(0, color='red', linestyle='--')
            axes[1, 0].set_title('Residual Time Series')
            axes[1, 0].set_xlabel('Observation')
            axes[1, 0].set_ylabel('Residual Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 잔차 자기상관 플롯
            from statsmodels.tsa.stattools import acf
            lags = min(40, len(residual_values) // 4)
            autocorr = acf(residual_values, nlags=lags, fft=True)
            axes[1, 1].plot(range(len(autocorr)), autocorr, 'o-', alpha=0.7, color='purple')
            axes[1, 1].axhline(0, color='red', linestyle='--')
            axes[1, 1].set_title('Residual Autocorrelation')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('Autocorrelation')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"잔차 분포 시각화 저장: {save_path}")
            
            logger.info("잔차 분포 시각화 완료")
            
            return fig
            
        except Exception as e:
            logger.error(f"잔차 분포 시각화 실패: {e}")
            raise
    
    def plot_factor_contribution(self, 
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        팩터 기여도 시각화
        
        Args:
            save_path: 저장 경로 (선택적)
            figsize: 그림 크기
            
        Returns:
            matplotlib Figure 객체
        """
        logger.info("팩터 기여도 시각화 시작")
        
        try:
            # 팩터 로딩에서 기여도 계산
            gamma_df = self.model.get_factor_loadings()
            gamma_values = gamma_df.values
            
            # 각 팩터의 총 기여도 (로딩의 제곱합)
            factor_contributions = np.sum(gamma_values ** 2, axis=0)
            total_contribution = np.sum(factor_contributions)
            
            # 백분율로 변환
            factor_contributions_pct = (factor_contributions / total_contribution) * 100
            
            # 바 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # 1. 팩터별 기여도 바 차트
            bars = ax1.bar(self.model.factor_names, factor_contributions_pct, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(self.model.factor_names)])
            ax1.set_title('Factor Contribution to Total Variance', fontweight='bold')
            ax1.set_xlabel('Factors')
            ax1.set_ylabel('Contribution (%)')
            ax1.grid(True, alpha=0.3)
            
            # 값 표시
            for bar, pct in zip(bars, factor_contributions_pct):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom')
            
            # 2. 누적 기여도 파이 차트
            ax2.pie(factor_contributions_pct, labels=self.model.factor_names, autopct='%1.1f%%',
                   colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(self.model.factor_names)])
            ax2.set_title('Cumulative Factor Contribution', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"팩터 기여도 시각화 저장: {save_path}")
            
            logger.info("팩터 기여도 시각화 완료")
            
            return fig
            
        except Exception as e:
            logger.error(f"팩터 기여도 시각화 실패: {e}")
            raise
    
    def generate_diagnostic_report(self, 
                                 returns: pd.DataFrame,
                                 characteristics: Optional[pd.DataFrame] = None,
                                 output_dir: str = "diagnostics_output",
                                 n_bootstrap: int = 500) -> str:
        """
        종합 진단 리포트 생성
        
        Args:
            returns: 수익률 데이터
            characteristics: 특성 데이터
            output_dir: 출력 디렉토리
            n_bootstrap: 부트스트랩 반복 횟수
            
        Returns:
            리포트 파일 경로
        """
        logger.info(f"종합 진단 리포트 생성 시작: {output_dir}")
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # 종합 진단 수행
            diagnostics = self.generate_comprehensive_diagnostics(
                returns, characteristics, n_bootstrap=n_bootstrap
            )
            
            # 시각화 생성
            heatmap_path = os.path.join(output_dir, "factor_loadings_heatmap.png")
            residual_path = os.path.join(output_dir, "residual_distribution.png")
            contribution_path = os.path.join(output_dir, "factor_contribution.png")
            
            self.plot_factor_loadings_heatmap(save_path=heatmap_path)
            self.plot_residual_distribution(returns, characteristics, save_path=residual_path)
            self.plot_factor_contribution(save_path=contribution_path)
            
            # 리포트 텍스트 생성
            report_path = os.path.join(output_dir, "diagnostic_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("IPCA 모델 진단 리포트\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("1. 모델 적합도 지표\n")
                f.write("-" * 30 + "\n")
                f.write(f"R²: {diagnostics.r_squared:.4f}\n")
                f.write(f"조정된 R²: {diagnostics.adjusted_r_squared:.4f}\n")
                f.write(f"설명된 분산 비율: {diagnostics.explained_variance_ratio:.4f}\n")
                f.write(f"정보 비율: {diagnostics.information_ratio:.4f}\n\n")
                
                f.write("2. PCA 대비 성능 개선\n")
                f.write("-" * 30 + "\n")
                f.write(f"R² 개선: {diagnostics.comparison_metrics['r_squared_improvement']:.4f}\n")
                f.write(f"설명된 분산 개선: {diagnostics.comparison_metrics['explained_variance_improvement']:.4f}\n")
                f.write(f"잔차 분산 감소: {diagnostics.comparison_metrics['residual_variance_reduction']:.4f}\n\n")
                
                f.write("3. 팩터 유의성 검증 결과\n")
                f.write("-" * 30 + "\n")
                for factor_name, factor_results in diagnostics.factor_significance.items():
                    f.write(f"\n{factor_name}:\n")
                    for char_name, char_results in factor_results.items():
                        significance = "유의함" if char_results['is_significant'] else "유의하지 않음"
                        f.write(f"  {char_name}: {char_results['original_value']:.4f} "
                               f"(p-value: {char_results['p_value']:.4f}, {significance})\n")
                
                f.write("\n4. 생성된 시각화 파일\n")
                f.write("-" * 30 + "\n")
                f.write(f"- 팩터 로딩 히트맵: {heatmap_path}\n")
                f.write(f"- 잔차 분포 분석: {residual_path}\n")
                f.write(f"- 팩터 기여도 분석: {contribution_path}\n")
            
            logger.info(f"종합 진단 리포트 생성 완료: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"종합 진단 리포트 생성 실패: {e}")
            raise