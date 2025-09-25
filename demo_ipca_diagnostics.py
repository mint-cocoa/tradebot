#!/usr/bin/env python3
"""
IPCA 진단 도구 데모 스크립트

이 스크립트는 IPCA 모델 진단 및 검증 도구의 기능을 시연합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 모듈 import
from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel
from crypto_dlsa_bot.ml.ipca_diagnostics import IPCADiagnostics


def generate_sample_data(n_assets=20, n_periods=200, n_factors=3):
    """샘플 데이터 생성"""
    logger.info(f"샘플 데이터 생성: {n_assets}개 자산, {n_periods}일, {n_factors}개 팩터")
    
    # 날짜 생성
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_periods)]
    
    # 심볼 생성
    symbols = [f'CRYPTO{i+1:02d}' for i in range(n_assets)]
    
    # 팩터 시계열 생성 (공통 팩터)
    np.random.seed(42)
    market_factor = np.random.normal(0, 0.02, n_periods)  # 시장 팩터
    size_factor = np.random.normal(0, 0.01, n_periods)    # 크기 팩터
    momentum_factor = np.random.normal(0, 0.015, n_periods)  # 모멘텀 팩터
    
    # 수익률 및 특성 데이터 생성
    returns_data = []
    characteristics_data = []
    
    for i, symbol in enumerate(symbols):
        # 각 자산의 팩터 로딩 (고정)
        market_loading = np.random.normal(1.0, 0.3)
        size_loading = np.random.normal(0.0, 0.5)
        momentum_loading = np.random.normal(0.0, 0.4)
        
        # 특성 데이터 (시간에 따라 변화)
        base_market_cap = np.random.uniform(8, 12)  # 로그 시가총액
        base_momentum = np.random.normal(0, 0.1)
        base_volatility = np.random.uniform(0.1, 0.5)
        
        for j, date in enumerate(dates):
            # 팩터 기반 수익률 생성
            factor_return = (market_loading * market_factor[j] + 
                           size_loading * size_factor[j] + 
                           momentum_loading * momentum_factor[j])
            
            # 특이적 수익률 추가
            idiosyncratic_return = np.random.normal(0, 0.01)
            total_return = factor_return + idiosyncratic_return
            
            # 가격 및 거래량 (수익률 계산용)
            price = 100 * np.exp(total_return * (j + 1) / 100)
            volume = np.random.uniform(1000, 10000)
            
            returns_data.append({
                'date': date,
                'symbol': symbol,
                'return': total_return,
                'close': price,
                'volume': volume
            })
            
            # 특성 데이터 (시간에 따라 약간 변화)
            market_cap = base_market_cap + np.random.normal(0, 0.1)
            momentum = base_momentum + np.random.normal(0, 0.05)
            volatility = base_volatility + np.random.normal(0, 0.02)
            
            characteristics_data.append({
                'date': date,
                'symbol': symbol,
                'log_market_cap': market_cap,
                'momentum': momentum,
                'volatility': max(0.01, volatility)  # 변동성은 양수
            })
    
    returns_df = pd.DataFrame(returns_data)
    characteristics_df = pd.DataFrame(characteristics_data)
    
    logger.info(f"데이터 생성 완료: 수익률 {returns_df.shape}, 특성 {characteristics_df.shape}")
    
    return returns_df, characteristics_df


def main():
    """메인 함수"""
    logger.info("IPCA 진단 도구 데모 시작")
    
    try:
        # 1. 샘플 데이터 생성
        returns_df, characteristics_df = generate_sample_data(n_assets=15, n_periods=150, n_factors=3)
        
        # 2. IPCA 모델 학습
        logger.info("IPCA 모델 학습 시작")
        model = CryptoIPCAModel(n_factors=3, max_iter=100)
        model.fit(returns_df, characteristics_df)
        logger.info("IPCA 모델 학습 완료")
        
        # 3. 진단 도구 초기화
        diagnostics = IPCADiagnostics(model)
        
        # 4. 모델 적합도 지표 계산
        logger.info("모델 적합도 지표 계산")
        fit_metrics = diagnostics.calculate_model_fit_metrics(returns_df, characteristics_df)
        
        print("\n=== 모델 적합도 지표 ===")
        print(f"R²: {fit_metrics['r_squared']:.4f}")
        print(f"조정된 R²: {fit_metrics['adjusted_r_squared']:.4f}")
        print(f"설명된 분산 비율: {fit_metrics['explained_variance_ratio']:.4f}")
        print(f"정보 비율: {fit_metrics['information_ratio']:.4f}")
        print(f"총 분산: {fit_metrics['total_variance']:.6f}")
        print(f"잔차 분산: {fit_metrics['residual_variance']:.6f}")
        print(f"관측치 수: {fit_metrics['n_observations']}")
        print(f"유효 관측치 수: {fit_metrics['n_valid_observations']}")
        
        # 5. 팩터 유의성 검증 (적은 부트스트랩 샘플)
        logger.info("팩터 유의성 검증 (부트스트랩)")
        significance_results = diagnostics.bootstrap_factor_significance(
            returns_df, characteristics_df, n_bootstrap=50
        )
        
        print("\n=== 팩터 유의성 검증 결과 ===")
        for factor_name, factor_results in significance_results.items():
            print(f"\n{factor_name}:")
            for char_name, char_results in factor_results.items():
                significance = "유의함" if char_results['is_significant'] else "유의하지 않음"
                print(f"  {char_name}: {char_results['original_value']:.4f} "
                     f"(p-value: {char_results['p_value']:.4f}, {significance})")
        
        # 6. IPCA vs PCA 비교
        logger.info("IPCA vs PCA 성능 비교")
        comparison_metrics = diagnostics.compare_with_pca(returns_df, characteristics_df)
        
        print("\n=== IPCA vs PCA 비교 ===")
        print(f"IPCA R²: {comparison_metrics['ipca_r_squared']:.4f}")
        print(f"PCA R²: {comparison_metrics['pca_r_squared']:.4f}")
        print(f"R² 개선: {comparison_metrics['r_squared_improvement']:.4f}")
        print(f"설명된 분산 개선: {comparison_metrics['explained_variance_improvement']:.4f}")
        
        # 7. 시각화 생성
        logger.info("시각화 생성")
        
        # 팩터 로딩 히트맵
        fig1 = diagnostics.plot_factor_loadings_heatmap(save_path="factor_loadings_demo.png")
        print("\n팩터 로딩 히트맵 저장: factor_loadings_demo.png")
        
        # 잔차 분포
        fig2 = diagnostics.plot_residual_distribution(
            returns_df, characteristics_df, save_path="residual_distribution_demo.png"
        )
        print("잔차 분포 시각화 저장: residual_distribution_demo.png")
        
        # 팩터 기여도
        fig3 = diagnostics.plot_factor_contribution(save_path="factor_contribution_demo.png")
        print("팩터 기여도 시각화 저장: factor_contribution_demo.png")
        
        # 8. 종합 진단 리포트 생성
        logger.info("종합 진단 리포트 생성")
        report_path = diagnostics.generate_diagnostic_report(
            returns_df, characteristics_df, 
            output_dir="diagnostics_demo_output", 
            n_bootstrap=30
        )
        print(f"\n종합 진단 리포트 생성 완료: {report_path}")
        
        # 9. 모델 진단 정보 출력
        model_diagnostics = model.get_model_diagnostics()
        print("\n=== 모델 기본 정보 ===")
        print(f"팩터 수: {model_diagnostics['n_factors']}")
        print(f"특성 수: {model_diagnostics['n_characteristics']}")
        print(f"자산 수: {model_diagnostics['n_assets']}")
        print(f"팩터 로딩 형태: ({model_diagnostics['gamma_shape_0']}, {model_diagnostics['gamma_shape_1']})")
        print(f"팩터 시계열 형태: ({model_diagnostics['factors_shape_0']}, {model_diagnostics['factors_shape_1']})")
        
        logger.info("IPCA 진단 도구 데모 완료")
        
    except Exception as e:
        logger.error(f"데모 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()