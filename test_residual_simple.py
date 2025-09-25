#!/usr/bin/env python3
"""
간단한 잔차 계산 테스트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from crypto_dlsa_bot.ml.residual_calculator import ResidualCalculator, ResidualCalculatorConfig

def test_simple_residual_calculation():
    """간단한 잔차 계산 테스트"""
    print("간단한 잔차 계산 테스트 시작...")
    
    # 간단한 테스트 데이터 생성
    np.random.seed(42)
    
    symbols = ['BTC', 'ETH', 'ADA']
    dates = pd.date_range('2023-01-01', '2023-01-30', freq='D')
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'symbol': symbol,
                'date': date,
                'return': np.random.normal(0, 0.02),
                'close': np.random.uniform(50, 200),
                'volume': np.random.lognormal(10, 0.5)
            })
    
    returns_df = pd.DataFrame(data)
    print(f"테스트 데이터 생성: {returns_df.shape}")
    
    # 잔차 계산기 설정
    config = ResidualCalculatorConfig(
        rolling_window_size=15,
        min_observations=10,
        refit_frequency=5
    )
    calculator = ResidualCalculator(config)
    
    try:
        # 잔차 계산
        residuals = calculator.calculate_rolling_residuals_optimized(
            returns_df, window_size=15, min_periods=10
        )
        
        print(f"잔차 계산 결과: {residuals.shape}")
        if len(residuals) > 0:
            print("잔차 통계:")
            print(residuals['residual'].describe())
            print("\n심볼별 잔차 수:")
            print(residuals['symbol'].value_counts())
        else:
            print("잔차 계산 결과가 없습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_residual_calculation()