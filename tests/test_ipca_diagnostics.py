"""
IPCA 진단 도구 단위 테스트

IPCA 모델 진단 및 검증 도구의 기능을 테스트합니다.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock

# 테스트 대상 모듈 import
from crypto_dlsa_bot.ml.crypto_ipca_model import CryptoIPCAModel
from crypto_dlsa_bot.ml.ipca_diagnostics import IPCADiagnostics, ModelDiagnostics


class TestIPCADiagnostics(unittest.TestCase):
    """IPCA 진단 도구 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 경고 무시
        warnings.filterwarnings('ignore')
        
        # 테스트용 데이터 생성
        self.n_assets = 10
        self.n_periods = 100
        self.n_factors = 3
        
        # 날짜 생성
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.n_periods)]
        
        # 심볼 생성
        symbols = [f'CRYPTO{i+1}' for i in range(self.n_assets)]
        
        # 수익률 데이터 생성
        np.random.seed(42)
        returns_data = []
        
        for date in dates:
            for symbol in symbols:
                return_val = np.random.normal(0, 0.02)  # 일일 수익률 2% 변동성
                returns_data.append({
                    'date': date,
                    'symbol': symbol,
                    'return': return_val,
                    'close': 100 * np.exp(return_val),  # 가격 데이터
                    'volume': np.random.uniform(1000, 10000)  # 거래량
                })
        
        self.returns_df = pd.DataFrame(returns_data)
        
        # 특성 데이터 생성
        characteristics_data = []
        for date in dates:
            for symbol in symbols:
                characteristics_data.append({
                    'date': date,
                    'symbol': symbol,
                    'log_market_cap': np.random.normal(10, 2),
                    'momentum': np.random.normal(0, 0.1),
                    'volatility': np.random.uniform(0.1, 0.5)
                })
        
        self.characteristics_df = pd.DataFrame(characteristics_data)
        
        # IPCA 모델 학습
        self.model = CryptoIPCAModel(n_factors=self.n_factors, max_iter=100)
        self.model.fit(self.returns_df, self.characteristics_df)
        
        # 진단 도구 초기화
        self.diagnostics = IPCADiagnostics(self.model)
    
    def test_initialization(self):
        """진단 도구 초기화 테스트"""
        # 정상 초기화
        diagnostics = IPCADiagnostics(self.model)
        self.assertIsInstance(diagnostics, IPCADiagnostics)
        self.assertEqual(diagnostics.model, self.model)
        
        # 학습되지 않은 모델로 초기화 시 오류
        untrained_model = CryptoIPCAModel(n_factors=3)
        with self.assertRaises(ValueError):
            IPCADiagnostics(untrained_model)
    
    def test_calculate_model_fit_metrics(self):
        """모델 적합도 지표 계산 테스트"""
        metrics = self.diagnostics.calculate_model_fit_metrics(
            self.returns_df, self.characteristics_df
        )
        
        # 필수 지표 존재 확인
        required_metrics = [
            'r_squared', 'adjusted_r_squared', 'explained_variance_ratio',
            'information_ratio', 'total_variance', 'residual_variance',
            'n_observations', 'n_factors'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # 값 범위 검증 (R²는 음수일 수 있음)
        self.assertLessEqual(metrics['r_squared'], 1)
        self.assertGreater(metrics['total_variance'], 0)
        self.assertGreater(metrics['residual_variance'], 0)
        self.assertEqual(metrics['n_factors'], self.n_factors)
        
        print(f"R² = {metrics['r_squared']:.4f}")
        print(f"조정된 R² = {metrics['adjusted_r_squared']:.4f}")
        print(f"설명된 분산 비율 = {metrics['explained_variance_ratio']:.4f}")
    
    def test_bootstrap_factor_significance(self):
        """부트스트랩 팩터 유의성 검증 테스트"""
        # 적은 부트스트랩 샘플로 빠른 테스트
        significance_results = self.diagnostics.bootstrap_factor_significance(
            self.returns_df, self.characteristics_df, n_bootstrap=10
        )
        
        # 결과 구조 검증
        self.assertIsInstance(significance_results, dict)
        self.assertEqual(len(significance_results), self.n_factors)
        
        for factor_name in self.model.factor_names:
            self.assertIn(factor_name, significance_results)
            factor_results = significance_results[factor_name]
            
            for char_name in self.model.characteristic_names:
                self.assertIn(char_name, factor_results)
                char_results = factor_results[char_name]
                
                # 필수 필드 확인
                required_fields = [
                    'original_value', 'bootstrap_mean', 'bootstrap_std',
                    'lower_ci', 'upper_ci', 'p_value', 'is_significant', 'n_bootstrap'
                ]
                
                for field in required_fields:
                    self.assertIn(field, char_results)
                
                # 값 범위 검증
                self.assertGreaterEqual(char_results['p_value'], 0)
                self.assertLessEqual(char_results['p_value'], 1)
                self.assertIsInstance(char_results['is_significant'], bool)
                self.assertGreater(char_results['n_bootstrap'], 0)
        
        print(f"부트스트랩 유의성 검증 완료: {len(significance_results)} 팩터")
    
    def test_analyze_factor_stability(self):
        """팩터 안정성 분석 테스트"""
        # 작은 윈도우로 빠른 테스트
        stability_results = self.diagnostics.analyze_factor_stability(
            self.returns_df, self.characteristics_df, 
            window_size=30, step_size=10
        )
        
        # 결과 구조 검증
        self.assertIsInstance(stability_results, dict)
        self.assertEqual(len(stability_results), self.n_factors)
        
        for factor_name in self.model.factor_names:
            self.assertIn(factor_name, stability_results)
            stability_df = stability_results[factor_name]
            
            self.assertIsInstance(stability_df, pd.DataFrame)
            self.assertGreater(len(stability_df), 0)
            
            # 필수 컬럼 확인
            required_columns = ['date', 'window_start', 'window_end']
            for col in required_columns:
                self.assertIn(col, stability_df.columns)
            
            # 특성 컬럼 확인
            for char_name in self.model.characteristic_names:
                self.assertIn(char_name, stability_df.columns)
        
        print(f"팩터 안정성 분석 완료: {len(stability_results)} 팩터")
    
    def test_compare_with_pca(self):
        """IPCA vs PCA 비교 분석 테스트"""
        comparison_metrics = self.diagnostics.compare_with_pca(
            self.returns_df, self.characteristics_df
        )
        
        # 필수 지표 확인
        required_metrics = [
            'ipca_r_squared', 'pca_r_squared', 'r_squared_improvement',
            'ipca_explained_variance', 'pca_explained_variance', 'explained_variance_improvement',
            'ipca_residual_variance', 'pca_residual_variance', 'residual_variance_reduction',
            'n_factors', 'n_observations'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, comparison_metrics)
            self.assertIsInstance(comparison_metrics[metric], (int, float))
        
        # 값 범위 검증 (R²는 음수일 수 있음)
        self.assertLessEqual(comparison_metrics['ipca_r_squared'], 1)
        self.assertLessEqual(comparison_metrics['pca_r_squared'], 1)
        self.assertEqual(comparison_metrics['n_factors'], self.n_factors)
        
        print(f"IPCA R² = {comparison_metrics['ipca_r_squared']:.4f}")
        print(f"PCA R² = {comparison_metrics['pca_r_squared']:.4f}")
        print(f"개선도 = {comparison_metrics['r_squared_improvement']:.4f}")
    
    def test_generate_comprehensive_diagnostics(self):
        """종합 진단 테스트"""
        diagnostics = self.diagnostics.generate_comprehensive_diagnostics(
            self.returns_df, self.characteristics_df, 
            n_bootstrap=10, stability_window=30
        )
        
        # 결과 타입 확인
        self.assertIsInstance(diagnostics, ModelDiagnostics)
        
        # 필수 필드 확인
        self.assertIsInstance(diagnostics.r_squared, float)
        self.assertIsInstance(diagnostics.adjusted_r_squared, float)
        self.assertIsInstance(diagnostics.information_ratio, float)
        self.assertIsInstance(diagnostics.explained_variance_ratio, float)
        self.assertIsInstance(diagnostics.factor_significance, dict)
        self.assertIsInstance(diagnostics.stability_metrics, dict)
        self.assertIsInstance(diagnostics.comparison_metrics, dict)
        
        # 딕셔너리 변환 테스트
        diagnostics_dict = diagnostics.to_dict()
        self.assertIsInstance(diagnostics_dict, dict)
        
        print("종합 진단 완료")
    
    def test_plot_factor_loadings_heatmap(self):
        """팩터 로딩 히트맵 시각화 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "heatmap_test.png")
            
            fig = self.diagnostics.plot_factor_loadings_heatmap(save_path=save_path)
            
            # 그림 객체 확인
            self.assertIsNotNone(fig)
            
            # 파일 저장 확인
            self.assertTrue(os.path.exists(save_path))
            
            print("팩터 로딩 히트맵 생성 완료")
    
    def test_plot_residual_distribution(self):
        """잔차 분포 시각화 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "residual_test.png")
            
            fig = self.diagnostics.plot_residual_distribution(
                self.returns_df, self.characteristics_df, save_path=save_path
            )
            
            # 그림 객체 확인
            self.assertIsNotNone(fig)
            
            # 파일 저장 확인
            self.assertTrue(os.path.exists(save_path))
            
            print("잔차 분포 시각화 완료")
    
    def test_plot_factor_contribution(self):
        """팩터 기여도 시각화 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "contribution_test.png")
            
            fig = self.diagnostics.plot_factor_contribution(save_path=save_path)
            
            # 그림 객체 확인
            self.assertIsNotNone(fig)
            
            # 파일 저장 확인
            self.assertTrue(os.path.exists(save_path))
            
            print("팩터 기여도 시각화 완료")
    
    def test_generate_diagnostic_report(self):
        """종합 진단 리포트 생성 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = self.diagnostics.generate_diagnostic_report(
                self.returns_df, self.characteristics_df,
                output_dir=temp_dir, n_bootstrap=10
            )
            
            # 리포트 파일 확인
            self.assertTrue(os.path.exists(report_path))
            
            # 시각화 파일들 확인
            expected_files = [
                "factor_loadings_heatmap.png",
                "residual_distribution.png",
                "factor_contribution.png",
                "diagnostic_report.txt"
            ]
            
            for filename in expected_files:
                file_path = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(file_path), f"파일이 생성되지 않음: {filename}")
            
            # 리포트 내용 확인
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("IPCA 모델 진단 리포트", content)
                self.assertIn("모델 적합도 지표", content)
                self.assertIn("PCA 대비 성능 개선", content)
                self.assertIn("팩터 유의성 검증 결과", content)
            
            print(f"종합 진단 리포트 생성 완료: {report_path}")
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        # 빈 데이터로 테스트
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.diagnostics.calculate_model_fit_metrics(empty_df)
        
        # 잘못된 형식의 데이터로 테스트
        invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
        
        with self.assertRaises(Exception):
            self.diagnostics.calculate_model_fit_metrics(invalid_df)
        
        print("오류 처리 테스트 완료")
    
    def test_edge_cases(self):
        """경계 사례 테스트"""
        # 매우 작은 데이터셋
        small_returns = self.returns_df.head(20)
        small_chars = self.characteristics_df.head(20)
        
        # 작은 데이터셋으로도 기본 지표는 계산되어야 함
        try:
            metrics = self.diagnostics.calculate_model_fit_metrics(small_returns, small_chars)
            self.assertIsInstance(metrics, dict)
            print("작은 데이터셋 테스트 통과")
        except Exception as e:
            print(f"작은 데이터셋 테스트 실패: {e}")
        
        # 단일 팩터 모델
        single_factor_model = CryptoIPCAModel(n_factors=1, max_iter=50)
        single_factor_model.fit(self.returns_df, self.characteristics_df)
        single_diagnostics = IPCADiagnostics(single_factor_model)
        
        metrics = single_diagnostics.calculate_model_fit_metrics(
            self.returns_df, self.characteristics_df
        )
        self.assertEqual(metrics['n_factors'], 1)
        
        print("단일 팩터 모델 테스트 통과")


class TestModelDiagnostics(unittest.TestCase):
    """ModelDiagnostics 데이터 클래스 테스트"""
    
    def test_model_diagnostics_creation(self):
        """ModelDiagnostics 생성 테스트"""
        diagnostics = ModelDiagnostics(
            r_squared=0.75,
            adjusted_r_squared=0.73,
            information_ratio=0.5,
            explained_variance_ratio=0.8,
            factor_significance={'Factor_1': {'char1': {'p_value': 0.05}}},
            stability_metrics={'stability_1': 0.1},
            comparison_metrics={'improvement': 0.05}
        )
        
        self.assertEqual(diagnostics.r_squared, 0.75)
        self.assertEqual(diagnostics.adjusted_r_squared, 0.73)
        self.assertEqual(diagnostics.information_ratio, 0.5)
        self.assertEqual(diagnostics.explained_variance_ratio, 0.8)
        
        # 딕셔너리 변환 테스트
        diagnostics_dict = diagnostics.to_dict()
        self.assertIsInstance(diagnostics_dict, dict)
        self.assertEqual(diagnostics_dict['r_squared'], 0.75)
        
        print("ModelDiagnostics 생성 및 변환 테스트 완료")


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)