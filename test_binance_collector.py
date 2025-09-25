#!/usr/bin/env python3
"""
Binance Data Collector 테스트 스크립트
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector
from crypto_dlsa_bot.utils.logging import get_logger

def test_binance_collector():
    """Binance 데이터 수집기 테스트"""
    logger = get_logger(__name__)
    
    # 테스트용 데이터 디렉토리 생성
    test_data_dir = project_root / "test_data" / "binance"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # BinanceDataCollector 인스턴스 생성
    collector = BinanceDataCollector(
        data_dir=str(test_data_dir),
        max_retries=2,
        retry_delay=1.0,
        rate_limit_delay=0.5,
        max_workers=2,
        use_public_data=True
    )
    
    logger.info("=== Binance Data Collector 테스트 시작 ===")
    
    # 1. 기본 설정 확인
    logger.info(f"데이터 디렉토리: {collector.data_dir}")
    logger.info(f"지원 시간프레임: {list(collector.TIMEFRAME_MAPPING.keys())}")
    logger.info(f"주요 심볼: {collector.MAJOR_SYMBOLS[:5]}...")
    
    # 2. 작은 범위 데이터 수집 테스트 (API 방식)
    logger.info("\n=== API를 통한 소량 데이터 수집 테스트 ===")
    
    # 최근 2일간의 1시간 데이터
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    try:
        # API를 통한 데이터 수집
        collector.use_public_data = False  # API 방식 강제
        
        api_data = collector.collect_ohlcv(
            symbols=['BTCUSDT'],
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            market_type='spot'
        )
        
        if not api_data.empty:
            logger.info(f"API 데이터 수집 성공: {len(api_data)} 레코드")
            logger.info(f"데이터 범위: {api_data['timestamp'].min()} ~ {api_data['timestamp'].max()}")
            logger.info(f"샘플 데이터:\n{api_data.head()}")
            
            # 데이터 검증
            is_valid = collector.validate_data(api_data)
            logger.info(f"데이터 검증 결과: {'통과' if is_valid else '실패'}")
            
        else:
            logger.warning("API를 통한 데이터 수집 실패")
            
    except Exception as e:
        logger.error(f"API 데이터 수집 중 오류: {e}")
    
    # 3. 사용 가능한 심볼 조회 테스트
    logger.info("\n=== 사용 가능한 심볼 조회 테스트 ===")
    
    try:
        available_symbols = collector.get_available_symbols('spot')
        logger.info(f"사용 가능한 심볼 수: {len(available_symbols)}")
        logger.info(f"처음 10개 심볼: {available_symbols[:10]}")
        
    except Exception as e:
        logger.error(f"심볼 조회 중 오류: {e}")
    
    # 4. 데이터 정규화 테스트
    logger.info("\n=== 데이터 정규화 테스트 ===")
    
    if 'api_data' in locals() and not api_data.empty:
        try:
            normalized_data = collector.normalize_data(api_data)
            logger.info(f"정규화 완료: {len(normalized_data)} 레코드")
            
        except Exception as e:
            logger.error(f"데이터 정규화 중 오류: {e}")
    
    # 5. Binance Public Data 스크립트 경로 확인
    logger.info("\n=== Binance Public Data 설정 확인 ===")
    
    public_data_repo = collector.public_data_repo
    download_script = public_data_repo / "python" / "download-kline.py"
    
    logger.info(f"Public Data 리포지토리 경로: {public_data_repo}")
    logger.info(f"리포지토리 존재 여부: {public_data_repo.exists()}")
    logger.info(f"다운로드 스크립트 존재 여부: {download_script.exists()}")
    
    if download_script.exists():
        logger.info("✅ Binance Public Data 스크립트 준비 완료")
    else:
        logger.warning("❌ Binance Public Data 스크립트를 찾을 수 없음")
    
    logger.info("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_binance_collector()