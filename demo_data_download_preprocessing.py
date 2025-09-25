#!/usr/bin/env python3
"""
Binance Public Data 다운로드 및 전처리 과정 시연 스크립트

이 스크립트는 다음 과정을 시연합니다:
1. binance-public-data 스크립트를 사용한 실제 데이터 다운로드
2. 다운로드된 ZIP 파일의 구조 확인
3. CSV 데이터 파싱 및 전처리
4. 데이터 품질 검증 및 정리
5. 최종 데이터 저장
"""

import sys
import os
import subprocess
import zipfile
import csv
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_dlsa_bot.services.binance_data_collector import BinanceDataCollector
from crypto_dlsa_bot.utils.logging import get_logger

def demonstrate_data_download_and_preprocessing():
    """데이터 다운로드 및 전처리 과정 시연"""
    logger = get_logger(__name__)
    
    logger.info("=== Binance Public Data 다운로드 및 전처리 시연 ===")
    
    # 1. 데이터 수집기 초기화
    collector = BinanceDataCollector(
        data_dir="demo_download_data",
        use_public_data=True
    )
    
    # 2. binance-public-data 저장소 확인
    logger.info("1. binance-public-data 저장소 확인")
    if not collector.public_data_repo.exists():
        logger.error(f"binance-public-data 저장소를 찾을 수 없습니다: {collector.public_data_repo}")
        logger.info("다음 명령어로 저장소를 클론하세요:")
        logger.info("git clone https://github.com/binance/binance-public-data.git")
        return
    
    logger.info(f"✓ binance-public-data 저장소 발견: {collector.public_data_repo}")
    
    # 3. 소량의 테스트 데이터 다운로드 (최근 2일간의 BTCUSDT 1시간 데이터)
    logger.info("2. 테스트 데이터 다운로드 시작")
    
    # 최근 데이터가 아닌 과거 데이터를 사용하여 public data 방식 강제
    end_date = datetime.now() - timedelta(days=30)  # 30일 전
    start_date = end_date - timedelta(days=2)  # 2일간의 데이터
    
    logger.info(f"다운로드 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    logger.info("심볼: BTCUSDT, 시간프레임: 1h")
    
    try:
        # 실제 데이터 수집 (public data 방식 사용)
        data = collector.collect_ohlcv(
            symbols=['BTCUSDT'],
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            market_type='spot'
        )
        
        if data.empty:
            logger.warning("다운로드된 데이터가 없습니다. API 방식으로 대체 시도...")
            # API 방식으로 최근 데이터 수집
            recent_end = datetime.now() - timedelta(days=1)
            recent_start = recent_end - timedelta(hours=24)
            
            data = collector._collect_using_api(
                symbols=['BTCUSDT'],
                timeframe='1h',
                start_date=recent_start,
                end_date=recent_end,
                market_type='spot'
            )
        
        if not data.empty:
            logger.info(f"✓ 데이터 다운로드 완료: {len(data)} 레코드")
            
            # 4. 다운로드된 데이터 구조 확인
            logger.info("3. 다운로드된 데이터 구조 확인")
            logger.info(f"데이터 형태: {data.shape}")
            logger.info(f"컬럼: {list(data.columns)}")
            logger.info(f"데이터 타입:\n{data.dtypes}")
            logger.info(f"시간 범위: {data['timestamp'].min()} ~ {data['timestamp'].max()}")
            
            # 샘플 데이터 출력
            logger.info("샘플 데이터 (처음 5개 레코드):")
            print(data.head().to_string())
            
            # 5. 데이터 품질 검증
            logger.info("4. 데이터 품질 검증")
            
            # 원본 데이터 품질 리포트
            quality_report = collector.get_data_quality_report(data)
            logger.info("원본 데이터 품질 리포트:")
            print(json.dumps(quality_report, indent=2, ensure_ascii=False, default=str))
            
            # 데이터 유효성 검사
            is_valid = collector.validate_data(data)
            logger.info(f"데이터 유효성 검사 결과: {is_valid}")
            
            # 6. 데이터 전처리 및 정리
            logger.info("5. 데이터 전처리 및 정리")
            
            # 인위적으로 일부 데이터 품질 문제 생성 (시연용)
            logger.info("시연을 위해 일부 데이터 품질 문제를 생성합니다...")
            dirty_data = data.copy()
            
            # 중복 레코드 추가
            if len(dirty_data) > 0:
                dirty_data = pd.concat([dirty_data, dirty_data.iloc[:2]], ignore_index=True)
                
                # 일부 null 값 추가
                dirty_data.loc[len(dirty_data)//2, 'volume'] = None
                
                # 잘못된 OHLC 관계 생성
                if len(dirty_data) > 3:
                    dirty_data.loc[3, 'high'] = dirty_data.loc[3, 'low'] - 100  # high < low
            
            logger.info(f"문제가 있는 데이터 생성 완료: {len(dirty_data)} 레코드")
            
            # 문제가 있는 데이터의 품질 리포트
            dirty_quality_report = collector.get_data_quality_report(dirty_data)
            logger.info("문제가 있는 데이터 품질 리포트:")
            print(json.dumps(dirty_quality_report, indent=2, ensure_ascii=False, default=str))
            
            # 데이터 정리
            cleaned_data = collector.clean_data(dirty_data)
            logger.info(f"✓ 데이터 정리 완료: {len(cleaned_data)} 레코드")
            
            # 정리된 데이터 품질 리포트
            cleaned_quality_report = collector.get_data_quality_report(cleaned_data)
            logger.info("정리된 데이터 품질 리포트:")
            print(json.dumps(cleaned_quality_report, indent=2, ensure_ascii=False, default=str))
            
            # 7. 전처리된 데이터 저장
            logger.info("6. 전처리된 데이터 저장")
            
            if len(cleaned_data) > 0:
                # 데이터 저장
                saved_file = collector.save_data(cleaned_data, 'preprocessed_btcusdt_demo.csv')
                logger.info(f"✓ 전처리된 데이터 저장 완료: {saved_file}")
                
                # 저장된 데이터 다시 로드하여 검증
                loaded_data = collector.load_data(saved_file)
                logger.info(f"✓ 저장된 데이터 로드 검증 완료: {len(loaded_data)} 레코드")
                
                # 8. 전처리 결과 요약
                logger.info("7. 전처리 결과 요약")
                logger.info(f"원본 데이터: {len(data)} 레코드")
                logger.info(f"문제 데이터: {len(dirty_data)} 레코드")
                logger.info(f"정리된 데이터: {len(cleaned_data)} 레코드")
                logger.info(f"제거된 레코드: {len(dirty_data) - len(cleaned_data)}")
                
                # 데이터 품질 점수 비교
                original_score = quality_report['data_quality']['overall_score']
                dirty_score = dirty_quality_report['data_quality']['overall_score']
                cleaned_score = cleaned_quality_report['data_quality']['overall_score']
                
                logger.info(f"품질 점수 변화:")
                logger.info(f"  원본: {original_score:.2f}")
                logger.info(f"  문제 데이터: {dirty_score:.2f}")
                logger.info(f"  정리 후: {cleaned_score:.2f}")
                
                # 정리
                saved_file.unlink()
                logger.info("임시 파일 정리 완료")
            
        else:
            logger.error("데이터를 다운로드할 수 없습니다.")
            
    except Exception as e:
        logger.error(f"데이터 다운로드 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("=== 시연 완료 ===")

def demonstrate_zip_file_structure():
    """ZIP 파일 구조 확인 시연 (binance-public-data 형식)"""
    logger = get_logger(__name__)
    
    logger.info("=== Binance Public Data ZIP 파일 구조 시연 ===")
    
    # 실제 binance-public-data에서 다운로드되는 파일 구조 설명
    logger.info("Binance Public Data ZIP 파일 구조:")
    logger.info("1. 파일명 형식: SYMBOL-INTERVAL-YEAR-MONTH.zip (월간 데이터)")
    logger.info("   예: BTCUSDT-1h-2023-01.zip")
    logger.info("2. 파일명 형식: SYMBOL-INTERVAL-DATE.zip (일간 데이터)")
    logger.info("   예: BTCUSDT-1h-2023-01-15.zip")
    
    logger.info("3. ZIP 파일 내부 구조:")
    logger.info("   - CSV 파일 1개 (ZIP 파일명과 동일한 이름)")
    logger.info("   - CSV 컬럼 구조:")
    logger.info("     [0] Open time (timestamp in milliseconds)")
    logger.info("     [1] Open price")
    logger.info("     [2] High price") 
    logger.info("     [3] Low price")
    logger.info("     [4] Close price")
    logger.info("     [5] Volume")
    logger.info("     [6] Close time")
    logger.info("     [7] Quote asset volume")
    logger.info("     [8] Number of trades")
    logger.info("     [9] Taker buy base asset volume")
    logger.info("     [10] Taker buy quote asset volume")
    logger.info("     [11] Ignore")
    
    logger.info("4. 데이터 전처리 과정:")
    logger.info("   - 타임스탬프 변환 (milliseconds → datetime)")
    logger.info("   - 가격 데이터 float 변환")
    logger.info("   - 불필요한 컬럼 제거 (OHLCV만 유지)")
    logger.info("   - 데이터 유효성 검증")
    logger.info("   - 중복 제거 및 정렬")

if __name__ == "__main__":
    # ZIP 파일 구조 설명
    demonstrate_zip_file_structure()
    
    print("\n" + "="*80 + "\n")
    
    # 실제 데이터 다운로드 및 전처리 시연
    demonstrate_data_download_and_preprocessing()