#!/usr/bin/env python3
"""
Binance Public Data 스크립트 직접 사용 시연

이 스크립트는 binance-public-data 저장소의 Python 스크립트를 직접 호출하여
실제 데이터를 다운로드하고 처리하는 과정을 보여줍니다.
"""

import os
import sys
import subprocess
import zipfile
import csv
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_dlsa_bot.utils.logging import get_logger

def check_binance_public_data_repo():
    """binance-public-data 저장소 확인"""
    logger = get_logger(__name__)
    
    repo_path = Path("binance-public-data")
    python_path = repo_path / "python"
    
    if not repo_path.exists():
        logger.error("binance-public-data 저장소가 없습니다.")
        logger.info("다음 명령어로 저장소를 클론하세요:")
        logger.info("git clone https://github.com/binance/binance-public-data.git")
        return False, None, None
    
    if not python_path.exists():
        logger.error("binance-public-data/python 폴더가 없습니다.")
        return False, None, None
    
    download_script = python_path / "download-kline.py"
    if not download_script.exists():
        logger.error("download-kline.py 스크립트가 없습니다.")
        return False, None, None
    
    logger.info(f"✓ binance-public-data 저장소 확인 완료: {repo_path}")
    return True, repo_path, python_path

def demonstrate_direct_download():
    """직접 다운로드 시연"""
    logger = get_logger(__name__)
    
    logger.info("=== Binance Public Data 직접 다운로드 시연 ===")
    
    # 1. 저장소 확인
    repo_exists, repo_path, python_path = check_binance_public_data_repo()
    if not repo_exists:
        return
    
    # 2. 다운로드 디렉토리 설정
    download_dir = Path("demo_direct_download")
    download_dir.mkdir(exist_ok=True)
    
    logger.info(f"다운로드 디렉토리: {download_dir.absolute()}")
    
    # 3. 다운로드 명령어 구성
    # 최근 몇 일간의 BTCUSDT 1시간 데이터 다운로드
    end_date = datetime.now() - timedelta(days=7)  # 1주일 전 (public data 사용 보장)
    start_date = end_date - timedelta(days=1)  # 1일간의 데이터
    
    cmd = [
        "python3", "download-kline.py",
        "-t", "spot",                                    # 현물 시장
        "-s", "BTCUSDT",                                # 심볼
        "-i", "1h",                                     # 1시간 간격
        "-startDate", start_date.strftime("%Y-%m-%d"),  # 시작일
        "-endDate", end_date.strftime("%Y-%m-%d"),      # 종료일
        "-folder", str(download_dir.absolute()),        # 저장 폴더
        "-skip-monthly", "1",                           # 월간 데이터 스킵
        "-skip-daily", "0"                              # 일간 데이터 사용
    ]
    
    logger.info("다운로드 명령어:")
    logger.info(" ".join(cmd))
    
    # 4. 환경 변수 설정
    env = os.environ.copy()
    env["STORE_DIRECTORY"] = str(download_dir.absolute())
    
    try:
        # 5. 다운로드 실행
        logger.info("다운로드 시작...")
        
        result = subprocess.run(
            cmd,
            cwd=python_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode == 0:
            logger.info("✓ 다운로드 완료")
            logger.info("다운로드 출력:")
            print(result.stdout)
            
            # 6. 다운로드된 파일 확인
            analyze_downloaded_files(download_dir, logger)
            
        else:
            logger.error(f"다운로드 실패 (exit code: {result.returncode})")
            logger.error("에러 출력:")
            print(result.stderr)
            
            # 파일이 없을 수 있으므로 (과거 데이터) 시연용 가짜 데이터 생성
            logger.info("시연을 위해 샘플 데이터를 생성합니다...")
            create_sample_data(download_dir, start_date, end_date, logger)
            
    except subprocess.TimeoutExpired:
        logger.error("다운로드 타임아웃")
    except Exception as e:
        logger.error(f"다운로드 중 오류: {e}")
        # 시연용 샘플 데이터 생성
        logger.info("시연을 위해 샘플 데이터를 생성합니다...")
        create_sample_data(download_dir, start_date, end_date, logger)

def create_sample_data(download_dir: Path, start_date: datetime, end_date: datetime, logger):
    """시연용 샘플 데이터 생성"""
    
    # Binance public data 구조 모방
    data_path = download_dir / "data" / "spot" / "daily" / "klines" / "BTCUSDT" / "1h"
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 샘플 ZIP 파일 생성
    date_str = start_date.strftime("%Y-%m-%d")
    zip_filename = f"BTCUSDT-1h-{date_str}.zip"
    csv_filename = f"BTCUSDT-1h-{date_str}.csv"
    
    zip_path = data_path / zip_filename
    
    # 샘플 CSV 데이터 생성
    sample_data = []
    current_time = start_date
    base_price = 50000.0
    
    while current_time < end_date:
        timestamp_ms = int(current_time.timestamp() * 1000)
        open_price = base_price + (hash(str(current_time)) % 1000 - 500)
        high_price = open_price + (hash(str(current_time + timedelta(minutes=1))) % 500)
        low_price = open_price - (hash(str(current_time + timedelta(minutes=2))) % 300)
        close_price = open_price + (hash(str(current_time + timedelta(minutes=3))) % 200 - 100)
        volume = 100 + (hash(str(current_time + timedelta(minutes=4))) % 500)
        
        sample_data.append([
            timestamp_ms,           # Open time
            f"{open_price:.2f}",    # Open
            f"{high_price:.2f}",    # High  
            f"{low_price:.2f}",     # Low
            f"{close_price:.2f}",   # Close
            f"{volume:.5f}",        # Volume
            timestamp_ms + 3599999, # Close time
            "5000000.00",          # Quote asset volume
            "1000",                # Number of trades
            "50.25",               # Taker buy base asset volume
            "2525000.00",          # Taker buy quote asset volume
            "0"                    # Ignore
        ])
        
        current_time += timedelta(hours=1)
        base_price = close_price  # 다음 캔들의 시작가
    
    # CSV 파일을 ZIP으로 압축
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        csv_content = "\n".join([",".join(row) for row in sample_data])
        zf.writestr(csv_filename, csv_content)
    
    logger.info(f"✓ 샘플 데이터 생성 완료: {zip_path}")
    logger.info(f"  - 레코드 수: {len(sample_data)}")
    logger.info(f"  - 시간 범위: {start_date} ~ {end_date}")

def analyze_downloaded_files(download_dir: Path, logger):
    """다운로드된 파일 분석"""
    
    logger.info("6. 다운로드된 파일 분석")
    
    # 데이터 디렉토리 탐색
    data_dir = download_dir / "data"
    if not data_dir.exists():
        logger.warning("data 디렉토리가 없습니다.")
        return
    
    # ZIP 파일 찾기
    zip_files = list(data_dir.rglob("*.zip"))
    logger.info(f"발견된 ZIP 파일 수: {len(zip_files)}")
    
    for zip_file in zip_files:
        logger.info(f"분석 중: {zip_file}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                file_list = zf.namelist()
                logger.info(f"  ZIP 내부 파일: {file_list}")
                
                # CSV 파일 찾기
                csv_files = [f for f in file_list if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    logger.info(f"  CSV 파일 분석: {csv_file}")
                    
                    with zf.open(csv_file) as f:
                        # 처음 몇 줄 읽기
                        content = f.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        logger.info(f"    총 라인 수: {len(lines)}")
                        
                        if lines:
                            # 첫 번째 라인 분석
                            first_line = lines[0].split(',')
                            logger.info(f"    첫 번째 레코드: {first_line}")
                            logger.info(f"    컬럼 수: {len(first_line)}")
                            
                            # 컬럼 설명
                            column_names = [
                                "Open time", "Open", "High", "Low", "Close", "Volume",
                                "Close time", "Quote asset volume", "Number of trades",
                                "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
                            ]
                            
                            logger.info("    컬럼 매핑:")
                            for i, (value, name) in enumerate(zip(first_line, column_names)):
                                logger.info(f"      [{i}] {name}: {value}")
                            
                            # 데이터를 DataFrame으로 변환
                            df = parse_csv_to_dataframe(content, logger)
                            if df is not None and not df.empty:
                                logger.info(f"    DataFrame 생성 완료: {df.shape}")
                                logger.info(f"    시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
                                
                                # 샘플 데이터 출력
                                logger.info("    샘플 데이터:")
                                print(df.head(3).to_string())
        
        except Exception as e:
            logger.error(f"ZIP 파일 분석 중 오류: {e}")

def parse_csv_to_dataframe(csv_content: str, logger) -> pd.DataFrame:
    """CSV 내용을 DataFrame으로 변환"""
    
    try:
        lines = csv_content.strip().split('\n')
        data = []
        
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 6:  # 최소 OHLCV 데이터 확인
                    try:
                        timestamp = datetime.fromtimestamp(int(parts[0]) / 1000)
                        data.append({
                            'timestamp': timestamp,
                            'symbol': 'BTCUSDT',  # 하드코딩 (실제로는 파일명에서 추출)
                            'open': float(parts[1]),
                            'high': float(parts[2]),
                            'low': float(parts[3]),
                            'close': float(parts[4]),
                            'volume': float(parts[5])
                        })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"잘못된 데이터 라인 스킵: {line[:50]}... 오류: {e}")
                        continue
        
        if data:
            df = pd.DataFrame(data)
            return df.sort_values('timestamp')
        else:
            logger.warning("유효한 데이터가 없습니다.")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"CSV 파싱 중 오류: {e}")
        return pd.DataFrame()

def cleanup_demo_files():
    """시연 파일 정리"""
    logger = get_logger(__name__)
    
    download_dir = Path("demo_direct_download")
    if download_dir.exists():
        import shutil
        shutil.rmtree(download_dir)
        logger.info("시연 파일 정리 완료")

if __name__ == "__main__":
    try:
        demonstrate_direct_download()
    finally:
        # 정리
        cleanup_demo_files()