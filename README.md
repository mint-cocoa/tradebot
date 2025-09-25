# Crypto-DLSA Bot

Deep Learning Statistical Arbitrage Bot for Cryptocurrency Markets

## Quick start (IPCA training & benchmark on real data)

This repository includes a practical IPCA workflow backed by real OHLCV parquet data and two entrypoint scripts:

- `scripts/run_real_ipca_training.py`: End-to-end training + optional rolling OOS backtest.
- `scripts/benchmark_residual_calculation.py`: Residual/model-performance benchmark (IC, R2_OS, long–short).

Setup
- Create/activate a Python 3.12 virtualenv (or use `./crypto-dlsa-env/`).
- Install: `pip install -r requirements.txt`

Train + OOS backtest example
```
PYTHONPATH=. ./crypto-dlsa-env/bin/python scripts/run_real_ipca_training.py \
	--symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT,XRPUSDT,DOTUSDT,MATICUSDT \
	--start 2024-01-01 --end 2025-09-01 --timeframe 1d \
	--input-parquet data/processed/ohlcv/1d/multi_symbol/data_20250923_173421.parquet \
	--output-dir models/ipca_2025_09_multi_bestcfg \
	--oos-backtest --oos-warmup 60 --oos-train-window 30 --oos-refit-frequency 5
```

Config-benchmark example
```
PYTHONPATH=. ./crypto-dlsa-env/bin/python scripts/benchmark_residual_calculation.py \
	--skip-scalability \
	--symbols BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,SOLUSDT,XRPUSDT,DOTUSDT,MATICUSDT \
	--timeframe 1d \
	--start-date 2024-01-01 --end-date 2025-09-01 \
	--output models/ipca_2025_09_multi/bench_config_oos.json
```

Notes
- Large artifacts (`data/`, `models/`) are git-ignored to keep the repo small.
- IC measures same-day cross-sectional correlation between expected and actual returns; LS Sharpe evaluates a daily equal-weight long–short portfolio built from the signal.

## 개요

Crypto-DLSA Bot은 Guijarro-Ordonez 등의 "Deep Learning Statistical Arbitrage" 논문을 기반으로 한 암호화폐 시장 특화 통계적 차익거래 시스템입니다. 이 시스템은 CNN+Transformer 아키텍처를 사용하여 암호화폐 시장의 비효율성을 포착하고 리스크 조정 수익률을 극대화합니다.

## 주요 기능

- **다중 데이터 소스 지원**: Binance, CoinGecko, Glassnode 등 주요 데이터 제공업체 통합
- **암호화폐 특화 팩터 모델**: Market, Size, NVT Ratio, Momentum, Volatility 팩터 계산
- **CNN+Transformer 딥러닝 모델**: 시계열 신호 추출 및 포트폴리오 최적화
- **현실적인 백테스팅**: 거래 비용, 슬리피지, 가스비를 고려한 정교한 시뮬레이션
- **실시간 모니터링**: 성과 분석 및 리스크 관리 대시보드

## 시스템 아키텍처

```
Data Layer → Processing Layer → ML Layer → Trading Layer → Visualization Layer
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/crypto-dlsa/crypto-dlsa-bot.git
cd crypto-dlsa-bot
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 등을 설정
```

### 5. 설정 파일 확인

`config.yaml` 파일을 확인하고 필요에 따라 수정합니다.

## 사용 방법

### 기본 사용법

```python
from crypto_dlsa_bot.config.settings import load_config
from crypto_dlsa_bot.services.data_collector import BinanceDataCollector
from crypto_dlsa_bot.services.factor_engine import CryptoFactorEngine
from crypto_dlsa_bot.ml.cnn_transformer import CNNTransformerModel
from crypto_dlsa_bot.backtest.engine import BacktestEngine

# 설정 로드
config = load_config('config.yaml')

# 데이터 수집
collector = BinanceDataCollector(config.data)
data = collector.collect_ohlcv(['BTCUSDT', 'ETHUSDT'], '1h', start_date, end_date)

# 팩터 계산
factor_engine = CryptoFactorEngine(config.factor)
factors = factor_engine.calculate_market_factors(data)
residuals = factor_engine.calculate_residuals(model, returns)

# 모델 학습
model = CNNTransformerModel(config.model)
model.train(features, targets)

# 백테스팅
backtest_engine = BacktestEngine(config.backtest)
results = backtest_engine.run_backtest(predictions, price_data, start_date, end_date)
```

## 프로젝트 구조

```
crypto-dlsa-bot/
├── crypto_dlsa_bot/
│   ├── __init__.py
│   ├── config/              # 설정 관리
│   ├── data/               # 데이터 수집 및 전처리
│   ├── models/             # 데이터 모델 정의
│   ├── services/           # 서비스 인터페이스
│   ├─�� ml/                 # 머신러닝 모델
│   ├── backtest/           # 백테스팅 엔진
│   └── utils/              # 유틸리티 함수
├── tests/                  # 테스트 코드
├── config.yaml            # 설정 파일
├── requirements.txt       # 의존성 목록
└── README.md             # 프로젝트 문서
```

## 설정

### API 키 설정

다음 API 키들이 필요합니다:

- **Binance API**: 가격 데이터 수집용
- **CoinGecko API**: 메타데이터 및 보완 데이터용
- **Glassnode API**: 온체인 메트릭용 (선택사항)
- **Dune API**: 고급 온체인 분석용 (선택사항)

### 하드웨어 요구사항

- **CPU**: 4코어 이상 권장
- **RAM**: 16GB 이상 권장
- **GPU**: NVIDIA GPU (CUDA 지원) 권장 (딥러닝 학습용)
- **저장공간**: 100GB 이상 (데이터 저장용)

## 개발

### 개발 환경 설정

```bash
pip install -e ".[dev]"
pre-commit install
```

### 테스트 실행

```bash
pytest tests/ -v --cov=crypto_dlsa_bot
```

### 코드 포맷팅

```bash
black crypto_dlsa_bot/
flake8 crypto_dlsa_bot/
mypy crypto_dlsa_bot/
```

## 라이선스

MIT License

## 기여

프로젝트에 기여하고 싶으시다면 다음 단계를 따라주세요:

1. 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 지원

문제가 있거나 질문이 있으시면 GitHub Issues를 통해 문의해주세요.

## 면책조항

이 소프트웨어는 교육 및 연구 목적으로 제공됩니다. 실제 거래에 사용하기 전에 충분한 테스트와 검증을 수행하시기 바랍니다. 투자에는 항상 위험이 따르며, 과거 성과가 미래 결과를 보장하지 않습니다.