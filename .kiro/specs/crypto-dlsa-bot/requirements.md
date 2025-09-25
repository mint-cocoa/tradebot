# Requirements Document

## Introduction

Crypto-DLSA (Cryptocurrency Deep Learning Statistical Arbitrage) Bot은 Guijarro-Ordonez 등의 "Deep Learning Statistical Arbitrage" 논문에서 제시된 딥러닝 기반 통계적 차익거래 프레임워크를 암호화폐 시장에 적용한 자동화된 트레이딩 전략 시스템입니다. 이 시스템은 암호화폐 시장의 비효율성을 포착하여 리스크 조정 수익률을 극대화하는 것을 목표로 하며, 백테스팅을 통해 전략의 유효성을 검증합니다.

## Requirements

### Requirement 1

**User Story:** 퀀트 분석가로서, 주요 CEX와 온체인 데이터를 자동으로 수집하고 전처리할 수 있는 시스템이 필요합니다. 이를 통해 정확하고 일관된 데이터를 기반으로 분석을 수행할 수 있습니다.

#### Acceptance Criteria

1. WHEN 시스템이 시작되면 THEN 바이낸스, 바이비트 등 주요 CEX API를 통해 OHLCV 및 거래량 데이터를 시간별/분별 주기로 수집해야 합니다
2. WHEN 온체인 데이터가 필요하면 THEN Glassnode, Dune 등의 API를 통해 토크노믹스 및 네트워크 활동 데이터(활성 주소 수, TVL)를 수집해야 합니다
3. WHEN 데이터 수집이 완료되면 THEN 결측치 처리, 정규화 등 전처리 과정을 거쳐 시계열 데이터베이스에 저장해야 합니다
4. IF API 오류가 발생하면 THEN 재시도 로직을 통해 99.9%의 데이터 수집 성공률을 달성해야 합니다

### Requirement 2

**User Story:** 퀀트 분석가로서, 암호화폐 시장 특성을 반영한 팩터 모델을 구현하여 각 암호화폐의 잔차를 계산할 수 있어야 합니다. 이를 통해 시장 중립적인 차익거래 기회를 식별할 수 있습니다.

#### Acceptance Criteria

1. WHEN 팩터 계산이 요청되면 THEN Market, Size, NVT Ratio, Momentum 등 암호화폐 시장 팩터를 계산해야 합니다
2. WHEN 수익률 및 특성 데이터가 준비되면 THEN PCA, IPCA 팩터 모델을 구현하여 적용해야 합니다
3. WHEN 팩터 모델이 정의되면 THEN 각 암호화폐의 잔차 시계열을 롤링 윈도우 방식으로 계산하고 저장해야 합니다
4. IF 새로운 팩터 모델이 추가되면 THEN 최소한의 코드 수정으로 시스템에 통합할 수 있어야 합니다

### Requirement 3

**User Story:** 딥러닝 엔지니어로서, CNN+Transformer 아키텍처를 구현하고 샤프 지수 최대화를 위한 학습 파이프라인을 구축할 수 있어야 합니다. 이를 통해 효과적인 포트폴리오 배분 전략을 학습할 수 있습니다.

#### Acceptance Criteria

1. WHEN 딥러닝 모델이 구현되면 THEN 논문에 명시된 CNN+Transformer 아키텍처를 PyTorch 또는 TensorFlow로 구현해야 합니다
2. WHEN 모델 학습이 시작되면 THEN 샤프 지수 최대화 또는 평균-분산 최적화를 위한 커스텀 손실 함수를 사용해야 합니다
3. WHEN 학습 파이프라인이 실행되면 THEN 롤링 윈도우 기반의 모델 학습, 검증, 추론을 자동화해야 합니다
4. IF GPU가 사용 가능하면 THEN GPU를 활용한 학습 가속을 지원해야 합니다

### Requirement 4

**User Story:** 포트폴리오 매니저로서, 현실적인 거래 비용을 고려한 정교한 백테스팅을 통해 전략의 실제 성과를 평가할 수 있어야 합니다. 이를 통해 전략의 실용성을 검증할 수 있습니다.

#### Acceptance Criteria

1. WHEN 모델이 거래 비중을 출력하면 THEN 해당 비중에 따라 포트폴리오를 구성하고 수익률을 시뮬레이션해야 합니다
2. WHEN 백테스팅이 실행되면 THEN 거래 수수료, 슬리피지, 네트워크 가스비 등 현실적인 거래 비용을 적용해야 합니다
3. WHEN 백테스팅이 완료되면 THEN 연율 샤프 지수, 알파, 최대 낙폭(MDD), 수익률 등 주요 KPI를 계산하고 리포트를 생성해야 합니다
4. WHEN 분석 결과가 생성되면 THEN 누적 수익률 곡선, 자산별 비중 변화, 팩터 노출도 등을 시각화해야 합니다

### Requirement 5

**User Story:** 시스템 관리자로서, 대용량 데이터 처리와 확장 가능한 아키텍처를 통해 안정적이고 효율적인 시스템 운영이 가능해야 합니다.

#### Acceptance Criteria

1. WHEN 5년치 시간별 데이터로 100개 암호화폐에 대한 백테스팅을 실행하면 THEN 12시간 이내에 완료되어야 합니다
2. WHEN 새로운 암호화폐나 데이터 소스가 추가되면 THEN 최소한의 코드 수정으로 시스템에 통합할 수 있어야 합니다
3. WHEN API 키 및 민감 정보를 처리하면 THEN 암호화하여 안전하게 관리해야 합니다
4. IF 시스템 장애가 발생하면 THEN 데이터 무결성을 보장하고 복구 가능해야 합니다