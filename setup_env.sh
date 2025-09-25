#!/bin/bash

# Crypto-DLSA Bot 환경 설정 스크립트

echo "🚀 Crypto-DLSA Bot 환경 설정을 시작합니다..."

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "❌ uv가 설치되어 있지 않습니다. 먼저 uv를 설치해주세요:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv 버전: $(uv --version)"

# 가상환경 생성
if [ ! -d "crypto-dlsa-env" ]; then
    echo "📦 가상환경을 생성합니다..."
    uv venv crypto-dlsa-env
else
    echo "✅ 가상환경이 이미 존재합니다."
fi

# 가상환경 활성화 및 패키지 설치
echo "📥 필요한 패키지들을 설치합니다..."
source crypto-dlsa-env/bin/activate

# 기본 패키지 설치
uv pip install pandas numpy scipy scikit-learn
uv pip install requests python-binance pycoingecko
uv pip install pyyaml python-dotenv
uv pip install pytest pytest-cov
uv pip install matplotlib seaborn plotly

echo "🧪 테스트를 실행합니다..."
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✅ 모든 테스트가 통과했습니다!"
    echo ""
    echo "🎉 환경 설정이 완료되었습니다!"
    echo ""
    echo "사용 방법:"
    echo "1. 가상환경 활성화: source crypto-dlsa-env/bin/activate"
    echo "2. 환경 변수 설정: cp .env.example .env (그리고 API 키 설정)"
    echo "3. 설정 파일 확인: config.yaml"
    echo ""
    echo "개발을 시작하세요! 🚀"
else
    echo "❌ 테스트 실패. 환경 설정을 확인해주세요."
    exit 1
fi