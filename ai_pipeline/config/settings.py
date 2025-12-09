"""
ai_pipeline.config.settings
프로젝트 전체에서 사용하는 공통 환경 설정
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# 1. 프로젝트 루트 경로 정의
# ============================================================
# 현재 파일: ai_pipeline/config/settings.py
# → parents[2] = MyEggBasket-AI 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# .env 파일 경로
ENV_PATH = PROJECT_ROOT / ".env"

# ============================================================
# 2. .env 로드
# ============================================================
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, verbose=True)
  
# ============================================================
# 3. Elasticsearch 설정
# ============================================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")

# ============================================================
# 4. 네이버 API 설정
# ============================================================
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# ============================================================
# 5. MongoDB 설정
# ============================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stock_data")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "reports")

# ============================================================
# 6. NLP / AI 모델 설정
# ============================================================
MAX_CHUNK_SIZE = 500
FINBERT_MODEL = "snunlp/KR-FinBert-SC"

# ============================================================
# 7. 한국투자증권(KIS) 설정
# ============================================================
KIS_IS_MOCK = os.getenv("KIS_IS_MOCK", "true").lower() == "true"

if KIS_IS_MOCK:
    KIS_APP_KEY = os.getenv("KIS_MOCK_APP_KEY")
    KIS_APP_SECRET = os.getenv("KIS_MOCK_APP_SECRET")
    KIS_ACCOUNT_NO = os.getenv("KIS_MOCK_ACCOUNT_NO")
    KIS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
  
else:
    KIS_APP_KEY = os.getenv("KIS_APP_KEY")
    KIS_APP_SECRET = os.getenv("KIS_APP_SECRET")
    KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")
    KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

