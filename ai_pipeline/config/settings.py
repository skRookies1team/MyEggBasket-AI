import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# Elasticsearch 설정
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")

# 네이버 API 설정
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# NLP 모델 설정
FINBERT_MODEL = "snunlp/KR-FINBERT-SC"
MAX_CHUNK_SIZE = 500

# 환경변수 검증
if not NAVER_CLIENT_ID:
    raise ValueError("⚠️ NAVER_CLIENT_ID 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

if not NAVER_CLIENT_SECRET:
    raise ValueError("⚠️ NAVER_CLIENT_SECRET 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")