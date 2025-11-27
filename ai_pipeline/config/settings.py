import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 절대경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 확인용 로그
print("🔍 settings.py PROJECT_ROOT =", PROJECT_ROOT)

# .env 로드
env_path = PROJECT_ROOT / ".env"
print("🔍 .env path =", env_path, ", exists =", env_path.exists())

load_dotenv(dotenv_path=env_path, verbose=True)

# 환경변수 확인
print("🔍 Loaded NAVER_CLIENT_ID =", os.getenv("NAVER_CLIENT_ID"))
print("🔍 Loaded NAVER_CLIENT_SECRET =", os.getenv("NAVER_CLIENT_SECRET"))

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
    raise ValueError("⚠ NAVER_CLIENT_ID 환경변수가 설정되지 않았습니다.")

if not NAVER_CLIENT_SECRET:
    raise ValueError("⚠ NAVER_CLIENT_SECRET 환경변수가 설정되지 않았습니다.")
