import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 확인용 로그
#print("🔍 settings.py PROJECT_ROOT =", PROJECT_ROOT)

# .env 로드
env_path = PROJECT_ROOT / ".env"
#print("🔍 .env path =", env_path, ", exists =", env_path.exists())

load_dotenv(dotenv_path=env_path, verbose=True)

# 환경변수 확인
#print(" Loaded NAVER_CLIENT_ID =", os.getenv("NAVER_CLIENT_ID"))
#print(" Loaded NAVER_CLIENT_SECRET =", os.getenv("NAVER_CLIENT_SECRET"))

# Elasticsearch 설정
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")

# 네이버 API 설정
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID:
    # (선택사항) 환경변수 누락 시 경고 출력
    print("⚠️ [경고] NAVER_CLIENT_ID가 설정되지 않았습니다.")

# ==========================================
# 2. 데이터베이스 / 엘라스틱서치 설정
# ==========================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")

# ==========================================
# 3. MongoDB 설정 (리포트 저장용)
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stock_data")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "reports")

# ==========================================
# 4. NLP 및 AI 모델 설정
# ==========================================
MAX_CHUNK_SIZE = 500
# 한국어 금융 특화 모델 (감성 분석을 나중에 다시 쓸 경우를 위해 설정 유지)
FINBERT_MODEL = "snunlp/KR-FinBert-SC"

# ==========================================
# 5. 한국투자증권 (KIS) 설정
# ==========================================
# 모의투자 모드 여부 (True: 모의투자, False: 실전투자)
KIS_IS_MOCK = True 

if KIS_IS_MOCK:
    # 모의투자용 정보 로드
    KIS_APP_KEY = os.getenv("KIS_MOCK_APP_KEY")
    KIS_APP_SECRET = os.getenv("KIS_MOCK_APP_SECRET")
    KIS_ACCOUNT_NO = os.getenv("KIS_MOCK_ACCOUNT_NO")
    KIS_BASE_URL = "https://openapivts.koreainvestment.com:29443" # 모의투자 서버
    print("📢 [설정] 현재 '모의투자' 모드로 동작합니다.")
else:
    # 실전투자용 정보 로드
    KIS_APP_KEY = os.getenv("KIS_APP_KEY")
    KIS_APP_SECRET = os.getenv("KIS_APP_SECRET")
    KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")
    KIS_BASE_URL = "https://openapi.koreainvestment.com:9443" # 실전투자 서버
    print("🚨 [설정] 현재 '실전투자' 모드로 동작합니다. (실제 자금 사용)")

# ==========================================
# 6. 키움증권 설정
# ==========================================
KIWOOM_ACCOUNT_NO = os.getenv("KIWOOM_ACCOUNT_NO")