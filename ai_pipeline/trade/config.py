import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트 경로 찾기 및 설정
def _find_project_root(target_pkg='ai_pipeline'):
    p = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(p, target_pkg)):
            return p
        newp = os.path.dirname(p)
        if newp == p:
            return os.path.abspath(os.getcwd())
        p = newp

PROJECT_ROOT = _find_project_root('ai_pipeline')

# .env 로드
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 주요 경로 및 설정
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ES_HOST = "http://localhost:9200"

# 학습/테스트에 사용할 날짜 리스트
TARGET_DATES = [
    "20251120", "20251121", "20251124", "20251125",
    "20251126", "20251127", "20251202", "20251203",
    "20251204", "20251205", "20251208", "20251209"
]