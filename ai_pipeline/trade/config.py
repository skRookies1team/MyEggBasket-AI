import os
from dotenv import load_dotenv
from ai_pipeline.config.settings import ES_HOST

# 프로젝트 루트 경로 찾기
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

# 주요 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# [핵심 기능] data 폴더 스캔하여 날짜 리스트 자동 생성
def get_auto_target_dates(data_dir):
    if not os.path.exists(data_dir):
        return []
    
    dates = []
    # data 폴더의 모든 파일 확인
    for f in os.listdir(data_dir):
        # 파일명이 8자리 숫자이고 .csv로 끝나는지 확인 (예: 20251209.csv)
        if f.endswith(".csv") and f[:8].isdigit() and len(f) == 12:
            dates.append(f[:8])
            
    # 날짜순 정렬
    return sorted(dates)

# 자동으로 날짜 리스트 채우기
TARGET_DATES = get_auto_target_dates(DATA_DIR)
