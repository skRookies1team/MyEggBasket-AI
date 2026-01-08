import sys
import os
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 전체 파이프라인(ETL -> 학습 -> 예측 -> 자문) 하나만 가져옵니다.
from ai_pipeline.pipeline_main import run_full_pipeline

def job_wrapper():
    """
    10분마다 실행되는 메인 작업
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[ 스케줄러 실행] 현재 시간: {now}")
    try:
        # 뉴스수집 -> 모델학습 -> 예측 -> 자문생성까지 한 번에 실행!
        run_full_pipeline()
    except Exception as e:
        print(f" 스케줄러 작업 중 에러 발생: {e}")

def start_scheduler():
    sched = BackgroundScheduler(timezone='Asia/Seoul')

    # [테스트용] 시작 5초 후 1회 실행
    sched.add_job(job_wrapper, 'date', run_date=datetime.now(), id='test_run')

    # [장중 모드] 평일 08:30 ~ 16:00 (10분 간격)
    sched.add_job(
        job_wrapper,
        'cron',
        day_of_week='mon-fri',
        hour='8-15',
        minute='*/10',
        id='market_mode'
    )

    sched.start()
    print(" [System] 자동화 스케줄러 시작됨 (10분 주기)")
    
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()

if __name__ == "__main__":
    start_scheduler()