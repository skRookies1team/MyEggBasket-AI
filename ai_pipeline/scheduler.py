import sys
import os
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# 프로젝트 루트 경로 잡기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT_DIR)

# 우리가 만든 ETL 실행 함수 가져오기
from ai_pipeline.news_source.news_etl_runner import run_finance_news_etl

def job_wrapper():
    """스케줄러가 실행할 작업 (로그 포함)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[⏰ 스케줄러 실행] 현재 시간: {now}")
    try:
        # 기존 ETL 함수 실행 (페이지 수는 조절 가능, 예: 3페이지)
        run_finance_news_etl() 
    except Exception as e:
        print(f"❌ 작업 실행 중 에러 발생: {e}")

def start_scheduler():
    # 백그라운드 스케줄러 생성
    sched = BackgroundScheduler(timezone='Asia/Seoul')

    # ====================================================
    # 🗓️ 스케줄 설정 (Cron 방식)
    # ====================================================

    # 1. 🔥 [장중 모드] 평일(월~금) 08:30 ~ 16:00 → 10분마다 실행
    # (장 시작 전후로 뉴스가 많이 나오므로 넉넉하게 잡음)
    sched.add_job(
        job_wrapper,
        'cron',
        day_of_week='mon-fri',
        hour='8-15',        # 8시부터 15시(오후 3시)대까지
        minute='*/10',      # 10분 간격 (0분, 10분, 20분...)
        id='market_mode'
    )

    # 2. 🌙 [퇴근/밤 모드] 매일 18:00 ~ 23:00 → 1시간마다 실행
    # (장 마감 후 공시나 글로벌 뉴스 체크용)
    sched.add_job(
        job_wrapper,
        'cron',
        hour='18-23',
        minute='0',         # 매 정각(0분)에 실행
        id='night_mode'
    )
    
    # 3. 💤 [심야/새벽] 00:00 ~ 07:00 → 실행 안 함 (또는 1번만 실행하고 싶으면 추가)
    # sched.add_job(job_wrapper, 'cron', hour='6', minute='0', id='morning_brief')

    # ====================================================

    sched.start()
    print("🚀 뉴스 수집 스케줄러가 시작되었습니다!")
    print("   - 장중 (08:00~16:00): 10분 간격")
    print("   - 야간 (18:00~23:00): 1시간 간격")
    print("   (종료하려면 Ctrl + C를 누르세요)")

    try:
        # 메인 스레드가 죽지 않도록 무한 루프 유지
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 스케줄러를 종료합니다.")
        sched.shutdown()

if __name__ == "__main__":
    start_scheduler()