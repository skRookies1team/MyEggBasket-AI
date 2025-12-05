import sys
import os
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# 1. 프로젝트 루트 경로 잡기 (모듈 import를 위해 필수)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 2. [핵심] 우리가 만든 '지휘자(전체 파이프라인)' 가져오기
from ai_pipeline.pipeline_main import run_full_pipeline

def job_wrapper():
    """
    스케줄러가 정해진 시간마다 실행할 작업 함수
    단순 크롤링뿐만 아니라 그래프 구축 -> GCN 실행까지 한 번에 처리합니다.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[ 스케줄러 실행] 현재 시간: {now}")
    
    try:
        # 전체 파이프라인 실행!
        run_full_pipeline()
    except Exception as e:
        print(f" 스케줄러 작업 중 에러 발생: {e}")

def start_scheduler():
    # 백그라운드 스케줄러 생성 (서울 시간 기준)
    sched = BackgroundScheduler(timezone='Asia/Seoul')

    # ====================================================
    #  스케줄 설정 (Cron 방식)
    # ====================================================

    # 0. [테스트용] 실행하자마자 5초 뒤에 한 번 돌려보기 (잘 되는지 확인용)
    # 나중에 필요 없으면 주석 처리하세요.
    # sched.add_job(job_wrapper, 'date', run_date=datetime.now(), id='test_run')

    # 1.  [장중 모드] 평일(월~금) 08:30 ~ 16:00 → 10분마다 실행
    sched.add_job(
        job_wrapper,
        'cron',
        day_of_week='mon-sat',
        hour='8-15',        # 8시~15시
        minute='*/10',      # 10분 간격
        id='market_mode'
    )

    # 2. 🌙 [퇴근/밤 모드] 매일 18:00 ~ 23:00 → 1시간마다 실행
    sched.add_job(
        job_wrapper,
        'cron',
        hour='18-23',
        minute='0',         # 정각마다
        id='night_mode'
    )

    # ====================================================

    sched.start()
    print("\n" + "="*50)
    print(" 뉴스 수집 스케줄러가 시작되었습니다!")
    print("   - 장중 (08:00~15:00): 10분 간격")
    print("   - 야간 (18:00~23:00): 1시간 간격")
    print("   (종료하려면 Ctrl + C를 누르세요)")
    print("="*50)

    try:
        # 메인 프로그램이 종료되지 않도록 무한 대기
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n 스케줄러를 종료합니다.")
        sched.shutdown()

if __name__ == "__main__":
    start_scheduler()