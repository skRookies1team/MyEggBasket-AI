import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import threading
import time
import sys
import os
from datetime import datetime

# [중요] 기존 코드들을 불러오기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------
# 사용자 정의 모듈 가져오기 (작성하신 파일들)
# -----------------------------------------------------------
try:
    # 1. 조언자 모드 (리밸런싱 제안만)
    from ai_pipeline.trade.ai_advisor import AIAdvisor
    # 2. 자동매매 모드 (실제 매수/매도)
    from ai_pipeline.trade.run_realtime_trade import AIAutoTrader
    # 3. 학습/예측 파이프라인
    from ai_pipeline.predict_main import run_pipeline_with_rebalancing
except ImportError as e:
    print(f"[오류] 모듈을 찾을 수 없습니다: {e}")
    print("폴더 구조가 ai_pipeline 폴더 상위에 main.py가 있는지 확인해주세요.")

# -----------------------------------------------------------
# 봇 관리자 (스레드로 백그라운드 실행 관리)
# -----------------------------------------------------------
class BotManager:
    def __init__(self):
        self.trader_running = False
        self.advisor_running = False
        self.trader_thread = None
        self.advisor_thread = None
        self.trader_instance = None
        self.advisor_instance = None

    def _trader_loop(self):
        """자동매매 봇이 1분마다 실행되는 무한 루프"""
        print(" >> [Start] 자동매매 봇 가동 시작")
        if not self.trader_instance:
            self.trader_instance = AIAutoTrader()
        
        while self.trader_running:
            try:
                self.trader_instance.run_cycle()
            except Exception as e:
                print(f" >> [Error] 매매 중 오류 발생: {e}")
            
            # 60초 대기 (중단 신호를 받으면 즉시 멈추도록 1초씩 끊어서 대기)
            for _ in range(60):
                if not self.trader_running: break
                time.sleep(1)
        print(" >> [Stop] 자동매매 봇 종료")

    def _advisor_loop(self):
        """조언자 봇이 5분마다 실행되는 무한 루프"""
        print(" >> [Start] AI 어드바이저 가동 시작")
        if not self.advisor_instance:
            self.advisor_instance = AIAdvisor()

        while self.advisor_running:
            try:
                self.advisor_instance.generate_advice()
            except Exception as e:
                print(f" >> [Error] 조언 생성 중 오류: {e}")

            # 300초(5분) 대기
            for _ in range(300):
                if not self.advisor_running: break
                time.sleep(1)
        print(" >> [Stop] AI 어드바이저 종료")

    def start_trader(self):
        if self.trader_running: return "이미 실행 중입니다."
        self.trader_running = True
        self.trader_thread = threading.Thread(target=self._trader_loop, daemon=True)
        self.trader_thread.start()
        return "자동매매가 시작되었습니다."

    def stop_trader(self):
        self.trader_running = False
        return "자동매매 종료 신호를 보냈습니다. (잠시 후 종료됨)"

    def start_advisor(self):
        if self.advisor_running: return "이미 실행 중입니다."
        self.advisor_running = True
        self.advisor_thread = threading.Thread(target=self._advisor_loop, daemon=True)
        self.advisor_thread.start()
        return "AI 어드바이저가 시작되었습니다."

    def stop_advisor(self):
        self.advisor_running = False
        return "AI 어드바이저 종료 신호를 보냈습니다."

    def get_status(self):
        return {
            "trader_status": "RUNNING" if self.trader_running else "STOPPED",
            "advisor_status": "RUNNING" if self.advisor_running else "STOPPED"
        }

bot_manager = BotManager()

# -----------------------------------------------------------
# FastAPI 앱 생성 (여기가 바로 AI Swagger를 만드는 부분!)
# -----------------------------------------------------------
app = FastAPI(
    title="AI Trading Controller",
    description="AI 자동매매 및 파이프라인 제어용 API",
    version="1.0.0"
)

# 1. 상태 확인 API
@app.get("/status")
def get_status():
    return bot_manager.get_status()

# 2. 자동매매 제어 API (팀원이 말한 그 Endpoint!)
@app.post("/bot/trade/{action}")
def control_trader(action: str):
    """
    action: 'start' (시작) 또는 'stop' (중지)
    """
    if action == "start":
        return {"msg": bot_manager.start_trader()}
    elif action == "stop":
        return {"msg": bot_manager.stop_trader()}
    else:
        raise HTTPException(status_code=400, detail="start 또는 stop만 입력 가능합니다.")

# 3. 어드바이저 제어 API
@app.post("/bot/advisor/{action}")
def control_advisor(action: str):
    """
    action: 'start' (시작) 또는 'stop' (중지)
    """
    if action == "start":
        return {"msg": bot_manager.start_advisor()}
    elif action == "stop":
        return {"msg": bot_manager.stop_advisor()}
    else:
        raise HTTPException(status_code=400, detail="start 또는 stop만 입력 가능합니다.")

# 4. 학습 파이프라인 실행 API
@app.post("/pipeline/train")
async def run_training(background_tasks: BackgroundTasks):
    """
    오래 걸리는 학습/예측 작업을 백그라운드에서 실행합니다.
    """
    background_tasks.add_task(run_pipeline_with_rebalancing)
    return {"msg": "AI 학습 및 예측 파이프라인이 백그라운드에서 시작되었습니다."}

# -----------------------------------------------------------
# 서버 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    # 포트 8000번에서 AI 서버 실행 (백엔드는 보통 8080이니 겹치지 않게)
    print("AI 컨트롤러 서버를 시작합니다... http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)