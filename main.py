import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import threading
import time
import sys
import os
import requests
from dotenv import load_dotenv

# [중요] 기존 코드들을 불러오기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# .env 로드 (백엔드 주소 설정을 위해)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# 백엔드 API URL 설정 (내부망/외부망 환경에 맞춰 수정)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8081")

# -----------------------------------------------------------
# 사용자 정의 모듈 가져오기
# -----------------------------------------------------------
try:
    from ai_pipeline.trade.ai_advisor import AIAdvisor
    from ai_pipeline.trade.run_realtime_trade import AIAutoTrader
    from ai_pipeline.predict_main import run_pipeline_with_rebalancing
    from ai_pipeline.news_etl.TrendAnalyzer import TrendAnalyzer
except ImportError as e:
    print(f"[오류] 모듈을 찾을 수 없습니다: {e}")
    print("폴더 구조가 ai_pipeline 폴더 상위에 main.py가 있는지 확인해주세요.")


# -----------------------------------------------------------
# 봇 관리자 (BotManager) - 기존 코드 유지
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
        print(" >> [Start] 자동매매 봇 가동 시작")
        if not self.trader_instance:
            self.trader_instance = AIAutoTrader()
        while self.trader_running:
            try:
                self.trader_instance.run_cycle()
            except Exception as e:
                print(f" >> [Error] 매매 중 오류 발생: {e}")
            for _ in range(60):
                if not self.trader_running: break
                time.sleep(1)
        print(" >> [Stop] 자동매매 봇 종료")

    def _advisor_loop(self):
        print(" >> [Start] AI 어드바이저 가동 시작")
        if not self.advisor_instance:
            self.advisor_instance = AIAdvisor()
        while self.advisor_running:
            try:
                self.advisor_instance.generate_advice()
            except Exception as e:
                print(f" >> [Error] 조언 생성 중 오류: {e}")
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
        return "자동매매 종료 신호를 보냈습니다."

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

# 트렌드 분석기 인스턴스 생성
try:
    trend_analyzer = TrendAnalyzer()
except Exception as e:
    print(f"[Warning] Elasticsearch 연결 실패 가능성: {e}")
    trend_analyzer = None

app = FastAPI(
    title="AI Trading Controller",
    description="AI 자동매매 및 파이프라인 제어용 API",
    version="1.0.0"
)


# -----------------------------------------------------------
# [Helper] 백엔드로 트렌드 데이터 전송 로직
# -----------------------------------------------------------
def send_trends_to_backend_logic():
    """
    TrendAnalyzer를 실행하고 결과를 백엔드 API로 전송(Push)합니다.
    """
    print("[Trend] 뉴스 트렌드 분석 시작...")
    if not trend_analyzer:
        print("[Trend Error] TrendAnalyzer가 초기화되지 않았습니다.")
        return

    try:
        # 1. AI 분석 수행
        raw_result = trend_analyzer.get_period_trends()

        # 3. 백엔드 API 호출 (AIBubbleChartController)
        # URL: /api/app/ai/keywords/trending (POST)
        target_url = f"{BACKEND_API_URL}/ai/keywords/trending"

        print(f"[Trend] 백엔드로 데이터 전송 시도: {target_url}")
        response = requests.post(target_url, json=raw_result, timeout=10)

        if response.status_code == 200:
            print("[Trend] ✅ 백엔드 전송 성공!")
        else:
            print(f"[Trend] ❌ 백엔드 전송 실패: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"[Trend] ❌ 프로세스 실행 중 오류 발생: {e}")


# -----------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    """
    Lifespan handler: 서버 시작 시 트렌드 분석을 백엔드로 전송하는 작업을 비동기 백그라운드 스레드로 시작합니다.
    """
    print(" >> [Auto-Start] 서버 시작과 동시에 뉴스 트렌드 분석 및 전송 작업을 시작합니다.")
    threading.Thread(target=send_trends_to_backend_logic, daemon=True).start()
    try:
        yield
    finally:
        # 필요한 종료 정리 작업이 있다면 여기에 추가
        pass

app = FastAPI(
    title="AI Trading Controller",
    description="AI 자동매매 및 파이프라인 제어용 API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/status")
def get_status():
    return bot_manager.get_status()


@app.post("/bot/trade/{action}")
def control_trader(action: str):
    if action == "start":
        return {"msg": bot_manager.start_trader()}
    elif action == "stop":
        return {"msg": bot_manager.stop_trader()}
    else:
        raise HTTPException(status_code=400, detail="start/stop only")


@app.post("/bot/advisor/{action}")
def control_advisor(action: str):
    if action == "start":
        return {"msg": bot_manager.start_advisor()}
    elif action == "stop":
        return {"msg": bot_manager.stop_advisor()}
    else:
        raise HTTPException(status_code=400, detail="start/stop only")

@app.post("/pipeline/train")
async def run_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline_with_rebalancing)
    return {"msg": "AI 학습 및 예측 파이프라인이 백그라운드에서 시작되었습니다."}

# [수정됨] 백엔드 API 사용하도록 변경
@app.post("/pipeline/sync/trends")
async def trigger_trend_sync(background_tasks: BackgroundTasks):
    """
    [트리거] 뉴스 트렌드 분석을 수행하고, 결과를 백엔드 API로 전송합니다.
    (기존의 데이터를 직접 반환하던 방식에서 -> 백엔드로 쏘는 방식으로 변경)
    """
    background_tasks.add_task(send_trends_to_backend_logic)
    return {"msg": "뉴스 트렌드 분석 및 백엔드 동기화 작업이 시작되었습니다."}

# -----------------------------------------------------------
# 서버 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    print("AI 컨트롤러 서버를 시작합니다... http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)