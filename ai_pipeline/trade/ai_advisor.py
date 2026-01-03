import sys
import os
import time
import numpy as np
import csv
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------
# 1. 프로젝트 경로 및 환경 설정
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# [변경] 백엔드 내부 API 설정
BACKEND_API_URL = "http://localhost:8081/api/internal"
AI_SECRET_KEY = os.getenv("AI_SECRET_KEY", "myeggbasketsecretkey00")  # .env에 설정된 키 사용

ADVICE_LOG_PATH = os.path.join(current_dir, "ai_advice_history1.csv")

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------
try:
    from ai_pipeline.feature_store import OnlineFeatureStore
    from ai_pipeline.boosting_model.train import StackingEnsemble
    from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy
except ImportError:
    # 경로 문제 시 로컬 경로 기준으로 다시 시도
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../../")))
    from ai_pipeline.feature_store import OnlineFeatureStore
    from ai_pipeline.boosting_model.train import StackingEnsemble
    from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy

# -----------------------------------------------------------
# 3. 종목명 매핑
# -----------------------------------------------------------
STOCK_NAME_MAP = {
    "005930": "삼성전자", "000660": "SK하이닉스", "207940": "삼성바이오로직스",
    "005380": "현대차", "000270": "기아", "055550": "신한지주",
    "105560": "KB금융", "068270": "셀트리온", "015760": "한국전력",
    "028260": "삼성물산", "032830": "삼성생명", "012330": "현대모비스",
    "035420": "NAVER", "006400": "삼성SDI", "086790": "하나금융지주",
    "006405": "삼성SDI우", "000810": "삼성화재", "010140": "삼성중공업",
    "064350": "현대로템", "138040": "메리츠금융지주", "051910": "LG화학",
    "010130": "고려아연", "009540": "HD한국조선해양", "267260": "HD현대",
    "066570": "LG전자", "066575": "LG전자우", "033780": "KT&G",
    "003550": "LG", "003555": "LG우", "310200": "애니플러스",
    "034020": "두산에너빌리티", "012450": "한화에어로스페이스", "009830": "한화솔루션",
    "011070": "LG이노텍", "071050": "한국금융지주", "081660": "휠라홀딩스",
    "046890": "서울반도체", "323410": "카카오뱅크", "017670": "SK텔레콤",
    "010620": "현대미포조선", "047050": "포스코인터내셔널", "009155": "삼성전기우",
    "275630": "에이치시티", "009835": "한화솔루션우", "001440": "대한전선",
    "138930": "BNK금융지주", "175330": "JB금융지주", "051900": "LG생활건강",
    "005490": "POSCO홀딩스", "034220": "LG디스플레이"
}


def get_stock_name(code):
    return STOCK_NAME_MAP.get(code, code)


# -----------------------------------------------------------
# 3.5. PortfolioRebalancer (기존 로직 유지)
# -----------------------------------------------------------
class PortfolioRebalancer:
    def __init__(self, risk_aversion='neutral'):
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings_detail, ai_scores_df, total_budget, last_sell_times, last_buy_times):
        if ai_scores_df is None or ai_scores_df.empty:
            return pd.DataFrame()

        if 'ai_score' not in ai_scores_df.columns:
            return pd.DataFrame()

        ai_scores_df = ai_scores_df.copy()
        ai_scores_df['code'] = ai_scores_df['code'].astype(str).str.strip().str.zfill(6)

        merged_df = ai_scores_df.copy()
        held_codes = set(current_holdings_detail.keys())

        # 파라미터 설정 (기존 유지)
        BUY_SCORE_THRESHOLD = 86
        SELL_SCORE_THRESHOLD = 50
        PROFIT_TAKE_RATE = 10.5
        STOP_LOSS_RATE = -10.2
        THRESHOLD_RATIO = 0.01
        MAX_INDIVIDUAL_WEIGHT = 0.50

        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= BUY_SCORE_THRESHOLD)
        cond_hold = merged_df['code'].isin(held_codes)
        candidates = merged_df[cond_new_buy | cond_hold].copy()

        if candidates.empty:
            return pd.DataFrame()

        candidates['calc_score'] = candidates['ai_score'].apply(lambda x: x if x >= SELL_SCORE_THRESHOLD else 0)
        candidates['weight_score'] = np.power(candidates['calc_score'], 2)
        total_weight_score = candidates['weight_score'].sum()

        if total_weight_score > 0:
            raw_ratios = candidates['weight_score'] / total_weight_score
            candidates['target_ratio'] = raw_ratios.apply(lambda x: min(x, MAX_INDIVIDUAL_WEIGHT))
        else:
            candidates['target_ratio'] = 0

        rebalancing_plan = []
        threshold_amt = total_budget * THRESHOLD_RATIO

        for _, row in candidates.iterrows():
            code = row['code']
            holding = current_holdings_detail.get(code, {'amt': 0, 'avg_price': 0, 'current_price': 0})

            current_amt = holding['amt']
            avg_price = holding['avg_price']
            current_price = row.get('current_price', holding.get('current_price', 0))

            target_amt = int(total_budget * row['target_ratio'])
            diff = target_amt - current_amt
            ai_score = row['ai_score']

            profit_rate = 0.0
            if avg_price > 0 and current_price > 0:
                profit_rate = ((current_price - avg_price) / avg_price * 100)

            # 1차 판단
            if diff > threshold_amt:
                base_action = '매수'
            elif diff < -threshold_amt:
                base_action = '비중축소'
            else:
                base_action = '유지'

            final_action = '유지'
            reason = f"목표비중 {row['target_ratio'] * 100:.1f}%"

            # 상세 매매 로직
            if target_amt == 0 and current_amt > 0:
                final_action = '전량매도'
                reason = f"AI 점수 미달({ai_score}점) / 목표비중 0%"
            elif profit_rate <= STOP_LOSS_RATE and current_amt > 0:
                final_action = '전량매도' if ai_score < 40 else '비중축소'
                reason = f"📉 손절매({profit_rate:.2f}%)"
            elif profit_rate >= PROFIT_TAKE_RATE and current_amt > 0:
                if ai_score < 90:
                    final_action = '비중축소'
                    if target_amt == 0: final_action = '전량매도'
                    reason = f"💰 익절({profit_rate:.2f}%)"
                else:
                    if base_action == '매수':
                        final_action = '매수'
                        reason = f"🚀 급등({profit_rate:.2f}%) + AI강세"
                    else:
                        final_action = '유지'
                        reason = f"💰 익절권이나 상승세 유지"
            elif ai_score < 20 and current_amt > 0:
                final_action = '전량매도'
                reason = f"AI 점수 위험수준({ai_score}점)"
            else:
                final_action = base_action
                if final_action == '비중축소':
                    if target_amt == 0:
                        final_action = '전량매도'
                        reason = "목표 비중 0%"
                    else:
                        reason = "비중 축소 (리밸런싱)"
                elif final_action == '매수':
                    reason = "추가/신규 매수"

            if final_action != '유지':
                rebalancing_plan.append({
                    "code": code,
                    "name": get_stock_name(code),
                    "ai_score": ai_score,
                    "current_amt": current_amt,
                    "target_amt": target_amt,
                    "target_ratio": row['target_ratio'],
                    "diff": diff,
                    "action": final_action,
                    "reason": reason
                })

        return pd.DataFrame(rebalancing_plan)


# -----------------------------------------------------------
# 4. AI Advisor (수정됨: Multi-User Support)
# -----------------------------------------------------------
class AIAdvisor:
    def __init__(self):
        print("\n" + "=" * 60)
        print("[AI Advisor] 투자 조언 및 리포트 생성기 (Internal API Mode)")
        print(f"    - Target API: {BACKEND_API_URL}")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        self.model = StackingEnsemble()
        self.rebalancer = PortfolioRebalancer()
        self.headers = {"X-AI-SECRET": AI_SECRET_KEY}  # [변경] Secret Key 헤더 사용

        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        try:
            self.model.load_model(model_path)
            print(" [Init] AI 모델 로드 완료")
        except Exception as e:
            print(f" [Error] 모델 로드 실패: {e}")

        self.target_codes = list(STOCK_NAME_MAP.keys())
        self.init_csv_log()

    def init_csv_log(self):
        if not os.path.exists(ADVICE_LOG_PATH):
            with open(ADVICE_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    "timestamp", "user_id", "code", "name", "ai_score",
                    "current_amt", "target_amt", "diff", "action", "reason"
                ])

    # [추가] 활성 사용자 목록 조회
    def get_active_users(self):
        try:
            url = f"{BACKEND_API_URL}/users"
            res = requests.get(url, headers=self.headers, timeout=20)
            if res.status_code == 200:
                return res.json()  # [1, 2, 5, ...]
            else:
                print(f"[Advisor] 유저 목록 조회 실패: {res.status_code}")
                return []
        except Exception as e:
            print(f"[Advisor] 유저 조회 중 에러: {e}")
            return []

    # [변경] 사용자별 포트폴리오(잔고) 조회
    def get_user_portfolio(self, user_id):
        try:
            url = f"{BACKEND_API_URL}/balance/{user_id}"
            res = requests.get(url, headers=self.headers, timeout=20)
            if res.status_code == 200:
                data = res.json()
                return data
            else:
                # 잔고가 없거나 조회 실패 시 None
                return None
        except Exception as e:
            print(f"[Advisor] 잔고 조회 에러 (User {user_id}): {e}")
            return None

    # [변경] 내부 API로 조언 전송
    def send_advice_to_server(self, user_id, row):
        """
        AI 리밸런싱 결과 1건을 백엔드 내부 API로 전송
        """
        url = f"{BACKEND_API_URL}/ai/recommendation"

        action_map = {
            "매수": "BUY",
            "추가 매수": "BUY",
            "강력매수": "BUY",
            "매수고려": "BUY",
            "비중축소": "SELL",
            "전량매도": "SELL",
            "손절매": "SELL",
            "익절": "SELL",
            "유지": "HOLD"
        }

        action_enum = action_map.get(row["action"], "HOLD")

        # 내부용 API Payload (InternalRecommendationRequest 구조에 맞춤)
        payload = {
            "userId": user_id,
            "stockCode": str(row["code"]),
            "type": action_enum,  # BUY, SELL, HOLD
            "reason": f"{row['reason']} (변동: {int(row['diff']):,}원)",  # 금액 정보 보존을 위해 reason에 추가
            "score": float(row["ai_score"])
        }

        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=3)
            if res.status_code == 200:
                print(f"   ✅ [User {user_id}] 서버 저장 완료: {row['name']} ({action_enum})")

                # 로컬 CSV 로그에도 저장
                with open(ADVICE_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        user_id, row["code"], row["name"], row["ai_score"],
                        row["current_amt"], row["target_amt"], row["diff"],
                        row["action"], row["reason"]
                    ])
            else:
                print(f"   ❌ [User {user_id}] 서버 저장 실패: {res.text}")
        except Exception as e:
            print(f"   ⚠️ [User {user_id}] 전송 중 에러: {e}")

    def analyze_stock(self, code):
        features = self.store.get_realtime_features(code)
        if features is None or features.empty:
            return None

        probs = self.model.predict_proba(features)
        return {
            "code": code,
            "name": get_stock_name(code),
            "ai_score": round(probs[0, 1] * 100, 2),
            "current_price": int(features["close"].values[0])
        }

    def generate_advice(self):
        print(f"\n[Cycle] {datetime.now().strftime('%H:%M:%S')} AI 조언 생성 프로세스 시작")

        # 1. 활성 사용자 조회
        users = self.get_active_users()
        print(f"[System] 대상 사용자: {len(users)}명")
        if not users:
            print("[System] 활성 사용자가 없어 대기합니다.")
            return

        # 2. [최적화] 전체 종목 AI 점수 산출 (유저별 반복 방지)
        print("[AI] 시장 전체 종목 분석 중...")
        ai_results = []
        count = 0
        for code in self.target_codes:
            data = self.analyze_stock(code)
            if data:
                ai_results.append(data)
                count += 1

        if not ai_results:
            print("[Warning] AI 분석 결과가 없습니다. (장 마감 or 데이터 부족)")
            return

        print(f"[AI] 총 {len(ai_results)}개 종목 분석 완료. 사용자별 포트폴리오 진단 시작.")
        ai_df_global = pd.DataFrame(ai_results)

        # 3. 사용자별 순차 처리
        for user_id in users:
            try:
                # 3-1. 사용자 잔고 조회
                portfolio = self.get_user_portfolio(user_id)
                if not portfolio:
                    continue

                # 3-2. 데이터 파싱 (KisBalanceDTO 구조 -> my_holdings)
                output1 = portfolio.get('output1') or []  # 보유종목 리스트
                output2 = portfolio.get('output2') or []  # 예수금 리스트

                # 1) 숫자 변환 헬퍼 함수
                def _parse(val):
                    if val is None: return 0
                    if isinstance(val, (int, float)): return int(val)
                    if isinstance(val, str):
                        val = val.replace(',', '').strip()
                        if val == '': return 0
                        return int(float(val))
                    return 0

                # 2) 예수금 파싱
                d2_cash = 0
                total_cash_api = 0
                if output2:
                    balance_info = output2[0]
                    # prvs_rcdl_excc_amt: 가수금제외 D+2 예수금
                    d2_cash = _parse(balance_info.get('prvs_rcdl_excc_amt', 0))
                    total_cash_api = _parse(balance_info.get('dnca_tot_amt', 0))
                
                # 예수금이 있는 쪽을 선택 (D+2 우선)
                cash = d2_cash if d2_cash > 0 else total_cash_api

                # 3) 보유종목 파싱 (stockCode -> pdno, quantity -> hldg_qty)
                my_holdings = {}
                for h in output1:
                    qty = _parse(h.get("hldg_qty", 0))  # 보유수량
                    if qty > 0:
                        code = h.get("pdno")            # 종목코드
                        avg_price = float(h.get("pchs_avg_pric", 0)) # 매입평균가
                        
                        my_holdings[code] = {
                            "qty": qty,
                            "avg_price": avg_price,
                            "current_price": 0, 
                            "amt": 0
                        }
                        
                # 3-3. 현재가 업데이트 (AI 분석 결과 활용)
                ai_df_user = ai_df_global.copy()
                for idx, row in ai_df_user.iterrows():
                    code = row['code']
                    if code in my_holdings:
                        price = row['current_price']
                        my_holdings[code]['current_price'] = price
                        my_holdings[code]['amt'] = price * my_holdings[code]['qty']

                # 3-4. 리밸런싱 전략 실행
                stock_assets = sum(h["amt"] for h in my_holdings.values())
                total_asset = cash + stock_assets
                
                print(f"   [Check] User {user_id} 자산: {total_asset:,}원 (현금: {cash:,}원 / 주식: {stock_assets:,}원)")
                
                # 자산이 너무 적으면 패스 (예: 1만원 미만)
                if total_asset < 10000:
                    continue

                plan_df = self.rebalancer.run_ai_rebalancing(
                    my_holdings, ai_df_user, total_asset, {}, {}
                )

                if not plan_df.empty:
                    print(f"  >> [User {user_id}] 조언 {len(plan_df)}건 생성됨")
                    for _, row in plan_df.iterrows():
                        self.send_advice_to_server(user_id, row)

            except Exception as e:
                print(f"  ⚠️ [User {user_id}] 처리 중 오류: {e}")

        print("[System] 전체 사용자 처리 완료. 5분 대기...\n")


# -----------------------------------------------------------
# 5. 자동 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    advisor = AIAdvisor()
    while True:
        advisor.generate_advice()
        time.sleep(300)