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



BACKEND_API_URL = os.getenv("BACKEND_API_URL")
TEST_EMAIL = os.getenv("TEST_EMAIL")
TEST_PASSWORD = os.getenv("TEST_PASSWORD")

ADVICE_LOG_PATH = os.path.join(current_dir, "ai_advice_history.csv")

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------

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
# 3.5. PortfolioRebalancer
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

        BUY_SCORE_THRESHOLD = 86
        SELL_SCORE_THRESHOLD = 50
        PROFIT_TAKE_RATE = 10.5
        STOP_LOSS_RATE = -10.2
        THRESHOLD_RATIO = 0.01
        MAX_INDIVIDUAL_WEIGHT = 0.20

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
            # [핵심] 20% Cap 적용
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

            # 1차 판단 (금액 차이 기반)
            if diff > threshold_amt:
                base_action = '매수'
            elif diff < -threshold_amt:
                base_action = '비중축소'
            else:
                base_action = '유지'

            # 2차 판단 (상세 로직 적용 - run_realtime_trade.py 로직 참조)
            final_action = '유지'
            reason = f"목표비중 {row['target_ratio']*100:.1f}%"

            # [CASE 0] 목표 금액이 0원인데 보유 중이면 -> 전량 매도
            if target_amt == 0 and current_amt > 0:
                final_action = '전량매도'
                reason = f"AI 점수 미달({ai_score}점) / 목표비중 0%"

            # [CASE 1] 손절매
            elif profit_rate <= STOP_LOSS_RATE and current_amt > 0:
                final_action = '전량매도' if ai_score < 40 else '비중축소'
                reason = f"📉 손절매({profit_rate:.2f}%)"

            # [CASE 2] 익절
            elif profit_rate >= PROFIT_TAKE_RATE and current_amt > 0:
                if ai_score < 90:
                    final_action = '비중축소'
                    # 익절인데 목표금액이 0이면 전량매도
                    if target_amt == 0: final_action = '전량매도'
                    reason = f"💰 익절({profit_rate:.2f}%)"
                else:
                    # 점수가 90점 이상으로 매우 높으면 익절 구간이어도 홀딩하거나 추매
                    if base_action == '매수':
                        final_action = '매수'
                        reason = f"🚀 급등({profit_rate:.2f}%) + AI강세"
                    else:
                        final_action = '유지'
                        reason = f"💰 익절권이나 상승세 유지"

            # [CASE 3] AI 점수 미달 (20점 미만) -> 전량 매도
            elif ai_score < 20 and current_amt > 0:
                final_action = '전량매도'
                reason = f"AI 점수 위험수준({ai_score}점)"

            # [CASE 4] 일반 리밸런싱 (위의 특수 케이스가 아닐 때)
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

            # 결과 저장
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
# 4. AI Advisor
# -----------------------------------------------------------
class AIAdvisor:
    def __init__(self):
        self.store = OnlineFeatureStore()
        self.model = StackingEnsemble()
        self.rebalancer = PortfolioRebalancer()

        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        self.model.load_model(model_path)

        self.target_codes = list(STOCK_NAME_MAP.keys())
        self.auth_token = None
        self.init_csv_log()

    def init_csv_log(self):
        if not os.path.exists(ADVICE_LOG_PATH):
            with open(ADVICE_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([
                    "timestamp","code","name","ai_score",
                    "current_amt","target_amt","diff","action","reason"
                ])

    def login(self):
        res = requests.post(
            f"{BACKEND_API_URL}/auth/login",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
        )
        if res.status_code == 200:
            self.auth_token = res.json().get("accessToken")
            return True
        return False

    def get_headers(self):
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    def get_balance(self):
        res = requests.get(
            f"{BACKEND_API_URL}/kis/trade/balance",
            headers=self.get_headers(),
            params={"virtual": "true"}
        )
        return res.json() if res.status_code == 200 else None

    def get_my_portfolio_id(self):
        """
        로그인한 사용자의 포트폴리오 목록 조회 후
        첫 번째 portfolioId 반환
        """
        try:
            url = f"{BACKEND_API_URL}/portfolios"
            res = requests.get(url, headers=self.get_headers(), timeout=5)

            if res.status_code != 200:
                print(f"[Portfolio] 조회 실패: {res.status_code}")
                return None

            portfolios = res.json()
            if not portfolios:
                print("[Portfolio] 포트폴리오 없음")
                return None

            portfolio_id = portfolios[0].get("portfolioId")
            print(f"[Portfolio] 사용 portfolioId = {portfolio_id}")
            return portfolio_id

        except Exception as e:
            print(f"[Portfolio] 조회 오류: {e}")
            return None

    def send_advice_to_server(self, row):
        """
        AI 리밸런싱 결과 1건을 백엔드에 저장
        """
        url = f"{BACKEND_API_URL}/ai-recommendations"

        action_map = {
            "매수": "BUY",
            "추가 매수": "BUY",
            "강력매수": "BUY",
            "매수고려": "BUY",
            "비중축소": "SELL",
            "전량매도": "SELL",
            "손절매": "SELL",
            "익절": "SELL"
        }

        action_enum = action_map.get(row["action"], "HOLD")
        portfolio_id = self.get_my_portfolio_id()

        if not portfolio_id:
            print(" [Server] portfolioId 없음 → 전송 스킵")
            return

        try:
            target_ratio_val = row.get("target_ratio", 0)
            target_percentage = target_ratio_val * 100  # 0.15 -> 15.0

            payload = {
                "portfolioId": int(portfolio_id),
                "stockCode": str(row["code"]),
                "aiScore": float(row["ai_score"]),
                "actionType": action_enum,  # "BUY" or "SELL"
                "currentHolding": int(row["current_amt"]),
                "targetHolding": int(row["target_amt"]),
                "targetHoldingPercentage": float(round(target_percentage, 2)),
                "adjustmentAmount": int(row["diff"]),
                "reasonSummary": str(row["reason"]),
                "riskWarning": "AI Risk Model Analysis"  # 고정값 혹은 row에서 추출
            }

            # 4. 서버 전송
            res = requests.post(
                url,
                json=payload,
                headers=self.get_headers(),
                timeout=5
            )

            if res.status_code in (200, 201):
                print(f"   ✅ 서버 저장 완료 [{row['name']}({row['code']})]: {action_enum}")
            else:
                print(f"   ❌ 서버 저장 실패 {res.status_code}: {res.text}")

        except Exception as e:
            print(f"   ⚠️ 데이터 변환 또는 전송 중 오류: {e}")
            # 디버깅을 위해 payload 출력 (필요시 주석 해제)
            # print("Payload:", payload)

    def analyze_stock(self, code):
        features = self.store.get_realtime_features(code)
        if features is None or features.empty:
            return None

        probs = self.model.predict_proba(features)
        return {
            "code": code,
            "name": get_stock_name(code),
            "ai_score": round(probs[0,1] * 100, 2),
            "current_price": int(features["close"].values[0])
        }

    def generate_advice(self):
        print("\n[System] --- AI 조언 생성 프로세스 시작 ---")

        # 1. 로그인 체크
        if not self.auth_token:
            print("[Auth] 토큰 없음, 로그인 시도 중...")
            if not self.login():
                print("[Error] 로그인 실패! 환경변수나 백엔드 상태를 확인하세요.")
                return

            else:
                print("[Auth] 로그인 성공.")

        # 2. 잔고 조회 체크
        balance = self.get_balance()
        if not isinstance(balance, dict):
            print("[Error] 잔고 조회 실패 (응답이 dict가 아님). API 상태를 확인하세요.")
            return

        # 3. 보유 종목 및 현금 파싱
        holdings = balance.get("holdings") or []
        summary = balance.get("summary", {})
        cash_raw = summary.get("totalCashAmount") or summary.get("d2CashAmount") or 0
        cash = int(cash_raw)

        print(f"[Account] 예수금: {cash:,}원 / 보유종목 수: {len(holdings)}개")

        my_holdings = {}
        for h in holdings:
            if int(h.get("quantity", 0)) > 0:
                my_holdings[h["stockCode"]] = {
                    "qty": int(h["quantity"]),
                    "avg_price": float(h.get("avgPrice", 0)),
                    "current_price": 0,
                    "amt": 0
                }

        # 4. AI 분석 수행
        print("[AI] 종목 분석 및 점수 산출 중...")
        ai_results = []
        target_list = list(set(self.target_codes) | set(my_holdings.keys()))
     
        # 진행상황을 보기 위해 tqdm이 없다면 카운터 출력
        count = 0
        for code in target_list:
            data = self.analyze_stock(code)
            if data:
                if code in my_holdings:
                    my_holdings[code]["current_price"] = data["current_price"]
                    my_holdings[code]["amt"] = data["current_price"] * my_holdings[code]["qty"]

                ai_results.append(data)
                count += 1
                
                # 너무 많으니 100개 단위로만 로그 찍기
                if count % 100 == 0:
                    print(f"  ... {count}개 종목 분석 완료")

        if not ai_results:
            print("[Warning] AI 분석 결과가 하나도 없습니다. 장 마감 시간이거나 데이터 문제일 수 있습니다.")
            return

        print(f"[AI] 총 {len(ai_results)}개 종목 분석 완료.")

        # 5. 리밸런싱 계산
        ai_df = pd.DataFrame(ai_results)
        total_asset = cash + sum(h["amt"] for h in my_holdings.values())
        print(f"[Strategy] 총 자산(현금+주식): {total_asset:,}원 기준으로 리밸런싱 계산 시작")

        plan_df = self.rebalancer.run_ai_rebalancing(
            my_holdings, ai_df, total_asset, {}, {}
        )

        if plan_df.empty:
            print("[Strategy] 리밸런싱 대상 종목이 없습니다. (조건을 만족하는 매수/매도 신호 없음)")
            return

        # 6. 서버 전송
        print(f"\n[Action] {len(plan_df)}건의 매매 신호 발생! 상세 내역 출력 및 서버 전송 시작...")
        
        for _, row in plan_df.iterrows():
            # 요청하신 포맷대로 출력
            print("-" * 50)
            print(f"📌 종목명(코드): {row['name']}({row['code']})")
            print(f"   • AI 점수   : {float(row['ai_score']):.2f}점")
            print(f"   • 현재 보유 : {int(row['current_amt']):,}원")
            print(f"   • 목표 보유 : {int(row['target_amt']):,}원")
            print(f"   • 조절 금액 : {int(row['diff']):,}원")
            print(f"   • 판단      : {row['action']}")
            print(f"   • 이유      : {row['reason']}")
            print("-" * 50)

            self.send_advice_to_server(row)        

        print("[System] 프로세스 완료. 5분 대기...\n")

# -----------------------------------------------------------
# 5. 자동 실행
# -----------------------------------------------------------

if __name__ == "__main__":
    advisor = AIAdvisor()
    while True:
        advisor.generate_advice()
        time.sleep(300)