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
# 3.5. PortfolioRebalancer (기준 완화 버전)
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

        # -------------------------------------------------------
        # [설정] 전략 파라미터 (테스트를 위해 매수 기준 완화)
        # -------------------------------------------------------
        # 원래 기준: BUY_SCORE_THRESHOLD = 86
        # 수정 기준: 30 (매수 의견이 나오는지 확인용)
        BUY_SCORE_THRESHOLD = 86  
        
        SELL_SCORE_THRESHOLD = 50
        PROFIT_TAKE_RATE = 10.5
        STOP_LOSS_RATE = -10.2
        THRESHOLD_RATIO = 0.01 # 1%만 차이나도 매매 추천 (민감도 증가)
        SELL_COOLDOWN_MINUTES = 0 # 쿨타임 해제

        # 필터링
        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= BUY_SCORE_THRESHOLD)
        cond_hold = (merged_df['code'].isin(held_codes))
        candidates = merged_df[cond_new_buy | cond_hold].copy()

        if candidates.empty:
            return pd.DataFrame()

        # 비중 산출
        candidates['calc_score'] = candidates['ai_score'].apply(lambda x: x if x >= SELL_SCORE_THRESHOLD else 0)
        candidates['weight_score'] = np.power(candidates['calc_score'], 2)
        total_weight_score = candidates['weight_score'].sum()

        if total_weight_score > 0:
            candidates['target_ratio'] = candidates['weight_score'] / total_weight_score
        else:
            candidates['target_ratio'] = 0

        # 주문 생성
        rebalancing_plan = []
        threshold_amt = total_budget * THRESHOLD_RATIO

        for _, row in candidates.iterrows():
            code = row['code']
            ai_score = row['ai_score']
            target_ratio = row['target_ratio']

            holding_info = current_holdings_detail.get(code, {'qty': 0, 'avg_price': 0, 'current_price': 0, 'amt': 0})
            current_amt = holding_info['amt']
            avg_price = holding_info['avg_price']
            
            current_price = row.get('current_price', 0)
            if current_price == 0: current_price = holding_info.get('current_price', 0)

            target_amt = int(total_budget * target_ratio)
            diff = target_amt - current_amt

            profit_rate = 0.0
            if avg_price > 0 and current_price > 0:
                profit_rate = ((current_price - avg_price) / avg_price) * 100

            # 액션 결정
            if diff > threshold_amt:
                base_action = '매수'
            elif diff < -threshold_amt:
                base_action = '비중축소'
            else:
                base_action = '유지'

            final_action = base_action
            reason = f"목표비중 {target_ratio*100:.1f}%"

            # 손절/익절 로직
            if profit_rate <= STOP_LOSS_RATE:
                final_action = '전량매도' if ai_score < 40 else '비중축소'
                reason = f"손절매({profit_rate:.2f}%)"
            elif profit_rate >= PROFIT_TAKE_RATE:
                final_action = '비중축소'
                reason = f"익절({profit_rate:.2f}%)"
            
            # 최종 추가
            if final_action != '유지':
                rebalancing_plan.append({
                    'code': code,
                    'name': get_stock_name(code),
                    'ai_score': ai_score,
                    'current_amt': current_amt,
                    'target_amt': target_amt,
                    'diff': int(diff),
                    'action': final_action,
                    'profit_rate': profit_rate,
                    'reason': reason
                })

        df_plan = pd.DataFrame(rebalancing_plan)
        if not df_plan.empty:
            df_plan['abs_diff'] = df_plan['diff'].abs()
            df_plan = df_plan.sort_values(by='abs_diff', ascending=False)
            
        return df_plan

# -----------------------------------------------------------
# 4. AI Advisor
# -----------------------------------------------------------
class AIAdvisor:
    def __init__(self):
        print("\n" + "=" * 60)
        print("[AI Advisor] 매매 의견 생성기 (TEST MODE)")
        print(" - 특징: 매수 기준을 30점으로 낮춰서 추천 종목 강제 출력")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        self.model = StackingEnsemble()
        self.rebalancer = PortfolioRebalancer()
        self.vc_strategy = ValueChainStrategy()

        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        self.model.load_model(model_path)

        self.target_codes = list(STOCK_NAME_MAP.keys())
        self.auth_token = None
        self.init_csv_log()

    # -------------------------------------------------------
    # CSV 로그
    # -------------------------------------------------------
    def init_csv_log(self):
        if not os.path.exists(ADVICE_LOG_PATH):
            with open(ADVICE_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "code", "name",
                    "ai_score", "current_amt",
                    "target_amt", "diff",
                    "action", "reason"
                ])

    def save_advice(self, row):
        with open(ADVICE_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                row["code"], row["name"],
                row["ai_score"], row["current_amt"],
                row["target_amt"], row["diff"],
                row["action"], row["reason"]
            ])

    # -------------------------------------------------------
    # 인증 / 잔고
    # -------------------------------------------------------
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
            params={"virtual": "true"} # 모의투자로 변경 확인
        )
        return res.json() if res.status_code == 200 else None

    # -------------------------------------------------------
    # AI 분석
    # -------------------------------------------------------
    def analyze_stock(self, code):
        try:
            features = self.store.get_realtime_features(code)
            if features is None or features.empty:
                return None

            probs = self.model.predict_proba(features)
            score = probs[0, 1] * 100
            
            if "close" not in features.columns:
                return None
                
            price = int(features["close"].values[0])
            
            return {
                "code": code,
                "name": get_stock_name(code),
                "ai_score": round(score, 2),
                "current_price": price
            }
        except Exception as e:
            print(f" [{code}] 분석 실패: {e}")
            return None

    # -------------------------------------------------------
    # 출력 블록
    # -------------------------------------------------------
    def print_block(self, row, total_asset):
        ratio = (row["target_amt"] / total_asset * 100) if total_asset > 0 else 0
        sign = "+" if row["diff"] > 0 else ""

        print(f"\n▶ {row['name']} ({row['code']})")
        print(f" - AI 점수     : {row['ai_score']}")
        print(f" - 현재 보유   : {int(row['current_amt']):,}원")
        print(f" - 목표 보유   : {int(row['target_amt']):,}원 ({ratio:.1f}%)")
        print(f" - 조절 금액   : {sign}{int(row['diff']):,}원")
        print(f" - 판단        : {row['action']}")
        print(f" - 이유        : {row['reason']}")

    # -------------------------------------------------------
    # 핵심 실행
    # -------------------------------------------------------
    def generate_advice(self):
        print(f"\n[Advisor] {datetime.now().strftime('%H:%M:%S')} 의견 생성 중...")

        if not self.auth_token and not self.login():
            print(" 로그인 실패")
            return

        balance = self.get_balance()
        if not balance:
            print(" 잔고 조회 실패")
            return

        holdings = balance.get("holdings")
        if not isinstance(holdings, list):
            holdings = []

        summary = balance.get("summary", {})
        cash_raw = summary.get("totalCashAmount") or summary.get("d2CashAmount") or 0
        cash = int(str(cash_raw).replace(",", "")) if cash_raw else 0

        my_holdings = {}
        for h in holdings:
            qty = int(h.get("quantity", 0))
            if qty > 0:
                my_holdings[h["stockCode"]] = {
                    "qty": qty,
                    "avg_price": float(h.get("avgPrice", 0)),
                    "current_price": 0,
                    "amt": 0
                }

        universe = set(self.target_codes) | set(my_holdings.keys())
        ai_results = []

        for code in universe:
            data = self.analyze_stock(code)
            if data:
                if code in my_holdings:
                    my_holdings[code]["current_price"] = data["current_price"]
                    my_holdings[code]["amt"] = data["current_price"] * my_holdings[code]["qty"]
                ai_results.append(data)

        if not ai_results:
            print(" 분석 결과 없음")
            return

        ai_df = pd.DataFrame(ai_results)
        total_stock_val = sum(h["amt"] for h in my_holdings.values())
        total_asset = cash + total_stock_val

        # 리밸런싱 실행
        plan_df = self.rebalancer.run_ai_rebalancing(
            my_holdings, ai_df, total_asset, {}, {}
        )

        # 1. 실행 기준 충족 (매수/매도/비중조절)
        if not plan_df.empty:
            print("\n[AI 매매 제안] (기준 충족)")
            for _, row in plan_df.iterrows():
                self.print_block(row, total_asset)
                self.save_advice(row)
        
        # 2. 유망 종목 무조건 출력 (매수 제안이 없어도 상위 종목은 보여줌)
        print("\n[AI 유망 종목 TOP 5] (참고용)")
        top_df = ai_df.sort_values("ai_score", ascending=False).head(5)
        
        # 이미 매매 제안에 있는 종목은 제외하고 출력하려면 로직 추가 가능
        # 여기서는 단순히 Top 5를 보여줌
        existing_codes = plan_df['code'].tolist() if not plan_df.empty else []
        
        VIRTUAL_RATIO = 0.10
        for _, r in top_df.iterrows():
            # 매매 제안에 이미 포함된 종목이면 스킵 (중복 출력 방지)
            if r['code'] in existing_codes:
                continue

            holding = my_holdings.get(r["code"])
            current_amt = holding["amt"] if holding else 0
            target_amt = int(total_asset * VIRTUAL_RATIO)

            row = {
                "code": r["code"],
                "name": r["name"],
                "ai_score": r["ai_score"],
                "current_amt": current_amt,
                "target_amt": target_amt,
                "diff": target_amt - current_amt,
                "action": "강력매수" if r['ai_score'] >= 80 else "매수고려",
                "reason": "AI 점수 상위"
            }
            self.print_block(row, total_asset)

# -----------------------------------------------------------
# 5. 자동 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    advisor = AIAdvisor()
    try:
        while True:
            advisor.generate_advice()
            time.sleep(300)
    except KeyboardInterrupt:
        print("\n[System] Advisor 종료")