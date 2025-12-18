import sys
import os
import time
import requests
import pandas as pd
import numpy as np
import csv
import json
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------
# 1. 프로젝트 경로 및 환경 설정
# -----------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# .env 로드
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# [수정] 백엔드 포트 8080 (application.properties 기준)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/app")
TEST_EMAIL = os.getenv("TEST_EMAIL", "testuser@example.com")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "password1234")

# CSV 로그 파일 경로
LOG_FILE_PATH = os.path.join(current_dir, "trade_record.csv")

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------
# [중요] 가격 정보는 오직 FeatureStore에서만 가져옵니다. (PriceLoader 사용 안 함)
from ai_pipeline.feature_store import OnlineFeatureStore
from ai_pipeline.boosting_model.train import StackingEnsemble


# -----------------------------------------------------------
# 3. 포트폴리오 리밸런서
# -----------------------------------------------------------
class PortfolioRebalancer:
    """AI 점수 기반 포트폴리오 리밸런싱 엔진"""

    def __init__(self, risk_aversion='neutral'):
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings_amt, ai_scores_df, total_budget=None):
        if ai_scores_df is None or ai_scores_df.empty:
            return pd.DataFrame()

        # 1. 데이터 정제
        cleaned_holdings = {str(k).strip().zfill(6): v for k, v in current_holdings_amt.items()}
        current_total = sum(cleaned_holdings.values())

        ai_scores_df = ai_scores_df.copy()
        ai_scores_df['code'] = ai_scores_df['code'].astype(str).str.strip().str.zfill(6)

        if total_budget is None:
            total_budget = current_total

        # 2. 보유 종목 데이터 보정
        merged_df = ai_scores_df.copy()
        held_codes = set(cleaned_holdings.keys())
        prediction_codes = set(merged_df['code'].values)

        missing_holdings = held_codes - prediction_codes
        if missing_holdings:
            missing_data = []
            for code in missing_holdings:
                # 점수 미확인 보유종목은 40점 부여 (유지/축소 유도)
                missing_data.append({'code': code, 'ai_score': 40.0})
            merged_df = pd.concat([merged_df, pd.DataFrame(missing_data)], ignore_index=True)

        # 3. 이중 필터링
        cond_new_buy = (~merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 60)
        cond_keep = (merged_df['code'].isin(held_codes)) & (merged_df['ai_score'] >= 40)

        buy_candidates = merged_df[cond_new_buy | cond_keep].copy()

        # 4. 비중 산출 (Score^2)
        if not buy_candidates.empty:
            buy_candidates['weight_score'] = np.power(buy_candidates['ai_score'], 2)
            total_weight_score = buy_candidates['weight_score'].sum()
            buy_candidates['target_ratio'] = buy_candidates['weight_score'] / total_weight_score

        # 5. 최종 주문 생성
        rebalancing_plan = []
        THRESHOLD_RATIO = 0.02
        threshold_amt = total_budget * THRESHOLD_RATIO

        # (1) 매수/유지/축소 대상
        if not buy_candidates.empty:
            for _, row in buy_candidates.iterrows():
                code = row['code']
                target_ratio = row['target_ratio']
                target_amt = int(total_budget * target_ratio)
                current_amt = int(cleaned_holdings.get(code, 0))
                diff = target_amt - current_amt

                if abs(diff) < threshold_amt:
                    action = '유지'
                elif diff > 0:
                    action = '매수'
                else:
                    action = '비중축소'

                rebalancing_plan.append({
                    'code': code,
                    'ai_score': row['ai_score'],
                    'current_amt': current_amt,
                    'target_amt': target_amt,
                    'diff': int(diff),
                    'action': action,
                    'reason': f"목표비중 {target_ratio * 100:.1f}%"
                })

        # (2) 전량 매도 대상 (탈락한 보유 종목)
        surviving_codes = set(buy_candidates['code'].values) if not buy_candidates.empty else set()
        for code, amt in cleaned_holdings.items():
            if code not in surviving_codes:
                rebalancing_plan.append({
                    'code': code,
                    'ai_score': 0.0,
                    'current_amt': int(amt),
                    'target_amt': 0,
                    'diff': -int(amt),
                    'action': '전량매도',
                    'reason': "점수 미달(40점 미만)"
                })

        df_plan = pd.DataFrame(rebalancing_plan)
        if not df_plan.empty:
            df_plan = df_plan.sort_values(by='diff', ascending=True)  # 매도 먼저 하도록 정렬

        return df_plan


# -----------------------------------------------------------
# 4. 메인 트레이더 (AIAutoTrader)
# -----------------------------------------------------------
class AIAutoTrader:
    def __init__(self):
        print("\n" + "=" * 60)
        print(" 🤖 [AI AutoTrader] 포트폴리오 리밸런싱 시스템")
        print("=" * 60)

        # [중요] FeatureStore만 사용하여 가격/피처 정보 수집
        self.store = OnlineFeatureStore()
        self.rebalancer = PortfolioRebalancer(risk_aversion='neutral')

        # 직접 주문 객체(KISMockTrader) 제거 -> 백엔드 API 사용

        self.model = StackingEnsemble()
        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        try:
            self.model.load_model(model_path)
            print(" [Init] AI 모델 로드 완료")
        except Exception as e:
            print(f" [Error] 모델 로드 실패: {e}")
            sys.exit(1)

        # 감시 대상 유니버스
        self.target_codes = ["005930", "000660", "207940", "005380", "000270", "055550", "105560", "068270", "015760",
                             "028260", "032830", "012330", "035420", "006400", "086790", "006405", "000810", "010140",
                             "064350", "138040", "051910", "010130", "009540", "267260", "066570", "066575", "033780",
                             "003550", "003555", "310200", "034020", "012450", "009830", "011070", "071050", "081660",
                             "046890", "323410", "017670", "010620", "047050", "009155", "275630", "009835", "001440",
                             "138930", "175330", "051900", "005490", "034220"]
        self.auth_token = None
        self.init_csv_log()

    def init_csv_log(self):
        if not os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'code', 'action', 'qty', 'price', 'total_amt', 'reason'])

    def save_trade_log(self, code, action, qty, price, reason):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_amt = qty * price
            with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, code, action, qty, price, total_amt, reason])
            print(f"      💾 로그 저장 완료")
        except Exception as e:
            print(f"      ⚠️ 로그 저장 실패: {e}")

    # --- 백엔드 API (로그인) ---
    def login(self):
        url = f"{BACKEND_API_URL}/auth/login"
        payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                self.auth_token = resp.json().get('accessToken')
                print(f" [Auth] 백엔드 로그인 성공")
                return True
            else:
                print(f" [Auth] 로그인 실패: Status={resp.status_code}, Msg={resp.text}")
            return False
        except Exception as e:
            print(f" [Auth] 서버 연결 오류: {e}")
            return False

    def get_headers(self):
        if not self.auth_token: return {}
        # Bearer Token 형식 맞춤
        return {'Authorization': f'Bearer {self.auth_token}'}

    # --- 백엔드 API (잔고 조회) ---
    def get_balance(self):
        """백엔드에서 잔고 조회 (예수금, 보유수량)"""
        url = f"{BACKEND_API_URL}/kis/trade/balance"

        # [주의] KIS AppKey가 실전용이면 'false', 모의투자용이면 'true'
        # EGW2004 오류 방지를 위해 false로 설정 (사용자 환경에 맞춤)
        params = {'virtual': 'true'}

        try:
            resp = requests.get(url, headers=self.get_headers(), params=params)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                print(f"      ⛔ [Balance] 토큰 만료 또는 인증 실패 (401). 재로그인 시도...")
                self.login()
                return None
            else:
                print(f"      ⚠️ [Balance] 조회 실패: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            print(f"      🚫 [Balance] 요청 중 예외 발생: {e}")
            return None

    # --- 백엔드 API (주문 전송) ---
    def send_order(self, code, action, price, qty, reason):
        print(f"      📡 주문 전송... [{code} {qty}주 {action}] ({reason})")

        url = f"{BACKEND_API_URL}/kis/trade"
        params = {'virtual': 'true'}  # 실전용 계좌인 경우 false

        # Action -> API orderType 매핑
        order_type = "BUY" if action == '매수' else "SELL"

        # 요청 바디 (사용자가 지정한 형식 준수)
        payload = {
            "stockCode": code,
            "orderType": order_type,
            "quantity": qty,
            "price": price,
            "triggerSource": "MANUAL"  # 자동매매지만 API 요구사항에 맞춰 MANUAL 사용
        }

        try:
            # POST 요청 전송 (Bearer Token 포함)
            res = requests.post(url, headers=self.get_headers(), params=params, json=payload)

            if res.status_code == 200:
                # 성공 시 응답 내용 출력
                print(f"      ✅ 주문 성공! - {res.json()}")
                self.save_trade_log(code, action, qty, price, reason)
                return True
            else:
                # 실패 시 에러 메시지 출력
                print(f"      ❌ 주문 실패: {res.status_code} - {res.text}")
                return False
        except Exception as e:
            print(f"      🚫 주문 중 에러 발생: {e}")
            return False

    def analyze_stock(self, code):
        # [중요] OnlineFeatureStore를 통해서만 가격 및 피처 로드
        features = self.store.get_realtime_features(code)
        if features is None or features.empty: return None

        probs = self.model.predict_proba(features)
        score = probs[0, 1] * 100 if hasattr(probs, 'ndim') and probs.ndim == 2 else probs[1] * 100

        return {
            'code': code,
            'ai_score': round(score, 2),
            'current_price': int(features['close'].values[0])  # 현재가 추출
        }

    def run_cycle(self):
        print(f"\n 🕒 [Cycle] {datetime.now().strftime('%H:%M:%S')} 리밸런싱 시작")

        if not self.auth_token:
            if not self.login(): return

        # 1. 잔고 조회 (실패 시 중단)
        balance_data = self.get_balance()
        if not balance_data: return

        summary = balance_data.get('summary', {})
        holdings_list = balance_data.get('holdings') or []

        try:
            cash = int(summary.get('totalCashAmount', 0))
        except:
            cash = 0

        # 보유 종목 파싱
        my_holdings_map = {}
        for h in holdings_list:
            code = h['stockCode']
            qty = h.get('quantity', 0)
            my_holdings_map[code] = {'qty': qty, 'price': 0, 'amt': 0}

        print(f" 💰 예수금: {cash:,}원 | 보유종목: {len(my_holdings_map)}개")

        # 2. 유니버스 분석 (전체 종목 AI 점수 산출)
        universe = set(self.target_codes) | set(my_holdings_map.keys())
        ai_results = []

        for code in universe:
            data = self.analyze_stock(code)
            if data:
                ai_results.append(data)
                if code in my_holdings_map:
                    price = data['current_price']
                    qty = my_holdings_map[code]['qty']
                    my_holdings_map[code]['price'] = price
                    my_holdings_map[code]['amt'] = price * qty

        if not ai_results:
            print(" [Info] 분석 가능한 종목 데이터가 없습니다.")
            return

        ai_scores_df = pd.DataFrame(ai_results)

        # 3. 리밸런서 실행
        current_holdings_amt = {code: info['amt'] for code, info in my_holdings_map.items()}
        total_stock_val = sum(current_holdings_amt.values())
        total_asset = cash + total_stock_val

        print(f" [Asset] 총 자산: {total_asset:,}원 (주식:{total_stock_val:,} + 현금:{cash:,})")

        plan_df = self.rebalancer.run_ai_rebalancing(current_holdings_amt, ai_scores_df, total_budget=total_asset)

        if plan_df.empty:
            print(" [Plan] 매매할 건이 없습니다.")
            return

        print("\n [리밸런싱 계획]")
        print(plan_df[['code', 'ai_score', 'action', 'diff', 'reason']].to_string(index=False))

        # 4. 주문 실행
        for _, row in plan_df.iterrows():
            action = row['action']
            if action == '유지': continue

            code = row['code']
            price = 0
            found = [x for x in ai_results if x['code'] == code]
            if found: price = found[0]['current_price']

            if price == 0: continue

            # 매도
            if action in ['비중축소', '전량매도']:
                amt_to_sell = abs(row['diff'])
                qty_to_sell = int(amt_to_sell // price)
                if qty_to_sell > 0:
                    self.send_order(code, action, price, qty_to_sell, row['reason'])
                    cash += (qty_to_sell * price)

            # 매수
            elif action == '매수':
                amt_to_buy = row['diff']
                safe_cash = cash * 0.95  # 미수 방지 안전 마진
                if safe_cash >= amt_to_buy:
                    qty_to_buy = int(amt_to_buy // price)
                    if qty_to_buy > 0:
                        if self.send_order(code, action, price, qty_to_buy, row['reason']):
                            cash -= (qty_to_buy * price)
                else:
                    if safe_cash > 100000:
                        print(f"      ⚠️ 예수금 부족 ({code}): 필요 {amt_to_buy:,} > 가능 {safe_cash:,.0f}")

        print(" [Cycle] 종료")


if __name__ == "__main__":
    trader = AIAutoTrader()
    try:
        while True:
            trader.run_cycle()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n [System] 자동매매 종료")