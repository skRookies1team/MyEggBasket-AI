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

# 백엔드 설정
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8081/api/app")
TEST_EMAIL = os.getenv("TEST_EMAIL", "testuser@example.com")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "password1234")

# CSV 로그 파일 경로
LOG_FILE_PATH = os.path.join(current_dir, "trade_record.csv")

# -----------------------------------------------------------
# 2. 모듈 Import
# -----------------------------------------------------------
from ai_pipeline.feature_store import OnlineFeatureStore
from ai_pipeline.boosting_model.train import StackingEnsemble


# -----------------------------------------------------------
# 3. KIS Direct Trader (직접 주문용)
# -----------------------------------------------------------
class KISMockTrader:
    def __init__(self):
        self.app_key = os.getenv("KIS_MOCK_APP_KEY")
        self.app_secret = os.getenv("KIS_MOCK_APP_SECRET")
        self.acc_no = os.getenv("KIS_MOCK_ACCOUNT_NO")  # 계좌번호 앞 8자리
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.access_token = None

        if not all([self.app_key, self.app_secret, self.acc_no]):
            print(" [KIS] .env 확인 필요: KIS_MOCK_APP_KEY, SECRET, ACCOUNT_NO 없음")
        else:
            self.auth()

    def auth(self):
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        try:
            res = requests.post(f"{self.base_url}/oauth2/tokenP", headers=headers, data=json.dumps(body))
            if res.status_code == 200:
                self.access_token = res.json()["access_token"]
                print(f" [KIS] 토큰 발급 완료")
            else:
                print(f" [KIS] 토큰 발급 실패: {res.text}")
        except Exception as e:
            print(f" [KIS] 인증 중 에러: {e}")

    def get_common_headers(self, tr_id):
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }

    def buy_limit(self, stock_code, price, qty):
        """지정가 매수 (현재가 주문)"""
        if not self.access_token: return None
        headers = self.get_common_headers("VTTC0802U")
        body = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": "01",
            "PDNO": stock_code,
            "ORD_DVSN": "00",  # 지정가
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price)
        }
        res = requests.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers,
                            data=json.dumps(body))
        return res.json()

    def sell_limit(self, stock_code, price, qty):
        """지정가 매도 (현재가 주문)"""
        if not self.access_token: return None
        headers = self.get_common_headers("VTTC0801U")
        body = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": "01",
            "PDNO": stock_code,
            "ORD_DVSN": "00",
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price)
        }
        res = requests.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers,
                            data=json.dumps(body))
        return res.json()


# -----------------------------------------------------------
# 4. 포트폴리오 리밸런서 (제공해주신 로직 통합)
# -----------------------------------------------------------
class PortfolioRebalancer:
    """AI 점수 기반 포트폴리오 리밸런싱 엔진"""

    def __init__(self, risk_aversion='neutral'):
        self.risk_aversion = risk_aversion

    def run_ai_rebalancing(self, current_holdings_amt, ai_scores_df, total_budget=None):
        """
        current_holdings_amt: {'005930': 5000000} (종목별 평가금액)
        ai_scores_df: DataFrame ['code', 'ai_score']
        """
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

        if buy_candidates.empty:
            # 유효 대상이 없으면 보유종목 전량 매도 시그널 생성 필요
            # 여기서는 빈 DF 리턴 -> 아래 로직에서 전량 매도 처리됨
            pass

        # 4. 비중 산출 (Score^2)
        if not buy_candidates.empty:
            buy_candidates['weight_score'] = np.power(buy_candidates['ai_score'], 2)
            total_weight_score = buy_candidates['weight_score'].sum()
            buy_candidates['target_ratio'] = buy_candidates['weight_score'] / total_weight_score
        else:
            # 후보가 아무도 없으면 빈 DF 유지
            pass

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
# 5. 메인 트레이더 (AIAutoTrader)
# -----------------------------------------------------------
class AIAutoTrader:
    def __init__(self):
        print("\n" + "=" * 60)
        print(" 🤖 [AI AutoTrader] 포트폴리오 리밸런싱 시스템")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        self.rebalancer = PortfolioRebalancer(risk_aversion='neutral')
        self.kis_direct = KISMockTrader()  # 주문용
        self.model = StackingEnsemble()

        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        try:
            self.model.load_model(model_path)
            print(" [Init] AI 모델 로드 완료")
        except Exception as e:
            print(f" [Error] 모델 로드 실패: {e}")
            sys.exit(1)

        # 감시 대상 유니버스
        self.target_codes = ["005930", "000660", "207940", "005380", "000270", "055550", "105560", "035420", "006400",
                             "051910", "000810", "010140", "009830", "011070", "047050", "009155", "051900"]
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

    # --- 백엔드 API (잔고 조회용) ---
    def login(self):
        url = f"{BACKEND_API_URL}/auth/login"
        payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                self.auth_token = resp.json().get('accessToken')
                print(f" [Auth] 백엔드 로그인 성공")
                return True
            return False
        except:
            return False

    def get_headers(self):
        if not self.auth_token: return {}
        return {'Authorization': f'Bearer {self.auth_token}'}

    def get_balance(self):
        """백엔드에서 잔고 조회 (예수금, 보유수량)"""
        url = f"{BACKEND_API_URL}/kis/trade/balance"
        params = {'virtual': 'true'}
        try:
            resp = requests.get(url, headers=self.get_headers(), params=params)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                self.login()
                return None
            return None
        except:
            return None

    # --- 주문 실행 (KIS Direct) ---
    def send_order(self, code, action, price, qty, reason):
        print(f"      📡 주문 전송... [{code} {qty}주 {action}] ({reason})")
        res = None
        if action == '매수':
            res = self.kis_direct.buy_limit(code, price, qty)
        elif action in ['비중축소', '전량매도']:  # 둘 다 매도 로직
            res = self.kis_direct.sell_limit(code, price, qty)

        if res:
            rt_cd = res.get('rt_cd')
            msg1 = res.get('msg1')
            if rt_cd == '0':
                print(f"      ✅ 주문 성공! - {msg1}")
                self.save_trade_log(code, action, qty, price, reason)
                return True
            else:
                print(f"      ❌ 주문 실패: {msg1} (코드:{rt_cd})")
                return False
        return False

    def analyze_stock(self, code):
        features = self.store.get_realtime_features(code)
        if features is None or features.empty: return None
        probs = self.model.predict_proba(features)
        score = probs[0, 1] * 100 if hasattr(probs, 'ndim') and probs.ndim == 2 else probs[1] * 100
        return {
            'code': code,
            'ai_score': round(score, 2),
            'current_price': int(features['close'].values[0])
        }

    def run_cycle(self):
        print(f"\n 🕒 [Cycle] {datetime.now().strftime('%H:%M:%S')} 리밸런싱 시작")

        if not self.auth_token:
            if not self.login(): return

        # 1. 잔고 조회
        balance_data = self.get_balance()
        if not balance_data: return

        summary = balance_data.get('summary', {})
        holdings_list = balance_data.get('holdings') or []

        try:
            cash = int(summary.get('totalCashAmount', 0))
        except:
            cash = 0

        # 보유 종목 파싱 {code: {qty, price, amt}}
        # 백엔드 DTO 필드명 주의 (quantity, currentPrice 등)
        my_holdings_map = {}
        for h in holdings_list:
            code = h['stockCode']
            qty = h.get('quantity', 0)
            # 현재가는 백엔드 데이터보다 실시간 조회가 더 정확하므로 나중에 업데이트
            my_holdings_map[code] = {'qty': qty, 'price': 0, 'amt': 0}

        print(f" 💰 예수금: {cash:,}원 | 보유종목: {len(my_holdings_map)}개")

        # 2. 유니버스 분석 (전체 종목 AI 점수 산출)
        universe = set(self.target_codes) | set(my_holdings_map.keys())
        ai_results = []

        for code in universe:
            data = self.analyze_stock(code)
            if data:
                ai_results.append(data)
                # 보유종목이면 현재가/평가금액 업데이트
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
        # (1) 보유 종목 평가금액 맵핑
        current_holdings_amt = {code: info['amt'] for code, info in my_holdings_map.items()}
        # (2) 총 자산 (예수금 + 주식평가액)
        total_stock_val = sum(current_holdings_amt.values())
        total_asset = cash + total_stock_val

        print(f" [Asset] 총 자산: {total_asset:,}원 (주식:{total_stock_val:,} + 현금:{cash:,})")

        # (3) 계획 산출
        plan_df = self.rebalancer.run_ai_rebalancing(current_holdings_amt, ai_scores_df, total_budget=total_asset)

        if plan_df.empty:
            print(" [Plan] 매매할 건이 없습니다.")
            return

        print("\n [리밸런싱 계획]")
        print(plan_df[['code', 'ai_score', 'action', 'diff', 'reason']].to_string(index=False))

        # 4. 주문 실행 (매도 먼저 -> 현금확보 -> 매수)
        # diff가 오름차순 정렬되어 있으므로 (음수=매도) 순서대로 실행하면 됨

        for _, row in plan_df.iterrows():
            action = row['action']
            if action == '유지': continue

            code = row['code']
            price = 0
            # 가격 찾기
            found = [x for x in ai_results if x['code'] == code]
            if found: price = found[0]['current_price']

            if price == 0: continue

            # 수량 계산
            # 매도(비중축소/전량매도): diff는 음수
            if action in ['비중축소', '전량매도']:
                amt_to_sell = abs(row['diff'])
                qty_to_sell = int(amt_to_sell // price)
                if qty_to_sell > 0:
                    self.send_order(code, action, price, qty_to_sell, row['reason'])
                    # 가상 잔고 반영 (매수 자금 확보용)
                    cash += (qty_to_sell * price)

                    # 매수
            elif action == '매수':
                amt_to_buy = row['diff']
                # 안전장치: 예수금 확인
                safe_cash = cash * 0.95
                if safe_cash >= amt_to_buy:
                    qty_to_buy = int(amt_to_buy // price)
                    if qty_to_buy > 0:
                        if self.send_order(code, action, price, qty_to_buy, row['reason']):
                            cash -= (qty_to_buy * price)
                else:
                    if safe_cash > 100000:  # 로그 너무 많이 뜨는 것 방지
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