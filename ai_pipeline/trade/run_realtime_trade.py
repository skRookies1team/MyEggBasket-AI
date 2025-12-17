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

# 백엔드 설정 (잔고 조회용)
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
# 3. KIS Direct Trader (사용자 제공 코드 통합)
# -----------------------------------------------------------
class KISMockTrader:
    def __init__(self):
        self.app_key = os.getenv("KIS_MOCK_APP_KEY")
        self.app_secret = os.getenv("KIS_MOCK_APP_SECRET")
        self.acc_no = os.getenv("KIS_MOCK_ACCOUNT_NO")  # 계좌번호 앞 8자리
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.access_token = None

        if not all([self.app_key, self.app_secret, self.acc_no]):
            print(" [KIS] .env에 모의투자 계좌 정보(APP_KEY, SECRET, ACCOUNT_NO)가 없습니다.")
        else:
            self.auth()

    def auth(self):
        """접근 토큰 발급"""
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

    # 잔고 조회는 백엔드를 쓰므로 여기서는 생략 가능하나, 비상용으로 유지

    def buy_limit(self, stock_code, price, qty):
        """지정가 매수 (현재가로 주문하여 시장가 효과)"""
        if not self.access_token: return None
        headers = self.get_common_headers("VTTC0802U")  # 모의투자 매수
        body = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": "01",
            "PDNO": stock_code,
            "ORD_DVSN": "00",  # 00: 지정가 (백엔드의 03 문제 해결)
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price)
        }
        res = requests.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers,
                            data=json.dumps(body))
        return res.json()

    def sell_limit(self, stock_code, price, qty):
        """지정가 매도"""
        if not self.access_token: return None
        headers = self.get_common_headers("VTTC0801U")  # 모의투자 매도
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
# 4. 전략 및 메인 트레이더
# -----------------------------------------------------------
class TradingStrategy:
    def __init__(self):
        self.state = {}

    def get_action(self, code, current_data, holding_info):
        score = current_data['score']
        price = current_data['price']
        vol_ratio = current_data.get('vol_ratio', 1.0)

        if code not in self.state:
            self.state[code] = {'prev_score': score, 'max_score': score}

        prev_score = self.state[code]['prev_score']
        self.state[code]['max_score'] = max(self.state[code]['max_score'], score)

        qty = holding_info.get('quantity', 0)
        avg_price = holding_info.get('avgPrice', 0)

        profit_pct = 0.0
        if qty > 0 and avg_price > 0:
            profit_pct = (price - avg_price) / avg_price

        # 1. 손절매 (-3%)
        if qty > 0 and profit_pct <= -0.03:
            return 'SELL', 1.0, f"손절매 (수익률 {profit_pct * 100:.2f}%)"

        # 2. AI 점수 급락 (40점 미만) -> 전량 매도
        if qty > 0 and score < 40:
            return 'SELL', 1.0, f"전량매도 (점수 {score}점)"

        # 3. 익절 (고점 찍고 하락 시 차익 실현)
        if qty > 0 and self.state[code]['max_score'] >= 90 and score < 85:
            return 'SELL', 0.5, "차익실현 (고점 대비 하락)"

        # 4. 매수 진입
        is_uptrend = score >= prev_score
        is_valid_vol = vol_ratio >= 0.8

        if is_uptrend and is_valid_vol:
            if qty == 0 and score >= 60:
                return 'BUY', 0.3, "1차 진입 (60점↑)"
            elif qty > 0 and score >= 80 and prev_score < 80:
                return 'BUY', 0.4, "2차 불타기 (80점 돌파)"
            elif qty > 0 and score >= 90 and prev_score < 90:
                return 'BUY', 0.3, "3차 확신 (90점 돌파)"

        self.state[code]['prev_score'] = score
        return 'HOLD', 0.0, ""


class AIAutoTrader:
    def __init__(self):
        print("\n" + "=" * 60)
        print(" 🤖 [AI AutoTrader] 하이브리드 자동매매 (조회:백엔드 / 주문:Direct)")
        print("=" * 60)

        self.store = OnlineFeatureStore()
        self.strategy = TradingStrategy()

        # [NEW] KIS 직접 호출 객체 생성
        self.kis_direct = KISMockTrader()

        self.model = StackingEnsemble()
        model_path = os.path.join(project_root, "ai_pipeline/boosting_model/models")
        try:
            self.model.load_model(model_path)
            print(" [Init] AI 모델 로드 완료")
        except Exception as e:
            print(f" [Error] 모델 로드 실패: {e}")
            sys.exit(1)

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
            print(f" [Log] 로그 파일 생성: {LOG_FILE_PATH}")

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

    # -----------------------------------------------------------
    # [백엔드 API - 잔고 조회]
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # [Direct KIS API - 주문 실행] (수정된 부분)
    # -----------------------------------------------------------
    def send_order(self, code, type_str, price, qty, reason):
        """
        백엔드 거치지 않고 직접 KIS API 호출
        """
        print(f"      📡 직접 주문 전송... [{code} {qty}주 @ {price}원] ({type_str})")

        res = None
        if type_str == "BUY":
            res = self.kis_direct.buy_limit(code, price, qty)
        elif type_str == "SELL":
            res = self.kis_direct.sell_limit(code, price, qty)

        # 결과 처리
        if res:
            rt_cd = res.get('rt_cd')
            msg1 = res.get('msg1')
            if rt_cd == '0':
                print(f"      ✅ 주문 성공! (KIS Direct) - {msg1}")
                self.save_trade_log(code, type_str, qty, price, reason)
                return True
            else:
                print(f"      ❌ 주문 실패 (KIS Direct): {msg1} (코드: {rt_cd})")
                return False
        else:
            print("      ❌ 주문 통신 오류 (KIS Direct)")
            return False

    def analyze_stock(self, code):
        features = self.store.get_realtime_features(code)
        if features is None or features.empty:
            return None

        probs = self.model.predict_proba(features)
        score = probs[0, 1] * 100 if hasattr(probs, 'ndim') and probs.ndim == 2 else probs[1] * 100

        return {
            'score': round(score, 2),
            'price': int(features['close'].values[0]),
            'vol_ratio': features.get('hist_Vol_Ratio', pd.Series([1.0])).values[0]
        }

    def run_cycle(self):
        print(f"\n 🕒 [Cycle] {datetime.now().strftime('%H:%M:%S')} 시장 감시 시작")

        if not self.auth_token:
            if not self.login(): return

        # 1. 잔고 조회 (백엔드 이용)
        balance_data = self.get_balance()
        if not balance_data: return

        summary = balance_data.get('summary', {})
        holdings_list = balance_data.get('holdings')
        if holdings_list is None: holdings_list = []

        try:
            cash = int(summary.get('totalCashAmount', 0))
        except:
            cash = 0

        my_holdings = {h['stockCode']: h for h in holdings_list}
        print(f" 💰 예수금: {cash:,}원 | 보유종목: {len(my_holdings)}개")

        universe = set(self.target_codes) | set(my_holdings.keys())

        for code in universe:
            market_data = self.analyze_stock(code)
            if not market_data: continue

            holding_info = my_holdings.get(code, {'quantity': 0, 'avgPrice': 0})

            # 전략 판단
            action, ratio, reason = self.strategy.get_action(code, market_data, holding_info)

            status_mk = "🔴보유" if holding_info.get('quantity', 0) > 0 else "⚪미보유"
            print(f"   [{code}] {market_data['score']}점 ({status_mk}) -> {action} ({reason})")

            # 주문 실행
            if action == 'BUY':
                amount_to_buy = 1_000_000 * ratio
                safe_cash = cash * 0.95

                if safe_cash >= amount_to_buy:
                    qty = int(amount_to_buy // market_data['price'])
                    if qty > 0:
                        if self.send_order(code, "BUY", market_data['price'], qty, reason):
                            cash -= amount_to_buy
                else:
                    if safe_cash >= 100000:
                        print(f"      ⚠️ 예수금 부족 (필요: {amount_to_buy:,.0f} > 가능: {safe_cash:,.0f})")

            elif action == 'SELL':
                current_qty = holding_info.get('quantity', 0)
                qty_to_sell = int(current_qty * ratio)
                if qty_to_sell > 0:
                    self.send_order(code, "SELL", market_data['price'], qty_to_sell, reason)

        print(" [Cycle] 종료")


if __name__ == "__main__":
    trader = AIAutoTrader()
    try:
        while True:
            trader.run_cycle()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n [System] 자동매매 종료")