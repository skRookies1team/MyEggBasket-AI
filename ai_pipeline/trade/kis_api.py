import os
import json
import requests
import time

class KISMockTrader:
    def __init__(self):
        self.app_key = os.getenv("KIS_MOCK_APP_KEY")
        self.app_secret = os.getenv("KIS_MOCK_APP_SECRET")
        self.acc_no = os.getenv("KIS_MOCK_ACCOUNT_NO") # 계좌번호 앞 8자리
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.access_token = None

        if not all([self.app_key, self.app_secret, self.acc_no]):
            print("❌ [KIS] .env에 모의투자 계좌 정보가 없습니다.")
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
        res = requests.post(f"{self.base_url}/oauth2/tokenP", headers=headers, data=json.dumps(body))
        if res.status_code == 200:
            self.access_token = res.json()["access_token"]
            print(f"✅ [KIS] 토큰 발급 완료")
        else:
            print(f"❌ [KIS] 토큰 발급 실패: {res.text}")

    def get_common_headers(self, tr_id):
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }

    def buy_limit(self, stock_code, price, qty):
        """지정가 매수"""
        if not self.access_token: return
        headers = self.get_common_headers("VTTC0802U") # 모의투자 매수 TR ID
        body = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": "01",
            "PDNO": stock_code,
            "ORD_DVSN": "00", # 00: 지정가
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price)
        }
        res = requests.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(body))
        msg = res.json().get('msg1', '알 수 없음')
        print(f" 🛒 [매수주문] {stock_code} {qty}주 @ {price}원 -> 결과: {msg}")
        return res.json()