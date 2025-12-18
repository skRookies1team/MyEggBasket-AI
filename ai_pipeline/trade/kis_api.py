import os
import json
import requests
import time
from dotenv import load_dotenv

load_dotenv()

class KISMockTrader:
    def __init__(self):
        self.app_key = os.getenv("KIS_MOCK_APP_KEY")
        self.app_secret = os.getenv("KIS_MOCK_APP_SECRET")
        self.acc_no = os.getenv("KIS_MOCK_ACCOUNT_NO") # 계좌번호 앞 8자리
        self.base_url = "https://openapivts.koreainvestment.com:29443"
        self.access_token = None

        if not all([self.app_key, self.app_secret, self.acc_no]):
            print(" [KIS] .env에 모의투자 계좌 정보가 없습니다.")
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
            print(f" [KIS] 토큰 발급 완료")
        else:
            print(f" [KIS] 토큰 발급 실패: {res.text}")


    def get_common_headers(self, tr_id):
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
    

    def get_balance(self):
        """주식 잔고 및 예수금 조회"""
        if not self.access_token: return None
        
        headers = self.get_common_headers("VTTC8434R") 
        
        params = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": "01",
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        try:
            res = requests.get(f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance", headers=headers, params=params)
            
            if res.status_code == 200:
                data = res.json()
                if 'rt_cd' in data and data['rt_cd'] != '0':
                    print(f" [KIS] 잔고 조회 실패 메시지: {data.get('msg1')}")
                    return None
                
                return data # 전체 데이터 반환 (output1: 보유종목, output2: 예수금 등)
            else:
                print(f" [KIS] 잔고 조회 요청 실패: {res.text}")
                return None
        except Exception as e:
            print(f" [KIS] 잔고 조회 시스템 에러: {e}")
            return None


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
        print(f"  [매수주문] {stock_code} {qty}주 @ {price}원 -> 결과: {msg}")
        return res.json()
    

    def sell_limit(self, stock_code, price, qty):
        """지정가 매도"""
        if not self.access_token: return
        headers = self.get_common_headers("VTTC0801U") # 모의투자 매도
        # 시장가 매도 시 ORD_DVSN="01", ORD_UNPR="0"
        # 여기선 지정가(00) 기준
        body = {
            "CANO": self.acc_no,
            "ACNT_PRDT_CD": "01",
            "PDNO": stock_code,
            "ORD_DVSN": "00",
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price)
        }
        res = requests.post(f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash", headers=headers, data=json.dumps(body))
        print(f"  [매도] {stock_code} {qty}주 @ {price}원 (메시지: {res.json().get('msg1')})")
        return res.json()