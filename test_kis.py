import os
import sys
import requests
import json
from dotenv import load_dotenv

# 프로젝트 경로 인식
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 설정 파일 불러오기
from ai_pipeline.config.settings import KIS_APP_KEY, KIS_APP_SECRET, KIS_BASE_URL, KIS_ACCOUNT_NO

def test_connection():
    print("🔌 한국투자증권 API 접속 테스트 시작...")
    print(f"   - 접속 서버: {KIS_BASE_URL}")
    
    # 1. 접근 토큰(Token) 발급 요청
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET
    }
    
    try:
        url = f"{KIS_BASE_URL}/oauth2/tokenP"
        res = requests.post(url, headers=headers, data=json.dumps(body))
        
        if res.status_code == 200:
            token = res.json().get("access_token")
            print(f"✅ [성공] 토큰 발급 완료! (앞 10자리: {token[:10]}...)")
            
            # 2. 삼성전자(005930) 현재가 조회로 최종 확인
            check_price(token)
        else:
            print(f"❌ [실패] 토큰 발급 에러: {res.text}")
            
    except Exception as e:
        print(f"❌ [에러] 연결 실패: {e}")

def check_price(token):
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST01010100"
    }
    # 삼성전자 코드: 005930
    url = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price?fid_cond_mrkt_div_code=J&fid_input_iscd=005930"
    
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        price = res.json()['output']['stck_prpr']
        print(f"💰 [확인] 삼성전자 현재가: {price}원")
        print("🎉 축하합니다! API 연결 설정이 완벽합니다.")
    else:
        print(f"⚠️ 시세 조회 실패: {res.text}")

if __name__ == "__main__":
    test_connection()