import requests
import json
import time
import pandas as pd
import os
from datetime import datetime
# [추가] .env 파일 로드를 위한 모듈
from dotenv import load_dotenv

# [중요] 에러 처리를 위한 모듈
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from urllib3.exceptions import IncompleteRead, ProtocolError

# ==========================================
# 0. 환경 변수 로드 (.env)
# ==========================================
# 현재 디렉토리의 .env 파일을 읽어옵니다.
load_dotenv()

# ==========================================
# 1. 설정 (환경 변수에서 가져오기)
# ==========================================
APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
URL_BASE = "https://openapi.koreainvestment.com:9443"

# 키 값 존재 여부 확인 (실수 방지용)
if not APP_KEY or not APP_SECRET:
    print("❌ Error: .env 파일에서 APP_KEY 또는 APP_SECRET을 찾을 수 없습니다.")
    print("    1. .env 파일이 같은 폴더에 있는지 확인하세요.")
    print("    2. .env 파일 안에 변수명이 정확한지 확인하세요.")
    exit()

# ==========================================
# 2. 수집할 종목 리스트 (50개)
# ==========================================
target_codes_raw = "005930,000660,207940,005380,000270,055550,105560,068270,015760,028260,032830,012330,035420,006400,086790,006405,000810,010140,064350,138040,051910,010130,009540,267260,066570,066575,033780,003550,003555,310200,034020,012450,009830,011070,071050,081660,046890,323410,017670,010620,047050,009155,275630,009835,001440, 138930 ,175330 ,051900,005490,034220"
target_codes = [code.strip() for code in target_codes_raw.split(',')]

# ==========================================
# 3. 토큰 발급 함수
# ==========================================
def get_access_token():
    url = f"{URL_BASE}/oauth2/tokenP"
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    try:
        res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(body))
        if res.status_code == 200: return res.json()['access_token']
    except Exception as e:
        print(f"Token Error: {e}")
    return None

# ==========================================
# 4. [핵심] 안전한 데이터 요청 함수 (재시도 로직 포함)
# ==========================================
def get_minute_chart_safe(token, code, date_str, time_str, max_retries=10):
    path = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    url = f"{URL_BASE}{path}"
    headers = {
        "content-type": "application/json; charset=utf-8", "authorization": f"Bearer {token}",
        "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "FHKST03010230", "custtype": "P"
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": date_str, "FID_INPUT_HOUR_1": time_str,
        "FID_PW_DATA_INCU_YN": "Y", "FID_ETC_CLS_CODE": "", "FID_FAKE_TICK_INCU_YN": "N"
    }

    for attempt in range(max_retries):
        try:
            # timeout을 30초로 넉넉하게 설정
            res = requests.get(url, headers=headers, params=params, timeout=30)
            
            if res.status_code == 200:
                data = res.json()
                if data['rt_cd'] == '0':
                    return data['output2']
                else:
                    # API 호출 제한이나 일시적 오류
                    print(f"\n   ⚠️ API Msg: {data['msg1']} (Retry {attempt+1}/{max_retries})...")
                    time.sleep(1.5)
            else:
                print(f"\n   ⚠️ HTTP Status: {res.status_code} (Retry {attempt+1}/{max_retries})...")
                time.sleep(1.5)
                
        # [중요] 방금 발생한 에러들을 여기서 다 잡습니다
        except (ChunkedEncodingError, ConnectionError, ReadTimeout, IncompleteRead, ProtocolError) as e:
            print(f"\n   🚨 네트워크 끊김 발생! ({e}) -> 3초 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(3) # 3초 쉬고 다시 시도
            
        except Exception as e:
            print(f"\n   🚨 알 수 없는 에러: {e} -> 재시도")
            time.sleep(1)

    print(f"\n❌ {max_retries}회 재시도 실패. 건너뜁니다.")
    return None

# ==========================================
# 5. 메인 실행
# ==========================================
if __name__ == "__main__":
    token = get_access_token()
    
    save_dir = "stock_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if token:
        total_count = len(target_codes)
        print(f"✅ 총 {total_count}개 종목 수집 시작 (불사신 모드 🛡️ + 보안 강화 🔒)\n")
        
        for idx, code in enumerate(target_codes):
            # 이미 파일이 있으면 건너뛰기 기능
            filename = f"{save_dir}/{code}_1Year.csv"
            if os.path.exists(filename):
                print(f"[{idx+1}/{total_count}] {code} 이미 수집됨. 건너뜀 ->")
                continue

            print(f"[{idx+1}/{total_count}] 수집 시작: {code} ...")
            
            all_data = []
            current_date = "20251211"
            current_time = "153000"
            target_limit = "20241211" 

            try:
                while True:
                    # 안전한 함수 호출
                    chunk = get_minute_chart_safe(token, code, current_date, current_time)
                    
                    if not chunk: 
                        break
                    
                    all_data.extend(chunk)
                    last_row = chunk[-1]
                    next_date = last_row['stck_bsop_date']
                    next_time = last_row['stck_cntg_hour']
                    
                    if next_date < target_limit: break
                    if next_date == current_date and next_time == current_time: break
                    
                    current_date = next_date
                    current_time = next_time
                    
                    # 진행률 표시
                    print(f"\r   Reading: {current_date} {current_time} ({len(all_data)} rows)", end="")
                    time.sleep(0.05) # 부하 조절
                
                print() # 줄바꿈

                if all_data:
                    df = pd.DataFrame(all_data)
                    col_map = {"stck_bsop_date": "Date", "stck_cntg_hour": "Time", "stck_prpr": "Close", "cntg_vol": "Volume"}
                    df = df[col_map.keys()].rename(columns=col_map)
                    df = df.sort_values(by=['Date', 'Time']).reset_index(drop=True)
                    df.drop_duplicates(subset=['Date', 'Time'], inplace=True)
                    
                    df.to_csv(filename, index=False)
                    print(f"   💾 저장 완료: {filename} (총 {len(df)}건)")
                else:
                    print(f"   ⚠️ 데이터 없음: {code}")

            except Exception as e:
                print(f"\n   ❌ 치명적 오류 발생 ({code}): {e}")
            
            print("-" * 40)
            time.sleep(0.5)

        print("\n🎉 모든 종목 수집이 완료되었습니다!")
    else:
        print("토큰 발급 실패: .env 키 값을 확인하세요.")