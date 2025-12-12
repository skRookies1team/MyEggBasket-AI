import os
import time
import pandas as pd
import requests
import re
import random
import uuid
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException

# 1. 설정
DATA_PATH = "data/data_2218_20251128.csv"
DOWNLOAD_DIR = os.path.abspath("data/reports")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 수집할 타겟 종목 50개
TARGET_CODES = [
    "005930", "000660", "207940", "005380", "000270", "055550", "105560", "068270", "015760", "028260",
    "032830", "012330", "035420", "006400", "086790", "006405", "000810", "010140", "064350", "138040",
    "051910", "010130", "009540", "267260", "066570", "066575", "033780", "003550", "003555", "310200",
    "034020", "012450", "009830", "011070", "071050", "081660", "046890", "323410", "017670", "010620",
    "047050", "009155", "275630", "009835", "001440", "138930", "175330", "051900", "092740", "034220"
]
TARGET_CODES = [code.strip() for code in TARGET_CODES]

def load_stock_list():
    if not os.path.exists(DATA_PATH):
        print(f" 데이터 파일이 없습니다: {DATA_PATH}")
        return []
    try:
        try:
            df = pd.read_csv(DATA_PATH, encoding='cp949', dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATH, encoding='utf-8', dtype=str)
        filtered_df = df[df['단축코드'].isin(TARGET_CODES)]
        return filtered_df[['한글 종목명', '단축코드']].values.tolist()
    except Exception as e:
        print(f" CSV 로드 실패: {e}")
        return []

def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "", text).strip()

def clean_stock_name(name):
    name = name.split("(")[0]
    remove_list = ["보통주", "우선주", "1우선주", "2우선주", "주식회사", "홀딩스", "금융지주", "지주", "그룹"]
    for word in remove_list:
        name = name.replace(word, "")
    return name.strip()

def download_pdf(url, filename):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=20)
        if response.status_code == 200:
            file_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"    다운로드 실패: {e}")
    return False

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def parse_date(date_text):
    """날짜 포맷을 유연하게 처리 (24.05.21 또는 2024.05.21)"""
    for fmt in ["%y.%m.%d", "%Y.%m.%d"]:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    return None

def crawl_reports(years=3): # [변경] 넉넉하게 3년치로 설정 (24년도 누락 방지)
    stocks = load_stock_list()
    if not stocks: return

    cutoff_date = datetime.now() - timedelta(days=365 * years)
    print(f"📅 수집 기준일: {cutoff_date.strftime('%Y-%m-%d')} ~ 현재")
    print(f"🚀 [진단 모드] 왜 안 받아지는지 로그를 확인하세요!")
    
    driver = get_driver()
    total_download = 0
    
    for idx, (name, target_code) in enumerate(stocks):
        clean_name = clean_stock_name(name)
        target_code = str(target_code).zfill(6)
        
        stock_download_count = 0
        print(f"\n[{idx+1}/{len(stocks)}] '{clean_name}' ({target_code}) 탐색 중...")
        
        stop_crawling = False
        try: driver.current_window_handle
        except: driver = get_driver()

        for page in range(1, 101):
            if stop_crawling: break

            search_url = f"https://finance.naver.com/research/company_list.naver?keyword={clean_name}&page={page}"
            
            try:
                driver.get(search_url)
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.type_1")))
                
                rows = driver.find_elements(By.CSS_SELECTOR, "table.type_1 tr")
                if len(rows) < 3: break 

                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if not cells or len(cells) < 5: continue
                    
                    real_stock_name = cells[0].text.strip()
                    date_text = cells[4].text.strip()
                    raw_title = cells[1].text.strip()

                    # [1. 날짜 확인]
                    report_date = parse_date(date_text)
                    if not report_date:
                        continue 
                    
                    if report_date < cutoff_date:
                        print(f"   🛑 {date_text} 도달 (날짜 지남). 다음 종목으로.")
                        stop_crawling = True
                        break

                    # [2. 종목 검증 - 유연하게]
                    is_match = False
                    
                    # 2-1. 코드 확인
                    try:
                        link_elm = cells[0].find_element(By.TAG_NAME, "a")
                        href = link_elm.get_attribute("href")
                        found_code = re.search(r'code=(\d+)', href)
                        if found_code and found_code.group(1) == target_code:
                            is_match = True
                    except: pass
                    
                    # 2-2. 이름 확인 (코드 실패했을 경우 대비)
                    if not is_match:
                        if (clean_name in real_stock_name) or (real_stock_name in clean_name):
                            is_match = True

                    if not is_match:
                        # 디버깅: 왜 안 받는지 출력
                        # print(f"   🚫 [Skip] 이름 불일치: 내꺼({clean_name}) != 표({real_stock_name})")
                        continue

                    # 다운로드 준비
                    pdf_link_elm = row.find_elements(By.CSS_SELECTOR, "td.file a")
                    if not pdf_link_elm: continue
                    pdf_url = pdf_link_elm[0].get_attribute("href")

                    safe_title = clean_filename(raw_title)
                    date_str = date_text.replace('.', '')
                    unique_id = uuid.uuid4().hex[:4]
                    
                    file_name = f"{target_code}_{clean_name}_{date_str}_{safe_title}_{unique_id}.pdf"
                    if len(file_name) > 150:
                        file_name = f"{target_code}_{clean_name}_{date_str}_{safe_title[:20]}_{unique_id}.pdf"

                    if download_pdf(pdf_url, file_name):
                        stock_download_count += 1
                        total_download += 1
                        # 10개마다 로그
                        if stock_download_count % 10 == 0:
                            print(f"   ...수집중: {date_text} | {safe_title[:15]}...")

            except TimeoutException: continue
            except WebDriverException:
                try: driver.quit()
                except: pass
                driver = get_driver()
                break
            except Exception as e:
                print(f"   ❌ 에러: {e}")
                break
        
        print(f"   👉 [{clean_name}] 총 {stock_download_count}개 저장됨")
            
    driver.quit()
    print(f"\n✅ 전체 완료! 총 {total_download}개 파일 저장됨.")

if __name__ == "__main__":
    # 혹시 몰라서 3년치로 늘려뒀습니다. (24년도 무조건 포함되게)
    crawl_reports(years=3)