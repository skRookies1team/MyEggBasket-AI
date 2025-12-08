import os
import time
import pandas as pd
import requests
import re
import random
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException

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
        print(f"❌ 데이터 파일이 없습니다: {DATA_PATH}")
        return []
    try:
        try:
            df = pd.read_csv(DATA_PATH, encoding='cp949', dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATH, encoding='utf-8', dtype=str)
        
        filtered_df = df[df['단축코드'].isin(TARGET_CODES)]
        print(f"📊 타겟 종목 {len(filtered_df)}개 로드 완료.")
        return filtered_df[['한글 종목명', '단축코드']].values.tolist()
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return []

def clean_filename(text):
    """파일명에 사용할 수 없는 특수문자 제거"""
    return re.sub(r'[\\/*?:"<>|]', "", text).strip()

def clean_stock_name(name):
    name = name.split("(")[0]
    remove_list = ["보통주", "우선주", "1우선주", "2우선주", "3우선주", "주식회사", "홀딩스"]
    for word in remove_list:
        name = name.replace(word, "")
    return name.strip()

def download_pdf(url, filename):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200:
            file_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"   ⚠️ 다운로드 실패: {e}")
    return False

def get_driver():
    """크롬 드라이버 생성 및 옵션 설정"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") # 화면 숨김
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def crawl_reports(years=2): 
    stocks = load_stock_list()
    if not stocks: return

    # 수집 제한 날짜 (오늘 - N년)
    cutoff_date = datetime.now() - timedelta(days=365 * years)
    print(f"📅 기준일: {cutoff_date.strftime('%Y-%m-%d')} 이후 리포트만 수집")
    print(f"🚀 [재접속 기능 포함] {len(stocks)}개 종목 수집 시작...")
    
    driver = get_driver() # 브라우저 켜기
    total_download = 0
    
    for idx, (name, code) in enumerate(stocks):
        clean_name = clean_stock_name(name)
        code = str(code).zfill(6)
        
        stock_download_count = 0
        print(f"\n[{idx+1}/{len(stocks)}] '{clean_name}' ({code}) 검색 중...")
        
        stop_crawling = False

        # [안전장치] 브라우저 세션이 유효한지 확인하고, 죽었으면 살려냄
        try:
            # 테스트로 현재 윈도우 핸들을 가져와봄 (에러나면 죽은 것)
            driver.current_window_handle
        except:
            print("   🔄 브라우저 재시작 중...")
            try:
                driver.quit()
            except:
                pass
            driver = get_driver()

        # 페이지 순회
        for page in range(1, 101):
            if stop_crawling: break

            search_url = f"https://finance.naver.com/research/company_list.naver?keyword={clean_name}&page={page}"
            
            try:
                driver.get(search_url)
                time.sleep(random.uniform(0.3, 0.6))
                
                rows = driver.find_elements(By.CSS_SELECTOR, "table.type_1 tr")
                if len(rows) < 3: break # 빈 페이지면 끝

                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if not cells or len(cells) < 5: continue
                    
                    # 날짜 확인
                    date_text = cells[4].text.strip() # 예: 24.05.21
                    try:
                        report_date = datetime.strptime(date_text, "%y.%m.%d")
                        if report_date < cutoff_date:
                            print(f"   🛑 {date_text} 도달 (오래된 자료). 다음 종목으로.")
                            stop_crawling = True
                            break
                    except ValueError:
                        continue

                    # PDF 링크
                    pdf_link_elm = row.find_elements(By.CSS_SELECTOR, "td.file a")
                    if not pdf_link_elm: continue
                    
                    pdf_url = pdf_link_elm[0].get_attribute("href")
                    title = cells[1].text.strip()
                    safe_title = clean_filename(title)
                    date_str = date_text.replace('.', '')
                    
                    file_name = f"{code}_{clean_name}_{date_str}_{safe_title}.pdf"
                    if len(file_name) > 150:
                        file_name = f"{code}_{clean_name}_{date_str}_{safe_title[:30]}.pdf"

                    # [중복 체크] 이미 받은 파일이면 스킵 (재실행 시 속도 UP)
                    if os.path.exists(os.path.join(DOWNLOAD_DIR, file_name)):
                        continue

                    if download_pdf(pdf_url, file_name):
                        stock_download_count += 1
                        total_download += 1
                        if stock_download_count % 5 == 0:
                            print(f"   ...{stock_download_count}개째 저장 ({date_str})")

            except WebDriverException as e:
                # 크롬이 죽었을 때 처리
                print(f"   ⚠️ 브라우저 에러 발생: {e}")
                print("   🔄 브라우저를 재시작합니다...")
                try:
                    driver.quit()
                except:
                    pass
                driver = get_driver() # 새 창 열기
                time.sleep(2)
                # 현재 페이지 다시 시도하려면 page -= 1 해야 하지만, 복잡해지니 다음 페이지/종목으로 넘어감
                break
                
            except Exception as e:
                print(f"   ❌ 일반 에러: {e}")
                break
        
        print(f"   👉 총 {stock_download_count}개 신규 저장")
            
    driver.quit()
    print(f"\n✅ 전체 작업 완료! 총 {total_download}개 파일이 저장되었습니다.")

if __name__ == "__main__":
    crawl_reports(years=2)