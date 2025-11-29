import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime

# 기업/종목 분석 뉴스 기본 URL
BASE_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"

def fetch_daily_news_list(target_date, max_pages=10):
    """
    특정 날짜(YYYYMMDD)의 뉴스를 수집하는 핵심 함수
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Referer": "https://finance.naver.com"
    }

    daily_urls = []
    
    print(f"   Searching Date: {target_date}")

    for page in range(1, max_pages + 1):
        # URL 조합: 기본URL + 날짜 + 페이지
        current_url = f"{BASE_URL}&date={target_date}&page={page}"
        
        try:
            res = requests.get(current_url, headers=headers)
            res.encoding = 'EUC-KR'
            
            if res.status_code != 200: 
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            
            # 뉴스 링크 추출 (기업분석 섹션 구조)
            # 1. 썸네일 있는 기사 (dl > dd.articleSubject > a)
            # 2. 썸네일 없는 기사 (dl > dt.articleSubject > a)
            links = soup.select(".articleSubject a")
            
            if not links:
                break # 링크 없으면 끝
                
            count = 0
            for link in links:
                href = link.get("href")
                if href and "news_read" in href:
                    href = href.replace("§", "&sect")
                    full_url = "https://finance.naver.com" + href
                    daily_urls.append(full_url)
                    count += 1
            
            if count == 0:
                break
            
            time.sleep(random.uniform(0.1, 0.3))

        except Exception:
            continue

    return list(set(daily_urls))

def fetch_finance_news_list(max_pages=5):
    """
    [스케줄러용] 오늘 날짜의 뉴스를 수집합니다.
    """
    # 오늘 날짜 구하기 (YYYYMMDD)
    today = datetime.now().strftime("%Y%m%d")
    
    print(f"🔍 [스케줄러] 오늘의 기업/종목 분석 뉴스 수집 시작 ({today})")
    
    # 위에서 만든 날짜별 수집 함수 재활용
    urls = fetch_daily_news_list(today, max_pages)
    
    print(f"🔥 총 수집된 뉴스 URL: {len(urls)}개")
    return urls