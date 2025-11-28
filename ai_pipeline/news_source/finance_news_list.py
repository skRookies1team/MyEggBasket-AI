import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta

# 기업/종목 분석 뉴스 기본 URL (날짜, 페이지 제외)
BASE_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402"

def fetch_daily_news_list(target_date, max_pages=10):
    """
    특정 날짜(YYYYMMDD)의 뉴스를 1~max_pages까지 수집합니다.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Referer": "https://finance.naver.com"
    }

    daily_urls = []
    
    # 1페이지부터 max_pages(10)까지 순회
    for page in range(1, max_pages + 1):
        # URL 조합: 기본URL + 날짜 + 페이지
        current_url = f"{BASE_URL}&date={target_date}&page={page}"
        
        try:
            res = requests.get(current_url, headers=headers)
            res.encoding = 'EUC-KR'
            
            if res.status_code != 200: 
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            
            # 뉴스 링크 추출 (기업분석 섹션 구조: dl > dd.articleSubject > a)
            # 또는 썸네일 없는 기사: dl > dt.articleSubject > a
            links = soup.select(".articleSubject a")
            
            if not links:
                # 링크가 없으면 해당 날짜의 페이지 끝임 -> 중단
                break
                
            count = 0
            for link in links:
                href = link.get("href")
                if href and "news_read" in href:
                    href = href.replace("§", "&sect")
                    full_url = "https://finance.naver.com" + href
                    daily_urls.append(full_url)
                    count += 1
            
            # 페이지에 뉴스가 없거나 적으면(마지막 페이지) 중단
            if count == 0:
                break
            
            # 차단 방지 딜레이
            time.sleep(random.uniform(0.1, 0.3))

        except Exception:
            continue

    # 중복 제거
    return list(set(daily_urls))

# (기존 함수와의 호환성을 위해 남겨두되, 이번 작업엔 안 씁니다)
def fetch_finance_news_list(max_pages=1):
    return []