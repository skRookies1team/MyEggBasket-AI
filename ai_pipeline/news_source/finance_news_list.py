import requests
from bs4 import BeautifulSoup
import time
import random

# 네이버 금융 뉴스 (실시간 속보) 기본 URL
BASE_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258"

def fetch_finance_news_list(max_pages=1):
    """
    네이버 금융 뉴스 리스트를 수집합니다.
    max_pages: 수집할 페이지 수 (기본값 1)
    """
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Referer": "https://finance.naver.com"
    }

    all_urls = []
    
    print(f"🔍 뉴스 리스트 수집 시작 (목표: {max_pages}페이지)...")

    for page in range(1, max_pages + 1):
        # URL 뒤에 &page=숫자 붙이기
        target_url = f"{BASE_URL}&page={page}"
        
        # 로그가 너무 많이 뜨면 지저분하니까 1페이지랑 10페이지 단위만 출력
        if page == 1 or page % 10 == 0:
            print(f"   Reading Page {page}...", end=" ")

        try:
            res = requests.get(target_url, headers=headers)
            res.encoding = 'EUC-KR' # 한글 깨짐 방지
            
            if res.status_code != 200:
                print(f"❌ 실패 ({res.status_code})")
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            
            # 뉴스 링크 추출 (제목에 걸린 링크)
            links = soup.select("dl .articleSubject a")
            
            count = 0
            for link in links:
                href = link.get("href")
                if href and "news_read" in href:
                    # 특수문자 깨짐 복구 (§ -> &sect)
                    href = href.replace("§", "&sect")
                    
                    full_url = "https://finance.naver.com" + href
                    all_urls.append(full_url)
                    count += 1
            
            if page == 1 or page % 10 == 0:
                print(f"✅ {count}개 추출")
            
            # 페이지 넘길 때마다 랜덤하게 쉬어주기 (차단 방지)
            time.sleep(random.uniform(0.3, 0.7))

        except Exception as e:
            print(f"\n❌ 에러 발생 (Page {page}): {e}")

    # 전체 중복 제거
    unique_urls = list(set(all_urls))
    print(f"🔥 총 수집된 고유 뉴스 URL: {len(unique_urls)}개")
    
    return unique_urls