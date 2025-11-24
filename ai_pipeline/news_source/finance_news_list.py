import requests
from bs4 import BeautifulSoup

# 실시간 속보 리스트
FINANCE_NEWS_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258"

def fetch_finance_news_list():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Referer": "https://finance.naver.com"
    }

    res = requests.get(FINANCE_NEWS_URL, headers=headers)
    res.encoding = 'EUC-KR'  # 인코딩 고정

    if res.status_code != 200:
        print("❌ 네이버 금융 뉴스 페이지 요청 실패:", res.status_code)
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.select("dl .articleSubject a")

    urls = []
    for link in links:
        href = link.get("href")
        if href and "news_read" in href:
            # [수정됨] § 기호를 &sect (section)으로 정확히 복구
            href = href.replace("§", "&sect")
            
            full_url = "https://finance.naver.com" + href
            urls.append(full_url)

    urls = list(set(urls))
    print(f"🔥 Finance 페이지에서 수집된 뉴스 URL: {len(urls)}개")
    return urls