import requests
from bs4 import BeautifulSoup
import time
import random
import re  # 정규표현식 모듈 추가 (URL 추출용)

def extract_real_article_url(finance_url):
    return finance_url

def fetch_article_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://finance.naver.com/"
    }
    
    try:
        time.sleep(random.uniform(0.2, 0.5)) # 대기 시간
        
        response = requests.get(url, headers=headers)
        
        # [1] 인코딩 처리 (금융은 EUC-KR, 일반 뉴스는 UTF-8)
        # 일단 EUC-KR로 시도해보고, 나중에 깨지면 UTF-8로 전환하는 방식도 가능하지만,
        # charset 체크가 가장 안전함.
        content_type = response.headers.get('content-type', '').lower()
        if 'charset=euc-kr' in content_type or b'charset=euc-kr' in response.content.lower():
            response.encoding = 'euc-kr'
        else:
            response.encoding = 'utf-8'

        text_data = response.text

        # ============================================================
        #  [핵심 추가] 자바스크립트 리다이렉트 감지 및 처리
        # ============================================================
        # <SCRIPT>top.location.href='...';</SCRIPT> 패턴이 있는지 검사
        if "top.location.href" in text_data:
            match = re.search(r"top\.location\.href='(.*?)'", text_data)
            if match:
                return fetch_article_text(match.group(1))

        soup = BeautifulSoup(text_data, "html.parser")
        content = None
        
        # 태그 찾기 (우선순위별)

        # 1. 제목 추출 
        title = ""
        title_tag = soup.select_one(".article_info h3, #articleTitle, h2.end_tit")
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            meta_title = soup.select_one("meta[property='og:title']")
            if meta_title:
                title = meta_title['content']


        # 1. 네이버 금융 본문
        content = None
        # 1. 네이버 금융
        candidate = soup.select_one(".articleCont")
        if candidate:
            for tag in candidate.select(".link_news, .guide, script, .img_desc"):
                tag.decompose()
            content = candidate.get_text(separator=" ", strip=True)

        # 2. 네이버 뉴스 (일반)
        if not content:
            candidate = soup.select_one("#dic_area")
            if candidate:
                for tag in candidate.select(".img_desc, .end_photo_org"):
                    tag.decompose()
                content = candidate.get_text(separator=" ", strip=True)
        
        # 3. 기타
        if not content:
            candidate = soup.select_one("#articeBody, #content")
            if candidate:
                content = candidate.get_text(separator=" ", strip=True)

        # --- [핵심 수정] 튜플(제목, 본문) 반환 ---
        if title and content and len(content) >= 30:
            return title, content  # (제목, 본문) 튜플 반환
        else:
            return None

    except Exception as e:
        # print(f" 크롤링 에러: {e}") # 로그가 너무 많으면 주석 처리
        return None
    
    