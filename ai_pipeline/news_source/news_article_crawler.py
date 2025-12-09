import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime

def extract_real_article_url(finance_url):
    """
    네이버 금융 뉴스 URL이 리다이렉트가 필요한 경우 처리
    (현재는 그대로 반환하지만, 추후 확장 가능성 유지)
    """
    return finance_url

def fetch_article_text(url):
    """
    뉴스 기사 URL에서 제목, 본문, 작성시간을 추출하여 반환
    Returns: (title, content, published_date) 튜플
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://finance.naver.com/"
    }
    
    try:
        # 랜덤 대기 (차단 방지)
        time.sleep(random.uniform(0.2, 0.5)) 
        
        response = requests.get(url, headers=headers)
        
        # [1] 인코딩 자동 감지 및 설정
        content_type = response.headers.get('content-type', '').lower()
        if 'charset=euc-kr' in content_type or b'charset=euc-kr' in response.content.lower():
            response.encoding = 'euc-kr'
        else:
            response.encoding = 'utf-8'

        text_data = response.text

        # [2] 자바스크립트 리다이렉트 처리 (금융 뉴스 -> 본 뉴스 이동 시)
        if "top.location.href" in text_data:
            match = re.search(r"top\.location\.href='(.*?)'", text_data)
            if match:
                return fetch_article_text(match.group(1))

        soup = BeautifulSoup(text_data, "html.parser")
        
        # ---------------------------------------------------------
        # 1. 제목 추출
        # ---------------------------------------------------------
        title = ""
        title_tag = soup.select_one(".article_info h3, #articleTitle, h2.end_tit")
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            meta_title = soup.select_one("meta[property='og:title']")
            if meta_title: 
                title = meta_title['content']

        # ---------------------------------------------------------
        # 2. 날짜 추출 (시:분:초 포함)
        # ---------------------------------------------------------
        published_date = None
        
        # (A) 모바일/PC 공통 (가장 정확한 data-date-time 속성 확인)
        date_tag = soup.select_one("span.media_end_head_info_datestamp_time")
        if date_tag:
            if date_tag.has_attr('data-date-time'):
                # 예: "2024-12-09 10:23:01" -> "2024-12-09T10:23:01"
                published_date = date_tag['data-date-time'].replace(" ", "T")
            else:
                published_date = _parse_date_text(date_tag.get_text(strip=True))

        # (B) 네이버 금융 뉴스 구버전 레이아웃
        if not published_date:
            date_tag = soup.select_one(".article_info span.t11")
            if date_tag:
                published_date = _parse_date_text(date_tag.get_text(strip=True))
        
        # (C) 실패 시 현재 시간 (Fallback)
        if not published_date:
             published_date = datetime.now().isoformat()

        # ---------------------------------------------------------
        # 3. 본문 추출
        # ---------------------------------------------------------
        content = None
        
        # (1) 네이버 금융
        candidate = soup.select_one(".articleCont") 
        if candidate:
            for tag in candidate.select(".link_news, .guide, script, .img_desc"): 
                tag.decompose()
            content = candidate.get_text(separator=" ", strip=True)

        # (2) 네이버 뉴스 (일반)
        if not content:
            candidate = soup.select_one("#dic_area") 
            if candidate:
                for tag in candidate.select(".img_desc, .end_photo_org"): 
                    tag.decompose()
                content = candidate.get_text(separator=" ", strip=True)
        
        # (3) 기타 fallback
        if not content:
            candidate = soup.select_one("#articeBody, #content") 
            if candidate: 
                content = candidate.get_text(separator=" ", strip=True)

        # ---------------------------------------------------------
        # 결과 반환
        # ---------------------------------------------------------
        if title and content and len(content) >= 30:
            return title, content, published_date
        else:
            return None

    except Exception as e:
        print(f"Crawling Error ({url}): {e}")
        return None

def _parse_date_text(date_text):
    """
    텍스트 형식의 날짜('2025.12.09. 오전 10:23')를 ISO 포맷으로 변환
    """
    try:
        is_pm = '오후' in date_text
        # 정규식으로 한글 및 불필요 문자 제거
        clean_text = re.sub(r'[오전|오후|기사입력]', '', date_text).strip()
        
        # 날짜 파싱
        dt = datetime.strptime(clean_text, "%Y.%m.%d. %H:%M")
        
        # 시간 보정
        if is_pm and dt.hour != 12: 
            dt = dt.replace(hour=dt.hour + 12)
        elif not is_pm and '오전' in date_text and dt.hour == 12: 
            dt = dt.replace(hour=0)
            
        return dt.isoformat()
    except:
        return datetime.now().isoformat()