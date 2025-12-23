import requests
from bs4 import BeautifulSoup
import time
import random
import re
from datetime import datetime

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
        time.sleep(random.uniform(0.2, 0.5)) 
        response = requests.get(url, headers=headers)
        
        # 인코딩 처리
        content_type = response.headers.get('content-type', '').lower()
        if 'charset=euc-kr' in content_type or b'charset=euc-kr' in response.content.lower():
            response.encoding = 'euc-kr'
        else:
            response.encoding = 'utf-8'

        text_data = response.text

        # 리다이렉트 처리
        if "top.location.href" in text_data:
            match = re.search(r"top\.location\.href='(.*?)'", text_data)
            if match:
                return fetch_article_text(match.group(1))

        soup = BeautifulSoup(text_data, "html.parser")
        
        # 1. 제목
        title = ""
        title_tag = soup.select_one(".article_info h3, #articleTitle, h2.end_tit")
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            meta_title = soup.select_one("meta[property='og:title']")
            if meta_title: title = meta_title['content']

        # 2. 날짜
        published_date = None
        date_tag = soup.select_one("span.media_end_head_info_datestamp_time")
        if date_tag:
            if date_tag.has_attr('data-date-time'):
                 published_date = date_tag['data-date-time'].replace(" ", "T")
            else:
                 published_date = _parse_date_text(date_tag.get_text(strip=True))

        if not published_date:
            date_tag = soup.select_one(".article_info span.t11")
            if date_tag:
                published_date = _parse_date_text(date_tag.get_text(strip=True))
        
        if not published_date:
             published_date = datetime.now().isoformat()

        # 3. 본문
        content = None
        candidate = soup.select_one(".articleCont") # 금융
        if candidate:
            for tag in candidate.select(".link_news, .guide, script, .img_desc"): tag.decompose()
            content = candidate.get_text(separator=" ", strip=True)

        if not content:
            candidate = soup.select_one("#dic_area") # 뉴스
            if candidate:
                for tag in candidate.select(".img_desc, .end_photo_org"): tag.decompose()
                content = candidate.get_text(separator=" ", strip=True)
        
        if not content:
            candidate = soup.select_one("#articeBody, #content") # 기타
            if candidate: content = candidate.get_text(separator=" ", strip=True)

        if title and content and len(content) >= 30:
            return title, content, published_date
        else:
            return None

    except Exception as e:
        print(f"Crawling Error: {e}")
        return None

def _parse_date_text(date_text):
    try:
        is_pm = '오후' in date_text
        clean_text = re.sub(r'[오전|오후|기사입력]', '', date_text).strip()
        dt = datetime.strptime(clean_text, "%Y.%m.%d. %H:%M")
        if is_pm and dt.hour != 12: dt = dt.replace(hour=dt.hour + 12)
        elif not is_pm and '오전' in date_text and dt.hour == 12: dt = dt.replace(hour=0)
        return dt.isoformat()
    except:
        return datetime.now().isoformat()