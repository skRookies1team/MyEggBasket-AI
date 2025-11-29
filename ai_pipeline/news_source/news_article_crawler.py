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
        time.sleep(random.uniform(0.3, 1.0)) # 대기 시간
        
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
        # 🚨 [핵심 추가] 자바스크립트 리다이렉트 감지 및 처리
        # ============================================================
        # <SCRIPT>top.location.href='...';</SCRIPT> 패턴이 있는지 검사
        if "top.location.href" in text_data:
            print("🔄 리다이렉트 감지! 실제 기사 페이지로 이동합니다...")
            
            # URL 추출 (작은 따옴표 안의 주소를 꺼냄)
            match = re.search(r"top\.location\.href='(.*?)'", text_data)
            if match:
                new_url = match.group(1)
                print(f"   ➡ 이동할 주소: {new_url}")
                
                # [재귀 호출] 추출한 새 주소로 다시 크롤링 시도
                return fetch_article_text(new_url)
        # ============================================================

        soup = BeautifulSoup(text_data, "html.parser")
        content = None
        
        # 태그 찾기 (우선순위별)
        # 1. 네이버 금융 본문
        candidate = soup.select_one(".articleCont")
        if candidate:
            for tag in candidate.select(".link_news, .guide, script"): # 불필요 태그 삭제
                tag.decompose()
            content = candidate.get_text(separator=" ", strip=True)

        # 2. 네이버 뉴스(일반) 본문 (리다이렉트 된 곳은 보통 여기임)
        if not content:
            candidate = soup.select_one("#dic_area")
            if candidate:
                for tag in candidate.select(".img_desc, .end_photo_org"):
                    tag.decompose()
                content = candidate.get_text(separator=" ", strip=True)

        # 3. 스포츠/연예 등 기타
        if not content:
            candidate = soup.select_one("#articeBody")
            if candidate:
                content = candidate.get_text(separator=" ", strip=True)
        
        # 4. 구형 레이아웃
        if not content:
            candidate = soup.select_one("#content")
            if candidate:
                content = candidate.get_text(separator=" ", strip=True)

        if content:
            # 너무 짧으면 광고나 에러일 수 있음
            if len(content) < 30: 
                return None
            return content
        else:
            # 리다이렉트도 아니고, 본문도 없으면 진짜 실패
            print("❌ 본문 태그 찾기 실패")
            return None

    except Exception as e:
        print(f"❌ 크롤링 에러: {e}")
        return None
    
    