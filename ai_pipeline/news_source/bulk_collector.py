import sys
import os
import time
from datetime import datetime, timedelta

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.news_source.finance_news_list import fetch_daily_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.nlp.text_splitter import split_text
from ai_pipeline.nlp.sentiment import analyze_sentiment
from ai_pipeline.mapping.stock_mapping import get_stock_mentions
from ai_pipeline.news_etl.es_uploader import save_news_to_es, exists_in_es

def run_date_range_collection(start_date_str, end_date_str):
    """
    기간별 대량 수집 실행기
    start_date_str: "2025-11-29"
    end_date_str: "2025-12-03"
    """
    print("\n" + "="*60)
    print(f"📚 [Bulk] 기간별 기업분석 뉴스 수집 시작")
    print(f"   📅 기간: {start_date_str} ~ {end_date_str}")
    print("="*60)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    current_date = start_date
    total_saved = 0

    while current_date <= end_date:
        target_date = current_date.strftime("%Y%m%d") # YYYYMMDD 형식
        print(f"\n📅 [Date: {target_date}] 뉴스 수집 중...", end=" ")
        
        # 1. 해당 날짜의 URL 목록 가져오기 (최대 10페이지)
        daily_urls = fetch_daily_news_list(target_date, max_pages=10)
        print(f"👉 {len(daily_urls)}개 발견", end="")
        
        if not daily_urls:
            current_date += timedelta(days=1)
            continue

        # 2. 크롤링 및 저장
        daily_saved = 0
        for url in daily_urls:
            real_url = extract_real_article_url(url)
            
            # 중복 체크
            if exists_in_es(real_url): continue

            # 본문 수집
            text = fetch_article_text(real_url)
            if not text or len(text) < 100: continue

            # 종목 매핑 (종목 없으면 스킵)
            stocks = get_stock_mentions(text)
            if not stocks: continue

            # 분석 및 저장
            chunks = split_text(text)
            sentiments = analyze_sentiment(chunks)
            save_news_to_es(real_url, text, chunks, sentiments, stocks)
            daily_saved += 1
            
            # 진행 표시 (.)
            print(".", end="", flush=True) 
            time.sleep(0.05) # 너무 빠르면 차단됨

        print(f" ✅ {daily_saved}건 저장 완료")
        total_saved += daily_saved
        
        # 하루 넘기기
        current_date += timedelta(days=1)

    print("\n" + "="*60)
    print(f"🎉 전체 기간 수집 완료! 총 저장된 뉴스: {total_saved}건")
    print("="*60)

if __name__ == "__main__":
    # 2025년 10월 1일 ~ 오늘(2025-11-28)
    run_date_range_collection("2025-11-29", "2025-12-03")