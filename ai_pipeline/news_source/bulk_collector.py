import sys
import os
import time

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.news_source.finance_news_list import fetch_finance_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.nlp.text_splitter import split_text
from ai_pipeline.nlp.sentiment import analyze_sentiment
from ai_pipeline.mapping.stock_mapping import get_stock_mentions
from ai_pipeline.news_etl.es_uploader import save_news_to_es, exists_in_es

def run_bulk_collection(target_pages=50):
    """
    과거 뉴스 대량 수집 (Backfill)
    target_pages: 수집할 페이지 수 (예: 50페이지 = 약 1,000개 기사)
    """
    print("\n" + "="*60)
    print(f"📚 [과거 데이터 확보] 대량 수집 시작 (목표: {target_pages}페이지)")
    print("="*60)

    # 1. URL 대량 확보
    # 기존 함수를 재활용하되, 페이지 수를 크게 잡음
    news_urls = fetch_finance_news_list(max_pages=target_pages)
    
    print(f"\n📌 확보된 URL 후보: {len(news_urls)}개")
    print("   이제 본문 크롤링 및 분석을 시작합니다... (시간이 좀 걸립니다)")
    
    saved_count = 0
    skipped_count = 0

    for idx, finance_url in enumerate(news_urls):
        # 진행 상황 표시 (10개마다 로그)
        if idx % 10 == 0:
            print(f"   ... 진행률 {idx}/{len(news_urls)} ({idx/len(news_urls)*100:.1f}%)")

        real_url = extract_real_article_url(finance_url)
        
        # 중복 체크 (이미 스케줄러가 수집한 건 패스)
        if exists_in_es(real_url):
            skipped_count += 1
            continue

        # 본문 수집
        article_text = fetch_article_text(real_url)
        if not article_text or len(article_text) < 100:
            continue

        # 종목 매핑 (종목 없으면 버림)
        related_stocks = get_stock_mentions(article_text)
        if not related_stocks:
            continue

        # 분석 및 저장
        chunks = split_text(article_text)
        sentiments = analyze_sentiment(chunks)
        save_news_to_es(real_url, article_text, chunks, sentiments, related_stocks)
        
        saved_count += 1
        
        # 네이버 차단 방지를 위해 약간의 텀 (0.1초)
        time.sleep(0.1)

    print("\n" + "="*60)
    print(f"✅ 대량 수집 완료!")
    print(f"   - 저장된 뉴스: {saved_count}건")
    print(f"   - 중복/스킵: {skipped_count}건")
    print("="*60)
    print("👉 이제 pipeline_main.py를 실행해서 그래프를 다시 구축하세요!")

if __name__ == "__main__":
    # 원하는 페이지 수 설정 (보통 1페이지당 20개)
    # 50페이지 = 약 1,000개 뉴스 (최근 2~3일치)
    run_bulk_collection(target_pages=50)