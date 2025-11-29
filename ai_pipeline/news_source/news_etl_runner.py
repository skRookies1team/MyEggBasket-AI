import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)


from ai_pipeline.news_source.finance_news_list import fetch_finance_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.nlp.text_splitter import split_text
from ai_pipeline.nlp.sentiment import analyze_sentiment
from ai_pipeline.news_etl.es_uploader import save_news_to_es, exists_in_es
from ai_pipeline.mapping.stock_mapping import get_stock_mentions

def run_finance_news_etl():
    print("🔥 ETL 시작됨")

    news_urls = fetch_finance_news_list()
    print("📌 수집된 URL 개수:", len(news_urls))

    if len(news_urls) == 0:
        print("❌ 뉴스 URL이 0개입니다. selector나 User-Agent 문제입니다.")
        return
    
    saved_count = 0
    skipped_dup = 0
    skipped_irrelevant = 0

    for idx, finance_url in enumerate(news_urls):
        real_url = extract_real_article_url(finance_url)
        print(f"➡ 실제 기사 URL: {real_url}")

        if exists_in_es(real_url):
            # print(f"   ⏭️ [Skip] 이미 수집된 뉴스") # 로그 너무 많으면 주석
            skipped_dup += 1
            continue

        print(f"[{idx+1}/{len(news_urls)}] 🆕 신규 뉴스 분석 중: {real_url}")
       

        article_text = fetch_article_text(real_url)
        if not article_text:
            print("❌ 본문 없음 → 스킵")
            continue

        if len(article_text) < 100:
             print("⚠️ 본문이 너무 짧음(단신) → 스킵")
             continue
        
        related_stocks = get_stock_mentions(article_text)

        if not related_stocks:
            print("   🗑️ [Pass] 주식 종목 언급 없음 (일반 경제 뉴스)")
            skipped_irrelevant += 1
            continue

        chunks = split_text(article_text)
        sentiments = analyze_sentiment(chunks)

        print(f"   ✅ 관련 종목 발견: {related_stocks}")
        save_news_to_es(real_url, article_text, chunks, sentiments, related_stocks)
        saved_count += 1

    print("\n" + "="*40)
    print(f"✅ ETL 완료 요약")
    print(f"   - 총 저장됨: {saved_count}건")
    print(f"   - 중복 스킵: {skipped_dup}건")
    print(f"   - 관련없음(종목X) 스킵: {skipped_irrelevant}건")
    print("="*40)

if __name__ == "__main__":
    run_finance_news_etl()