import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)


from ai_pipeline.news_source.finance_news_list import fetch_finance_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.nlp.text_splitter import split_text
from ai_pipeline.nlp.sentiment import analyze_sentiment
from ai_pipeline.news_etl.es_uploader import save_news_to_es

def run_finance_news_etl():

    print("🔥 ETL 시작됨")

    news_urls = fetch_finance_news_list()
    print("📌 수집된 URL 개수:", len(news_urls))

    if len(news_urls) == 0:
        print("❌ 뉴스 URL이 0개입니다. selector나 User-Agent 문제입니다.")
        return



    for idx, finance_url in enumerate(news_urls):
        print(f"\n[{idx+1}/{len(news_urls)}] 처리 중: {finance_url}")

        real_url = extract_real_article_url(finance_url)
        print(f"➡ 실제 기사 URL: {real_url}")

        article_text = fetch_article_text(real_url)
        if not article_text:
            print("❌ 본문 없음 → 스킵")
            continue

        chunks = split_text(article_text)
        sentiments = analyze_sentiment(chunks)

        save_news_to_es(real_url, article_text, chunks, sentiments)

    print("\n✅ ETL 전체 완료 (Finance 뉴스 → ES 저장)")

if __name__ == "__main__":
    run_finance_news_etl()