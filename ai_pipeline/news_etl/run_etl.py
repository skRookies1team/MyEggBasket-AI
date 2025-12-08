from ai_pipeline.news_etl.news_crawler import fetch_article_text
from ai_pipeline.nlp.text_splitter import split_text
from ai_pipeline.nlp.sentiment import analyze_sentiment
from ai_pipeline.news_etl.es_uploader import save_news_to_es

def run_news_etl():
    urls = fetch_news_urls(query="증시", display=20)

    for url in urls:
        print("▶", url)

        text = fetch_article_text(url)
        chunks = split_text(text)
        sentiments = analyze_sentiment(chunks)

        save_news_to_es(url, text, chunks, sentiments)

    print(" ETL 완료 (뉴스 → ES 저장)")
