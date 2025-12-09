# ai_pipeline/news_etl/news_crawler.py 에서 import 한다고 가정
from ai_pipeline.news_etl.news_crawler import fetch_article_text
from ai_pipeline.news_etl.es_uploader import save_news_to_es

# fetch_news_urls 함수가 정의되어 있거나 import 되어 있어야 합니다.
def fetch_news_urls(query, display=20):
    return []  # 실제 URL 리스트를 반환하도록 구현 필요

def run_news_etl():
    print(" ETL 파이프라인 시작...")
    urls = fetch_news_urls(query="증시", display=5)

    for url in urls:
        print(f"▶ 처리 중: {url}")

        # 1. 크롤링 (제목, 본문, 날짜 3개 반환)
        crawled_data = fetch_article_text(url)
        
        if not crawled_data:
            print("    크롤링 실패 또는 내용 부족")
            continue

        title, text, p_date = crawled_data

        # 2. NLP 분석 (현재는 비워둠, 필요시 연결)
        # chunks = split_text(text)
        # sentiments = analyze_sentiment(chunks)
        
        # 임시 데이터 (나중에 실제 NLP 모듈 결과로 대체하세요)
        related_stocks = []      # 예: ["005930"]
        analysis_results = []    # 예: [{"stock_code": "005930", "sentiment": 0.5}]
        sentence_details = [] 
        value_chain_info = []

        # 3. ES 저장 (인자 순서 중요: es_uploader.py와 일치시킴)
        save_success = save_news_to_es(
            url=url,
            title=title,
            text=text,
            published_date=p_date,
            related_stocks=related_stocks,
            analysis_results=analysis_results,
            sentence_details=sentence_details,
            value_chain_info=value_chain_info
        )

        if save_success:
            print(f"    저장 완료: {title[:10]}...")
        else:
            print("    저장 실패")

    print(" ETL 완료")

if __name__ == "__main__":
    run_news_etl()