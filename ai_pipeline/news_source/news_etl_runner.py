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
from ai_pipeline.nlp.news_analyzer import NewsAnalyzer

def run_finance_news_etl():
    print("🔥 ETL 시작됨")

    news_urls = fetch_finance_news_list()
    print("📌 수집된 URL 개수:", len(news_urls))

    if len(news_urls) == 0:
        print("❌ 뉴스 URL이 0개입니다. selector나 User-Agent 문제입니다.")
        return
    
    analyzer = NewsAnalyzer()

    saved_count = 0
    skipped_dup = 0

    for idx, finance_url in enumerate(news_urls):
        real_url = extract_real_article_url(finance_url)
        print(f"➡ 실제 기사 URL: {real_url}")

        if exists_in_es(real_url):
            # print(f"   ⏭️ [Skip] 이미 수집된 뉴스") # 로그 너무 많으면 주석
            skipped_dup += 1
            continue

        print(f"[{idx+1}/{len(news_urls)}] 🆕 분석 중: {real_url}")
        article_text = fetch_article_text(real_url)
        
        if not article_text or len(article_text) < 50:
            continue

        # ---------------------------------------------------------
        # 🧠 문장 단위 정밀 분석 실행
        # ---------------------------------------------------------
        # results: 종목별 지표 (score, trend, volatility)
        # details: 문장별 점수 리스트
        results, details = analyzer.analyze_article(article_text)

        if not results:
            print("   🗑️ [Pass] 종목 관련 내용 없음")
            continue

        # 관련 종목 추출
        related_stocks = list(results.keys())
        
        # 구버전 호환용 (대표 감성 점수 하나씩만 리스트로)
        legacy_sentiments = [results[code]['sentiment_score'] for code in related_stocks]

        # 저장
        save_news_to_es(
            url=real_url, 
            text=article_text, 
            related_stocks=related_stocks,
            analysis_results=results,   # [NEW] 상세 지표
            sentence_details=details,   # [NEW] 문장별 근거
            sentiments=legacy_sentiments # [Legacy] 호환용
        )
        
        print(f"   ✅ 저장 완료: {related_stocks}")
        saved_count += 1

    print("\n" + "="*40)
    print(f"✅ ETL 완료")
    print(f"   - 저장됨: {saved_count}건")
    print(f"   - 중복됨: {skipped_dup}건")
    print("="*40)

if __name__ == "__main__":
    run_finance_news_etl()