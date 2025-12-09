import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from ai_pipeline.news_source.finance_news_list import fetch_finance_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.news_etl.es_uploader import save_news_to_es, exists_in_es
from ai_pipeline.nlp.news_analyzer import NewsAnalyzer
from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer

def run_finance_news_etl():
    print(" ETL 시작됨 (문장단위 정밀분석 + 밸류체인)")

    news_urls = fetch_finance_news_list(max_pages=3)
    
    if not news_urls:
        print(" 수집된 뉴스가 없습니다.")
        return
    
    # 분석기들 초기화
    news_analyzer = NewsAnalyzer()
    vc_analyzer = ValueChainAnalyzer()
    
    saved_count = 0
    skipped_dup = 0

    for idx, finance_url in enumerate(news_urls):
        real_url = extract_real_article_url(finance_url)
        
        # 1. 중복 체크
        if exists_in_es(real_url):
            skipped_dup += 1
            continue

        print(f"[{idx+1}]   분석 중: {real_url}")
        
        # 2. 제목, 본문, 날짜 수집 (수정된 크롤러는 3개를 반환함)
        crawled_data = fetch_article_text(real_url)
        
        if not crawled_data:
            print("     본문 없음/짧음 -> Pass")
            continue

        # [수정] 반환값 3개 언패킹
        title, article_text, p_date = crawled_data
        
        if len(article_text) < 50:
             print("     본문 너무 짧음 -> Pass")
             continue

        # ---------------------------------------------------------
        #  [Core 1] 문장 단위 정밀 감성 분석
        # ---------------------------------------------------------
        # results: { '005930': {'sentiment_score': 0.9, ...} }
        # details: [ {'sentence': '...', 'ticker': '...', 'sentiment': 0.9}, ... ]
        analysis_results, sentence_details = news_analyzer.analyze_article(article_text)

        if not analysis_results:
            print("    [Pass] 종목 언급 없음 (일반 뉴스)")
            continue

        related_stocks = list(analysis_results.keys())

        # ---------------------------------------------------------
        #  [Core 2] 밸류체인 연관 종목 추출
        # ---------------------------------------------------------
        value_chain_info = []
        for stock_code in related_stocks:
            # 각 종목별로 연관된 친구들을 찾음 (CSV 기반)
            recs = vc_analyzer.find_similar_stocks(stock_code, top_n=3)
            if recs:
                for r in recs:
                    value_chain_info.append({
                        "source_code": stock_code,
                        "related_code": r['code'],
                        "related_name": r['name'],
                        "reason": r['reason']
                    })

        # ---------------------------------------------------------
        #  [Core 3] 최종 저장
        # ---------------------------------------------------------
        print(f"    발견: {related_stocks} | 문장수: {len(sentence_details)} | VC연관: {len(value_chain_info)}개")
        
        # 변수명 매칭 (url -> real_url, text -> article_text, published_date 추가)
        save_news_to_es(
            url=real_url,
            title=title,
            text=article_text,
            published_date=p_date,
            related_stocks=related_stocks,
            analysis_results=analysis_results,
            sentence_details=sentence_details,
            value_chain_info=value_chain_info
        )
        saved_count += 1

    print("\n" + "="*40)
    print(f" ETL 완료")
    print(f"   - 새로 저장됨: {saved_count}건")
    print(f"   - 중복 건너뜀: {skipped_dup}건")
    print("="*40)

if __name__ == "__main__":
    run_finance_news_etl()