import sys
import os
import time
from datetime import datetime, timedelta

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.news_source.finance_news_list import fetch_daily_news_list
from ai_pipeline.news_source.news_article_crawler import extract_real_article_url, fetch_article_text
from ai_pipeline.news_etl.es_uploader import save_news_to_es, exists_in_es


# [NEW] 신규 분석 모듈 가져오기
from ai_pipeline.nlp.news_analyzer import NewsAnalyzer
from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer

def run_date_range_collection(start_date_str, end_date_str):
    """
    기간별 대량 수집 실행기 (문장 분석 + 밸류체인 포함)
    """
    print("\n" + "="*60)
    print(f"📚 [Bulk] 기간별 기업분석 뉴스 정밀 수집 시작")
    print(f"   📅 기간: {start_date_str} ~ {end_date_str}")
    print("="*60)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    current_date = start_date
    total_saved = 0

    # 분석기 초기화
    news_analyzer = NewsAnalyzer()
    vc_analyzer = ValueChainAnalyzer()

    while current_date <= end_date:
        target_date = current_date.strftime("%Y%m%d") # YYYYMMDD 형식
        print(f"\n📅 [Date: {target_date}] 뉴스 수집 중...", end=" ")
        
        # 1. 해당 날짜의 URL 목록 가져오기 (최대 10페이지)
        daily_urls = fetch_daily_news_list(target_date, max_pages=10)
        print(f"👉 {len(daily_urls)}개 발견", end="")
        
        if not daily_urls:
            current_date += timedelta(days=1)
            continue

        # 2. 크롤링 및 정밀 분석
        daily_saved = 0
        for url in daily_urls:
            real_url = extract_real_article_url(url)
            
            if exists_in_es(real_url): continue

            # 여기서 (제목, 본문)을 받습니다.
            result = fetch_article_text(real_url)
            
            # result가 None이거나 길이가 2가 아니면 건너뜀
            if not result:
                continue

            title = result[0]
            article_text = result[1]

            # 본문 길이 2차 체크
            if len(article_text) < 50:
                continue

            # ---------------------------------------------------------
            # 🧠 [Core 1] 문장 단위 정밀 감성 분석
            # ---------------------------------------------------------
            analysis_results, sentence_details = news_analyzer.analyze_article(article_text)

            if not analysis_results: continue

            related_stocks = list(analysis_results.keys())

            # ---------------------------------------------------------
            # 🔗 [Core 2] 밸류체인 연관 종목 추출
            # ---------------------------------------------------------
            value_chain_info = []
            for stock_code in related_stocks:
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
            # 💾 [Core 3] 저장 (es_uploader 최신 규격 맞춤)
            # ---------------------------------------------------------
            save_news_to_es(
                url=real_url,
                title=title,
                text=article_text,
                related_stocks=related_stocks,
                analysis_results=analysis_results, # 1h, trend, volatility 포함됨
                sentence_details=sentence_details, # 문장별 점수 포함됨
                value_chain_info=value_chain_info  # 밸류체인 정보 포함됨
            )
            
            daily_saved += 1
            
            # 진행 표시 (.)
            print(".", end="", flush=True) 
            time.sleep(0.05) 

        print(f" ✅ {daily_saved}건 저장 완료")
        total_saved += daily_saved
        
        # 하루 넘기기
        current_date += timedelta(days=1)

    print("\n" + "="*60)
    print(f"🎉 전체 기간 수집 완료! 총 저장된 뉴스: {total_saved}건")
    print("="*60)

if __name__ == "__main__":
    # 2025년 10월 1일 ~ 오늘(2025-12-04)
    run_date_range_collection("2025-09-01", "2025-10-31")