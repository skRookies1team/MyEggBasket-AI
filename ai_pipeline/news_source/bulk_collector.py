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
    print(f" [Bulk] 기간별 기업분석 뉴스 정밀 수집 시작")
    print(f"    기간: {start_date_str} ~ {end_date_str}")
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
              
        print(f"\n [Date: {target_date}] 뉴스 수집 중...", end=" ")
        
        daily_urls = fetch_daily_news_list(target_date, max_pages=10)
        print(f" {len(daily_urls)}개 발견", end="")
        
        if not daily_urls:
            current_date += timedelta(days=1)
            continue

        daily_saved = 0
        for url in daily_urls:
            real_url = extract_real_article_url(url)
            
            if exists_in_es(real_url): continue

            # [중요] 크롤러에서 (제목, 본문, 실제작성시간) 3개를 받아야 합니다.
            result = fetch_article_text(real_url)
            
            if not result:
                continue

            # 결과 언패킹 (Unpacking)
            if len(result) == 3:
                # 크롤러가 날짜까지 정상적으로 긁어온 경우
                title = result[0]
                article_text = result[1]
                p_date = result[2]  # <--- 실제 기사 작성 시간 ("2025-11-01 14:23:05" 등)
            elif len(result) == 2:
                # 크롤러가 날짜를 못 가져오고 (제목, 본문)만 준 경우 (비상용)
                title = result[0]
                article_text = result[1]
                p_date = current_date.strftime("%Y-%m-%d 00:00:00") # 어쩔 수 없이 00시로 설정
            else:
                continue

            # 본문 길이 2차 체크
            if len(article_text) < 50:
                continue

            # ---------------------------------------------------------
            #  [Core 1] 문장 단위 정밀 감성 분석
            # ---------------------------------------------------------
            analysis_results, sentence_details = news_analyzer.analyze_article(article_text)

            if not analysis_results: continue

            related_stocks = list(analysis_results.keys())

            # ---------------------------------------------------------
            #  [Core 2] 밸류체인 연관 종목 추출
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
            # 💾 [Core 3] 저장
            # ---------------------------------------------------------
            save_news_to_es(
                url=real_url,
                title=title,
                text=article_text,
                published_date=p_date, # 크롤링 된 실제 시간 사용
                related_stocks=related_stocks,
                analysis_results=analysis_results,
                sentence_details=sentence_details,
                value_chain_info=value_chain_info
            )
            
            daily_saved += 1
            print(".", end="", flush=True) 
            time.sleep(0.05) 

        print(f"  {daily_saved}건 저장 완료")
        total_saved += daily_saved
        
        current_date += timedelta(days=1)

    print("\n" + "="*60)
    print(f" 전체 기간 수집 완료! 총 저장된 뉴스: {total_saved}건")
    print("="*60)

if __name__ == "__main__":
    run_date_range_collection("2025-12-07", "2025-12-11")