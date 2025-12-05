import sys
import os
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.config.settings import ES_HOST
from ai_pipeline.nlp.news_analyzer import NewsAnalyzer
from ai_pipeline.news_etl.es_uploader import save_news_to_es

def reprocess_all_news():
    print("\n" + "="*60)
    print(" [재처리] 기존 뉴스 데이터 문장 단위 재분석 시작")
    print("="*60)

    es = Elasticsearch(ES_HOST)
    index_name = "news_articles"
    
    # 분석기 초기화
    analyzer = NewsAnalyzer()

    # 1. 모든 문서 가져오기 (scan 사용 - 대량 데이터용)
    # query: stock_analysis 필드가 없는(아직 재처리 안 된) 문서만 골라낼 수도 있음
    # 여기서는 그냥 전부 다시 돌림
    query = {"query": {"match_all": {}}}
    
    print(" 데이터 로딩 중...")
    docs = scan(es, index=index_name, query=query)
    
    count = 0
    updated = 0
    
    for hit in docs:
        count += 1
        doc_id = hit['_id']
        source = hit['_source']
        
        url = source.get('url')
        text = source.get('text')
        
        if not text: continue
        
        # 이미 처리된 건지 확인 (선택 사항)
        # if 'stock_analysis' in source: continue

        # -----------------------------------------------
        # 🧠 재분석 실행
        # -----------------------------------------------
        results, details = analyzer.analyze_article(text)
        
        if not results:
            print(f"[{count}]  종목 없음: {url[:30]}...")
            # (선택) 종목 없는 뉴스는 삭제할 수도 있음: es.delete(...)
            continue
            
        related_stocks = list(results.keys())
        legacy_sentiments = [results[code]['sentiment_score'] for code in related_stocks]

        # 덮어쓰기 (Update)
        # save_news_to_es 함수가 id를 url 해시로 생성하므로, url이 같으면 덮어씌워짐.
        save_news_to_es(
            url=url,
            related_stocks=related_stocks,
            analysis_results=results,
            sentiments=legacy_sentiments
        )
        
        updated += 1
        if updated % 10 == 0:
            print(f"[{count}]  {updated}개 업데이트 완료 (최근: {related_stocks})")

    print("\n" + "="*60)
    print(f" 재처리 완료!")
    print(f"   - 총 스캔 문서: {count}개")
    print(f"   - 업데이트됨: {updated}개")
    print("="*60)

if __name__ == "__main__":
    reprocess_all_news()