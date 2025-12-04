# ai_pipeline/run_batch.py
import sys
import os
# 프로젝트 루트 경로를 잡아주기 위한 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from elasticsearch import Elasticsearch
from ai_pipeline.analysis.sentiment_aggregator import calculate_aggregated_features
from ai_pipeline.storage import ElasticStorage
import pandas as pd

def run_backfill():
    print("🏭 뉴스 데이터 가공 공장 가동 시작...")
    
    es = Elasticsearch("http://localhost:9200")
    
    # 1. ES에서 뉴스 데이터 다 꺼내오기 (최대 10,000개)
    # 실제 운영에선 scroll api를 써야 하지만 지금은 size=10000으로 충분
    resp = es.search(
        index="news_articles",
        body={
            "size": 10000, 
            "_source": ["timestamp", "related_stocks", "sentiment_score"]
        }
    )
    
    hits = resp['hits']['hits']
    print(f"📦 원본 뉴스 {len(hits)}개를 로딩했습니다.")

    if len(hits) == 0:
        print("가공할 뉴스가 없습니다.")
        return

    # 2. Aggregator가 좋아하는 형태로 변환
    # (aggregator는 {'date':..., 'code':..., 'score':...} 리스트를 원함)
    raw_data = []
    
    for hit in hits:
        src = hit['_source']
        timestamp = src.get('timestamp')
        score = src.get('sentiment_score')
        stocks = src.get('related_stocks', []) # 리스트 형태라고 가정

        # 종목이 여러 개면 쪼개서 넣기 (Explode)
        if isinstance(stocks, list):
            for stock_code in stocks:
                raw_data.append({
                    'date': timestamp,
                    'code': stock_code,
                    'score': score
                })
        elif isinstance(stocks, str): # 혹시 문자열로 저장되어 있다면
             raw_data.append({
                'date': timestamp,
                'code': stocks,
                'score': score
            })

    print(f"🔧 종목별 데이터로 변환 완료: 총 {len(raw_data)}건 처리 중...")

    # 3. 데이터 집계 (우리가 만든 함수)
    df_features = calculate_aggregated_features(raw_data)
    
    print(f"📊 집계 완료! 생성된 피처 데이터: {len(df_features)}행")
    print(df_features.head()) # 눈으로 확인

    # 4. ES에 저장 (우리가 만든 스토리지)
    storage = ElasticStorage()
    storage.save_features(df_features)
    
    print("✅ 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    run_backfill()