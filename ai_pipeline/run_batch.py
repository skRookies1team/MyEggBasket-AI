import sys
import os
import pandas as pd
from elasticsearch import Elasticsearch
from ai_pipeline.config.settings import ES_HOST

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from ai_pipeline.analysis.sentiment_aggregator import calculate_aggregated_features
from ai_pipeline.storage import ElasticStorage

def run_backfill():
    print(" 뉴스 데이터 가공 공장 가동 시작...")
    
    es = Elasticsearch(ES_HOST)
    
    resp = es.search(
        index="news_articles",
        body={
            "size": 10000, 
            "sort": [{"timestamp": "desc"}],
            "_source": ["timestamp", "related_stocks", "sentiment_score", "analysis_results"]
        }
    )
    
    hits = resp['hits']['hits']
    print(f" 원본 뉴스 {len(hits)}개를 로딩했습니다.")

    if len(hits) == 0:
        print(" 가공할 뉴스가 없습니다.")
        return

    raw_data = []
    valid_score_count = 0
    
    for hit in hits:
        src = hit['_source']
        timestamp = src.get('timestamp')
        
        # [핵심 수정] 리스트 형태의 analysis_results 처리
        analysis_res = src.get('analysis_results')
        
        # Case 1: 새로운 리스트 구조 (추천)
        if analysis_res and isinstance(analysis_res, list):
            for item in analysis_res:
                stock_code = item.get('stock_code')
                # 점수 가져오기 (없으면 0)
                score = float(item.get('sentiment_score', 0.0))
                
                if stock_code:
                    if abs(score) > 0.0001: valid_score_count += 1
                    raw_data.append({
                        'date': timestamp,
                        'code': stock_code,
                        'score': score
                    })

        # Case 2: 혹시 예전 딕셔너리 구조 데이터가 섞여있을 경우 대비 (호환성)
        elif analysis_res and isinstance(analysis_res, dict):
            for stock_code, stats in analysis_res.items():
                if isinstance(stats, dict):
                    score = float(stats.get('sentiment_score', 0.0))
                else:
                    score = float(stats)
                
                if abs(score) > 0.0001: valid_score_count += 1
                raw_data.append({
                    'date': timestamp,
                    'code': stock_code,
                    'score': score
                })
                
        # Case 3: 상세 데이터가 없는 경우 (기존 방식)
        else:
            stocks = src.get('related_stocks', [])
            raw_score = src.get('sentiment_score', 0.0)
            score = float(raw_score) if raw_score is not None else 0.0
                
            if isinstance(stocks, list) and stocks:
                if abs(score) > 0.0001: valid_score_count += 1
                for stock_code in stocks:
                    raw_data.append({
                        'date': timestamp,
                        'code': stock_code,
                        'score': score
                    })

    print(f" 변환 완료: 총 {len(raw_data)}행 데이터 생성")
    print(f"    (진단) 감성 점수가 0이 아닌 유효 데이터: {valid_score_count}건")
    
    if not raw_data:
        print(" 집계할 데이터가 없습니다.")
        return

    df_features = calculate_aggregated_features(raw_data)
    print(f" 집계 완료! 생성된 피처 데이터: {len(df_features)}행")
    
    storage = ElasticStorage()
    storage.save_features(df_features)
    print(" 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    run_backfill()