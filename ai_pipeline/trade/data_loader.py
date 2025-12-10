import os
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch

# 기존 모듈 가져오기 (경로 주의)
from ai_pipeline.boosting_model.realtime_feature_loader import RealtimeFeatureLoader

def load_data_and_merge_news(date_str, data_dir, es_client):
    """특정 날짜의 CSV를 읽고, 해당 날짜의 뉴스 감성점수를 매핑"""
    csv_path = os.path.join(data_dir, f"{date_str}.csv")
    if not os.path.exists(csv_path):
        print(f"   ⚠️ 파일 없음: {csv_path}")
        return None

    # 1. 시세 데이터 로드 및 기술적 지표 생성
    loader = RealtimeFeatureLoader(csv_path)
    df = loader.load_and_preprocess()
    if df.empty: return None

    df = loader.create_technical_features(df)
    if df.empty: return None
    
    df = df.dropna()
    if len(df) == 0: return None

    # 필요한 컬럼 정의
    feature_cols = [
        'prdy_ctrt', 'price_change_1', 'price_change_5', 'price_change_10',
        'price_vs_ma5', 'price_vs_ma20', 'tr_amount_change', 'spread', 'spread_pct',
        'buy_pressure', 'buy_strength', 'volatility_5', 'volatility_10',
        'momentum_5', 'momentum_10'
    ]
    
    # 없는 컬럼 채우기
    for c in feature_cols:
        if c not in df.columns: df[c] = 0.0

    keep_cols = feature_cols + ['target', 'stock_code', 'stck_prpr']
    if 'stck_prpr' not in df.columns: df['stck_prpr'] = 0
    
    df_result = df[keep_cols].copy()

    # 2. ES에서 뉴스 감성 가져오기
    dt_obj = datetime.strptime(date_str, "%Y%m%d")
    start_dt = dt_obj.replace(hour=0, minute=0, second=0).isoformat()
    end_dt = dt_obj.replace(hour=23, minute=59, second=59).isoformat()

    query = {
        "range": {
            "timestamp": {
                "gte": start_dt,
                "lte": end_dt
            }
        }
    }

    try:
        resp = es_client.search(
            index="news_articles", # 인덱스명 확인 필요 (stock_features_v1 인지 news_articles 인지)
            body={
                "size": 0,
                "query": query,
                "aggs": {
                    "by_stock": {
                        "terms": {"field": "related_stocks.keyword", "size": 1000},
                        "aggs": {"avg_sentiment": {"avg": {"field": "sentiment_score"}}}
                    }
                }
            }
        )
        sentiment_map = {}
        if 'aggregations' in resp:
            for bucket in resp['aggregations']['by_stock']['buckets']:
                sentiment_map[bucket['key']] = bucket['avg_sentiment']['value'] or 0.0
    except Exception as e:
        print(f"   [ES Error] 뉴스 로드 실패: {e}")
        sentiment_map = {}

    # 3. 매핑
    df_result['sentiment_score'] = df_result['stock_code'].map(sentiment_map).fillna(0.0)

    return df_result