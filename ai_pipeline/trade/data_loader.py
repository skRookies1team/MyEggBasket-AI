import os
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch

# 기존 모듈 가져오기
from ai_pipeline.boosting_model.realtime_feature_loader import RealtimeFeatureLoader

def load_data_and_merge_news(date_str, data_dir, es_client):
    """특정 날짜의 CSV를 읽고, 해당 날짜의 뉴스 감성점수를 매핑"""
    csv_path = os.path.join(data_dir, f"{date_str}.csv")
    if not os.path.exists(csv_path):
        print(f"   ⚠️ 파일 없음: {csv_path}")
        return None

    # 변수 초기화
    target_csv_path = csv_path
    temp_csv_path = None
    df_result = None

    try:
        # ------------------------------------------------------------------
        # [전처리] CSV 시간 포맷 정규화 (stck_cntg_hour -> 6자리 HHMMSS)
        # 이 과정이 있어야 1208, 1209 같은 파일도 시계열로 인식됩니다.
        # ------------------------------------------------------------------
        try:
            # 1. 일단 문자열로 읽기
            df_raw = pd.read_csv(csv_path, dtype=str)
            
            # 2. timestamp만 있고 stck_cntg_hour가 없는 경우 생성
            if 'stck_cntg_hour' not in df_raw.columns and 'timestamp' in df_raw.columns:
                df_raw['stck_cntg_hour'] = pd.to_datetime(df_raw['timestamp']).dt.strftime('%H%M%S')
            
            # 3. stck_cntg_hour 포맷 통일 (콜론/소수점 제거, 6자리 패딩)
            if 'stck_cntg_hour' in df_raw.columns:
                df_raw['stck_cntg_hour'] = (
                    df_raw['stck_cntg_hour']
                    .astype(str)
                    .str.replace(':', '')
                    .str.split('.').str[0]
                    .str.zfill(6)
                )
                
                # 임시 파일로 저장
                temp_csv_path = os.path.join(data_dir, f"temp_{date_str}.csv")
                df_raw.to_csv(temp_csv_path, index=False)
                target_csv_path = temp_csv_path  # 로더가 임시 파일을 읽도록 설정
        except Exception as e:
            print(f"   ⚠️ CSV 전처리 중 오류 (원본 사용): {e}")

        # ------------------------------------------------------------------
        # 1. 시세 데이터 로드 및 기술적 지표 생성
        # ------------------------------------------------------------------
        loader = RealtimeFeatureLoader(target_csv_path) # 수정된 경로 사용
        
        df = loader.load_and_preprocess()
        if df.empty: return None

        df = loader.create_technical_features(df)
        if df.empty: return None
        
        # [수정] dropna 대신 fillna(0) 사용 -> 데이터가 적은 날도 살리기 위함
        df = df.fillna(0)
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
        if 'target' not in df.columns: df['target'] = 0
        
        df_result = df[keep_cols].copy()

        # ------------------------------------------------------------------
        # 2. ES에서 뉴스 감성 가져오기
        # ------------------------------------------------------------------
        dt_obj = datetime.strptime(date_str, "%Y%m%d")
        start_dt = dt_obj.replace(hour=0, minute=0, second=0).isoformat()
        end_dt = dt_obj.replace(hour=23, minute=59, second=59).isoformat()

        query = {"range": {"timestamp": {"gte": start_dt, "lte": end_dt}}}

        try:
            resp = es_client.search(
                index="stock_features_v1", 
                body={
                    "size": 0,
                    "query": query,
                    "aggs": {
                        "by_stock": {
                            "terms": {"field": "stock_code", "size": 1000},
                            "aggs": {"avg_sentiment": {"avg": {"field": "sentiment_score"}}}
                        }
                    }
                }
            )
            sentiment_map = {}
            if 'aggregations' in resp:
                for bucket in resp['aggregations']['by_stock']['buckets']:
                    sentiment_map[bucket['key']] = bucket['avg_sentiment']['value'] or 0.0
        except Exception:
            sentiment_map = {}

        # 3. 매핑
        df_result['sentiment_score'] = df_result['stock_code'].map(sentiment_map).fillna(0.0)

    except Exception as e:
        print(f"   ❌ 데이터 로드 중 치명적 오류: {e}")
        return None

    finally:
        # [청소] 에러가 나든 성공하든 임시 파일은 무조건 삭제 (핵심 부분)
        if temp_csv_path and os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except Exception as e:
                print(f"   ⚠️ 임시 파일 삭제 실패: {e}")

    return df_result