import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from elasticsearch import Elasticsearch
from ai_pipeline.config.settings import ES_HOST

# ---------------------------------------------------------
# 1. 환경 설정 및 모듈 로드
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Feature Engineering (기술적 지표 + 실시간 체결량 담당)
from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

# 감성 점수 시간 가중치 계산 함수 로드 (없으면 내부 함수 사용)
try:
    from ai_pipeline.preprocessing.sentiment_decay import calculate_time_weighted_sentiment
except ImportError:
    print(" [경고] sentiment_decay 모듈 없음. 기본 함수를 사용합니다.")
    def calculate_time_weighted_sentiment(news_items, base_time_str, decay_rate=0.9):
        if not news_items: return 0.0
        # (간이 구현: 필요시 파일 확인)
        return news_items[0]['sentiment_score']

# 경로 설정
DATA_DIR = os.path.join(project_root, 'data')
GCN_DIR = os.path.join(project_root, 'ai_pipeline/gcn_model')
EMBEDDING_PATH = os.path.join(DATA_DIR, 'gcn_embeddings.npy')
NODE_LIST_PATH = os.path.join(DATA_DIR, 'gcn_node_list.csv')
OUTPUT_PATH = os.path.join(project_root, 'final_dataset_with_gcn.csv')

def make_complete_dataset():
    print(" [최종 통합] 기술적 지표(38개) + GCN + 시간가중 감성점수 병합 시작...")

    # ---------------------------------------------------------
    # 2. GCN 임베딩 데이터 로드
    # ---------------------------------------------------------
    if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(NODE_LIST_PATH):
        print(f" GCN 파일이 없습니다. 경로 확인: {GCN_DIR}")
        return

    try:
        node_df = pd.read_csv(NODE_LIST_PATH, dtype=str)
        if 'code' in node_df.columns:
            node_df.rename(columns={'code': 'stock_code'}, inplace=True)
            
        embeddings = np.load(EMBEDDING_PATH)
        emb_cols = [f'gcn_emb_{i}' for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols)
        
        # GCN 기본 데이터셋
        df_gcn_base = pd.concat([node_df, df_emb], axis=1)
        print(f" GCN 데이터 로드 완료: {len(df_gcn_base)}개 종목")
    
    except Exception as e:
        print(f" GCN 데이터 로드 중 오류: {e}")
        return

    # ---------------------------------------------------------
    # 3. 기술적 지표 생성 (기존 코드 복구 - 체결량 등 포함)
    # ---------------------------------------------------------
    print(" 기술적 지표(RSI, SMA, 체결량 등) 생성 중...")
    
    try:
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir): data_dir = project_root
            
        engineer = FeatureEngineer(data_dir=data_dir)
        features_ret = engineer.create_final_features()
        
        # 반환값 처리 (X, y, codes 또는 X, codes)
        if len(features_ret) == 3:
            X_tech, y, codes = features_ret
        elif len(features_ret) == 2:
            X_tech, codes = features_ret
        else:
            print(" FeatureEngineer 반환값 형식 오류")
            return

        # DataFrame 변환
        col_names = engineer.final_columns if hasattr(engineer, 'final_columns') else None
        df_tech = pd.DataFrame(X_tech, columns=col_names)
        df_tech['stock_code'] = [str(c).zfill(6) for c in codes]
        
        # 중복 제거 (최신 날짜 기준 1행만 남김)
        if 'date' in df_tech.columns:
            df_tech = df_tech.sort_values('date').groupby('stock_code').tail(1)
        else:
            df_tech = df_tech.drop_duplicates(subset=['stock_code'], keep='last')
            
        # 충돌 방지를 위해 기존 ES 관련 컬럼이나 GCN 컬럼이 있다면 미리 제거
        cols_to_drop = ['sentiment_score', 'ai_score', 'volatility'] + [c for c in df_tech.columns if c.startswith('gcn_emb_')]
        existing_drop_cols = [c for c in cols_to_drop if c in df_tech.columns]
        if existing_drop_cols:
            df_tech.drop(columns=existing_drop_cols, inplace=True)
            
        print(f" 기술적 지표 생성 완료: {len(df_tech)}개 종목 (피처 수: {df_tech.shape[1]})")

    except Exception as e:
        print(f" 기술적 지표 생성 중 오류: {e}")
        return

    # ---------------------------------------------------------
    # 4. Elasticsearch 데이터 조회 (시간 가중치 로직 적용)
    # ---------------------------------------------------------
    print(" ES 데이터 조회 및 시간 가중 감성 점수 계산 중...")
    
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print(" Elasticsearch 연결 실패")
        return

    es_data_list = []
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # GCN 데이터에 있는 종목들을 기준으로 조회
    target_codes = df_gcn_base['stock_code'].unique()

    for idx, code in enumerate(target_codes):
        sentiment = 0.0
        ai_score = 50.0
        
        # (1) 감성 점수 (최신 10개 뉴스 + 시간 가중치)
        try:
            resp_sent = es.search(
                index="stock_features_v1",
                body={
                    "query": {"term": {"stock_code": code}}, 
                    "size": 10,  #  수정: 1개가 아니라 10개를 가져옴
                    "sort": [{"timestamp": "desc"}]
                }
            )
            
            hits = resp_sent['hits']['hits']
            if hits:
                news_items = []
                for h in hits:
                    src = h['_source']
                    s_score = src.get('sentiment_score', 0.0)
                    t_stamp = src.get('timestamp', current_time_str)
                    
                    #  수정: 날짜 포맷팅 (T 제거)
                    if 'T' in t_stamp:
                        t_stamp = t_stamp.replace('T', ' ')
                        if '.' in t_stamp: t_stamp = t_stamp.split('.')[0]
                            
                    news_items.append({
                        'sentiment_score': s_score,
                        'published_at': t_stamp
                    })
                
                #  수정: 시간 가중치 함수 호출
                sentiment = calculate_time_weighted_sentiment(
                    news_items=news_items,
                    base_time_str=current_time_str,
                    decay_rate=0.9
                )
        except Exception:
            pass # 에러 시 0.0 유지

        # (2) AI 기술적 점수 (최신 1개)
        try:
            resp_tech = es.search(
                index="stock_technicals",
                body={
                    "query": {"term": {"stock_code": code}}, 
                    "size": 1, 
                    "sort": [{"timestamp": "desc"}]
                }
            )
            if resp_tech['hits']['hits']:
                ai_score = resp_tech['hits']['hits'][0]['_source'].get('ai_score', 50.0)
        except Exception:
            pass

        es_data_list.append({
            'stock_code': code,
            'sentiment_score': sentiment,
            'ai_score': ai_score
        })

    df_es = pd.DataFrame(es_data_list)
    
    # ---------------------------------------------------------
    # 5. 전체 데이터 병합 (Tech + GCN + ES)
    # ---------------------------------------------------------
    print(" 전체 데이터 병합 중...")
    
    # 1단계: 기술적 지표(Tech) + GCN
    # (기술적 지표가 있는 종목만 살리기 위해 inner join 혹은 left join 선택. 보통 Tech가 기준이 됨)
    merged_1 = pd.merge(df_tech, df_gcn_base, on='stock_code', how='inner')
    
    # 2단계: + ES 감성/AI 점수
    final_df = pd.merge(merged_1, df_es, on='stock_code', how='left')
    
    # 결측치 채우기
    final_df['sentiment_score'] = final_df['sentiment_score'].fillna(0.0)
    final_df['ai_score'] = final_df['ai_score'].fillna(50.0)

    # ---------------------------------------------------------
    # 6. 저장 및 결과 확인
    # ---------------------------------------------------------
    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("=" * 60)
    print(f" [완료] 최종 데이터셋 생성 및 저장 성공!")
    print(f" 파일 경로: {OUTPUT_PATH}")
    print(f" 데이터 크기: {final_df.shape} (행, 열)")
    print(f"    예상 컬럼 수: 약 38개 + GCN임베딩 + 감성점수")
    print("=" * 60)
    
    # 주요 컬럼 확인
    check_cols = ['stock_code', 'sentiment_score', 'ai_score']
    # 기술적 지표 컬럼 중 하나가 있는지 확인 (예: close, rsi 등 FeatureEngineer에서 만드는 것)
    tech_sample = [c for c in final_df.columns if c not in check_cols and not c.startswith('gcn')]
    
    print("\n[데이터 미리보기]")
    print(final_df[check_cols + tech_sample[:2]].head())

if __name__ == "__main__":
    make_complete_dataset()