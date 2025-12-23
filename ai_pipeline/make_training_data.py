import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
from ai_pipeline.config.settings import ES_HOST

# 프로젝트 루트 경로 설정 (ai_pipeline 폴더의 상위 폴더)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)

# 1. Feature Engineering (기술적 지표 + 타겟 생성 담당)
from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

# 2. 감성 점수 시간 가중치 계산 함수 (없으면 기본값 처리)
try:
    from ai_pipeline.preprocessing.sentiment_decay import calculate_time_weighted_sentiment
except ImportError:
    print(" [경고] sentiment_decay 모듈 없음. 기본 로직을 사용합니다.")
    def calculate_time_weighted_sentiment(news_items, base_time_str, decay_rate=0.9):
        if not news_items: return 0.0
        return news_items[0]['sentiment_score']
    
# 경로 설정
GCN_DIR = os.path.join(project_root, 'ai_pipeline/gcn_model')
EMBEDDING_PATH = os.path.join(GCN_DIR, 'gcn_embeddings.npy')
NODE_LIST_PATH = os.path.join(GCN_DIR, 'gcn_node_list.csv')
    

def create_labeled_dataset(data_dir, output_path):
    print("=" * 60)
    print(" [Data Gen] 학습용 라벨링 데이터셋 생성 시작 (Tech + GCN + Sentiment + AI Score) ")
    print("=" * 60)

    # 1. FeatureEngineer 초기화
    engineer = FeatureEngineer(data_dir=data_dir)

    # 2. 전체 파일 로드 및 피처 생성
    features_ret = engineer.create_final_features()

    if features_ret is None:
        print(" [Error] 데이터를 생성하지 못했습니다. (features_ret is None)")
        return

    # 반환값 언패킹 (X: 피처, y: 타겟, codes: 종목코드)
    # create_final_features가 None을 반환하는 경우를 대비
    if len(features_ret) == 3:
        X, y, codes = features_ret
    elif len(features_ret) == 2:
        X, y = features_ret
        codes = None
    else:
        print(" [Error] 피처 엔지니어링 반환값 형식이 맞지 않습니다.")
        return

    if X is None:
        print(" [Error] 생성된 피처 데이터(X)가 없습니다.")
        return
    
    # 3. 데이터 병합 (X + y)
    col_names = engineer.final_columns if hasattr(engineer, 'final_columns') else None
    df_train = pd.DataFrame(X, columns=col_names)
    df_train['target'] = y


    # 종목 코드가 있다면 식별을 위해 추가
    if codes is not None:
        df_train['stock_code'] = [str(c).zfill(6) for c in codes]

    # 충돌 방지: 미리 0으로 채워진 컬럼들 제거
    cols_to_drop = []
    # 1) GCN 임베딩 컬럼 제거 (gcn_emb_...)
    cols_to_drop.extend([c for c in df_train.columns if c.startswith('gcn_emb_')])
    
    # 2) 감성점수, AI 스코어, 변동성 등 제거
    for col in ['sentiment_score', 'ai_score', 'volatility', 'sentiment_volatility']:
        if col in df_train.columns:
            cols_to_drop.append(col)
            
    if cols_to_drop:
        print(f" 중복 방지를 위해 임시 컬럼 제거: {len(cols_to_drop)}개")
        df_train.drop(columns=cols_to_drop, inplace=True)
    
    print(f" 기술적 지표 생성 완료: {len(df_train)}개 종목")

    # [핵심 수정 2] GCN 임베딩 병합
    print("GCN 임베딩 로드 및 병합...")
    if os.path.exists(EMBEDDING_PATH) and os.path.exists(NODE_LIST_PATH):
        try:
            node_df = pd.read_csv(NODE_LIST_PATH, dtype=str)
            if 'code' in node_df.columns:
                node_df.rename(columns={'code': 'stock_code'}, inplace=True)
            
            embeddings = np.load(EMBEDDING_PATH)
            emb_cols = [f'gcn_emb_{i}' for i in range(embeddings.shape[1])]
            df_emb = pd.DataFrame(embeddings, columns=emb_cols)
            
            # GCN 데이터프레임
            df_gcn = pd.concat([node_df, df_emb], axis=1)
            
            # 병합
            df_train = pd.merge(df_train, df_gcn, on='stock_code', how='left')
            print(f" GCN 병합 완료 (임베딩 차원: {embeddings.shape[1]})")
            
        except Exception as e:
            print(f" GCN 병합 중 오류 (건너뜀): {e}")
    else:
        print(" GCN 파일이 없어 스킵합니다.")


    # [핵심 수정 3] ES 데이터(감성점수, AI Score) 병합
    print(" Elasticsearch 데이터 조회 (시간가중 감성 + AI Score)...")
    es = Elasticsearch(ES_HOST)
    
    if es.ping():
        es_data_list = []
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 현재 df_train에 있는 종목들만 조회
        target_codes = df_train['stock_code'].unique()
        
        for idx, code in enumerate(target_codes):
            sentiment = 0.0
            ai_score = 50.0 # 기본값
            
            # (1) 감성 점수 (최신 10개 + 시간 가중치)
            try:
                resp_sent = es.search(
                    index="stock_features_v1",
                    body={
                        "query": {"term": {"stock_code": code}}, 
                        "size": 10, 
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
                        if 'T' in t_stamp:
                            t_stamp = t_stamp.replace('T', ' ').split('.')[0]
                        news_items.append({'sentiment_score': s_score, 'published_at': t_stamp})
                    
                    sentiment = calculate_time_weighted_sentiment(news_items, current_time_str, decay_rate=0.9)
            except: pass

            # (2) AI Score
            try:
                resp_tech = es.search(
                    index="stock_technicals",
                    body={"query": {"term": {"stock_code": code}}, "size": 1, "sort": [{"timestamp": "desc"}]}
                )
                if resp_tech['hits']['hits']:
                    ai_score = resp_tech['hits']['hits'][0]['_source'].get('ai_score', 50.0)
            except: pass
            
            es_data_list.append({
                'stock_code': code,
                'sentiment_score': sentiment,
                'ai_score': ai_score
            })
            
        df_es = pd.DataFrame(es_data_list)
        df_train = pd.merge(df_train, df_es, on='stock_code', how='left')
        print(f" ES 데이터 병합 완료")
    else:
        print(" ES 연결 실패. 감성 점수는 0으로 처리됩니다.")
        df_train['sentiment_score'] = 0.0
        df_train['ai_score'] = 50.0


    # [핵심 수정] 병합이 안 됐거나 컬럼이 없는 경우를 대비해 강제 생성
    if 'sentiment_score' not in df_train.columns:
        print(" 'sentiment_score' 컬럼이 없어 생성합니다 (값: 0.0)")
        df_train['sentiment_score'] = 0.0
        
    if 'ai_score' not in df_train.columns:
        print(" 'ai_score' 컬럼이 없어 생성합니다 (값: 50.0)")
        df_train['ai_score'] = 50.0


    # 4. 결측치 제거
    # GCN/ES가 없는 종목들은 0 또는 기본값으로 채움
    df_train['sentiment_score'] = df_train['sentiment_score'].fillna(0.0)
    df_train['ai_score'] = df_train['ai_score'].fillna(50.0)
    gcn_cols = [c for c in df_train.columns if c.startswith('gcn_emb_')]
    df_train[gcn_cols] = df_train[gcn_cols].fillna(0.0)

    original_len = len(df_train)
    df_train = df_train.dropna()
    df_train['target'] = df_train['target'].astype(int)
    print(f" 결측치 제거: {original_len} -> {len(df_train)} (NaN 포함 행 삭제)")

    # 5. 저장
    df_train.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n [완료] 학습용 데이터 저장됨: {output_path}")
    print(f" 데이터 크기: {df_train.shape}")


if __name__ == "__main__":
    # 현재 파일 위치: .../MyEggBasket-AI/ai_pipeline/make_training_data.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 프로젝트 루트: .../MyEggBasket-AI
    project_root = os.path.dirname(current_dir)

    # [수정] data 폴더는 프로젝트 루트 바로 아래에 있습니다. ("../data" -> "data")
    data_dir = os.path.join(project_root, "data")

    # 출력 파일 경로
    output_csv = os.path.join(project_root, "train_dataset.csv")

    print(f" 데이터 폴더 경로: {data_dir}")

    if os.path.exists(data_dir):
        create_labeled_dataset(data_dir, output_csv)
    else:
        print(f" [Error] 데이터 폴더를 찾을 수 없습니다: {data_dir}")