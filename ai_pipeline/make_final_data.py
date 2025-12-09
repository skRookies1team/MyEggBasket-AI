import numpy as np
import pandas as pd
import os
import sys
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 기존 모듈 import
from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

# 경로 설정
GCN_DIR = os.path.join(project_root, 'ai_pipeline/gcn_model')
EMBEDDING_PATH = os.path.join(GCN_DIR, 'gcn_embeddings.npy')
NODE_LIST_PATH = os.path.join(GCN_DIR, 'gcn_node_list.csv')
OUTPUT_PATH = os.path.join(project_root, 'final_dataset_with_gcn.csv')

def make_complete_dataset():
    print("🚀 [최종 데이터셋 생성] ES 데이터(감성/AI) + GCN 임베딩 + 기술적 지표 병합 시작...")

    # ---------------------------------------------------------
    # 1. GCN 임베딩 데이터 로드
    # ---------------------------------------------------------
    if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(NODE_LIST_PATH):
        print(f"❌ GCN 파일이 없습니다. run_gcn.py를 먼저 실행하세요.")
        return

    try:
        # 노드 리스트
        node_df = pd.read_csv(NODE_LIST_PATH, dtype=str)
        if 'code' in node_df.columns:
            node_df.rename(columns={'code': 'stock_code'}, inplace=True)
            
        # 임베딩 데이터
        embeddings = np.load(EMBEDDING_PATH)
        emb_cols = [f'gcn_emb_{i}' for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols)
        
        # GCN 기본 데이터셋 생성
        df_gcn_base = pd.concat([node_df, df_emb], axis=1)
        
    except Exception as e:
        print(f"❌ GCN 데이터 로드 중 오류: {e}")
        return

    # ---------------------------------------------------------
    # 2. [핵심 수정] ES에서 감성 점수 & AI 스코어 각각 가져오기
    # ---------------------------------------------------------
    print("📡 Elasticsearch에서 데이터 조회 중...")
    
    es = Elasticsearch("http://localhost:9200")
    
    es_data_list = []
    
    # 두 개의 인덱스를 각각 조회해야 함
    idx_sentiment = "stock_features_v1" # 감성 점수 저장소
    idx_technical = "stock_technicals"  # AI 스코어 저장소
    
    for code in df_gcn_base['stock_code']:
        # 기본값 설정
        sentiment = 0.0
        volatility = 0.0
        ai_score = 50.0
        
        # (1) 감성 점수 가져오기
        try:
            resp_sent = es.search(
                index=idx_sentiment,
                body={"query": {"term": {"stock_code": code}}, "size": 1, "sort": [{"timestamp": "desc"}]}
            )
            if resp_sent['hits']['hits']:
                src = resp_sent['hits']['hits'][0]['_source']
                sentiment = src.get('sentiment_score', 0.0)
                volatility = src.get('sentiment_volatility', 0.0)
        except: pass

        # (2) AI 스코어(기술적 점수) 가져오기
        try:
            resp_tech = es.search(
                index=idx_technical,
                body={"query": {"term": {"stock_code": code}}, "size": 1, "sort": [{"timestamp": "desc"}]}
            )
            if resp_tech['hits']['hits']:
                src = resp_tech['hits']['hits'][0]['_source']
                ai_score = src.get('ai_score', 50.0)
        except: pass

        es_data_list.append({
            'stock_code': code,
            'sentiment_score': sentiment,
            'ai_score': ai_score,
            'volatility': volatility
        })

    # ES 데이터프레임 생성
    df_es = pd.DataFrame(es_data_list)
    
    # AI 스코어가 50점이 아닌(제대로 가져온) 데이터 개수 확인
    valid_ai_score = (df_es['ai_score'] != 50.0).sum()
    print(f"✅ ES 데이터 로드 완료: {len(df_es)}개 종목")
    print(f"   👉 AI 스코어 유효 데이터(50점 아님): {valid_ai_score}개")

    # ---------------------------------------------------------
    # 3. 기술적 지표 생성 (Feature Engineering)
    # ---------------------------------------------------------
    print("📊 기술적 지표(RSI, SMA 등) 생성 중...")
    
    try:
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir): data_dir = project_root
            
        engineer = FeatureEngineer(data_dir=data_dir)
        features_ret = engineer.create_final_features()
        
        if len(features_ret) == 3:
            X_tech, y, codes = features_ret
        elif len(features_ret) == 2:
            X_tech, codes = features_ret
        else:
            print("❌ FeatureEngineer 반환값 오류")
            return

        df_tech = pd.DataFrame(X_tech, columns=engineer.final_columns if hasattr(engineer, 'final_columns') else None)
        df_tech['stock_code'] = [str(c).zfill(6) for c in codes]
        
        if 'date' in df_tech.columns:
            df_tech = df_tech.sort_values('date').groupby('stock_code').tail(1)
        else:
            df_tech = df_tech.drop_duplicates(subset=['stock_code'], keep='last')
            
        # 충돌 컬럼 제거
        cols_to_drop = ['sentiment_score', 'ai_score', 'volatility'] + [c for c in df_tech.columns if c.startswith('gcn_emb_')]
        existing_drop_cols = [c for c in cols_to_drop if c in df_tech.columns]
        if existing_drop_cols:
            df_tech.drop(columns=existing_drop_cols, inplace=True)

    except Exception as e:
        print(f"❌ 기술적 지표 생성 중 오류: {e}")
        return

    # ---------------------------------------------------------
    # 4. 전체 병합 (GCN + ES + Tech)
    # ---------------------------------------------------------
    print("🔗 전체 데이터 병합 중...")
    
    # 1단계: GCN + ES 데이터 합치기
    df_step1 = pd.merge(df_gcn_base, df_es, on='stock_code', how='left')
    
    # 2단계: 위의 결과 + 기술적 지표 합치기
    final_df = pd.merge(df_tech, df_step1, on='stock_code', how='inner')
    
    # ---------------------------------------------------------
    # 5. 저장 및 검증
    # ---------------------------------------------------------
    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("=" * 60)
    print(f"💾 [최종 데이터 저장 완료] {OUTPUT_PATH}")
    print(f"   👉 총 종목 수: {len(final_df)}개")
    print("=" * 60)
    
    # 데이터 검증
    print("\n🔍 [데이터 검증: 삼성전자/하이닉스 등 샘플]")
    print(final_df[['stock_code', 'sentiment_score', 'ai_score', 'gcn_emb_0']].head())

if __name__ == "__main__":
    make_complete_dataset()