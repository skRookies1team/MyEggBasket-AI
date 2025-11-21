import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pipeline_module import NLPExtractor, GCNPipeline, XGBoostModel


def run_pipeline():
    """
    파이프라인의 각 모듈을 순차적으로 실행하고, 데이터를 연결하여 최종 XGBoost에 전달합니다.
    """
    print("=============================================")
    print("         🚀 금융 AI 파이프라인 시작         ")
    print("=============================================")
    
    # ----------------------------------------------------------------------
    # A. 더미 데이터 생성 (실제 DB 로드 대체)
    # ----------------------------------------------------------------------
    TICKER_COUNT = 100
    TIME_STEPS = 100
    FEATURE_COUNT = 10 
    
    # NLP 입력 데이터 (뉴스 기사 리스트)
    nlp_texts = [
        "엔비디아가 역대 최대 분기 실적을 기록하며 주가가 급등했다. AI 수요 폭발.",
        "경쟁사의 리콜 이슈로 인해 주가가 급락하며 시장에 부정적인 영향을 주었다.",
        "시장에 특별한 변화는 없었으나, 유가 급등으로 인해 운송주가 상승했다."
    ]
    
    # GCN 입력 데이터 (초기 피처 & 관계)
    ticker_list = [f'TKR{i:03d}' for i in range(TICKER_COUNT)]
    
    initial_features = pd.DataFrame(
        np.random.rand(TICKER_COUNT, FEATURE_COUNT),
        index=ticker_list,
        columns=[f'initial_feature_{i}' for i in range(FEATURE_COUNT)]
    )

    relations = pd.DataFrame({
        'from_ticker': np.random.choice(ticker_list, 1000),
        'to_ticker': np.random.choice(ticker_list, 1000),
        'weight': np.random.rand(1000) 
    })
    
    # XGBoost 학습 데이터 (Feature Store 시뮬레이션 - N_SAMPLES = 100 * 100 = 10000)
    N_SAMPLES = TICKER_COUNT * TIME_STEPS
    FINAL_FEATURE_COUNT = 120 
    X_final = pd.DataFrame(
        np.random.randn(N_SAMPLES, FINAL_FEATURE_COUNT),
        columns=[f'feature_{i}' for i in range(FINAL_FEATURE_COUNT)]
    )
    y_final = pd.Series(np.random.randint(0, 2, N_SAMPLES), name='target')

    # ----------------------------------------------------------------------
    # B. 파이프라인 실행
    # ----------------------------------------------------------------------
    
    # 1. NLP 모듈 실행
    nlp_module = NLPExtractor()
    sentiment_results = nlp_module.extract_sentiment(nlp_texts)
    print("\n[단계 1. NLP 결과 (샘플)]")
    print(sentiment_results)
    
    # 2. GCN 모듈 실행
    # GCN 입력 차원은 initial_features의 컬럼 수
    gcn_module = GCNPipeline(in_dim=FEATURE_COUNT, out_dim=32)
    node_embeddings_df = gcn_module.extract_embedding(
        initial_features=initial_features, 
        relations_df=relations
    )
    print("\n[단계 2. GCN Embedding 결과 (샘플)]")
    print(node_embeddings_df.head(3))
    
    # GCN Embedding 결과는 Feature Store에 저장될 최종 피처에 통합되어야 함.
    # (여기서는 X_final에 이미 통합되었다고 가정하고 XGBoost로 넘어갑니다.)
    
    # 3. XGBoost 모듈 실행 (더미 데이터로 학습 및 평가)
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'eta': 0.1, 'max_depth': 6, 'use_label_encoder': False
    }
    xgb_module = XGBoostModel(params=xgb_params)
    
    print("\n[단계 3. XGBoost 실행]")
    xgb_module.train_and_evaluate(X=X_final, y=y_final)
    
    print("=============================================")
    print("         ✅ 파이프라인 전체 실행 완료         ")
    print("=============================================")

if __name__ == '__main__':
    run_pipeline()