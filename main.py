import sys
sys.path.append(r"C:\Users\user\project\MyEggBasket-AI")

from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np

from ai_pipeline.gcn_model.model import GCNPipeline
from ai_pipeline.boosting_model.train import XGBoostModel

# ES 설정
ES_HOST = "http://localhost:9200"


# --------------------------
# ES에서 뉴스 불러오기
# --------------------------
def load_news_from_es(limit=100):
    es = Elasticsearch(ES_HOST)

    res = es.search(
        index="news_articles",
        size=limit,
        query={"match_all": {}}
    )

    news_list = [hit["_source"] for hit in res["hits"]["hits"]]
    return news_list



# --------------------------
# 메인 파이프라인 실행
# --------------------------
def run_pipeline():
    print("=============================================")
    print(" 금융 AI 파이프라인 실행 (ES 기반)")
    print("=============================================")

    # ---------------------------------------------------
    # A. ES에서 감성 데이터 불러오기
    # ---------------------------------------------------
    news_data = load_news_from_es(limit=200)

    if not news_data:
        print(" ES에 뉴스 데이터가 없습니다. 먼저 ETL(news_etl_runner.py)을 실행하세요.")
        return

    print(f" ES에서 {len(news_data)}개 뉴스 로드 완료")

    # 여러 뉴스가 존재 → 감성 score를 모두 모아서 feature로 활용
    sentiment_scores = []
    for n in news_data:
        sentiment_scores.extend(n["sentiments"])

    # 상위 N개만 사용하여 feature로 구성 (임시 feature)
    sentiment_features = sentiment_scores[:100]

    sentiment_df = pd.DataFrame({
        "sentiment": sentiment_features
    })

    print("\n[Sentiment Feature Sample]")
    print(sentiment_df.head())


    # ---------------------------------------------------
    # B. 더미 GCN 입력 데이터 생성
    # ---------------------------------------------------
    TICKER_COUNT = 100
    TIME_STEPS = 100
    FEATURE_COUNT = 10

    ticker_list = [f'TKR{i:03d}' for i in range(TICKER_COUNT)]

    # 랜덤 초기 피처
    initial_features = pd.DataFrame(
        np.random.rand(TICKER_COUNT, FEATURE_COUNT),
        index=ticker_list,
        columns=[f'initial_feature_{i}' for i in range(FEATURE_COUNT)]
    )

    # 랜덤 엣지 (관계도)
    relations = pd.DataFrame({
        'from_ticker': np.random.choice(ticker_list, 1000),
        'to_ticker': np.random.choice(ticker_list, 1000),
        'weight': np.random.rand(1000)
    })

    # ---------------------------------------------------
    # C. GCN Embedding 생성
    # ---------------------------------------------------
    print("\n[GCN Embedding 생성중...]")

    gcn = GCNPipeline(in_dim=FEATURE_COUNT, out_dim=32)
    gcn_embeddings = gcn.extract_embedding(initial_features, relations)

    print("\n[GCN Embedding Sample]")
    print(gcn_embeddings.head())


    # ---------------------------------------------------
    # D. XGBoost 학습
    # ---------------------------------------------------
    print("\n[XGBoost 학습 시작]")

    N_SAMPLES = TICKER_COUNT * TIME_STEPS
    FINAL_FEATURE_COUNT = 120

    # 랜덤 feature (현재는 더미 → 나중에 GCN + NLP 합칠 구조)
    X_final = pd.DataFrame(
        np.random.randn(N_SAMPLES, FINAL_FEATURE_COUNT),
        columns=[f'feature_{i}' for i in range(FINAL_FEATURE_COUNT)]
    )
    y_final = pd.Series(np.random.randint(0, 2, N_SAMPLES))

    xgb_model = XGBoostModel(params={
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 6
    })

    xgb_model.train_and_evaluate(X_final, y_final)

    print("\n=============================================")
    print("   파이프라인 전체 실행 완료")
    print("=============================================")



# --------------------------
# 실행 Entry Point
# --------------------------
if __name__ == "__main__":
    run_pipeline()
