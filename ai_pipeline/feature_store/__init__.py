import pandas as pd
import os
import sys

# 로더 모듈 임포트 (현재 폴더 . 위치 기준)
try:
    from .price_loader import PriceFeatureLoader
    from .sentiment_loader import SentimentFeatureLoader
    from .graph_loader import GraphFeatureLoader
except ImportError:
    # 경로 문제 발생 시 처리
    from price_loader import PriceFeatureLoader
    from sentiment_loader import SentimentFeatureLoader
    from graph_loader import GraphFeatureLoader


class OnlineFeatureStore:
    def __init__(self):
        print(" [FeatureStore] 실시간 피처 스토어 초기화 중...")

        # 각 로더 초기화
        self.price_loader = PriceFeatureLoader()
        self.sentiment_loader = SentimentFeatureLoader()
        self.graph_loader = GraphFeatureLoader()

    def get_realtime_features(self, stock_code):
        """
        특정 종목의 현재 시점 모든 피처를 가져와 1행 DataFrame으로 반환
        """
        stock_code = str(stock_code).zfill(6)

        # 1. 가격 및 TA 피처 (MongoDB + 계산)
        price_feats = self.price_loader.get_latest_technical_features(stock_code)
        if price_feats is None:
            return None

        # 2. 감성 피처 (Elasticsearch)
        sentiment_score = self.sentiment_loader.get_latest_sentiment(stock_code)

        # 3. 그래프 피처 (Memory)
        graph_feats = self.graph_loader.get_embedding_features(stock_code)

        # 4. 통합
        final_features = price_feats.copy()
        final_features['sentiment_score'] = sentiment_score
        final_features.update(graph_feats)

        # 5. DataFrame 변환
        df_row = pd.DataFrame([final_features])

        return df_row