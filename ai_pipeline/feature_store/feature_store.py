import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


# 같은 폴더 내 모듈 임포트
try:
    from .price_loader import PriceFeatureLoader
    from .sentiment_loader import SentimentFeatureLoader
    from .graph_loader import GraphFeatureLoader
except ImportError:
    # 경로 문제 시 상대 경로 처리
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from price_loader import PriceFeatureLoader
    from sentiment_loader import SentimentFeatureLoader
    from graph_loader import GraphFeatureLoader

class OnlineFeatureStore:
    def __init__(self):
        print(" [FeatureStore] 실시간 피처 스토어 초기화 중...")
        
        # 1. 각 로더 초기화
        # (주의: 실제 환경에 맞게 몽고DB URI 등을 수정하세요)
        self.price_loader = PriceFeatureLoader()
        self.sentiment_loader = SentimentFeatureLoader()
        self.graph_loader = GraphFeatureLoader() # data 폴더 자동 탐색

    def get_realtime_features(self, stock_code):
        """
        [핵심] 특정 종목의 현재 시점 모든 피처를 가져와 1행 DataFrame으로 반환
        """
        stock_code = str(stock_code).zfill(6)
        
        # 1. 가격 및 TA 피처 (MongoDB + 계산)
        price_feats = self.price_loader.get_latest_technical_features(stock_code)
        if price_feats is None:
            # 데이터가 없으면 예측 불가
            print(f" [FeatureStore] {stock_code}: 가격 데이터 없음")
            return None
            
        # 2. 감성 피처 (Elasticsearch)
        sentiment_score = self.sentiment_loader.get_latest_sentiment(stock_code)
        
        # 3. 그래프 피처 (Memory)
        graph_feats = self.graph_loader.get_embedding_features(stock_code)
        
        # 4. 통합 (Dictionary Merge)
        final_features = price_feats.copy()
        final_features['sentiment_score'] = sentiment_score
        final_features.update(graph_feats)
        
        # 5. DataFrame 변환 (모델 입력용)
        df_row = pd.DataFrame([final_features])
        
        # 모델 학습시 사용하지 않는 컬럼 제거/정렬 등 후처리가 필요하다면 여기서 수행
        
        return df_row

# --- 테스트 실행 ---
if __name__ == "__main__":
    store = OnlineFeatureStore()
    
    # 예시: 기아(000270)
    print("\n [Test] 기아(000270) 실시간 피처 조회:")
    features = store.get_realtime_features("000270")
    
    if features is not None:
        print(features.T) # 보기 좋게 전치해서 출력
        print("\n -> 이 데이터를 XGBoost 모델의 predict() 입력으로 사용하면 됩니다.")