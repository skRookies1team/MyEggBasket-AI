from elasticsearch import Elasticsearch, helpers
import pandas as pd
import math

class ElasticStorage:
    def __init__(self, host="http://localhost:9200"):
        self.es = Elasticsearch(host)
        
        # 집계된 데이터가 들어갈 인덱스 이름
        self.feature_index = "stock_features_v1"
        
        # 인덱스가 없으면 생성
        if not self.es.indices.exists(index=self.feature_index):
            self._create_feature_index()

    def _create_feature_index(self):
        """AI 모델용 피처 인덱스 매핑"""
        mapping = {
            "mappings": {
                "properties": {
                    "stock_code": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    
                    # [핵심] 우리가 만든 파생 변수들
                    "sentiment_score": {"type": "float"},
                    "sentiment_decay": {"type": "float"},
                    "sentiment_volatility": {"type": "float"},
                    "news_count": {"type": "integer"}  # 정수형
                }
            }
        }
        self.es.indices.create(index=self.feature_index, body=mapping)
        print(f"✅ Feature 인덱스 생성 완료: {self.feature_index}")

    def save_features(self, df_features):
        """
        sentiment_aggregator의 결과(DataFrame)를 저장
        """
        if df_features.empty:
            print("⚠️ 저장할 집계 데이터가 없습니다.")
            return

        actions = []
        for _, row in df_features.iterrows():
            
            # [안전장치] NaN(빈 값) 처리 및 타입 강제 변환
            # 데이터가 비어있으면 0으로 채움
            s_score = 0.0 if pd.isna(row['sentiment_1h']) else float(row['sentiment_1h'])
            s_decay = 0.0 if pd.isna(row['sentiment_decay']) else float(row['sentiment_decay'])
            s_vol = 0.0 if pd.isna(row['sentiment_volatility']) else float(row['sentiment_volatility'])
            
            # 뉴스 개수는 무조건 정수(int)로 변환
            n_count = 0 if pd.isna(row['news_count']) else int(row['news_count'])

            doc = {
                "_index": self.feature_index,
                "_source": {
                    "stock_code": row['code'],
                    "timestamp": row['last_updated'].isoformat(),
                    "sentiment_score": s_score,
                    "sentiment_decay": s_decay,
                    "sentiment_volatility": s_vol,
                    "news_count": n_count  # 이제 0.0이 아니라 0으로 들어갑니다
                }
            }
            actions.append(doc)

        try:
            # raise_on_error=False로 하면 일부 실패해도 멈추지 않음 (디버깅용)
            success, errors = helpers.bulk(self.es, actions, raise_on_error=False)
            
            if errors:
                print(f"⚠️ {len(errors)}건 저장 실패!")
                # 첫 번째 에러 원인 출력 (디버깅용)
                print(f"🔍 첫 번째 에러 원인: {errors[0]}")
            
            print(f"🚀 가공된 피처 {success}건을 '{self.feature_index}'에 저장했습니다.")
            
        except Exception as e:
            print(f"❌ 저장 시스템 에러: {e}")

if __name__ == "__main__":
    storage = ElasticStorage()
    print(storage.es.info())