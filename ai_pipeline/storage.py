from elasticsearch import Elasticsearch, helpers

class ElasticStorage:
    def __init__(self, host="http://localhost:9200"):
        self.es = Elasticsearch(host)
        
        # [NEW] 집계된 데이터가 들어갈 새로운 인덱스 이름
        self.feature_index = "stock_features_v1"
        
        # 인덱스가 없으면 생성
        if not self.es.indices.exists(index=self.feature_index):
            self._create_feature_index()

    def _create_feature_index(self):
        """AI 모델용 피처 인덱스 매핑"""
        mapping = {
            "mappings": {
                "properties": {
                    "stock_code": {"type": "keyword"},  # 종목코드 (검색용)
                    "timestamp": {"type": "date"},      # 기준 시간
                    
                    # [핵심] 우리가 만든 파생 변수들
                    "sentiment_score": {"type": "float"},      # 단순 평균
                    "sentiment_decay": {"type": "float"},      # 시간 가중
                    "sentiment_volatility": {"type": "float"}, # 변동성
                    "news_count": {"type": "integer"}          # 뉴스 개수
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
            doc = {
                "_index": self.feature_index, # 여기로 저장!
                "_source": {
                    "stock_code": row['code'],
                    "timestamp": row['last_updated'].isoformat(),
                    "sentiment_score": row['sentiment_1h'],
                    "sentiment_decay": row['sentiment_decay'],
                    "sentiment_volatility": row['sentiment_volatility'],
                    "news_count": row['news_count']
                }
            }
            actions.append(doc)

        try:
            success, _ = helpers.bulk(self.es, actions)
            print(f"🚀 가공된 피처 {success}건을 '{self.feature_index}'에 저장했습니다.")
        except Exception as e:
            print(f"❌ 저장 실패: {e}")