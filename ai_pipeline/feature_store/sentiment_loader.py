from elasticsearch import Elasticsearch
from datetime import datetime, timedelta

class SentimentFeatureLoader:
    def __init__(self, es_url="http://localhost:9200"):
        self.es = Elasticsearch(es_url)
        self.index_name = "news_articles"

    def get_latest_sentiment(self, stock_code, hours=24):
        """
        현재 시점 기준 최근 N시간 동안의 뉴스 감성 점수를 집계합니다.
        """
        now = datetime.now()
        start_time = now - timedelta(hours=hours)

        # 쿼리: 해당 종목 + 최근 시간
        should_query = [
            {"term": {"related_stocks.keyword": stock_code}},
            {"term": {"related_stocks.keyword": str(int(stock_code))}} # 270 대응
        ]

        body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"range": {"timestamp": {"gte": start_time.isoformat(), "lte": now.isoformat()}}}
                    ],
                    "should": should_query,
                    "minimum_should_match": 1
                }
            },
            "aggs": {
                "avg_sentiment": {"avg": {"field": "sentiment_score"}}
            }
        }

        try:
            res = self.es.search(index=self.index_name, body=body)
            avg_score = res['aggregations']['avg_sentiment']['value']
            
            # 뉴스가 없으면 중립(0.0) 반환
            if avg_score is None:
                return 0.0
            
            return float(avg_score)

        except Exception as e:
            print(f" [Sentiment] ES 조회 에러: {e}")
            return 0.0