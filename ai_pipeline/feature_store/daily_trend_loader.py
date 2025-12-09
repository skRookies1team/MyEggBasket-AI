import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta

class DailyTrendLoader:
    def __init__(self):
        self.es = Elasticsearch("http://localhost:9200")
        self.index_name = "stock_features_v1"  # ✅ 인덱스 이름 확인 (v1)

    def get_daily_trend(self, stock_code, days=30):
        # 조회 기간 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 1. Elasticsearch Aggregation 쿼리
        # (날짜별로 그룹화 -> 문서 개수(count) + 감성점수 평균(avg) 계산)
        body = {
            "size": 0,  # 개별 문서는 안 가져오고 집계 결과만 가져옴
            "query": {
                "bool": {
                    "must": [
                        {"term": {"stock_code": stock_code}},
                        {"range": {"timestamp": {"gte": start_date.strftime("%Y-%m-%d"), "lte": end_date.strftime("%Y-%m-%d")}}}
                    ]
                }
            },
            "aggs": {
                "daily_stats": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "day",
                        "format": "yyyy-MM-dd",
                        "time_zone": "+09:00"  # 한국 시간 기준
                    },
                    "aggs": {
                        "avg_sentiment": {
                            "avg": {"field": "sentiment_score"}  # ✅ 필드명 확인
                        }
                    }
                }
            }
        }

        try:
            res = self.es.search(index=self.index_name, body=body)
            buckets = res['aggregations']['daily_stats']['buckets']

            data = []
            for b in buckets:
                date_str = b['key_as_string']
                # key_as_string이 날짜+시간(T)으로 나올 경우 앞부분만 자름
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                
                count = b['doc_count']
                avg_score = b['avg_sentiment']['value']
                
                # avg_score가 None이면 0.0 처리
                if avg_score is None:
                    avg_score = 0.0

                data.append({
                    'date': date_str,
                    'avg_sentiment': avg_score,
                    'buzz_volume': count
                })

            if not data:
                print(f"⚠️ [{stock_code}] 해당 기간 데이터가 없습니다.")
                return pd.DataFrame()

            # DataFrame 변환
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # 이동 평균 (MA3) 계산
            df['ma3_sentiment'] = df['avg_sentiment'].rolling(window=3, min_periods=1).mean()
            
            # 소수점 정리
            df['avg_sentiment'] = df['avg_sentiment'].round(4)
            df['ma3_sentiment'] = df['ma3_sentiment'].round(4)

            return df

        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------
# 실행 테스트
# -----------------------------------------------------------
if __name__ == "__main__":
    loader = DailyTrendLoader()
    
    # 삼성전자(005930) 테스트
    print("\n🔍 삼성전자(005930) 데이터 조회 시도...")
    df = loader.get_daily_trend("000660")
    
    if not df.empty:
        print(f"\n📊 [005930] 시계열 트렌드 분석 결과 (최근 10일):")
        print(df.tail(10))
    else:
        print("결과가 비어있습니다.")