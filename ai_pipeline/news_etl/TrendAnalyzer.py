from datetime import datetime
from dateutil.relativedelta import relativedelta
from elasticsearch import Elasticsearch
import json

class TrendAnalyzer:
    def __init__(self, es_url="http://localhost:9200"):
        self.es = Elasticsearch(es_url)
        self.index_name = "news_articles"

    def get_period_trends(self):
        """
        최근 1주일, 2주일, 1개월, 3개월 기간별 트렌드 키워드 및 카테고리 추출
        """
        # [수정] 기간 정의를 relativedelta 객체로 변경하여 주/월 단위 혼용 지원
        periods = {
            "1_week": relativedelta(weeks=1),
            "2_weeks": relativedelta(weeks=2),
            "1_month": relativedelta(months=1),
            "3_months": relativedelta(months=3)
        }

        # 최종 결과를 담을 딕셔너리
        final_results = {}

        now = datetime.now()

        print(f"[{now}] 기간별 트렌드 분석 시작...")

        for label, delta in periods.items():
            # 1. 기간 설정 (start_date ~ now)
            # relativedelta 객체(delta)를 직접 빼서 날짜 계산
            start_date = now - delta

            # 2. ES 쿼리 작성 (Aggregation)
            body = {
                "size": 0,  # 문서는 가져오지 않고 집계 결과만 가져옴
                "query": {
                    "range": {
                        "published_date": {  # 기사 발행일 기준
                            "gte": start_date.isoformat(),
                            "lte": now.isoformat()
                        }
                    }
                },
                "aggs": {
                    "top_keywords": {
                        "terms": {
                            "field": "trend_keyword.keyword",  # 키워드 필드
                            "size": 10  # 상위 10개 추출
                        }
                    },
                    "top_categories": {
                        "terms": {
                            "field": "trend_category.keyword",  # 카테고리 필드
                            "size": 5  # 상위 5개 추출
                        }
                    }
                }
            }

            try:
                # 3. ES 검색 실행
                res = self.es.search(index=self.index_name, body=body)

                # 4. 결과 파싱
                keywords = [
                    {"name": b['key'], "count": b['doc_count']}
                    for b in res['aggregations']['top_keywords']['buckets']
                ]

                categories = [
                    {"name": b['key'], "count": b['doc_count']}
                    for b in res['aggregations']['top_categories']['buckets']
                ]

                # 5. 결과 저장
                final_results[label] = {
                    "period_start": start_date.strftime("%Y-%m-%d"),
                    "period_end": now.strftime("%Y-%m-%d"),
                    "keywords": keywords,
                    "categories": categories
                }

            except Exception as e:
                print(f"Error processing {label}: {e}")
                final_results[label] = {"error": str(e)}

        return final_results


# -----------------------------------------------------------
# 실행 예시
# -----------------------------------------------------------
if __name__ == "__main__":
    analyzer = TrendAnalyzer()

    # 분석 실행
    trend_data = analyzer.get_period_trends()

    # 프론트엔드로 보낼 결과 출력 (JSON 형태)
    print(json.dumps(trend_data, indent=2, ensure_ascii=False))