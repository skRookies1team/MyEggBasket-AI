from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

print("🔍 날짜별 저장된 종목(stock_code) 분포 확인 중...\n")

# 12월 7일, 8일 각각 어떤 종목이 있는지 집계
for target_date in ["2025-12-07", "2025-12-08"]:
    resp = es.search(
        index="stock_features_v1",
        body={
            "size": 0,
            "query": {
                "range": {
                    "timestamp": {
                        "gte": f"{target_date}T00:00:00",
                        "lte": f"{target_date}T23:59:59"
                    }
                }
            },
            "aggs": {
                "stocks": {
                    "terms": {"field": "stock_code", "size": 10} # 상위 10개 종목만 표시
                }
            }
        }
    )

    print(f"📅 [{target_date}] 데이터 총 {resp['hits']['total']['value']}개")
    buckets = resp['aggregations']['stocks']['buckets']
    
    if not buckets:
        print("   👉 데이터 없음")
    else:
        for b in buckets:
            print(f"   👉 종목코드 {b['key']}: {b['doc_count']}개")
    print("-" * 30)