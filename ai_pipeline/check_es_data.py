from elasticsearch import Elasticsearch
import json

def check_es():
    print("🕵️‍♂️ ES 데이터 정밀 진단 시작")
    
    try:
        es = Elasticsearch("http://localhost:9200")
        if not es.ping():
            print("❌ ES 연결 실패: 실행 중인지 확인하세요.")
            return

        # 1. 전체 데이터 개수 확인
        count = es.count(index="news_articles")['count']
        print(f"📊 저장된 뉴스 개수: {count}개")
        
        if count == 0:
            print("⚠️ 인덱스는 있는데 데이터가 비어있습니다.")
            return

        # 2. 샘플 데이터 3개만 꺼내보기
        resp = es.search(
            index="news_articles",
            body={
                "size": 3,
                "sort": [{"timestamp": "desc"}], # 최신순
                "_source": ["timestamp", "title", "related_stocks", "sentiment_score"]
            }
        )
        
        print("\n🔍 [데이터 샘플 확인]")
        for i, hit in enumerate(resp['hits']['hits']):
            source = hit['_source']
            print(f"\n[Sample {i+1}]")
            print(f" - 날짜(timestamp): {source.get('timestamp', '없음')}")
            print(f" - 종목(related_stocks): {source.get('related_stocks', '없음')}")
            print(f" - 점수(sentiment_score): {source.get('sentiment_score', '없음')}")
            print(f" - 제목: {source.get('title', '제목없음')[:30]}...")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    check_csv()