import sys
import os

# [핵심 수정] 프로젝트 루트 경로를 강제로 추가해야 'ai_pipeline'을 찾을 수 있습니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from ai_pipeline.config.settings import ES_HOST

def fix_missing_scores():
    print("🚑 기존 데이터 감성 점수 복구 작업 시작...")
    
    try:
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            print("❌ ES 연결 실패")
            return

        # 'sentiment_score' 필드가 없는 문서만 검색
        query = {
            "query": {
                "bool": {
                    "must_not": {
                        "exists": {"field": "sentiment_score"}
                    }
                }
            }
        }

        # scan을 사용하여 모든 데이터 순회
        rel = scan(es, query=query, index="news_articles")
        
        updated_count = 0
        
        for hit in rel:
            doc_id = hit['_id']
            source = hit['_source']
            
            # es_uploader.py와 동일하게 'sentiments' 필드를 사용
            sentiments_list = source.get('sentiments', [])
            
            # 리스트가 비어있지 않으면 평균 계산
            if sentiments_list and len(sentiments_list) > 0:
                avg_score = sum(sentiments_list) / len(sentiments_list)
            else:
                avg_score = 0.0
            
            # 해당 문서 업데이트 (평균 점수 추가)
            es.update(
                index="news_articles",
                id=doc_id,
                body={
                    "doc": {
                        "sentiment_score": avg_score
                    }
                }
            )
            updated_count += 1
            
            if updated_count % 500 == 0:
                print(f"   ⚡ {updated_count}개 복구 완료...")

        print(f"\n✅ 복구 완료! 총 {updated_count}개 문서에 점수가 추가되었습니다.")

    except Exception as e:
        print(f"❌ 작업 중 에러 발생: {e}")

if __name__ == "__main__":
    fix_missing_scores()