from elasticsearch import Elasticsearch
from datetime import datetime
from ai_pipeline.config.settings import ES_HOST

es = Elasticsearch(ES_HOST)

def save_news_to_es(url, text, chunks, sentiments, related_stocks):
    doc = {
        "url": url,
        "text": text,
        "chunks": chunks,
        "sentiments": sentiments,
        "related_stocks": related_stocks,  # 연관 종목 코드
        "timestamp": datetime.now().isoformat() # 수집 시간 기록
    }

    try:
        resp = es.index(index="news_articles", document=doc)
        print(f"💾 저장 완료: {resp['_id']}") # 로그가 너무 많으면 주석처리
        
    except Exception as e:
        print(f"❌ ES 저장 실패: {e}")


