import hashlib
from elasticsearch import Elasticsearch
from datetime import datetime
from ai_pipeline.config.settings import ES_HOST

# ES 연결
es = Elasticsearch(ES_HOST)
INDEX_NAME = "news_articles"

def generate_id_from_url(url):
    """URL 문자열을 MD5 해시로 변환하여 고유 ID 생성"""
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def exists_in_es(url):
    """이미 저장된 뉴스인지 확인"""
    doc_id = generate_id_from_url(url)
    try:
        return es.exists(index=INDEX_NAME, id=doc_id)
    except Exception:
        return False

def save_news_to_es(url, text, chunks, sentiments, related_stocks):
    """
    뉴스 데이터 저장 (평균 감성 점수 계산 포함)
    """
    doc_id = generate_id_from_url(url)
    
    # [핵심] 리스트(sentiments)의 평균을 구해 단일 점수(sentiment_score)로 변환
    if sentiments and len(sentiments) > 0:
        avg_score = sum(sentiments) / len(sentiments)
    else:
        avg_score = 0.0

    doc = {
        "url": url,
        "text": text,
        "chunks": chunks,
        "sentiments": sentiments,    # 개별 문장 점수 리스트
        "related_stocks": related_stocks,
        "timestamp": datetime.now().isoformat(),
        "sentiment_score": avg_score # [중요] AI가 사용할 평균 점수
    }

    try:
        # ES 버전에 따라 document=doc 또는 body=doc 사용
        resp = es.index(index=INDEX_NAME, id=doc_id, document=doc)
        # print(f"💾 저장 완료: {resp['_id']}") 
    except Exception as e:
        print(f"❌ ES 저장 실패: {e}")