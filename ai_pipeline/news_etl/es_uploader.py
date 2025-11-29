import hashlib
from elasticsearch import Elasticsearch
from datetime import datetime
from ai_pipeline.config.settings import ES_HOST

es = Elasticsearch(ES_HOST)
INDEX_NAME = "news_articles"

def generate_id_from_url(url):
    """
    URL 문자열을 MD5 해시로 변환하여 고유 ID 생성
    (긴 URL을 그대로 ID로 쓰기보다 해시값이 인덱싱에 유리함)
    """
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def exists_in_es(url):
    """
    이미 저장된 뉴스인지 확인 (중복 수집 방지용)
    """
    doc_id = generate_id_from_url(url)
    try:
        # HEAD 요청: 문서를 다 가져오지 않고 존재 여부만 가볍게 확인
        return es.exists(index=INDEX_NAME, id=doc_id)
    except Exception:
        # 인덱스가 없거나 에러 시 없다고 가정
        return False


def save_news_to_es(url, text, chunks, sentiments, related_stocks):
    doc_id = generate_id_from_url(url)
    doc = {
        "url": url,
        "text": text,
        "chunks": chunks,
        "sentiments": sentiments,
        "related_stocks": related_stocks,  # 연관 종목 코드
        "timestamp": datetime.now().isoformat() # 수집 시간 기록
    }

    try:
        resp = es.index(index=INDEX_NAME, id=doc_id, document=doc)
        print(f"💾 저장 완료: {resp['_id']}") # 로그가 너무 많으면 주석처리
        
    except Exception as e:
        print(f"❌ ES 저장 실패: {e}")


