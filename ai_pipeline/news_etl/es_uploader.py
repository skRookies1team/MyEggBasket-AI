import hashlib
from elasticsearch import Elasticsearch
from datetime import datetime
from ai_pipeline.config.settings import ES_HOST

# ES 연결
es = Elasticsearch("http://localhost:9200")
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

def save_news_to_es(url, text=None, related_stocks=None,
                    analysis_results=None, sentiments=None):
    doc_id = generate_id_from_url(url)

    # 업데이트할 필드만 묶기
    update_fields = {}

    if analysis_results is not None:
        update_fields["stock_analysis"] = analysis_results

    if sentiments is not None:
        update_fields["sentiments"] = sentiments

    # 기존 문서는 유지하고 지정된 필드만 덮어쓰기
    try:
        es.update(
            index=INDEX_NAME,
            id=doc_id,
            body={"doc": update_fields}
        )
    except Exception as e:
        print(f"❌ ES 업데이트 실패: {e}")
