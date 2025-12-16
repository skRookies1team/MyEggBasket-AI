import hashlib
from elasticsearch import Elasticsearch
from datetime import datetime
from ai_pipeline.config.settings import ES_HOST

es = Elasticsearch(ES_HOST)
INDEX_NAME = "news_articles"

def generate_id_from_url(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def exists_in_es(url):
    doc_id = generate_id_from_url(url)
    try:
        return es.exists(index=INDEX_NAME, id=doc_id)
    except Exception:
        return False

def save_news_to_es(url, title, text, published_date, related_stocks, analysis_results, sentence_details, value_chain_info, category=None, keyword=None):
    """
    뉴스 데이터 저장 (구조 변경: Map -> List)
    """
    doc_id = generate_id_from_url(url)
    
    # 1. 대표 감성 점수 계산
    avg_score = 0.0
    if analysis_results:
        scores = [float(v.get('sentiment_score', 0)) for v in analysis_results.values()]
        if scores:
            avg_score = sum(scores) / len(scores)

    # 2. [핵심 수정] analysis_results (Dict) -> List[Dict] 변환
    # 예: {"005930": {...}} -> [{"stock_code": "005930", ...}]
    # 이렇게 해야 ES 필드 개수 제한(1000개)에 걸리지 않음
    analysis_list = []
    if analysis_results and isinstance(analysis_results, dict):
        for code, data in analysis_results.items():
            # data가 딕셔너리라고 가정하고 복사 후 코드 추가
            if isinstance(data, dict):
                item = data.copy()
                item['stock_code'] = code
                analysis_list.append(item)
            else:
                # 데이터가 단순 값일 경우 대비
                analysis_list.append({'stock_code': code, 'value': data})
    
    # 만약 이미 리스트라면 그대로 사용
    elif isinstance(analysis_results, list):
        analysis_list = analysis_results

    doc = {
        "url": url,
        "title": title,
        "text": text,
        "related_stocks": related_stocks,
        
        "analysis_results": analysis_list,      # [수정됨] 리스트 형태로 저장
        "sentence_details": sentence_details,
        "value_chain_info": value_chain_info,
        
        "sentiment_score": avg_score,
        "published_date": published_date,
        "timestamp": datetime.now().isoformat(),
        "trend_category": category,
        "trend_keyword": keyword,
    }

    try:
        resp = es.index(index=INDEX_NAME, id=doc_id, document=doc)
    except Exception as e:
        print(f" ES 저장 실패: {e}")