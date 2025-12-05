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


def save_news_to_es(url, title, text, related_stocks, analysis_results, sentence_details, value_chain_info):
    """
    [최종] 뉴스 및 정밀 분석 데이터 저장
    
    Parameters:
    - url: 뉴스 링크
    - title: 뉴스 제목
    - text: 뉴스 원문
    - related_stocks: 원문 내 발견된 종목 코드 리스트 ['005930', ...]
    - analysis_results: 종목별 지표 (score, trend, volatility) -> sentiment_1h 등
    - sentence_details: 문장별 분석 데이터 (문장, 티커, 점수)
    - value_chain_info: 밸류체인 연관 종목 정보
    """
    doc_id = generate_id_from_url(url)
    
    # ES에 저장될 최종 문서 구조
    doc = {
        "url": url,
        "title": title,
        "text": text,
        "related_stocks": related_stocks,  # 원문 등장 종목
        "value_chain_stocks": value_chain_info, # 밸류체인 연관 종목 (JSON List)
        "timestamp": datetime.now().isoformat(),
        
        # 1. 문장 단위 상세 데이터
        "sentences": sentence_details, 
        # 예: [{"sentence": "...", "ticker": "005930", "sentiment": 0.9}, ...]

        # 2. 종목별 종합 지표 (sentiment_1h, trend, volatility)
        "sentiment_analysis": [] 
    }

    # analysis_results 딕셔너리를 리스트로 변환 (매핑 폭발 방지)
    if analysis_results:
        for code, metrics in analysis_results.items():
            doc["sentiment_analysis"].append({
                "code": code,
                "sentiment_1h": metrics['sentiment_score'],
                "sentiment_trend": metrics['sentiment_trend'],
                "sentiment_volatility": metrics['sentiment_volatility'],
                "mention_count": metrics['mention_count']
            })

    try:
        es.index(index=INDEX_NAME, id=doc_id, document=doc)
        # print(f" 저장 완료: {title[:20]}...")
    except Exception as e:
        print(f" ES 저장 실패: {e}")