import os
import sys
from elasticsearch import Elasticsearch, helpers
from datetime import datetime
from tqdm import tqdm

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from ai_pipeline.analysis.technical_analysis import TechnicalAnalyzer
from ai_pipeline.mapping.stock_mapping import StockMapper

def run_technical_etl():
    print("📈 [AI Score] 전 종목 기술적 점수 계산 시작...")
    
    es = Elasticsearch("http://localhost:9200")
    analyzer = TechnicalAnalyzer()
    mapper = StockMapper() # 종목 리스트 로드
    
    codes = list(mapper.stock_dict.values())
    print(f"📊 대상 종목: {len(codes)}개")

    actions = []
    timestamp = datetime.now().isoformat()

    # 진행률 표시 (tqdm)
    for code in tqdm(codes):
        # 1. 점수 계산
        ai_score = analyzer.get_technical_score(code)
        
        # 2. ES 저장 데이터 생성
        doc = {
            "_index": "stock_technicals", # 기술적 점수 전용 인덱스
            "_source": {
                "stock_code": code,
                "ai_score": ai_score, # 0~100점
                "timestamp": timestamp
            }
        }
        actions.append(doc)
        
        # 500개씩 끊어서 저장 (Bulk Insert)
        if len(actions) >= 500:
            helpers.bulk(es, actions)
            actions = []

    # 남은 거 저장
    if actions:
        helpers.bulk(es, actions)
    
    print(f"✅ AI 스코어 계산 및 저장 완료! (인덱스: stock_technicals)")

if __name__ == "__main__":
    run_technical_etl()