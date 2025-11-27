import sys
import os
import time
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 우리가 만든 모듈들 가져오기
from ai_pipeline.news_source.news_etl_runner import run_finance_news_etl
from ai_pipeline.graph_build.build_edges import build_graph_structure
from ai_pipeline.graph_build.build_gcn_dataset import create_pytorch_dataset
from ai_pipeline.gcn_model.run_gcn import run_gcn_inference
from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer
# from ai_pipeline.boosting_model.predict import run_prediction (이건 나중에 만들 파일)


def show_value_chain_recommendations():
    """
    [Step 5] 방금 수집된 뉴스에 등장한 종목들의 '밸류체인(유사 종목)'을 분석해서 보여줍니다.
    """
    print("\n📍 [Step 5/5] 최신 뉴스 기반 밸류체인 분석 & 추천")
    
    # 1. ES에서 가장 최신 뉴스 3개만 가져오기
    es = Elasticsearch("http://localhost:9200")
    try:
        resp = es.search(
            index="news_articles",
            body={
                "query": {"match_all": {}},
                "sort": [{"timestamp": "desc"}],
                "size": 3  # 최신 3개만 확인
            }
        )
    except Exception as e:
        print(f"   ⚠️ ES 접속 실패: {e}")
        return

    hits = resp['hits']['hits']
    if not hits:
        print("   ⚠️ 분석할 최신 뉴스가 없습니다.")
        return

    # 2. 분석기 초기화 (GCN 임베딩 로드)
    analyzer = ValueChainAnalyzer()
    
    # 3. 뉴스별로 종목 꺼내서 추천
    checked_stocks = set() # 중복 분석 방지

    for hit in hits:
        source = hit['_source']
        title = source.get('title', '제목 없음') # 제목이 없으면 URL이라도
        if title == '제목 없음': title = source.get('url', '')[:50] + "..."
        
        related_stocks = source.get('related_stocks', [])
        
        if not related_stocks:
            continue

        print(f"\n   📰 뉴스: {title}")
        print(f"      연관 종목: {related_stocks}")

        for stock_code in related_stocks:
            if stock_code in checked_stocks:
                continue
            
            # 밸류체인 분석 실행!
            recommendations = analyzer.find_similar_stocks(stock_code, top_n=3)
            
            if recommendations:
                print(f"      👉 [{stock_code}]의 GCN 밸류체인 추천 (유사그룹):")
                for item in recommendations:
                    print(f"         - {item['code']} (유사도: {item['score']})")
            else:
                print(f"      👉 [{stock_code}] 데이터 부족으로 추천 불가")
            
            checked_stocks.add(stock_code)


def run_full_pipeline():
    print("\n" + "="*60)
    print("🚀 [전체 파이프라인] 실행 시작")
    print("="*60)

    start_time = time.time()

    try:

         # 1. 뉴스 수집 및 ES 저장 (ETL)
        print("\n[Step 1] 뉴스 크롤링 & 저장 (ETL)")
        run_finance_news_etl()

        # 2. 그래프 데이터 구축 (Node/Edge 생성)
        # 새로운 뉴스가 들어왔으니 그래프를 다시 그려야 함
        print("\n[Step 2] 그래프 구조 생성 (Nodes/Edges)")
        build_graph_structure()

        # 3. GCN 데이터셋 변환 (.pt 생성)
        print("\n[Step 3] GCN 데이터셋 변환")
        create_pytorch_dataset()

        # 4. GCN 모델 실행 (임베딩 추출)
        # 학습된 모델을 불러와서 특징만 뽑아냄
        print("\n[Step 4] GCN 임베딩 추출")
        run_gcn_inference()

        # [Step 5] 밸류체인 분석 결과 출력 
        show_value_chain_recommendations()

        # 5. (예정) XGBoost 예측
        # print("\n[Step 5] 최종 등락 예측")
        # run_prediction() 

        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ [전체 파이프라인] 완료! (소요시간: {elapsed:.2f}초)")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 파이프라인 실행 중 치명적인 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_pipeline()