import sys
import os
import time
from elasticsearch import Elasticsearch

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 모듈 가져오기
from ai_pipeline.news_source.news_etl_runner import run_finance_news_etl
from ai_pipeline.graph_build.build_edges import build_graph_structure
from ai_pipeline.graph_build.build_gcn_dataset import create_pytorch_dataset
from ai_pipeline.gcn_model.run_gcn import train_gcn
from ai_pipeline.gcn_model.value_chain import ValueChainAnalyzer
from ai_pipeline.boosting_model.predict import run_prediction_pipeline
# from ai_pipeline.boosting_model.train_pipeline import run_full_training_pipeline
from ai_pipeline.config.settings import ES_HOST

try:
    from ai_pipeline.trade.ai_advisor import AIAdvisor
except ImportError:
    print(" [Warning] AIAdvisor 모듈을 찾을 수 없습니다. (ai_pipeline/trade/ai_advisor.py 파일 확인 필요)")
    AIAdvisor = None


def show_value_chain_recommendations():
    """
    [Step 5] 최신 뉴스 기반 밸류체인(CSV) 분석 & 추천
    """
    print("\n [Step 5/6] 최신 뉴스 기반 밸류체인 분석 (CSV 팩트 기반)")
    
    es = Elasticsearch(ES_HOST)
    try:
        # 최신 뉴스 10개만 샘플링
        resp = es.search(
            index="news_articles",
            body={
                "query": {"match_all": {}},
                "sort": [{"timestamp": "desc"}],
                "size": 10
            }
        )
    except Exception as e:
        print(f"    ES 접속 실패: {e}")
        return

    hits = resp['hits']['hits']
    if not hits:
        print("    분석할 최신 뉴스가 없습니다.")
        return

    # 분석기 초기화 (CSV 로드)
    analyzer = ValueChainAnalyzer()
    checked_stocks = set()

    for hit in hits:
        source = hit['_source']
        title = source.get('title', source.get('url', ''))[:40] + "..."
        related_stocks = source.get('related_stocks', [])
        
        if not related_stocks: continue

        print(f"\n    뉴스: {title}")
        print(f"      연관 종목: {related_stocks}")

        for stock_code in related_stocks:
            if stock_code in checked_stocks: continue
            
            # 밸류체인 분석 실행
            recommendations = analyzer.find_similar_stocks(stock_code, top_n=3)
            
            if recommendations:
                print(f"       [{stock_code}] 관련 밸류체인 추천:")
                for r in recommendations:
                    # CSV 분석기는 'reason' 필드를 줍니다 (예: 반도체 > 장비)
                    print(f"         - {r['name']} ({r['code']}) | 이유: {r['reason']}")
            else:
                print(f"       [{stock_code}] 밸류체인 데이터에 없음")
            
            checked_stocks.add(stock_code)

def run_full_pipeline():
    print("\n" + "="*60)
    print(" [전체 파이프라인] 자동 실행 시작")
    print("="*60)

    start_time = time.time()

    try:
        # [Step 1] 뉴스 크롤링 & 저장 (ETL)
        # 이제 '오늘의 기업분석 뉴스'를 가져옵니다.
        print("\n [Step 1/6] 뉴스 수집 및 저장 (ETL)")
        run_finance_news_etl()

        # [Step 2] 그래프 구조 업데이트
        print("\n [Step 2/6] 그래프 구조 생성 (Nodes/Edges)")
        build_graph_structure()

        # [Step 3] GCN 데이터셋 변환
        print("\n [Step 3/6] PyTorch 데이터셋 변환")
        create_pytorch_dataset()

        # [Step 4] GCN 모델 실행 (임베딩 추출 - Boosting 모델용)
        print("\n [Step 4/6] GCN 임베딩 벡터 추출")
        train_gcn()
        
        # [Step 5] 밸류체인 분석 (CSV 기반 추천 보여주기)
        show_value_chain_recommendations()

        # [Step 6] 모델 학습
        #print("\n [Step 6/7] Boosting Model 학습 (공시 피처 포함)")
        #try:
        #    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        #    data_dir = os.path.join(project_root, "data")
        #    
        #    if os.path.exists(data_dir):
        #        # 학습 실행 (튜닝 없이 빠른 학습)
        #        model, results = run_full_training_pipeline(data_dir, do_tuning=False)
        #        print(" 모델 학습 완료 (공시 피처가 포함된 상태로 학습됨)")
        #    else:
        #        print(f" 학습 데이터 폴더를 찾을 수 없음: {data_dir}")
        #except Exception as e:
        #    print(f" 모델 학습 중 오류 발생: {e}")
        #    import traceback
        #    traceback.print_exc()

        # [Step 7] XGBoost/LightGBM 최종 예측
        print("\n [Step 7/7] Boosting Model 최종 예측")
        run_prediction_pipeline()
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f" [전체 파이프라인] 완료! (소요시간: {elapsed:.2f}초)")
        print("="*60)
        
        # [Step 8] AI 투자 자문 실행 (Advisor)
        print("\n [Step 8/8] AI 투자 자문 보고서 생성")
        if AIAdvisor:
            try:
                advisor = AIAdvisor()
                advisor.generate_advice() # 분석하고 CSV에 저장함
                print(" -> 자문 보고서 생성 완료 (ai_advice_history.csv)")
            except Exception as e:
                print(f" 자문 생성 중 오류: {e}")
        else:
            print(" AIAdvisor 모듈을 찾을 수 없습니다.")

    except Exception as e:
        print(f"\n 파이프라인 실행 중 에러 발생: {e}")
        
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f" [전체 파이프라인] 완료! (소요시간: {elapsed:.2f}초)")
    print("="*60)
    
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    run_full_pipeline()