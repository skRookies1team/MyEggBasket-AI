import sys
import os
import time

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 우리가 만든 모듈들 가져오기
from ai_pipeline.news_source.news_etl_runner import run_finance_news_etl
from ai_pipeline.graph_build.build_edges import build_graph_structure
from ai_pipeline.graph_build.build_gcn_dataset import create_pytorch_dataset
from ai_pipeline.gcn_model.run_gcn import run_gcn_inference
# from ai_pipeline.boosting_model.predict import run_prediction (이건 나중에 만들 파일)

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

        # 5. (예정) XGBoost 예측
        # print("\n[Step 5] 최종 등락 예측")
        # run_prediction() 

        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ [전체 파이프라인] 완료! (소요시간: {elapsed:.2f}초)")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 파이프라인 실행 중 치명적인 에러 발생: {e}")

if __name__ == "__main__":
    run_full_pipeline()