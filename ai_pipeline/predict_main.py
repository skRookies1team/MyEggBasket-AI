import sys
import os
import time
import pandas as pd

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 모듈 가져오기
from ai_pipeline.news_source.news_etl_runner import run_finance_news_etl
from ai_pipeline.graph_build.build_edges import build_graph_structure
from ai_pipeline.boosting_model.predict import run_prediction_pipeline
from ai_pipeline.portfolio.rebalancer import PortfolioRebalancer

from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy


# [New] GCN Inference 모듈 (필요시 주석 해제하여 사용)
# from ai_pipeline.gcn_model.inference_gcn import GCNInference

def run_pipeline_with_rebalancing():
    print("\n" + "=" * 60)
    print(" [AI Financial Pipeline] 전체 파이프라인 시작")
    print("=" * 60)

    start_time = time.time()

    # ---------------------------------------------------------
    # 1. 데이터 수집 (ETL)
    # ---------------------------------------------------------
    print("\n [Step 1] 뉴스 및 시세 데이터 수집 (ETL)")
    run_finance_news_etl()

    # ---------------------------------------------------------
    # 2. 피처 엔지니어링 & 그래프 빌드
    # ---------------------------------------------------------
    print("\n [Step 2] 그래프 구조 및 피처 생성")
    build_graph_structure()

    # (참고) 실제 파이프라인에서는 여기서 build_gcn_dataset -> create_final_features가 수행됨
    # predict.py 내부에서 FeatureEngineer가 이 역할을 수행하므로 생략 가능

    # ---------------------------------------------------------
    # 3. AI 모델 예측 (Boosting + GCN Features)
    # ---------------------------------------------------------
    print("\n [Step 3] AI 모델(Boosting) 예측 실행")

    # 예측 결과 DataFrame 받기
    prediction_df = run_prediction_pipeline()

    if prediction_df is None or prediction_df.empty:
        print(" [Stop] 예측 결과가 없어 파이프라인을 중단합니다.")
        return

    print("\n [AI 예측 결과 Top 5]")
    print(prediction_df.sort_values('ai_score', ascending=False).head(5))


    # ---------------------------------------------------------
    # ✅ [Step 3.5] 밸류체인 전략 분석 (근거 도출 & 동반 매수 추천)
    # ---------------------------------------------------------
    print("\n [Step 3.5] 밸류체인 전략 분석 (근거 도출)")
    
    vc_strategy = ValueChainStrategy()
    recommendation_df = vc_strategy.analyze_predictions(prediction_df)
    
    if not recommendation_df.empty:
        print(f"\n >>> 밸류체인 동반 상승 추천 종목 ({len(recommendation_df)}건) <<<")
        # 상위 5개 출력
        for idx, row in recommendation_df.head(5).iterrows():
            print(f" [{idx+1}] {row['Rationale']}")
            print(f"      👉 매수 추천: {row['Main_Stock']} & {row['Target_Stock']}")
            print("-" * 50)
            
        # 결과 저장
        rec_save_path = os.path.join(os.path.dirname(__file__), "value_chain_recommendations.csv")
        recommendation_df.to_csv(rec_save_path, index=False, encoding='utf-8-sig')
        print(f" 밸류체인 추천 결과 저장 완료: {rec_save_path}")
    else:
        print(" -> 조건에 맞는 밸류체인 동반 상승 종목이 없습니다.")


    # ---------------------------------------------------------
    # 4. 포트폴리오 리밸런싱 (제안)
    # ---------------------------------------------------------
    print("\n [Step 4] 포트폴리오 리밸런싱 제안 생성")

    # [가정] 현재 내 계좌 보유 현황 (API 연동시 이 부분을 실제 계좌 잔고로 교체)
    my_current_portfolio = {
        '005930': 10000000,  # 삼성전자 1000만원
        '000660': 5000000,  # 하이닉스 500만원
        '035420': 2000000  # NAVER 200만원
    }

    rebalancer = PortfolioRebalancer(risk_aversion='neutral')

    # 리밸런싱 계산 (현재 자산 총액 유지 가정)
    plan_df = rebalancer.run_ai_rebalancing(my_current_portfolio, prediction_df)

    if not plan_df.empty:
        print("\n [최종 매매 제안서]")
        print("-" * 70)
        # 출력 포맷팅
        print(plan_df.to_string(index=False))
        print("-" * 70)

        # 파일로 저장 (주문 연동용)
        save_path = os.path.join(os.path.dirname(__file__), "final_order_plan.csv")
        plan_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f" 매매 제안서 저장 완료: {save_path}")
    else:
        print(" 리밸런싱 제안 내역이 없습니다.")

    elapsed = time.time() - start_time
    print(f"\n [Done] 모든 작업 완료 ({elapsed:.2f}초)")


if __name__ == "__main__":
    run_pipeline_with_rebalancing()