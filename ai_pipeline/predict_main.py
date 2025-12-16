import sys
import os
import time
import pandas as pd
import numpy as np

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
    print(" [AI Financial Pipeline] 밸류체인 전략 기반 리밸런싱")
    print("=" * 60)

    # ---------------------------------------------------------
    # [Step 1] 내 계좌 보유 현황 (API 연동 가정)
    # ---------------------------------------------------------
    my_portfolio = {
        '047050': 10000000,  # 포스코인터내셔널
        '006405': 5000000,  # 윤성에프앤씨
        '066570': 2000000,  # LG전자
        '005930': 15000000,  # 삼성전자
    }
    held_codes = list(my_portfolio.keys())
    print(f" [Step 1] 현재 보유 종목: {len(held_codes)}개")

    # ---------------------------------------------------------
    # [Step 2] AI 모델 전체 종목 예측
    # ---------------------------------------------------------

    # [Step 1~2] 데이터 수집 및 그래프 생성 (필요시 주석 해제하여 실행)
    run_finance_news_etl()
    build_graph_structure()
    # (주의: GCN 학습은 시간이 오래 걸리므로 여기서는 생략하고 기존 모델 사용 가정)

    print("\n [Step 2] AI 모델 예측 실행 (전체 종목)")
    prediction_df = run_prediction_pipeline()  # 전체 시장 예측

    if prediction_df is None or prediction_df.empty:
        print(" [Stop] 예측 데이터가 없습니다.")
        return

    # ---------------------------------------------------------
    # [Step 3] 유망 종목 필터링 (Strategy 적용)
    # ---------------------------------------------------------
    print("\n [Step 3] 밸류체인 전략으로 '진짜배기' 신규 종목 발굴")

    # 전략 실행
    vc_strategy = ValueChainStrategy()
    rec_df = vc_strategy.analyze_predictions(prediction_df)

    recommended_codes = []
    if not rec_df.empty:
        # 상위 5개 테마(쌍)만 선정하여 집중 투자
        top_picks = rec_df.head(5)
        recommended_codes = top_picks['Target_Code'].unique().tolist()

        print(f" -> 밸류체인 시너지 종목 {len(recommended_codes)}개 발견:")
        for _, row in top_picks.iterrows():
            print(f"    * {row['Target_Stock']}({row['Target_Score']}점): {row['Rationale']}")
    else:
        print(" -> 밸류체인 조건에 부합하는 강력한 신규 추천 종목이 없습니다.")


    # ---------------------------------------------------------
    #  [Step 3.5] 밸류체인 전략 분석 (근거 도출 & 동반 매수 추천)
    # ---------------------------------------------------------
    print("\n [Step 3.5] 밸류체인 전략 분석 (근거 도출)")
    
    vc_strategy = ValueChainStrategy()
    recommendation_df = vc_strategy.analyze_predictions(prediction_df)
    
    if not recommendation_df.empty:
        print(f"\n >>> 밸류체인 동반 상승 추천 종목 ({len(recommendation_df)}건) <<<")
        # 상위 5개 출력
        for idx, row in recommendation_df.head(5).iterrows():
            print(f" [{idx+1}] {row['Rationale']}")
            print(f"       매수 추천: {row['Main_Stock']} & {row['Target_Stock']}")
            print("-" * 50)
            
        # 결과 저장
        rec_save_path = os.path.join(os.path.dirname(__file__), "value_chain_recommendations.csv")
        recommendation_df.to_csv(rec_save_path, index=False, encoding='utf-8-sig')
        print(f" 밸류체인 추천 결과 저장 완료: {rec_save_path}")
    else:
        print(" -> 조건에 맞는 밸류체인 동반 상승 종목이 없습니다.")


    # ---------------------------------------------------------
    # [Step 4] 최종 포트폴리오 유니버스 구성
    # ---------------------------------------------------------
    # 유니버스 = (내 보유 종목) + (밸류체인 추천 종목)
    # * 잡주는 제외되고, 검증된 종목만 리밸런싱 대상이 됨
    final_universe_codes = list(set(held_codes + recommended_codes))

    print(f"\n [Step 4] 최종 리밸런싱 대상 확정: 총 {len(final_universe_codes)}개")
    print(f"   - 보유 종목(유지/관리): {len(held_codes)}개")
    print(f"   - 신규 후보(매수/편입): {len(recommended_codes)}개")

    # 예측 결과에서 유니버스에 해당하는 데이터만 추출
    target_prediction_df = prediction_df[prediction_df['code'].isin(final_universe_codes)].copy()

    # ---------------------------------------------------------
    # [Step 5] 리밸런싱 시뮬레이션
    # ---------------------------------------------------------
    print("\n [Step 5] 최적 비중 산출 (Risk Neutral)")
    rebalancer = PortfolioRebalancer(risk_aversion='neutral')

    # 내 계좌 + 선별된 유니버스로 리밸런싱 진행
    plan_df = rebalancer.run_ai_rebalancing(my_portfolio, target_prediction_df)

    if not plan_df.empty:
        print("\n [최종 매매 제안서]")
        print("-" * 80)
        # 출력 포맷 정리
        display_cols = ['code', 'ai_score', 'action', 'diff', 'target_ratio']
        print(plan_df[display_cols].to_string(index=False))

        # 파일 저장
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_order_plan.csv")
        plan_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print("-" * 80)
        print(f" 저장 완료: {save_path}")

        # (옵션) 밸류체인 근거 파일도 저장
        if not rec_df.empty:
            rec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "value_chain_rationale.csv")
            rec_df.to_csv(rec_path, index=False, encoding='utf-8-sig')
            print(f" 추천 근거 저장 완료: {rec_path}")

    print("\n [Done] 파이프라인 종료")


if __name__ == "__main__":
    run_pipeline_with_rebalancing()