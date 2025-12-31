import sys
import os
import time
import pandas as pd
import numpy as np
import requests  # [추가] API 요청용
from dotenv import load_dotenv  # [추가] 환경변수 로드용

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 환경 변수 로드
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

# 설정 값 가져오기
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/app")
TEST_EMAIL = os.getenv("TEST_EMAIL", "testuser@example.com")
TEST_PASSWORD = os.getenv("TEST_PASSWORD", "password1234")

# 모듈 가져오기
from ai_pipeline.news_source.news_etl_runner import run_finance_news_etl
from ai_pipeline.graph_build.build_edges import build_graph_structure
from ai_pipeline.boosting_model.predict import run_prediction_pipeline
from ai_pipeline.portfolio.rebalancer import PortfolioRebalancer
from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy


# [Helper] 로그인 및 포트폴리오 조회 함수
def fetch_my_account_portfolio():
    """
    백엔드 API를 통해 실제 계좌의 보유 종목 및 평가 금액을 조회합니다.
    Returns: { '종목코드': 평가금액(int), ... }
    """
    print(" [API] 백엔드 로그인 시도 중...")
    try:
        # 1. 로그인
        login_res = requests.post(
            f"{BACKEND_API_URL}/auth/login",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
            timeout=5
        )

        if login_res.status_code != 200:
            print(f" [Error] 로그인 실패: {login_res.status_code} {login_res.text}")
            return {}

        token = login_res.json().get("accessToken")
        headers = {"Authorization": f"Bearer {token}"}

        # 2. 잔고 조회 (실전/모의 여부는 쿼리 파라미터로 조정 가능, 여기선 false/true 확인 필요)
        # ai_advisor.py는 virtual=true를, run_realtime_trade.py는 virtual=false를 사용 중입니다.
        # 필요에 따라 'true' 또는 'false'로 변경하세요.
        balance_res = requests.get(
            f"{BACKEND_API_URL}/kis/trade/balance",
            headers=headers,
            params={"virtual": "true"},
            timeout=5
        )

        if balance_res.status_code != 200:
            print(f" [Error] 잔고 조회 실패: {balance_res.status_code} {balance_res.text}")
            return {}

        data = balance_res.json()
        holdings = data.get("holdings", [])

        portfolio = {}
        print(f" [API] 보유 종목 {len(holdings)}개 조회 성공")

        for h in holdings:
            qty = int(h.get("quantity", 0))
            if qty > 0:
                code = h.get("stockCode")
                # 현재가(currentPrice)가 있으면 사용하고, 없으면 평단가(avgPrice) 사용
                try:
                    price = float(h.get("currentPrice") or h.get("avgPrice", 0))
                except:
                    price = 0

                amount = int(qty * price)
                portfolio[code] = amount

        return portfolio

    except Exception as e:
        print(f" [Error] API 연동 중 예외 발생: {e}")
        return {}


def run_pipeline_with_rebalancing():
    print("\n" + "=" * 60)
    print(" [AI Financial Pipeline] 밸류체인 전략 기반 리밸런싱")
    print("=" * 60)

    # ---------------------------------------------------------
    # [Step 1] 내 계좌 보유 현황 (API 연동 적용)
    # ---------------------------------------------------------
    my_portfolio = fetch_my_account_portfolio()

    # API 연동 실패 시 빈 딕셔너리일 수 있음 -> 예외 처리 혹은 빈 상태로 진행
    if not my_portfolio:
        print(" [Warning] 보유 종목이 없거나 조회에 실패했습니다. (빈 포트폴리오로 진행)")

    held_codes = [str(c).strip().zfill(6) for c in my_portfolio.keys()]
    # 포트폴리오 키 포맷팅 (005930 등 6자리 문자열 보장)
    my_portfolio = {str(k).strip().zfill(6): v for k, v in my_portfolio.items()}

    print(f" [Step 1] 현재 보유 종목: {len(held_codes)}개")
    if held_codes:
        print(f"          {held_codes}")

    # ---------------------------------------------------------
    # [Step 2] AI 모델 전체 종목 예측
    # ---------------------------------------------------------

    # [Step 1~2] 데이터 수집 및 그래프 생성 (필요시 주석 해제하여 실행)
    # run_finance_news_etl()
    # build_graph_structure()

    print("\n [Step 2] AI 모델 예측 실행 (전체 종목)")
    prediction_df = run_prediction_pipeline()  # 전체 시장 예측

    if prediction_df is None or prediction_df.empty:
        print(" [Stop] 예측 데이터가 없습니다.")
        return

    prediction_df['code'] = prediction_df['code'].astype(str).str.strip().str.zfill(6)

    # 보유종목과 관계없는 단순 ai 점수 높은 종목 5개 출력
    print("\n" + "-" * 50)
    print(" [전체 종목 AI 예측 Score Top 5]")
    print("-" * 50)
    top_simple = prediction_df.sort_values('ai_score', ascending=False).head(5)
    print(top_simple[['code', 'ai_score', 'opinion']].to_string(index=False))
    print("-" * 50)

    # ---------------------------------------------------------
    # [Step 3] 밸류체인 전략으로 '진짜배기' 신규 종목 발굴 (통합)
    # ---------------------------------------------------------
    print("\n [Step 3] 밸류체인 전략 분석: 주도주 기반 파급 효과(Spillover) 종목 발굴")

    vc_strategy = ValueChainStrategy()

    # 1. 전략 분석 실행
    rec_df = vc_strategy.analyze_predictions(prediction_df)

    recommended_codes = []

    if not rec_df.empty:
        print(f" -> 💡 밸류체인 시너지 종목 {len(rec_df)}개 발견!")

        # 2. 결과 출력 및 저장 (Top 5만 화면 표시)
        print("\n [밸류체인 추천 Top 5]")
        for idx, row in rec_df.head(5).iterrows():
            print(f"  [{idx + 1}] {row['Rationale']}")

        # 전체 결과 파일 저장
        rec_save_path = os.path.join(os.path.dirname(__file__), "value_chain_recommendations.csv")
        rec_df.to_csv(rec_save_path, index=False, encoding='utf-8-sig')
        print(f"\n    * 상세 리포트 저장 완료: {rec_save_path}")

        # 3. 리밸런싱용 코드 추출 (Target_Code)
        recommended_codes = rec_df['Target_Code'].unique().tolist()
        recommended_codes = [str(c).strip().zfill(6) for c in recommended_codes]

    else:
        print(" -> 밸류체인 조건(대장주 급등 & 연관주 동반 상승)에 부합하는 종목이 없습니다.")

    # ---------------------------------------------------------
    # [Step 4] 최종 포트폴리오 유니버스 구성
    # ---------------------------------------------------------
    # 유니버스 = (내 보유 종목) + (밸류체인 추천 종목)
    final_universe_codes = list(set(held_codes + recommended_codes))

    print(f"\n [Step 4] 최종 리밸런싱 대상 확정: 총 {len(final_universe_codes)}개")
    print(f"   - 보유 종목(유지/관리): {len(held_codes)}개")
    print(f"   - 신규 후보(매수/편입): {len(recommended_codes)}개")

    # 예측 결과에서 유니버스에 해당하는 데이터만 추출
    target_prediction_df = prediction_df[prediction_df['code'].isin(final_universe_codes)].copy()

    # 보유종목 중 예측 데이터가 없는 경우 처리
    existing_in_pred = target_prediction_df['code'].tolist()
    missing_codes = [c for c in held_codes if c not in existing_in_pred]

    if missing_codes:
        print(f" [Info] 일부 보유종목 예측 데이터 부재 (0점 처리): {missing_codes}")
        missing_data = [{'code': c, 'ai_score': 0.0, 'opinion': '관망'} for c in missing_codes]
        target_prediction_df = pd.concat([target_prediction_df, pd.DataFrame(missing_data)], ignore_index=True)

    print(f"\n [Step 4] 리밸런싱 대상: 총 {len(target_prediction_df)}개 종목")

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
        # 컬럼 존재 여부 확인 후 출력
        valid_cols = [c for c in display_cols if c in plan_df.columns]
        print(plan_df[valid_cols].to_string(index=False))

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