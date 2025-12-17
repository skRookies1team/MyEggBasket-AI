import sys
import os
import pandas as pd
import json
import numpy as np

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ai_pipeline.feature_store import OnlineFeatureStore
from ai_pipeline.boosting_model.train import StackingEnsemble


def test_realtime_prediction(target_code="005930"):
    print("=" * 60)
    print(f" 🚀 [{target_code}] 실시간 AI 예측 테스트")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Feature Store 초기화 (DB 연결)
    # ---------------------------------------------------------
    try:
        fs = OnlineFeatureStore()
        print(" [Init] Feature Store 연결 성공")
    except Exception as e:
        print(f" [Error] Feature Store 연결 실패: {e}")
        return

    # ---------------------------------------------------------
    # 2. 실시간 피처 조회 (MongoDB + ES + GCN)
    # ---------------------------------------------------------
    print(" [Step 1] 실시간 피처 조회 중...")
    feature_df = fs.get_realtime_features(target_code)

    if feature_df is None:
        print(f" ❌ {target_code}에 대한 피처를 가져오지 못했습니다. (DB 데이터 확인 필요)")
        return

    print(" ✅ 피처 조회 완료")
    # print(feature_df.T) # 피처 값 확인하고 싶으면 주석 해제

    # ---------------------------------------------------------
    # 3. 학습된 모델 로드
    # ---------------------------------------------------------
    print(" [Step 2] AI 모델 로드 중...")
    model_dir = os.path.join("ai_pipeline", "boosting_model", "models")

    if not os.path.exists(os.path.join(model_dir, "meta_model.pkl")):
        print(f" ❌ 모델 파일이 없습니다: {model_dir}")
        print("    -> 먼저 train.py를 실행하여 모델을 학습시켜주세요.")
        return

    model = StackingEnsemble()
    try:
        model.load_model(model_dir)
    except Exception as e:
        print(f" ❌ 모델 로드 중 에러: {e}")
        return

    # ---------------------------------------------------------
    # 4. 예측 실행
    # ---------------------------------------------------------
    print(" [Step 3] 예측 수행 중...")

    # 모델 학습 시 사용한 피처 순서와 맞추기 (OnlineFeatureStore가 DataFrame을 반환하므로 자동 매칭됨)
    try:
        # predict_proba는 (N, 2) 배열 반환 -> [하락확률, 상승확률]
        probs = model.predict_proba(feature_df)

        # 상승 확률 추출
        if hasattr(probs, 'ndim') and probs.ndim == 2:
            up_prob = probs[0, 1]
        else:
            up_prob = probs[0]

        ai_score = round(up_prob * 100, 2)

        # 의견 결정
        if ai_score >= 90:
            opinion = "강력매수"
        elif ai_score >= 70:
            opinion = "매수"
        elif ai_score <= 40:
            opinion = "매도"
        else:
            opinion = "관망"

        print("\n" + "-" * 40)
        print(f" 예측 결과: [{target_code}]")
        print(f" AI 점수 : {ai_score}점")
        print(f" 투자 의견: {opinion}")
        print("-" * 40)

        # 주요 근거 데이터 출력 (참고용)
        print("\n [주요 지표 확인]")
        print(f" - 현재가: {feature_df['close'].values[0]}")
        print(f" - 감성점수: {feature_df['sentiment_score'].values[0]:.4f}")
        print(f" - RSI (14): {feature_df['hist_RSI_14'].values[0]:.2f}")
        print(f" - 변동성(5): {feature_df['volatility_5'].values[0]:.5f}")

    except Exception as e:
        print(f" ❌ 예측 실행 중 에러: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 테스트할 종목 코드 변경 가능
    test_realtime_prediction("005930")  # SK하이닉스 예시