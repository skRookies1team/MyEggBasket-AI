import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.train import StackingEnsemble

def run_prediction(csv_path=None):
    print("\n" + "="*60)
    print("🔮 [Step 6] XGBoost/LightGBM 최종 예측 및 추천")
    print("="*60)

    # 1. 모델 로드
    # 모델 파일이 저장된 경로 (boosting_model/models/)
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    
    if not os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
        print("⚠️ 학습된 모델이 없습니다. 먼저 train_pipeline.py를 실행해 모델을 만드세요.")
        return

    model = StackingEnsemble()
    try:
        model.load_model(model_dir)
        print("✅ 학습된 Stacking 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 2. 예측할 데이터 준비 (CSV + GCN)
    # csv_path가 없으면 프로젝트 루트에서 가장 최근 csv를 찾거나 고정된 파일 사용
    if csv_path is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        # [주의] 실제 서비스에선 API로 실시간 데이터를 받아와야 합니다.
        # 지금은 테스트용으로 기존 CSV를 사용합니다.
        csv_path = os.path.join(root_dir, "20251120.csv") 

    if not os.path.exists(csv_path):
        print(f"❌ 예측에 사용할 주식 데이터(CSV)가 없습니다: {csv_path}")
        return

    print(f"📊 데이터 로딩 및 피처 생성 중... ({os.path.basename(csv_path)})")
    
    # FeatureEngineer를 통해 GCN 임베딩까지 합친 데이터 생성
    engineer = FeatureEngineer(csv_path=csv_path)
    X, _, stock_codes = engineer.create_final_features() # y(타겟)는 예측 땐 필요 없음

    if X is None:
        print("❌ 피처 생성 실패")
        return

    # 3. 예측 실행 (확률값 추출)
    print("🚀 AI 예측 실행 중...")
    probs = model.predict(X) # 0~1 사이의 확률값 (상승 확률)

    # 4. 결과 정리 (데이터프레임)
    results_df = pd.DataFrame({
        'stock_code': stock_codes,
        'ai_score': np.round(probs * 100, 2) # 100점 만점 점수로 변환
    })

    # 점수 높은 순 정렬
    top_picks = results_df.sort_values(by='ai_score', ascending=False).head(10)

    # 5. 결과 출력
    print("\n🏆 [AI 강력 추천 종목 TOP 10]")
    print("-" * 50)
    print(f"{'순위':<4} {'종목코드':<10} {'AI점수':<10} {'추천의견':<10}")
    print("-" * 50)
    
    for rank, row in enumerate(top_picks.itertuples(), 1):
        score = row.ai_score
        
        # 점수에 따른 의견
        if score >= 80: opinion = "강력매수"
        elif score >= 60: opinion = "매수"
        else: opinion = "관망"

        print(f"{rank:<4} {row.stock_code:<10} {score:>5.1f}점      {opinion}")
    
    print("-" * 50)
    print("✅ 예측 완료. 이 데이터를 포트폴리오 관리에 활용하세요.")

if __name__ == "__main__":
    run_prediction()