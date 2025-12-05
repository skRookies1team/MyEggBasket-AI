import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가 (상황에 따라 유동적으로 작동)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.train import StackingEnsemble

def run_prediction(csv_path=None):
    print("\n" + "="*60)
    print(" [Step 6] XGBoost/LightGBM 최종 예측 및 추천")
    print("="*60)

    # 1. 모델 로드 경로 수정
    # train_pipeline.py에서 저장한 위치와 동일하게 설정 (boosting_model/models)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    
    # 모델 파일 존재 확인
    if not os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
        print(f" 학습된 모델이 없습니다. ({model_dir})")
        print("   -> 먼저 train_pipeline.py를 실행해 모델을 만드세요.")
        return

    model = StackingEnsemble()
    try:
        model.load_model(model_dir)
        print(" 학습된 Stacking 모델 로드 완료")
    except Exception as e:
        print(f" 모델 로드 실패: {e}")
        return

    # 2. 데이터 파일 경로 설정 (폴더 지원 추가)
    if csv_path is None:
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        
        # [수정] 우선순위 변경: 'data' 폴더가 있으면 폴더를 선택
        data_dir_path = os.path.join(project_root, "data")
        legacy_file_path = os.path.join(project_root, "20251120.csv")
        
        if os.path.exists(data_dir_path):
            csv_path = data_dir_path
        elif os.path.exists(legacy_file_path):
            csv_path = legacy_file_path

    if not csv_path or not os.path.exists(csv_path):
        print(f" 예측에 사용할 주식 데이터(폴더 또는 CSV)를 찾을 수 없습니다.")
        return

    print(f" 데이터 로딩 및 피처 생성 중... ({os.path.basename(csv_path)})")
    
    # 3. 피처 엔지니어링
    try:
        # [수정] FeatureEngineer가 이제 data_dir 인자를 받으므로 매개변수명 맞춤
        engineer = FeatureEngineer(data_dir=csv_path)
        features_ret = engineer.create_final_features()

        # 반환값 개수에 따라 유연하게 대처
        X = None
        stock_codes = []
        
        if len(features_ret) == 3:
            X, _, stock_codes = features_ret # 학습용 (X, y, codes)
        elif len(features_ret) == 2:
            X, stock_codes = features_ret    # 예측용 (X, codes)
        else:
            print(f" 피처 생성 반환값 오류: {len(features_ret)}개")
            return

    except Exception as e:
        print(f" 피처 생성 중 에러 발생: {e}")
        return

    if X is None or X.empty:
        print(" 생성된 피처 데이터가 없습니다.")
        return

    # 4. 예측 실행
    print(" AI 예측 실행 중...")
    
    try:
        # predict_proba를 사용하여 '상승 확률' 추출
        probs = model.predict_proba(X)
        
        # 모델에 따라 [하락확률, 상승확률] 2차원 배열일 수 있음
        if hasattr(probs, 'ndim') and probs.ndim == 2 and probs.shape[1] >= 2:
            up_probs = probs[:, 1] # 상승 확률만 추출
        else:
            up_probs = probs # 이미 1차원인 경우
            
    except AttributeError:
        # predict_proba 미지원 시
        print(" predict_proba 미지원 -> predict 결과 사용")
        up_probs = model.predict(X)

    # 5. 결과 정리
    # 종목코드 6자리 포맷팅
    fmt_codes = [str(c).zfill(6) for c in stock_codes]

    temp_df = pd.DataFrame({
        'stock_code': fmt_codes,
        'ai_score': np.round(up_probs * 100, 2),
        'original_index': range(len(fmt_codes)) # 순서 보존용
    })

    final_df = temp_df.drop_duplicates(subset=['stock_code'], keep='last')
    
    print(f"\n 중복 제거 완료: {len(temp_df)}행 -> {len(final_df)}행 (최신 데이터만 유지)")

    top_picks = final_df.sort_values(by='ai_score', ascending=False).head(10)
   

    # 6. 결과 출력
    print("\n [AI 강력 추천 종목 TOP 10]")
    print("-" * 50)
    print(f"{'순위':<4} {'종목코드':<10} {'AI점수':<10} {'추천의견':<10}")
    print("-" * 50)
    
    for rank, row in enumerate(top_picks.itertuples(), 1):
        score = row.ai_score
        
        # 점수에 따른 의견
        if score >= 80: opinion = "강력매수"
        elif score >= 60: opinion = "매수"
        elif score <= 40: opinion = "매도/관망"
        else: opinion = "중립"

        print(f"{rank:<4} {row.stock_code:<10} {score:>5.1f}점      {opinion}")
    
    print("-" * 50)
    
    # (선택사항) 결과 파일 저장
    output_path = os.path.join(project_root, "final_prediction_result.csv")
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f" 전체 결과 저장 완료: {output_path}")

if __name__ == "__main__":
    run_prediction()