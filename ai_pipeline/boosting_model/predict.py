import sys
import os
import pandas as pd
import numpy as np


# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.train import StackingEnsemble
from ai_pipeline.strategy.value_chain_strategy import ValueChainStrategy

def run_prediction_pipeline(csv_path=None):

    # 1. 모델 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")

    if not os.path.exists(os.path.join(model_dir, 'meta_model.pkl')):
        print(" [Error] 학습된 모델이 없습니다. 먼저 학습을 진행하세요.")
        return None

    model = StackingEnsemble()
    try:
        model.load_model(model_dir)
    except Exception as e:
        print(f" 모델 로드 실패: {e}")
        return None

    # 2. 데이터 준비
    if csv_path is None:
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        data_dir = os.path.join(project_root, "data", "krx_data")
        if not os.path.exists(data_dir):
            data_dir = os.path.join(project_root, "data")

        engineer = FeatureEngineer(data_dir=data_dir)
    else:
        engineer = FeatureEngineer(csv_path=csv_path)

    # 3. 피처 생성 및 예측
    try:
        # 캐시 사용 (속도 최적화)
        features_ret = engineer.create_final_features(use_cache=True)

        if len(features_ret) == 3:
            X, _, stock_codes = features_ret
        elif len(features_ret) == 2:
            X, stock_codes = features_ret
        else:
            return None

        if X is None or X.empty:
            print(" 예측할 데이터가 없습니다.")
            return None

        print(f" 총 {len(X)}개 종목에 대해 예측 수행 중...")
        probs = model.predict_proba(X)
        if hasattr(probs, 'ndim') and probs.ndim == 2:
            up_probs = probs[:, 1]
        else:
            up_probs = probs

        # 4. 결과 생성
        fmt_codes = [str(c).zfill(6) for c in stock_codes]
        result_df = pd.DataFrame({
            'code': fmt_codes,
            'ai_score': np.round(up_probs * 100, 2)
        })
        result_df = result_df.drop_duplicates(subset=['code'], keep='last')

        conditions = [
            (result_df['ai_score'] >= 90),  # 강력매수 기준 상향
            (result_df['ai_score'] >= 70),  # 매수 기준 상향
            (result_df['ai_score'] <= 40)
        ]
        choices = ['강력매수', '매수', '매도/관망']
        result_df['opinion'] = np.select(conditions, choices, default='중립')

        return result_df

    except Exception as e:
        print(f" 예측 파이프라인 에러: {e}")
        return None


if __name__ == "__main__":
    df = run_prediction_pipeline()
    if df is not None:
        print("\n [AI 예측 결과 Top 10]")
        print(df.sort_values('ai_score', ascending=False).head(10))

        # ---------------------------------------------------------
        #  [Step 3.5] 밸류체인 분석 (이 파일 단독 실행 시 작동)
        # ---------------------------------------------------------
        if ValueChainStrategy:
            print("\n" + "=" * 60)
            print(" [Step 3.5] 밸류체인 전략 분석 (Predict 단독 실행)")
            print("=" * 60)
            
            vc_strategy = ValueChainStrategy()
            recommendation_df = vc_strategy.analyze_predictions(df)
            
            if not recommendation_df.empty:
                print(f"\n >>> 밸류체인 동반 상승 추천 종목 ({len(recommendation_df)}건) <<<")
                for idx, row in recommendation_df.head(5).iterrows():
                    print("-" * 70)
                    print(f"[{idx+1}] 메인: {row['Main_Stock']}({row['Main_Score']}점) -> 타겟: {row['Target_Stock']}({row['Target_Score']}점)")
                    print(f" 근거: {row['Rationale']}") 
                    print("-" * 70)
            else:
                print(" -> 조건에 맞는 밸류체인 동반 상승 종목이 없습니다.")
