"""
통합 학습 파이프라인
- 피처 생성 1회
- Optuna 튜닝 (선택)
- 모델 학습
- 평가
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.optuna_tuning import HyperparameterTuner
from ai_pipeline.boosting_model.train import StackingEnsemble
from sklearn.model_selection import train_test_split
import json

def run_full_training_pipeline(csv_path, do_tuning=False, n_trials=3):
    """
    전체 학습 파이프라인 실행
    
    Parameters:
    - csv_path: CSV 파일 경로
    - do_tuning: True면 Optuna 튜닝 실행, False면 기본 파라미터 사용
    - n_trials: Optuna trial 횟수 (do_tuning=True일 때만 사용)
    """
    print("="*60)
    print("🚀 통합 학습 파이프라인 시작")
    print("="*60)
    
    # ============================================================
    # 1단계: 피처 생성 (1회만!)
    # ============================================================
    print("\n[1단계] 피처 생성 (CSV + GCN 임베딩)")
    print("-"*60)
    
    engineer = FeatureEngineer(csv_path=csv_path)
    X, y = engineer.create_final_features()
    
    if X is None:
        print("❌ 피처 생성 실패")
        return None
    
    # ============================================================
    # 2단계: 데이터 분할
    # ============================================================
    print("\n[2단계] 데이터 분할")
    print("-"*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   학습: {len(X_train):,}개")
    print(f"   검증: {len(X_val):,}개")
    print(f"   테스트: {len(X_test):,}개")
    
    # ============================================================
    # 3단계: 하이퍼파라미터 튜닝 (선택)
    # ============================================================
    if do_tuning:
        print("\n[3단계] 하이퍼파라미터 튜닝")
        print("-"*60)
        
        tuner = HyperparameterTuner(X_train, y_train, n_trials=n_trials, cv_folds=3)
        best_params = tuner.tune_all()
        
        # 저장
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print("\n💾 최적 파라미터 저장: best_params.json")
    else:
        print("\n[3단계] 하이퍼파라미터 튜닝 스킵")
        print("-"*60)
        print("   기본 파라미터 또는 기존 best_params.json 사용")
    
    # ============================================================
    # 4단계: 모델 학습
    # ============================================================
    print("\n[4단계] 모델 학습")
    print("-"*60)
    
    model = StackingEnsemble()  # best_params.json이 있으면 자동 로드
    model.train(X_train, y_train, X_val, y_val)
    
    # ============================================================
    # 5단계: 최종 평가
    # ============================================================
    print("\n[5단계] 최종 평가")
    print("-"*60)
    
    results = model.evaluate(X_test, y_test)
    
    # ============================================================
    # 6단계: 모델 저장
    # ============================================================
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model.save_model(save_dir=save_path)    
    
    print("\n" + "="*60)
    print("✅ 전체 파이프라인 완료!")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    csv_path = "20251120.csv"
    
    # 옵션 1: 튜닝 없이 빠르게 학습만
    # model, results = run_full_training_pipeline(csv_path, do_tuning=False)
    
    # 옵션 2: 튜닝 + 학습 (시간이 더 걸림)
    model, results = run_full_training_pipeline(csv_path, do_tuning=True, n_trials=3)