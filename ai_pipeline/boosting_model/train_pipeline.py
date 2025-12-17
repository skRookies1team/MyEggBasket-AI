"""
통합 학습 파이프라인 (수정됨)
- 피처 생성
- Optuna 튜닝 (선택)
- 모델 학습
- 평가 (직접 구현)
"""
import sys
import os
import json
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
from ai_pipeline.boosting_model.optuna_tuning import HyperparameterTuner
from ai_pipeline.boosting_model.train import StackingEnsemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

def run_full_training_pipeline(data_dir, do_tuning=False, n_trials=3):
    """
    전체 학습 파이프라인 실행

    Parameters:
    - data_dir: CSV 파일 경로
    - do_tuning: True면 Optuna 튜닝 실행, False면 기본 파라미터 사용
    - n_trials: Optuna trial 횟수 (do_tuning=True일 때만 사용)
    """
    print("="*60)
    print(" 통합 학습 파이프라인 시작")
    print("="*60)

    # ============================================================
    # 1단계: 피처 생성 (1회만!)
    # ============================================================
    print("\n[1단계] 피처 생성 (CSV + GCN 임베딩)")
    print("-" * 60)

    engineer = FeatureEngineer(data_dir=data_dir)
    # 캐시된 파일이 있으면 빠르게 로드
    features = engineer.create_final_features(use_cache=True)

    if features is None:
        print(" 피처 생성 실패")
        return None, None

    if len(features) == 3:
        X, y, _ = features # 학습할 땐 종목코드(_) 필요 없음
    elif len(features) == 2:
        X, y = features
    else:
        print(" 피처 생성 실패 (반환값 불일치)")
        return None, None

    if X is None:
        print(" 피처 생성 실패 (데이터 없음)")
        return None, None

    # ============================================================
    # 2단계: 데이터 분할
    # ============================================================
    print("\n[2단계] 데이터 분할")
    print("-" * 60)

    # Stratify로 클래스 비율 유지하며 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Validation Set (Threshold 튜닝 및 Early Stopping용)
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
        print("-" * 60)

        # 튜너 초기화 및 실행
        tuner = HyperparameterTuner(X_train, y_train, n_trials=n_trials, cv_folds=3)
        best_params = tuner.tune_all()

        # 최적 파라미터 저장
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print("\n 최적 파라미터 저장: best_params.json")
    else:
        print("\n[3단계] 하이퍼파라미터 튜닝 스킵")

    # ============================================================
    # 4단계: 모델 학습
    # ============================================================
    print("\n[4단계] 모델 학습")
    print("-" * 60)

    # best_params.json이 있으면 자동으로 로드하여 학습함
    model = StackingEnsemble()
    model.train(X_train, y_train, X_val, y_val)

    # ============================================================
    # 5단계: 최종 평가 (수정됨: 직접 구현)
    # ============================================================
    print("\n[5단계] 최종 평가")
    print("-" * 60)

    # 예측 수행 (Threshold 적용된 Class 0/1 반환)
    y_pred = model.predict(X_test)

    # 지표 계산
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f" [Threshold: {model.best_threshold:.2f}]")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print("-" * 50)
    print(classification_report(y_test, y_pred))

    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    # ============================================================
    # 6단계: 모델 저장
    # ============================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_model_dir = os.path.join(current_dir, "models")

    model.save_model(save_dir=target_model_dir)

    print("\n" + "=" * 60)
    print(" 전체 파이프라인 완료!")
    print("=" * 60)

    return model, results


if __name__ == "__main__":
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))

    # 데이터 폴더 지정 (data 폴더)
    data_dir = os.path.join(project_root, "data")

    # 실행: 튜닝 포함 (이미 튜닝된 파라미터가 있다면 do_tuning=False로 해도 됨)
    # model, results = run_full_training_pipeline(data_dir, do_tuning=False) # 학습만
    model, results = run_full_training_pipeline(data_dir, do_tuning=True, n_trials=3) # 튜닝 후 학습