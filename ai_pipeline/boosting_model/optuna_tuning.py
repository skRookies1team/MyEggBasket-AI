import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import numpy as np
import warnings
import json
import sys
import os

warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer


class HyperparameterTuner:
    """Optuna를 사용한 XGBoost & LightGBM 하이퍼파라미터 최적화"""

    def __init__(self, X, y, n_trials=100, cv_folds=3):
        # [수정] 인덱스 정렬
        if hasattr(X, 'reset_index'):
            self.X = X.reset_index(drop=True)
        else:
            self.X = X

        if hasattr(y, 'reset_index'):
            self.y = y.reset_index(drop=True)
        else:
            self.y = y

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_xgb_params = None
        self.best_lgb_params = None

    def objective_xgboost(self, trial):
        """XGBoost 최적화 목적 함수 (안정성 위해 CPU 모드 사용)"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': 1.0,
            # [알림] GPU를 쓰려면 xgboost>=2.1.0 설치 후 아래 주석 해제
            # 'tree_method': 'hist',
            # 'device': 'apple',
            # 'n_jobs': 1,

            'n_jobs': -1,  # CPU 모드일 땐 모든 코어 사용
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        scores = cross_val_score(
            model, self.X, self.y,
            cv=cv,
            scoring='average_precision',
            n_jobs=-1
        )

        return scores.mean()

    def objective_lightgbm(self, trial):
        """LightGBM 최적화 목적 함수"""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': 1.0,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring='average_precision', n_jobs=-1)
        return scores.mean()

    def tune_xgboost(self):
        print("\n XGBoost 하이퍼파라미터 튜닝 시작... (CPU)")
        study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
        study.optimize(self.objective_xgboost, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)
        self.best_xgb_params = study.best_params
        print(f"\n XGBoost 최적 파라미터 (AUC: {study.best_value:.4f})")
        return self.best_xgb_params

    def tune_lightgbm(self):
        print("\n LightGBM 하이퍼파라미터 튜닝 시작... (CPU)")
        study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
        study.optimize(self.objective_lightgbm, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)
        self.best_lgb_params = study.best_params
        print(f"\n LightGBM 최적 파라미터 (AUC: {study.best_value:.4f})")
        return self.best_lgb_params

    def tune_all(self):
        print("\n" + "=" * 60)
        print(" 하이퍼파라미터 최적화 시작")
        print("=" * 60)
        xgb_params = self.tune_xgboost()
        lgb_params = self.tune_lightgbm()
        return {'xgboost': xgb_params, 'lightgbm': lgb_params}


def run_tuning(data_path, n_trials=3):
    """실제 데이터로 하이퍼파라미터 튜닝 실행"""
    print("=" * 60)
    print(" 하이퍼파라미터 튜닝 (실제 데이터)")
    print("=" * 60)

    engineer = FeatureEngineer(data_dir=data_path)
    features_ret = engineer.create_final_features()

    if len(features_ret) == 3:
        X, y, _ = features_ret
    elif len(features_ret) == 2:
        X, y = features_ret
    else:
        print(" 피처 생성 반환값 오류")
        return None

    if X is None:
        print(" 피처 생성 실패")
        return None

    # [추가] 튜닝 속도 향상을 위한 데이터 샘플링 (Downsampling)
    # 데이터가 50만개를 넘으면 튜닝 때는 50만개만 사용
    if len(X) > 500000:
        print(f" [Tuning] 데이터가 너무 많습니다 ({len(X):,}개). 튜닝 속도를 위해 50만개로 샘플링합니다.")
        # Stratified Sampling으로 비율 유지하며 추출
        sample_size = 500000
        _, X_tune, _, y_tune = train_test_split(X, y, test_size=sample_size, stratify=y, random_state=42)

        # 인덱스 초기화 (안전장치)
        X = X_tune.reset_index(drop=True)
        y = y_tune.reset_index(drop=True)
        print(f" [Tuning] 샘플링 완료: {len(X):,}개 사용")

    tuner = HyperparameterTuner(X, y, n_trials=n_trials, cv_folds=3)
    best_params = tuner.tune_all()

    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    print("\n 최적 파라미터 저장 완료: best_params.json")
    return best_params


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    data_path = os.path.join(project_root, "data")

    if os.path.exists(data_path):
        run_tuning(data_path, n_trials=3)
    else:
        print(f" 데이터 경로를 찾을 수 없습니다: {data_path}")