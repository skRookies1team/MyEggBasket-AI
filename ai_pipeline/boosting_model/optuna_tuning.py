import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
        # [수정] 인덱스 정렬 (Cross-validation 시 에러 방지)
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
        """XGBoost 최적화 목적 함수"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        
        # StratifiedKFold로 클래스 비율 유지
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            model, self.X, self.y, 
            cv=cv, 
            scoring='roc_auc',
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
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            model, self.X, self.y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def tune_xgboost(self):
        """XGBoost 튜닝 실행"""
        print("\n🔧 XGBoost 하이퍼파라미터 튜닝 시작...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_tuning'
        )
        
        study.optimize(
            self.objective_xgboost,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1
        )
        
        self.best_xgb_params = study.best_params
        
        print(f"\n✅ XGBoost 최적 파라미터 (AUC: {study.best_value:.4f})")
        return self.best_xgb_params
    
    def tune_lightgbm(self):
        """LightGBM 튜닝 실행"""
        print("\n🔧 LightGBM 하이퍼파라미터 튜닝 시작...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='lightgbm_tuning'
        )
        
        study.optimize(
            self.objective_lightgbm,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1
        )
        
        self.best_lgb_params = study.best_params
        
        print(f"\n✅ LightGBM 최적 파라미터 (AUC: {study.best_value:.4f})")
        return self.best_lgb_params
    
    def tune_all(self):
        """XGBoost와 LightGBM 모두 튜닝"""
        print("\n" + "="*60)
        print("🎯 하이퍼파라미터 최적화 시작")
        print("="*60)
        
        xgb_params = self.tune_xgboost()
        lgb_params = self.tune_lightgbm()
        
        print("\n" + "="*60)
        print("✅ 하이퍼파라미터 최적화 완료!")
        print("="*60)
        
        return {
            'xgboost': xgb_params,
            'lightgbm': lgb_params
        }


def run_tuning(data_path, n_trials=3):
    """실제 데이터로 하이퍼파라미터 튜닝 실행"""
    print("="*60)
    print("🎯 하이퍼파라미터 튜닝 (실제 데이터)")
    print("="*60)
    
    # 1. 피처 생성 (data_path는 파일일 수도 있고 폴더일 수도 있음)
    # FeatureEngineer가 알아서 처리함
    engineer = FeatureEngineer(data_dir=data_path)
    
    # [핵심 수정] 3개 반환값 처리
    features_ret = engineer.create_final_features()
    
    if len(features_ret) == 3:
        X, y, _ = features_ret
    elif len(features_ret) == 2:
        X, y = features_ret
    else:
        print("❌ 피처 생성 반환값 오류")
        return None
    
    if X is None:
        print("❌ 피처 생성 실패")
        return None
    
    # 2. 튜닝 실행
    tuner = HyperparameterTuner(X, y, n_trials=n_trials, cv_folds=3)
    best_params = tuner.tune_all()
    
    # 3. 결과 저장
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n💾 최적 파라미터 저장 완료: best_params.json")
    
    return best_params


if __name__ == "__main__":
    # 프로젝트 루트의 data 폴더 경로 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    data_path = os.path.join(project_root, "data")
    if not os.path.exists(data_path):
        # 없으면 단일 파일 경로 사용 (테스트용)
        data_path = os.path.join(project_root, "20251120.csv")
    
    if os.path.exists(data_path):
        run_tuning(data_path, n_trials=3)
    else:
        print(f"❌ 데이터 경로를 찾을 수 없습니다: {data_path}")