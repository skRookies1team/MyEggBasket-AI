import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """Optuna를 사용한 XGBoost & LightGBM 하이퍼파라미터 최적화"""
    
    def __init__(self, X, y, n_trials=50):
        self.X = X
        self.y = y
        self.n_trials = n_trials
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
        
        # 3-Fold CV로 평가 (빠른 속도를 위해 3폴드)
        scores = cross_val_score(
            model, self.X, self.y, 
            cv=3, 
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
        
        scores = cross_val_score(
            model, self.X, self.y,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return scores.mean()
    
    def tune_xgboost(self):
        """XGBoost 튜닝 실행"""
        print("\n🔧 XGBoost 하이퍼파라미터 튜닝 시작...")
        print(f"   Trial 횟수: {self.n_trials}")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_tuning'
        )
        
        study.optimize(
            self.objective_xgboost,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_xgb_params = study.best_params
        
        print(f"\n✅ XGBoost 최적 파라미터:")
        for key, value in self.best_xgb_params.items():
            print(f"   {key}: {value}")
        print(f"   최고 ROC-AUC: {study.best_value:.4f}")
        
        return self.best_xgb_params
    
    def tune_lightgbm(self):
        """LightGBM 튜닝 실행"""
        print("\n🔧 LightGBM 하이퍼파라미터 튜닝 시작...")
        print(f"   Trial 횟수: {self.n_trials}")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='lightgbm_tuning'
        )
        
        study.optimize(
            self.objective_lightgbm,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_lgb_params = study.best_params
        
        print(f"\n✅ LightGBM 최적 파라미터:")
        for key, value in self.best_lgb_params.items():
            print(f"   {key}: {value}")
        print(f"   최고 ROC-AUC: {study.best_value:.4f}")
        
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


# 실행 예시
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    
    from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer
    
    # 피처 생성
    engineer = FeatureEngineer()
    engineer.load_stock_mapping()
    stock_codes = list(engineer.stock_mapping.keys())[:10]
    
    X, y = engineer.create_final_features(stock_codes, use_dummy=True)
    
    if X is not None:
        # 튜닝 실행 (Trial 수를 줄여서 빠르게 테스트)
        tuner = HyperparameterTuner(X, y, n_trials=20)
        best_params = tuner.tune_all()
        
        # 결과 저장
        import json
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print("\n💾 최적 파라미터 저장 완료: best_params.json")