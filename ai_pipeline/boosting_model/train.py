import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, classification_report)
import pickle
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

class StackingEnsemble:
    """
    XGBoost + LightGBM 스태킹 앙상블
    
    1단계: Base Models (XGBoost, LightGBM) 각각 학습
    2단계: Meta Model (Logistic Regression)이 Base Model 예측값으로 최종 예측
    """
    
    def __init__(self, xgb_params=None, lgb_params=None):
        # 기본 파라미터 (best_params.json이 있으면 사용)
        if os.path.exists('best_params.json'):
            with open('best_params.json', 'r') as f:
                best_params = json.load(f)
                self.xgb_params = best_params.get('xgboost', {})
                self.lgb_params = best_params.get('lightgbm', {})
                print(" best_params.json에서 최적 파라미터 로드")
        else:
            self.xgb_params = xgb_params or {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            
            self.lgb_params = lgb_params or {
                'objective': 'binary',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'random_state': 42,
                'verbose': -1,
                'metric': 'binary_logloss'
            }
        
        # 모델 초기화
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.is_trained = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """스태킹 모델 학습"""
        print("\n" + "="*60)
        print(" 스태킹 앙상블 학습 시작")
        print("="*60)
        
        # Step 1: Base Models 학습
        print("\n[Step 1] Base Models 학습")
        
        # XGBoost
        print(" XGBoost 학습 중...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        xgb_train_pred = self.xgb_model.predict_proba(X_train)[:, 1]
        
        if X_val is not None:
            xgb_val_pred = self.xgb_model.predict_proba(X_val)[:, 1]
            xgb_auc = roc_auc_score(y_val, xgb_val_pred)
            print(f"    XGBoost 검증 AUC: {xgb_auc:.4f}")
        
        # LightGBM
        print(" LightGBM 학습 중...")
        callbacks = [lgb.log_evaluation(0)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(50))
        
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        lgb_train_pred = self.lgb_model.predict_proba(X_train)[:, 1]
        
        if X_val is not None:
            lgb_val_pred = self.lgb_model.predict_proba(X_val)[:, 1]
            lgb_auc = roc_auc_score(y_val, lgb_val_pred)
            print(f"    LightGBM 검증 AUC: {lgb_auc:.4f}")
        
        # Step 2: Meta Model 학습
        print("\n[Step 2] Meta Model 학습")
        
        meta_train = np.column_stack([xgb_train_pred, lgb_train_pred])
        self.meta_model.fit(meta_train, y_train)
        
        if X_val is not None:
            meta_val = np.column_stack([xgb_val_pred, lgb_val_pred])
            meta_val_pred = self.meta_model.predict_proba(meta_val)[:, 1]
            meta_auc = roc_auc_score(y_val, meta_val_pred)
            print(f"    Stacking 검증 AUC: {meta_auc:.4f}")
        
        self.is_trained = True
        print("\n 스태킹 모델 학습 완료!")

        # 학습 시 사용된 피처명 저장
        try:
            # X_train이 pandas DataFrame이면 컬럼명을 저장
            if hasattr(X_train, 'columns'):
                self.feature_names = list(X_train.columns)
        except Exception:
            self.feature_names = None
    
    #  predict_proba 추가 및 predict 표준화
    def predict_proba(self, X):
        """
        [수정] 학습된 컬럼과 들어온 컬럼을 동적으로 맞춰주는 안전장치 추가
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")

        # X가 DataFrame인 경우 컬럼 매칭 수행
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            # 1. 학습 때 썼는데 지금 없는 컬럼 -> 0으로 채움
            missing_cols = set(self.feature_names) - set(X.columns)
            for c in missing_cols:
                X[c] = 0

            if hasattr(self, 'feature_names') and self.feature_names is not None:
                # 학습 때 안 쓴 컬럼(공시 등)이 들어오면 과감히 제거
                extra_cols = set(X.columns) - set(self.feature_names)
                if extra_cols:
                    X = X.drop(columns=list(extra_cols))

                # 학습 때 썼던 컬럼만 순서대로 유지
                X = X[self.feature_names]

        # 1. Base Model 예측
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        
        # 2. Meta Model 입력 생성
        meta_features = np.column_stack([xgb_pred, lgb_pred])
        
        # 3. 최종 확률 (sklearn 표준 포맷인 (N, 2)로 반환)
        # col 0: 하락(0) 확률, col 1: 상승(1) 확률
        final_probs_class1 = self.meta_model.predict_proba(meta_features)[:, 1]
        final_probs_class0 = 1 - final_probs_class1
        
        return np.column_stack([final_probs_class0, final_probs_class1])

    def predict(self, X):
        """
        [표준] 클래스 예측 (0 또는 1 반환)
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)
    
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        print("\n" + "="*60)
        print(" 최종 모델 평가 (테스트 데이터)")
        print("="*60)
        
        y_pred_proba = self.predict_proba(X_test)[:, 1] # 상승 확률만 추출
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n Accuracy: {accuracy:.4f}")
        print(f" ROC-AUC: {auc:.4f}")
        
        print("\n[분류 리포트]")
        print(classification_report(y_test, y_pred, 
                                   target_names=['하락(0)', '상승(1)']))
        
        print("\n[혼동 행렬]")
        cm = confusion_matrix(y_test, y_pred)
        print(f"              예측 하락  예측 상승")
        print(f"실제 하락       {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"실제 상승       {cm[1,0]:4d}      {cm[1,1]:4d}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': auc,
            'confusion_matrix': cm
        }
    
    def save_model(self, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'wb') as f: pickle.dump(self.xgb_model, f)
        with open(os.path.join(save_dir, 'lgb_model.pkl'), 'wb') as f: pickle.dump(self.lgb_model, f)
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'wb') as f: pickle.dump(self.meta_model, f)
        # 학습에 사용된 피처명도 함께 저장 (있다면)
        try:
            if hasattr(self, 'feature_names') and self.feature_names is not None:
                with open(os.path.join(save_dir, 'feature_names.json'), 'w', encoding='utf-8') as fh:
                    json.dump(self.feature_names, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

        print(f"\n 모델 저장 완료: {save_dir}/")
    
    def load_model(self, save_dir='models'):
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'rb') as f: self.xgb_model = pickle.load(f)
        with open(os.path.join(save_dir, 'lgb_model.pkl'), 'rb') as f: self.lgb_model = pickle.load(f)
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'rb') as f: self.meta_model = pickle.load(f)
        # 저장된 feature_names가 있으면 불러오기
        try:
            fn_path = os.path.join(save_dir, 'feature_names.json')
            if os.path.exists(fn_path):
                with open(fn_path, 'r', encoding='utf-8') as fh:
                    self.feature_names = json.load(fh)
        except Exception:
            self.feature_names = None

        self.is_trained = True
        print(f" 모델 로드 완료: {save_dir}/")


def plot_feature_importance(model, feature_names):
    """XGBoost의 피처 중요도 시각화"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Top 20 Feature Importances")
    plt.bar(range(20), importances[indices], align="center")
    plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def train_with_real_data(data_dir):
    """실제 체결 데이터로 학습"""
    print("="*60)
    print(" 실제 데이터로 스태킹 모델 학습")
    print("="*60)
    
    # 1. 피처 생성
    engineer = FeatureEngineer(data_dir=data_dir)
    result = engineer.create_final_features()

    # 데이터가 없으면 종료
    if result is None or result[0] is None:
        print(" 학습할 데이터가 없습니다.")
        return None, None

    if len(result) == 3:
        X, y, codes = result
    elif len(result) == 2:
        X, y = result
    else:
        print(" 피처 생성 반환값 오류")
        return None, None

    # 불필요 컬럼 제거 (FeatureEngineer에서 이미 처리했어도 안전장치)
    drop_cols = ['stck_shrn_iscd', 'code', 'stock_code', 'date', 'timestamp']
    real_drop = [c for c in drop_cols if c in X.columns]
    if real_drop:
        X = X.drop(columns=real_drop)

    
    for col in X.select_dtypes(include=['object']).columns:
        print(f" 경고: '{col}' 컬럼이 문자열입니다. 숫자로 변환을 시도하거나 삭제합니다.")
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        except:
            X = X.drop(columns=[col])

    print(f" 최종 학습 피처({X.shape[1]}개): {list(X.columns)}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    # 3. 모델 학습
    model = StackingEnsemble()
    # 학습에 사용된 피처명을 모델에 기록하여 저장
    try:
        model.feature_names = list(X_train.columns)
    except Exception:
        model.feature_names = None

    model.train(X_train, y_train)
    
    # 4. 테스트 평가
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print(" [최종 성능 평가]")
    print(f"   - 정확도(Accuracy):  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   - 정밀도(Precision): {precision_score(y_test, y_pred):.4f}")
    print(f"   - 재현율(Recall):    {recall_score(y_test, y_pred):.4f}")
    print(f"   - F1 점수(F1 Score): {f1_score(y_test, y_pred):.4f}")
    print("-" * 50)
    
    
    # 중요도 시각화 (XGBoost 기준)
    try:
        plot_feature_importance(model.xgb_model, X.columns)
    except:
        pass

    # 모델 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "models")
    model.save_model(save_dir)
    
    return model, {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

if __name__ == "__main__":
    # 데이터 폴더 경로 (본인 환경에 맞게)
    data_dir = r"C:\Users\user\project\MyEggBasket-AI\data"
    
    if os.path.exists(data_dir):
        train_with_real_data(data_dir)
    else:
        print(f" 경로 없음: {data_dir}")