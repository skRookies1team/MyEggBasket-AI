import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             classification_report, confusion_matrix)
import pickle
import os
import sys
import warnings
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
                print("✅ best_params.json에서 최적 파라미터 로드")
        else:
            self.xgb_params = xgb_params or {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42
            }
            
            self.lgb_params = lgb_params or {
                'objective': 'binary',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'random_state': 42,
                'verbose': -1
            }
        
        # 모델 초기화
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.is_trained = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """스태킹 모델 학습"""
        print("\n" + "="*60)
        print("🚀 스태킹 앙상블 학습 시작")
        print("="*60)
        
        # Step 1: Base Models 학습
        print("\n[Step 1] Base Models 학습")
        print("-" * 40)
        
        # XGBoost
        print("🔸 XGBoost 학습 중...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        xgb_train_pred = self.xgb_model.predict_proba(X_train)[:, 1]
        
        if X_val is not None:
            xgb_val_pred = self.xgb_model.predict_proba(X_val)[:, 1]
            xgb_auc = roc_auc_score(y_val, xgb_val_pred)
            print(f"   ✅ XGBoost 검증 AUC: {xgb_auc:.4f}")
        
        # LightGBM
        print("🔸 LightGBM 학습 중...")
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
            print(f"   ✅ LightGBM 검증 AUC: {lgb_auc:.4f}")
        
        # Step 2: Meta Model 학습
        print("\n[Step 2] Meta Model 학습")
        print("-" * 40)
        
        meta_train = np.column_stack([xgb_train_pred, lgb_train_pred])
        self.meta_model.fit(meta_train, y_train)
        
        if X_val is not None:
            meta_val = np.column_stack([xgb_val_pred, lgb_val_pred])
            meta_val_pred = self.meta_model.predict_proba(meta_val)[:, 1]
            meta_auc = roc_auc_score(y_val, meta_val_pred)
            print(f"   ✅ Stacking 검증 AUC: {meta_auc:.4f}")
        
        self.is_trained = True
        print("\n✅ 스태킹 모델 학습 완료!")
    
    # 🛠️ [수정] predict_proba 추가 및 predict 표준화
    def predict_proba(self, X):
        """
        [표준] 확률 반환 (N행 2열: [하락확률, 상승확률])
        이 함수가 있어야 analyzer.py나 predict.py에서 에러가 안 남!
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
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
        print("📊 최종 모델 평가 (테스트 데이터)")
        print("="*60)
        
        y_pred_proba = self.predict_proba(X_test)[:, 1] # 상승 확률만 추출
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✅ Accuracy: {accuracy:.4f}")
        print(f"✅ ROC-AUC: {auc:.4f}")
        
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
        """모델 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        with open(os.path.join(save_dir, 'lgb_model.pkl'), 'wb') as f:
            pickle.dump(self.lgb_model, f)
        
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        print(f"\n💾 모델 저장 완료: {save_dir}/")
    
    def load_model(self, save_dir='models'):
        """모델 불러오기"""
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'rb') as f:
            self.xgb_model = pickle.load(f)
        
        with open(os.path.join(save_dir, 'lgb_model.pkl'), 'rb') as f:
            self.lgb_model = pickle.load(f)
        
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'rb') as f:
            self.meta_model = pickle.load(f)
        
        self.is_trained = True
        print(f"✅ 모델 로드 완료: {save_dir}/")


def train_with_real_data(csv_path):
    """실제 체결 데이터로 학습"""
    print("="*60)
    print("🚀 실제 데이터로 스태킹 모델 학습")
    print("="*60)
    
    # 1. 피처 생성
    engineer = FeatureEngineer(csv_path=csv_path)
    X, y = engineer.create_final_features()
    
    if X is None:
        print("❌ 피처 생성 실패")
        return None
    
    # 2. 데이터 분할 (시간순 분할 권장)
    # 여기서는 단순 분할로 시작
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 데이터 분할:")
    print(f"   학습: {len(X_train):,}")
    print(f"   검증: {len(X_val):,}")
    print(f"   테스트: {len(X_test):,}")
    
    # 3. 모델 학습
    model = StackingEnsemble()
    model.train(X_train, y_train, X_val, y_val)
    
    # 4. 테스트 평가
    results = model.evaluate(X_test, y_test)
    
    # 5. 모델 저장
    model.save_model()
    
    print("\n" + "="*60)
    print("✅ 전체 프로세스 완료!")
    print("="*60)
    
    return model, results


# 실행
if __name__ == "__main__":
    csv_path = "20251120.csv"
    
    model, results = train_with_real_data(csv_path)