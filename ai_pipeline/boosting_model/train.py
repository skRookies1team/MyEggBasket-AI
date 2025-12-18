import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, classification_report)
import pickle
import os
import sys
import warnings
import json
import matplotlib.pyplot as plt

# 경고 무시
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer

class StackingEnsemble:
    """
    XGBoost + LightGBM 스태킹 앙상블 (불균형 데이터 처리 강화 버전)
    """

    def __init__(self, xgb_params=None, lgb_params=None):
        # 파라미터 로드
        if os.path.exists('best_params.json'):
            try:
                with open('best_params.json', 'r') as f:
                    best_params = json.load(f)
                    self.xgb_params = best_params.get('xgboost', {})
                    self.lgb_params = best_params.get('lightgbm', {})
                    print(" [Info] best_params.json 로드 완료")
            except:
                self.xgb_params, self.lgb_params = xgb_params, lgb_params
        else:
            self.xgb_params = xgb_params or {
                'objective': 'binary:logistic', 'max_depth': 6, 'learning_rate': 0.05,
                'n_estimators': 200, 'random_state': 42, 'eval_metric': 'logloss'
            }
            self.lgb_params = lgb_params or {
                'objective': 'binary', 'num_leaves': 31, 'learning_rate': 0.05,
                'n_estimators': 200, 'random_state': 42, 'verbose': -1, 'metric': 'binary_logloss'
            }

        # 모델 초기화
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)

        # [수정 2] 메타 모델에도 'balanced' 가중치 적용
        self.meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # 핵심!
        )

        self.feature_names = None
        self.is_trained = False
        self.best_threshold = 0.5  # 최적 임계값 저장용

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("\n" + "=" * 60)
        print(" [Model] 스태킹 앙상블 학습 시작 (Imbalance Handling)")
        print("=" * 60)

        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)

        # [수정 1] 클래스 불균형 비율 계산 및 적용
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = np.sqrt(num_neg / num_pos)

        print(f" [Info] 데이터 비율 - Positive: {num_pos}, Negative: {num_neg}")
        print(f" [Info] 적용할 scale_pos_weight: {scale_pos_weight:.4f}")

        # 가중치 파라미터 업데이트
        self.xgb_model.set_params(scale_pos_weight=scale_pos_weight)
        self.lgb_model.set_params(scale_pos_weight=scale_pos_weight)

        # 1. Base Models 학습
        print(" 1) XGBoost 학습 중...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        xgb_train_pred = self.xgb_model.predict_proba(X_train)[:, 1]

        print(" 2) LightGBM 학습 중...")
        callbacks = [lgb.log_evaluation(0)]
        self.lgb_model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        lgb_train_pred = self.lgb_model.predict_proba(X_train)[:, 1]

        # 2. Meta Model 학습
        print(" 3) Meta Model 학습 중...")
        meta_train = np.column_stack([xgb_train_pred, lgb_train_pred])
        self.meta_model.fit(meta_train, y_train)

        self.is_trained = True

        # [수정 3] 최적 임계값(Threshold) 찾기
        if X_val is not None and y_val is not None:
            print(" 4) 최적 임계값(Threshold) 튜닝 중...")
            val_probs = self.predict_proba(X_val)[:, 1]
            thresholds = np.arange(0.3, 0.8, 0.01)  # 0.3 ~ 0.8 사이 탐색
            f1_scores = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thresholds]

            best_idx = np.argmax(f1_scores)
            self.best_threshold = thresholds[best_idx]
            print(f"    -> Best Threshold: {self.best_threshold:.2f} (Max F1: {f1_scores[best_idx]:.4f})")

        print(" 학습 완료!")

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError(" 모델이 학습되지 않았습니다.")

        # DataFrame 컬럼 보정
        if hasattr(self, 'feature_names') and self.feature_names is not None and isinstance(X, pd.DataFrame):
            missing_cols = set(self.feature_names) - set(X.columns)
            for c in missing_cols: X[c] = 0
            X = X[self.feature_names]

        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]

        meta_features = np.column_stack([xgb_pred, lgb_pred])

        final_probs_class1 = self.meta_model.predict_proba(meta_features)[:, 1]
        final_probs_class0 = 1 - final_probs_class1
        return np.column_stack([final_probs_class0, final_probs_class1])

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        # 학습된 최적 임계값 사용
        return (probabilities >= self.best_threshold).astype(int)

    def save_model(self, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'wb') as f: pickle.dump(self.xgb_model, f)
        with open(os.path.join(save_dir, 'lgb_model.pkl'), 'wb') as f: pickle.dump(self.lgb_model, f)
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'wb') as f: pickle.dump(self.meta_model, f)

        # 추가 정보 저장 (Threshold 포함)
        meta_info = {'feature_names': self.feature_names, 'best_threshold': self.best_threshold}
        with open(os.path.join(save_dir, 'model_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=2)

        print(f"\n [Save] 모델 및 메타정보 저장 완료: {save_dir}")

    def load_model(self, save_dir='models'):
        with open(os.path.join(save_dir, 'xgb_model.pkl'), 'rb') as f: self.xgb_model = pickle.load(f)
        with open(os.path.join(save_dir, 'lgb_model.pkl'), 'rb') as f: self.lgb_model = pickle.load(f)
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'rb') as f: self.meta_model = pickle.load(f)

        # 메타 정보 로드
        meta_path = os.path.join(save_dir, 'model_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self.feature_names = info.get('feature_names')
                self.best_threshold = info.get('best_threshold', 0.5)

        self.is_trained = True
        print(f" [Load] 모델 로드 완료 (Threshold: {self.best_threshold})")


def train_with_real_data(data_dir):
    """실제 데이터로 학습 파이프라인 실행"""
    print("=" * 60)
    print(" [Pipeline] 실제 데이터로 학습 시작")
    print("=" * 60)

    # 1. Feature Engineering
    engineer = FeatureEngineer(data_dir=data_dir)
    result = engineer.create_final_features()

    if result is None:
        print(" [Error] 학습할 데이터가 생성되지 않았습니다.")
        return None, None

    if len(result) == 3:
        X, y, codes = result
    elif len(result) == 2:
        X, y = result
    else:
        return None, None

    # 2. 데이터 전처리
    print(f"\n [Preprocess] 데이터 전처리 시작 (Raw Shape: {X.shape})")

    # 식별자 제거
    drop_cols = ['stck_shrn_iscd', 'code', 'stock_code', 'date', 'timestamp', 'stock_name']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

    # 결측치 처리
    fill_zero_cols = ['sentiment_score', 'sentiment_volatility', 'sentiment_trend', 'ai_score']
    gcn_cols = [c for c in X.columns if c.startswith('gcn_emb_')]
    fill_zero_cols.extend(gcn_cols)

    for col in fill_zero_cols:
        if col in X.columns: X[col] = X[col].fillna(0.0)

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    X = X.fillna(0)
    print(f" [Preprocess] 전처리 완료. 최종 피처 수: {X.shape[1]}개")

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Validation Set 추가 분리 (Threshold 튜닝용)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f" 학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")

    # 4. 모델 학습 (Validation Set 전달)
    model = StackingEnsemble()
    model.train(X_train, y_train, X_val, y_val)

    # 5. 평가
    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print(f" [최종 성능 평가] (Threshold: {model.best_threshold:.2f})")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("-" * 50)
    print(classification_report(y_test, y_pred))

    # 6. 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "models")
    model.save_model(save_dir)

    return model

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data"))

    if os.path.exists(data_dir):
        train_with_real_data(data_dir)
    else:
        print(f" [Error] 데이터 폴더를 찾을 수 없습니다: {data_dir}")