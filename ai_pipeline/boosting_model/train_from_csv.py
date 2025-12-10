# 파일 위치: ai_pipeline/boosting_model/train_from_csv.py
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ai_pipeline.boosting_model.train import StackingEnsemble


def train_manual_csv(csv_path):
    print("=" * 60)
    print(f" [Manual Train] CSV 파일로 학습 시작: {os.path.basename(csv_path)}")
    print("=" * 60)

    # 1. CSV 로드
    if not os.path.exists(csv_path):
        print(f" [Error] 파일을 찾을 수 없습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f" 데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}열")

    # 2. 타겟(Label) 확인
    # 기존 코드상 타겟 컬럼명은 'target' 입니다. (상승:1, 하락/보합:0)
    if 'target' not in df.columns:
        print(" [Error] CSV 안에 'target' 컬럼이 없습니다.")
        print(" 학습을 위해서는 정답지(target)가 반드시 필요합니다.")
        return

    # 3. 데이터 전처리 (학습에 불필요한 컬럼 제거)
    # 날짜, 종목코드, 이름 등 비수치형 데이터와 정답(target)을 분리
    ignore_cols = ['date', 'timestamp', 'code', 'stock_code', 'stck_shrn_iscd', 'target', 'stock_name', 'opinion']

    # [수정] 피처 선택 로직 강화: 'disc_' 또는 'report_'로 시작하는 컬럼 제외
    feature_cols = [
        c for c in df.columns
        if c not in ignore_cols
           and not c.startswith('disc_')  # 공시 데이터 제외
           and not c.startswith('report_')  # 리포트 데이터 제외
    ]

    # 선택된 피처로 데이터셋 구성
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['target']

    print(f" [설정] 시계열/기술적 피처만 사용합니다.")
    print(f" 제외된 패턴: 'disc_*', 'report_*'")
    print(f" 최종 학습 피처 수: {X.shape[1]}개")
    # print(f" 최종 피처 목록: {list(X.columns)}")

    # 4. Train / Test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f" 학습 데이터: {len(X_train)}개 / 테스트 데이터: {len(X_test)}개")

    # 5. 모델 학습 (StackingEnsemble 사용)
    model = StackingEnsemble()

    # 피처 이름 저장 (나중에 predict 할 때 순서 맞추기 위함)
    model.feature_names = list(X.columns)

    model.train(X_train, y_train)

    # 6. 평가
    print("\n [모델 평가 결과]")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" 정확도(Accuracy): {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 7. 모델 저장
    # ai_pipeline/boosting_model/models 폴더에 저장됩니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "models")

    model.save_model(save_dir)
    print(f"\n [완료] 모델 파일이 저장되었습니다: {save_dir}")


if __name__ == "__main__":
    # 사용자가 업로드한 파일 경로 지정
    # 프로젝트 루트 기준 경로를 맞춰주세요.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))

    # 업로드한 파일 이름이 'train_dataset.csv' 라고 가정
    csv_file_path = os.path.join(project_root, "train_dataset.csv")

    train_manual_csv(csv_file_path)