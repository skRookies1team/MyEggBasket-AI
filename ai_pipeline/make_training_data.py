import os
import sys
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 설정 (ai_pipeline 폴더의 상위 폴더)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from ai_pipeline.boosting_model.feature_engineering import FeatureEngineer


def create_labeled_dataset(data_dir, output_path):
    print("=" * 60)
    print(" [Data Gen] 학습용 라벨링 데이터셋 생성 시작")
    print("=" * 60)

    # 1. FeatureEngineer 초기화
    engineer = FeatureEngineer(data_dir=data_dir)

    # 2. 전체 파일 로드 및 피처 생성
    features_ret = engineer.create_final_features()

    if features_ret is None:
        print(" [Error] 데이터를 생성하지 못했습니다. (features_ret is None)")
        return

    # 반환값 언패킹 (X: 피처, y: 타겟, codes: 종목코드)
    # create_final_features가 None을 반환하는 경우를 대비
    if len(features_ret) == 3:
        X, y, codes = features_ret
    elif len(features_ret) == 2:
        X, y = features_ret
        codes = None
    else:
        print(" [Error] 피처 엔지니어링 반환값 형식이 맞지 않습니다.")
        return

    if X is None:
        print(" [Error] 생성된 피처 데이터(X)가 없습니다.")
        return

    # 3. 데이터 병합 (X + y)
    df_train = X.copy()
    df_train['target'] = y

    # 종목 코드가 있다면 식별을 위해 추가
    if codes is not None:
        df_train['stock_code'] = codes

    # 4. 결측치 제거
    original_len = len(df_train)
    df_train = df_train.dropna()
    print(f" 결측치 제거: {original_len} -> {len(df_train)} (NaN 포함 행 삭제)")

    # 5. 저장
    df_train.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n [완료] 학습용 데이터 저장됨: {output_path}")
    print(f" 데이터 크기: {df_train.shape}")


if __name__ == "__main__":
    # 현재 파일 위치: .../MyEggBasket-AI/ai_pipeline/make_training_data.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 프로젝트 루트: .../MyEggBasket-AI
    project_root = os.path.dirname(current_dir)

    # [수정] data 폴더는 프로젝트 루트 바로 아래에 있습니다. ("../data" -> "data")
    data_dir = os.path.join(project_root, "data")

    # 출력 파일 경로
    output_csv = os.path.join(project_root, "train_dataset.csv")

    print(f" 데이터 폴더 경로: {data_dir}")

    if os.path.exists(data_dir):
        create_labeled_dataset(data_dir, output_csv)
    else:
        print(f" [Error] 데이터 폴더를 찾을 수 없습니다: {data_dir}")