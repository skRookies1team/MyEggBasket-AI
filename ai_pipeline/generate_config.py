# ai_pipeline/generate_config.py
import os
import sys
import pandas as pd

# 프로젝트 루트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def generate_feature_columns_file():
    # 1. 학습 데이터 읽기
    csv_path = "train_dataset.csv"
    if not os.path.exists(csv_path):
        print("❌ train_dataset.csv가 없습니다. 먼저 생성해주세요.")
        return

    df = pd.read_csv(csv_path, nrows=5)
    
    # 2. 학습에 쓰이는 피처만 추출 (메타데이터 제외)
    # (주의: target은 정답지이므로 피처 아님)
    exclude_cols = ['target', 'stock_code', 'date', 'timestamp', 'stck_prpr', 'code', 'Date']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"📊 감지된 피처 개수: {len(feature_cols)}개")
    
    # 3. 파이썬 파일로 코드 생성
    config_path = os.path.join("ai_pipeline", "config")
    os.makedirs(config_path, exist_ok=True)
    file_path = os.path.join(config_path, "feature_columns.py")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# 이 파일은 train_dataset.csv를 기준으로 자동 생성되었습니다.\n")
        f.write("# 모델 학습과 예측 시 피처 순서를 통일하기 위해 사용됩니다.\n\n")
        f.write("ALL_MODEL_FEATURES = [\n")
        for col in feature_cols:
            f.write(f"    '{col}',\n")
        f.write("]\n")
        
    print(f"✅ 설정 파일 생성 완료: {file_path}")
    print("   이제 모든 코드에서 이 리스트를 불러와 사용하면 순서가 꼬일 일이 없습니다.")

if __name__ == "__main__":
    generate_feature_columns_file()