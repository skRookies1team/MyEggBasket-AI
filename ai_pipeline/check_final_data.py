import os
import pandas as pd
import numpy as np

# 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
# create_dataset.py가 저장한 경로 (data 폴더)
data_path = os.path.join(current_dir, "../../final_train_data.csv")

def inspect_data():
    print("🕵️‍♂️ [데이터 품질 검사] XGBoost용 데이터 확인 중...")
    
    if not os.path.exists(data_path):
        print(f"❌ 파일이 없습니다: {data_path}")
        print("   -> create_dataset.py를 먼저 실행해주세요.")
        return

    # 데이터 로드
    df = pd.read_csv(data_path)
    
    print(f"✅ 파일 로드 성공! 총 {len(df)}개 종목")
    print("-" * 60)
    
    # 1. 컬럼 확인
    cols = df.columns.tolist()
    print(f"📌 컬럼 개수: {len(cols)}개")
    print(f"   - 필수 피처: sentiment_score, ai_score, volatility")
    print(f"   - 임베딩 피처: gcn_emb_0 ~ gcn_emb_63")
    
    # 2. 데이터 샘플 (SK하이닉스 찾기)
    target_code = '000660' # 하이닉스
    if target_code in df['stock_code'].astype(str).values:
        row = df[df['stock_code'].astype(str) == target_code].iloc[0]
        print(f"\n🔍 [샘플 검사] SK하이닉스 ({target_code}) 데이터:")
        print(f"   - 감성 점수: {row.get('sentiment_score', 'N/A')}")
        print(f"   - AI 스코어: {row.get('ai_score', 'N/A')}")
        print(f"   - 임베딩(앞 5개): {[row.get(f'gcn_emb_{i}') for i in range(5)]}")
    else:
        print(f"\n⚠️ SK하이닉스({target_code}) 데이터가 없습니다. (샘플로 첫번째 행 출력)")
        print(df.iloc[0])

    # 3. 0점(결측) 비율 확인
    # 임베딩이 전부 0인지 확인 (학습 실패 여부)
    emb_cols = [c for c in cols if 'gcn_emb' in c]
    if emb_cols:
        zeros = (df[emb_cols] == 0).all(axis=1).sum()
        if zeros > 0:
            print(f"\n⚠️ [경고] 임베딩이 전부 0인 종목이 {zeros}개 있습니다!")
        else:
            print(f"\n✅ [합격] 모든 종목의 임베딩이 정상적으로 채워져 있습니다.")
    
    print("-" * 60)
    print("👉 이 데이터가 있다면 XGBoost 학습 준비는 완벽합니다.")

if __name__ == "__main__":
    inspect_data()