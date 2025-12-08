import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트 경로 찾기
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 파일 경로 설정
npy_path = os.path.join(current_dir, "gcn_embeddings.npy")
csv_path = os.path.join(current_dir, "gcn_node_list.csv")

def find_similar_stocks(target_code, top_k=5):
    # 1. 데이터 로드
    if not os.path.exists(npy_path) or not os.path.exists(csv_path):
        print(" 학습된 데이터가 없습니다. run_gcn.py를 먼저 실행하세요.")
        return

    embeddings = np.load(npy_path)
    
    # CSV 로드 시 컬럼명을 확인
    node_df = pd.read_csv(csv_path, dtype=str)
    
    # 컬럼명이 'stock_code'인지 'code'인지 확인 후 통일
    col_name = 'stock_code' if 'stock_code' in node_df.columns else 'code'
    
    # 2. 입력한 종목이 리스트에 있는지 확인
    if target_code not in node_df[col_name].values:
        print(f" '{target_code}'는 학습된 데이터에 없는 종목입니다.")
        print(f"   (총 {len(node_df)}개 종목이 학습되었습니다)")
        return

    # 3. 타겟 종목의 벡터(임베딩) 찾기
    target_idx = node_df[node_df[col_name] == target_code].index[0]
    target_vector = embeddings[target_idx].reshape(1, -1)

    # 4. 코사인 유사도 계산
    similarities = cosine_similarity(target_vector, embeddings)[0]

    # 5. 유사도 순 정렬
    sorted_indices = similarities.argsort()[::-1]
    
    print(f"\n [{target_code}] AI 추천 유사 종목 TOP {top_k}")
    print("=" * 50)
    print(f"{'순위':<5} | {'종목코드':<10} | {'유사도':<10}")
    print("-" * 50)

    count = 0
    for idx in sorted_indices:
        if idx == target_idx: continue # 자기 자신 제외
            
        sim_score = similarities[idx]
        similar_code = node_df.iloc[idx][col_name]
        
        print(f"{count+1:<5} | {similar_code:<10} | {sim_score*100:.2f}%")
        
        count += 1
        if count >= top_k:
            break
    print("=" * 50)

if __name__ == "__main__":
    input_code = input(" 유사도를 분석할 종목 코드를 입력하세요 (예: 000660): ").strip()
    find_similar_stocks(input_code)