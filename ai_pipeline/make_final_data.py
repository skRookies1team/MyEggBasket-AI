## GCN 데이터 확인

import numpy as np
import pandas as pd
import os

# 👉 경로 설정
EMBEDDING_PATH = 'ai_pipeline/gcn_model/gcn_embeddings.npy'
NODE_LIST_PATH = 'ai_pipeline/gcn_model/gcn_node_list.csv'  # CSV 파일

# 출력 파일
OUTPUT_PATH = 'gcn_output_check.csv'

def create_and_check_data():
    print(f"🚀 [데이터 확인] GCN 결과물을 {OUTPUT_PATH}로 변환합니다...")

    # 1. 파일 존재 여부 확인
    if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(NODE_LIST_PATH):
        print(f"❌ 입력 파일을 찾을 수 없습니다.")
        print(f"   - {EMBEDDING_PATH}")
        print(f"   - {NODE_LIST_PATH}")
        return

    # 2. 파일 로드
    try:
        # (1) 임베딩 로드 (.npy 파일)
        embeddings = np.load(EMBEDDING_PATH)
        
        # (2) 노드 리스트 로드 (수정됨: .csv 파일은 read_csv로 읽기)
        # CSV를 읽어서 첫 번째 컬럼을 종목 코드로 사용합니다.
        node_df_temp = pd.read_csv(NODE_LIST_PATH)
        node_list = node_df_temp.iloc[:, 0].values  # 첫 번째 열의 값만 추출
        
        print(f"✅ 데이터 로드 성공!")
        print(f"   - 종목 수: {len(node_list)}개")
        print(f"   - 임베딩 차원: {embeddings.shape[1]}")
        
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return

    # 3. 데이터프레임 만들기
    # 종목 코드 (이미 CSV에서 읽었지만, 포맷 통일을 위해 다시 생성)
    df_nodes = pd.DataFrame(node_list, columns=['stock_code'])
    
    # 임베딩 값 (gcn_emb_0, gcn_emb_1, ...)
    col_names = [f'gcn_emb_{i}' for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=col_names)

    # 4. 병합 (종목코드 + 임베딩)
    # 행 개수가 맞는지 확인
    if len(df_nodes) != len(df_emb):
        print(f"⚠️ 경고: 종목 수({len(df_nodes)})와 임베딩 개수({len(df_emb)})가 다릅니다!")
        # 개수가 적은 쪽에 맞춰서 병합 (inner join 방식과 유사하게 인덱스 기준 병합)
        min_len = min(len(df_nodes), len(df_emb))
        final_df = pd.concat([df_nodes.iloc[:min_len], df_emb.iloc[:min_len]], axis=1)
    else:
        final_df = pd.concat([df_nodes, df_emb], axis=1)

    # 5. CSV로 저장
    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("-" * 50)
    print(f"💾 [생성 완료] {OUTPUT_PATH}")
    print("   👉 이 파일을 열어서 숫자가 꽉 차 있는지 확인하세요.")
    print("-" * 50)
    
    # 6. 미리보기 출력
    print("\n🔍 [데이터 미리보기]")
    print(final_df.head().to_markdown(index=False))

if __name__ == "__main__":
    create_and_check_data()