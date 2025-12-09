import numpy as np
import pandas as pd
import os

# 파일 경로
base_path = "ai_pipeline/gcn_model"
npy_path = os.path.join(base_path, "gcn_embeddings.npy")
csv_path = os.path.join(base_path, "gcn_node_list.csv")

# 데이터 로드
embeddings = np.load(npy_path)
nodes = pd.read_csv(csv_path, dtype={'code': str})

# 확인할 종목들 (하이닉스 + 아까 이상하게 추천된 녀석들)
target_codes = ['000660', '000810', '012630'] 

print(f"{'종목코드':<10} | {'임베딩 벡터 일부 (앞 5개 숫자)'}")
print("-" * 50)

for code in target_codes:
    if code in nodes['code'].values:
        idx = nodes[nodes['code'] == code].index[0]
        vec = embeddings[idx]
        # 벡터의 앞부분 5개만 출력해서 숫자가 다 0인지, 아니면 비슷한지 확인
        print(f"{code:<10} | {vec[:5]}")
    else:
        print(f"{code:<10} | 데이터 없음")