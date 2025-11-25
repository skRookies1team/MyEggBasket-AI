import sys
import os
import torch
from ai_pipeline.gcn_model.model import NewsStockGCN  
from torch_geometric.data import Data

# 경로 설정 (데이터셋 찾기 위함)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def run_gcn_inference():
    print("🧠 GCN 모델 초기화 및 실행 준비...")

    # 1. 데이터셋 로드 (.pt 파일)
    dataset_path = os.path.join(os.path.dirname(__file__), "../../finance_graph_data.pt")
    
    if not os.path.exists(dataset_path):
        print(f"❌ 데이터셋이 없습니다: {dataset_path}")
        return

    data = torch.load(dataset_path, weights_only=False)
    print(f"   📊 입력 데이터: 노드 {data.num_nodes}개, 엣지 {data.num_edges}개")
    print(f"   📊 입력 특징(Feature) 크기: {data.num_node_features} (감성,뉴스여부,종목여부)")

    # 2. 모델 설정
    # 입력=3, 히든=16, 출력=16 (벡터 크기)
    model = NewsStockGCN(in_channels=3, hidden_channels=16, out_channels=16)
    model.eval() # 평가 모드 (학습 아님)

    # 3. 실행 (Forward Pass)
    with torch.no_grad():
        node_embeddings = model(data)

    print("\n⚡ GCN 연산 완료!")
    print(f"   👉 결과 임베딩 형태(Shape): {node_embeddings.shape}")
    print("   (행: 노드 개수, 열: 16차원 벡터)")

    # 4. 결과 확인 (종목 노드 하나만 찍어보기)
    # 뒤쪽 노드들이 보통 종목 노드이므로 마지막 노드 확인
    last_node_idx = data.num_nodes - 1
    print(f"\n🔍 예시: 마지막 노드({last_node_idx}번)의 GCN 임베딩 결과:")
    print(node_embeddings[last_node_idx])

    # 5. 저장 (나중에 XGBoost가 가져다 쓰도록)
    save_path = "gcn_node_embeddings.pt"
    torch.save(node_embeddings, save_path)
    print(f"\n💾 임베딩 벡터 저장 완료: {save_path}")

if __name__ == "__main__":
    run_gcn_inference()