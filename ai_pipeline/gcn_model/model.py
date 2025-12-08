import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # Layer 1: 입력 특징 -> 128차원
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        # Layer 2: 128차원 -> 64차원 (최종 임베딩)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        # 1. 첫 번째 층 + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2. 두 번째 층 (Linear) -> 여기서 나온 값이 'Node Embedding'이 됨
        return self.conv2(x, edge_index)
    
def get_gae_model(in_channels, out_channels=64):
    """외부에서 호출하기 편한 헬퍼 함수"""
    encoder = GCNEncoder(in_channels, out_channels)
    
    # PyG의 GAE는 인코더만 넣어주면 디코더(InnerProduct)는 자동 생성됨
    # 목적: Z(임베딩) * Z.T(전치) = Adjacency Matrix(인접행렬) 복원
    model = GAE(encoder)
    return model