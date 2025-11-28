import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class NewsStockGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NewsStockGCN, self).__init__()
        
        # 첫 번째 층: 입력(3개 특징) -> 히든(16개 특징)으로 뻥튀기
        # 그래프 정보를 섞는 핵심 층입니다.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # 두 번째 층: 히든 -> 출력(Embedding Dimension)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. 첫 번째 그래프 합성곱 (Convolution)
        x = self.conv1(x, edge_index)
        x = F.relu(x) # 활성화 함수 (비선형성 추가)
        x = F.dropout(x, training=self.training) # 과적합 방지

        # 2. 두 번째 그래프 합성곱
        x = self.conv2(x, edge_index)
        
        # 결과값: 각 노드별로 새로 만들어진 '상태 벡터(Embedding)'
        return x