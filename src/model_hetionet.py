import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import degree

class HeteroPULLModel(torch.nn.Module):
    """
    HGT(Heterogeneous Graph Transformer)를 PULL 로직과 결합한 모델
    """
    def __init__(self, data, hidden_channels=128, out_channels=64, num_heads=4, num_layers=2):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 1. 초기 노드 임베딩
        self.node_embeds = nn.ModuleDict()
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            self.node_embeds[node_type] = nn.Embedding(num_nodes, hidden_channels)

        # 2. 이종 GNN (HGT) 레이어
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)
            
        # 3. 최종 출력용 Linear 레이어
        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(hidden_channels, out_channels)
        
        # 🆕 4. Dropout 추가 (과적합 방지)
        self.dropout = nn.Dropout(0.3)

    def encode(self, data, edge_index_dict, edge_weight_dict=None):
        """ GNN을 통과시켜 모든 노드의 최종 임베딩(z)을 계산 """
        x_dict = {}
        for node_type, embed_layer in self.node_embeds.items():
            x_dict[node_type] = embed_layer.weight
            
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # 🆕 Dropout 적용
            for node_type in x_dict.keys():
                x_dict[node_type] = self.dropout(x_dict[node_type])
            
        z_dict = {}
        for node_type, lin in self.lin_dict.items():
            z_dict[node_type] = lin(x_dict[node_type])
            # 🆕 L2 정규화 (임베딩 크기 제한)
            z_dict[node_type] = F.normalize(z_dict[node_type], p=2, dim=-1)
            
        return z_dict

    def decode(self, z_dict, edge_label_index):
        """ 'treats' 관계의 존재 확률을 계산 """
        compound_embeds = z_dict['Compound'][edge_label_index[0]]
        disease_embeds = z_dict['Disease'][edge_label_index[1]]
        # 🆕 내적 결과를 직접 반환 (sigmoid는 loss 함수에서)
        return (compound_embeds * disease_embeds).sum(dim=-1)

    def decode_all(self, z_dict, train_edge_index, ratio, epoch):
        """ PULL 로직: 'Unlabeled'에서 Top-K 후보(새로운 P)를 발굴 """
        compound_embeds = z_dict['Compound']
        disease_embeds = z_dict['Disease']

        # 🆕 내적 계산 (sigmoid 제거)
        prob_adj = compound_embeds @ disease_embeds.t()
        prob_adj[train_edge_index[0], train_edge_index[1]] = -float('inf')  # 훈련 엣지 제외
        
        n_edge = train_edge_index.shape[1]
        # 🆕 ratio 감소 (0.05 → 0.02)
        n_edge_add = int(n_edge * ratio * 0.4 * (epoch - 1))  # 더 보수적으로
        
        if n_edge_add == 0:
            return torch.tensor([[],[]], dtype=torch.long, device=prob_adj.device), \
                   torch.tensor([], device=prob_adj.device)
                   
        flat_probs = prob_adj.flatten()
        top_k_indices = torch.topk(flat_probs, n_edge_add).indices
        
        row_indices = top_k_indices // prob_adj.shape[1]
        col_indices = top_k_indices % prob_adj.shape[1]
        
        edge_index_add = torch.stack([row_indices, col_indices], dim=0)
        # 🆕 Soft weight (0~1 범위로 정규화)
        edge_weight_add = torch.sigmoid(prob_adj[row_indices, col_indices])
        
        return edge_index_add, edge_weight_add

    def merge_edge(self, edge_index, edge_weight, edge_index_add, edge_weight_add):
        """ 새 P 엣지를 기존 엣지 리스트에 추가 """
        edge_index = torch.cat((edge_index, edge_index_add), 1)
        edge_weight = torch.cat((edge_weight, edge_weight_add))
        return edge_index, edge_weight