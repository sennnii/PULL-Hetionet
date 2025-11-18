import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HeteroPULLModel(torch.nn.Module):
    """
    HGT + PULL 모델 (안정화 버전)
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

        # 2. HGT 레이어
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)
            
        # 3. 출력 Linear
        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(hidden_channels, out_channels)
        
        # 4. Dropout
        self.dropout = nn.Dropout(0.2)

    def encode(self, data, edge_index_dict, edge_weight_dict=None):
        """ 노드 임베딩 생성 """
        x_dict = {}
        for node_type, embed_layer in self.node_embeds.items():
            x_dict[node_type] = embed_layer.weight
            
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict.keys():
                x_dict[node_type] = self.dropout(F.relu(x_dict[node_type]))
            
        z_dict = {}
        for node_type, lin in self.lin_dict.items():
            z_dict[node_type] = lin(x_dict[node_type])
            
        return z_dict

    def decode(self, z_dict, edge_label_index):
        """ 엣지 예측 (Logits 반환) """
        compound_embeds = z_dict['Compound'][edge_label_index[0]]
        disease_embeds = z_dict['Disease'][edge_label_index[1]]
        
        # ✅ 단순 내적 (안정적)
        logits = (compound_embeds * disease_embeds).sum(dim=-1)
        
        # ✅ Clipping으로 안정화
        logits = torch.clamp(logits, min=-10, max=10)
        
        return logits

    def decode_all(self, z_dict, train_edge_index, ratio, epoch):
        """ PULL: 새로운 positive 엣지 발굴 """
        compound_embeds = z_dict['Compound']
        disease_embeds = z_dict['Disease']

        # 내적 계산
        raw_scores = compound_embeds @ disease_embeds.t()
        
        # ✅ 학습된 엣지 마스킹
        raw_scores[train_edge_index[0], train_edge_index[1]] = -1e9
        
        n_edge = train_edge_index.shape[1]
        n_edge_add = int(n_edge * ratio * (epoch - 1))
        
        if n_edge_add == 0:
            return torch.tensor([[],[]], dtype=torch.long, device=raw_scores.device), \
                   torch.tensor([], device=raw_scores.device)
                   
        flat_scores = raw_scores.flatten()
        top_k_indices = torch.topk(flat_scores, n_edge_add).indices
        
        row_indices = top_k_indices // raw_scores.shape[1]
        col_indices = top_k_indices % raw_scores.shape[1]
        
        edge_index_add = torch.stack([row_indices, col_indices], dim=0)
        
        # ✅ Weight 계산 (안정화)
        selected_scores = raw_scores[row_indices, col_indices]
        selected_scores = torch.clamp(selected_scores, min=-10, max=10)
        edge_weight_add = torch.sigmoid(selected_scores)
        
        return edge_index_add, edge_weight_add

    def merge_edge(self, edge_index, edge_weight, edge_index_add, edge_weight_add):
        """ 엣지 병합 """
        edge_index = torch.cat((edge_index, edge_index_add), 1)
        edge_weight = torch.cat((edge_weight, edge_weight_add))
        return edge_index, edge_weight