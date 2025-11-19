import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HeteroPULLModel(nn.Module):
    def __init__(self, data, hidden_channels=128, out_channels=64, num_heads=4, num_layers=2):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 1. 입력 Linear (도메인 특징 → hidden)
        self.input_lins = nn.ModuleDict()
        
        # Compound: Morgan Fingerprint (512) → hidden
        if hasattr(data['Compound'], 'x') and data['Compound'].x is not None:
            compound_dim = data['Compound'].x.shape[1]
            self.input_lins['Compound'] = nn.Sequential(
                Linear(compound_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            print(f"  [모델] Compound 입력: {compound_dim}D → {hidden_channels}D (Morgan FP)")
        
        # Disease: One-hot → hidden
        if hasattr(data['Disease'], 'x') and data['Disease'].x is not None:
            disease_dim = data['Disease'].x.shape[1]
            self.input_lins['Disease'] = nn.Sequential(
                Linear(disease_dim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            print(f"  [모델] Disease 입력: {disease_dim}D → {hidden_channels}D (One-hot)")
        
        # 2. 나머지 노드는 임베딩
        self.node_embeds = nn.ModuleDict()
        for node_type in data.node_types:
            if node_type not in self.input_lins:
                num_nodes = data[node_type].num_nodes
                self.node_embeds[node_type] = nn.Embedding(num_nodes, hidden_channels)
                print(f"  [모델] {node_type}: {num_nodes}개 → {hidden_channels}D (임베딩)")

        # 3. HGT 레이어
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)
            
        # 4. 출력 Linear
        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(hidden_channels, out_channels)
        
        # 5. Dropout
        self.dropout = nn.Dropout(0.4)

    def encode(self, data, edge_index_dict, edge_weight_dict=None):
        """ 노드 임베딩 생성 """
        x_dict = {}
        
        # 입력 특징이 있는 노드는 Linear, 없으면 Embedding
        for node_type in data.node_types:
            if node_type in self.input_lins:
                # 도메인 특징 사용
                x_dict[node_type] = self.input_lins[node_type](data[node_type].x)
            else:
                # 랜덤 임베딩 사용
                x_dict[node_type] = self.node_embeds[node_type].weight
            
        # HGT 레이어
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict.keys():
                x_dict[node_type] = self.dropout(F.relu(x_dict[node_type]))
            
        # 출력 Linear
        z_dict = {}
        for node_type, lin in self.lin_dict.items():
            z_dict[node_type] = lin(x_dict[node_type])
            
        return z_dict

    def decode(self, z_dict, edge_label_index):
        """ 엣지 예측 (Logits 반환) """
        compound_embeds = z_dict['Compound'][edge_label_index[0]]
        disease_embeds = z_dict['Disease'][edge_label_index[1]]
        
        # 내적
        logits = (compound_embeds * disease_embeds).sum(dim=-1)
        
        # Clipping으로 안정화
        logits = torch.clamp(logits, min=-10, max=10)
        
        return logits