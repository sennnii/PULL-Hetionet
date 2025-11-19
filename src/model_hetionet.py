import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, Linear
from collections import defaultdict

class HeteroPULLModel(nn.Module):
    def __init__(self, data, hidden_channels=128, out_channels=64, num_heads=4, num_layers=2):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_types = data.edge_types
        self.node_types = data.node_types
        
        # 1. 입력 Projection
        self.input_lins = nn.ModuleDict()
        self.node_embeds = nn.ModuleDict()
        
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                in_dim = data[node_type].x.shape[1]
                self.input_lins[node_type] = nn.Sequential(
                    Linear(in_dim, hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                print(f"  [Model] {node_type}: Feat {in_dim} -> Hidden {hidden_channels}")
            else:
                num_nodes = data[node_type].num_nodes
                self.node_embeds[node_type] = nn.Embedding(num_nodes, hidden_channels)
                nn.init.xavier_uniform_(self.node_embeds[node_type].weight)
                print(f"  [Model] {node_type}: Emb {num_nodes} -> Hidden {hidden_channels}")

        # 2. Encoder: HeteroConv를 쓰지 않고 ModuleDict로 직접 관리
        # 이렇게 해야 가중치를 엣지별로 정확히 넣어줄 수 있습니다.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict()
            for edge_type in data.edge_types:
                # 엣지 타입 이름을 문자열 키로 변환 ('src__rel__dst')
                edge_key = '__'.join(edge_type)
                conv_dict[edge_key] = GraphConv(hidden_channels, hidden_channels, aggr='mean')
            self.convs.append(conv_dict)

        # 3. Decoder
        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(hidden_channels, out_channels)

    def encode(self, data, edge_index_dict, edge_weight_dict=None):
        x_dict = {}
        
        # 1. Feature/Embedding 준비
        for node_type in self.node_types:
            if node_type in self.input_lins:
                x_dict[node_type] = self.input_lins[node_type](data[node_type].x)
            else:
                x_dict[node_type] = self.node_embeds[node_type].weight
        
        # 2. Custom Hetero Convolution Loop
        for conv_layer in self.convs:
            x_dict_out = defaultdict(list)
            
            # 모든 엣지 타입에 대해 루프 수행
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                edge_key = '__'.join(edge_type)
                
                if edge_key not in conv_layer:
                    continue
                
                # 소스/타겟 노드 임베딩 가져오기
                x_src = x_dict[src]
                x_dst = x_dict[dst]
                
                # 해당 엣지 타입에 맞는 가중치 가져오기
                edge_weight = None
                if edge_weight_dict is not None and edge_type in edge_weight_dict:
                    edge_weight = edge_weight_dict[edge_type]
                
                # GraphConv 수행
                # (src, dst) 튜플 입력 지원, edge_weight 지원
                out = conv_layer[edge_key]((x_src, x_dst), edge_index, edge_weight=edge_weight)
                
                x_dict_out[dst].append(out)
            
            # Aggregation (Sum)
            for node_type, outs in x_dict_out.items():
                if len(outs) > 0:
                    x_dict[node_type] = torch.stack(outs).sum(dim=0)
                    # Skip connection이나 Self-loop가 없으므로 기존 값과 합치거나 Activation만 적용
                    x_dict[node_type] = F.relu(x_dict[node_type])
                    x_dict[node_type] = F.dropout(x_dict[node_type], p=0.4, training=self.training)
        
        # 3. Final Projection
        z_dict = {}
        for node_type, lin in self.lin_dict.items():
            if node_type in x_dict:
                z_dict[node_type] = lin(x_dict[node_type])
            
        return z_dict

    def decode(self, z_dict, edge_label_index):
        src_emb = z_dict['Compound'][edge_label_index[0]]
        dst_emb = z_dict['Disease'][edge_label_index[1]]
        logits = (src_emb * dst_emb).sum(dim=-1)
        return logits