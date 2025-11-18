from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

def train(model, optimizer, data, train_data, criterion, epoch, z_dict=None):
    """ PULL 로직으로 이종 GNN 모델을 학습 """
    
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    
    # Epoch 1 (초기 학습)
    if epoch == 1:
        pos_edge_index = train_data[edge_type_to_predict].edge_index
        pos_edge_weight = torch.ones(pos_edge_index.shape[1], device=pos_edge_index.device)
        edge_index_dict = train_data.edge_index_dict
        
    # Epoch 2+ (PU Learning)
    else:
        pos_edge_index_add, pos_edge_weight_add = model.decode_all(
            z_dict, 
            train_data[edge_type_to_predict].edge_index, 
            ratio=0.02,  # 🆕 0.05 → 0.02로 감소
            epoch=epoch
        )
        
        original_pos_edge_index = train_data[edge_type_to_predict].edge_index
        original_pos_edge_weight = torch.ones(original_pos_edge_index.shape[1], device=original_pos_edge_index.device)
        
        pos_edge_index, pos_edge_weight = model.merge_edge(
            original_pos_edge_index, original_pos_edge_weight,
            pos_edge_index_add, pos_edge_weight_add
        )
        
        pos_edge_index = pos_edge_index.detach()
        pos_edge_weight = pos_edge_weight.detach()

        edge_index_dict = train_data.edge_index_dict.copy()
        edge_index_dict[edge_type_to_predict] = pos_edge_index

    # 🆕 모델 학습 (Inner Loop 감소: 100 → 50)
    for inner_epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, 
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=pos_edge_index.size(1), 
            method='sparse'
        )

        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        z_dict_new = model.encode(data, edge_index_dict, None)
        out = model.decode(z_dict_new, edge_label_index).view(-1)
        
        neg_edge_labels = torch.zeros(neg_edge_index.size(1), device=pos_edge_index.device)
        edge_label = torch.cat([pos_edge_weight, neg_edge_labels], dim=0)

        loss = criterion(out, edge_label)
        loss.backward()
        
        # 🆕 Gradient Clipping (gradient exploding 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

    return loss, z_dict_new, pos_edge_index, pos_edge_weight


@torch.no_grad()
def test(data, model, split_edge_index, criterion):
    """ 테스트/검증 데이터로 모델 성능(AUC) 평가 """
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    pos_edge_index = split_edge_index['pos']
    pos_out = model.decode(z_dict, pos_edge_index)
    
    neg_edge_index = split_edge_index['neg']
    neg_out = model.decode(z_dict, neg_edge_index)

    out = torch.cat([pos_out, neg_out]).view(-1)
    edge_label = torch.cat([
        torch.ones(pos_out.size(0)), 
        torch.zeros(neg_out.size(0))
    ], dim=0).to(out.device)
    
    loss = criterion(out, edge_label)
    # 🆕 Sigmoid 적용 (AUC 계산용)
    out_sigmoid = torch.sigmoid(out)
    auc = roc_auc_score(edge_label.cpu().numpy(), out_sigmoid.cpu().numpy())
    
    return loss, auc

@torch.no_grad()
def get_drug_repurposing_candidates(data, model, num_candidates=20):
    """ 학습된 모델로 새로운 약물-질병 치료 관계(P)를 발굴 """
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    compound_embeds = z_dict['Compound']
    disease_embeds = z_dict['Disease']
    
    # 🆕 내적 계산 (sigmoid는 나중에)
    prob_adj = compound_embeds @ disease_embeds.t()
    
    # 훈련/검증/테스트에 이미 사용된 엣지는 후보에서 제외
    for split in ['train', 'val', 'test']:
        key_name = f'{split}_pos_edge_index'
        if key_name in data['Compound', 'treats', 'Disease']:
            edge_index = data['Compound', 'treats', 'Disease'][key_name]
            prob_adj[edge_index[0], edge_index[1]] = -float('inf')
        
    flat_probs = prob_adj.flatten()
    top_k_indices = torch.topk(flat_probs, num_candidates).indices
    # 🆕 Sigmoid 적용
    top_k_probs = torch.sigmoid(flat_probs[top_k_indices])

    row_indices = top_k_indices // prob_adj.shape[1]
    col_indices = top_k_indices % prob_adj.shape[1]
    
    compound_names = data.node_names['Compound']
    disease_names = data.node_names['Disease']
    
    print("\n--- 새로운 약물 재창출 후보 Top 20 ---")
    for i in range(num_candidates):
        compound_idx = row_indices[i].item()
        disease_idx = col_indices[i].item()
        prob = top_k_probs[i].item()
        
        print(f"{i+1:02d}. [약물] {compound_names[compound_idx]} -> [질병] {disease_names[disease_idx]} (예측 확률: {prob:.4f})")
        
    return