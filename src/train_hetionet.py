from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import gc

def train(model, optimizer, data, train_data, criterion, epoch, z_dict=None):
    """ PULL 로직 - 개선 버전 """
    
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    
    if epoch == 1:
        pos_edge_index = train_data[edge_type_to_predict].edge_index
        pos_edge_weight = torch.ones(pos_edge_index.shape[1], device=pos_edge_index.device)
        edge_index_dict = train_data.edge_index_dict
        
    else:
        print("  - PULL: 새로운 positive 엣지 탐색 중... (CPU)")
        
        with torch.no_grad():
            z_dict_cpu = {k: v.cpu() for k, v in z_dict.items()}
            train_edge_cpu = train_data[edge_type_to_predict].edge_index.cpu()
            
            compound_embeds = z_dict_cpu['Compound']
            disease_embeds = z_dict_cpu['Disease']
            
            # ✅ 수정 1: Sigmoid 제거, 음수 infinity로 제외
            prob_adj = compound_embeds @ disease_embeds.t()
            prob_adj[train_edge_cpu[0], train_edge_cpu[1]] = -float('inf')  # 🔧 0 → -inf
            
            n_edge = train_edge_cpu.shape[1]
            n_edge_add = int(n_edge * 0.03 * (epoch - 1))  # 🔧 0.02 → 0.03
            
            if n_edge_add > 0:
                flat_probs = prob_adj.flatten()
                top_k_indices = torch.topk(flat_probs, n_edge_add).indices
                
                row_indices = top_k_indices // prob_adj.shape[1]
                col_indices = top_k_indices % prob_adj.shape[1]
                
                device = train_data[edge_type_to_predict].edge_index.device
                pos_edge_index_add = torch.stack([row_indices, col_indices], dim=0).to(device)
                
                # ✅ 수정 2: Soft weight 적용 (상위 결과만 sigmoid)
                raw_probs = prob_adj[row_indices, col_indices]
                pos_edge_weight_add = torch.sigmoid(raw_probs).to(device)  # 🔧 여기서만 sigmoid
                
                print(f"  - PULL: {n_edge_add}개 엣지 추가됨")
            else:
                device = train_data[edge_type_to_predict].edge_index.device
                pos_edge_index_add = torch.tensor([[],[]], dtype=torch.long, device=device)
                pos_edge_weight_add = torch.tensor([], device=device)
        
        # ✅ 수정 3: 메모리 정리 개선
        del z_dict_cpu, compound_embeds, disease_embeds, prob_adj
        if 'flat_probs' in locals():
            del flat_probs
        gc.collect()
        torch.cuda.empty_cache()
        
        original_pos_edge_index = train_data[edge_type_to_predict].edge_index
        original_pos_edge_weight = torch.ones(original_pos_edge_index.shape[1], device=original_pos_edge_index.device)
        
        pos_edge_index = torch.cat((original_pos_edge_index, pos_edge_index_add), 1)
        pos_edge_weight = torch.cat((original_pos_edge_weight, pos_edge_weight_add))
        
        edge_index_dict = train_data.edge_index_dict.copy()
        edge_index_dict[edge_type_to_predict] = pos_edge_index

    # ✅ 수정 4: Inner Loop 증가 및 전체 데이터 사용
    num_inner_epochs = 50  # 🔧 30 → 50
    batch_size = 200       # 🔧 128 → 200
    
    model.train()
    total_loss = 0
    
    for inner_epoch in range(1, num_inner_epochs + 1):
        # ✅ 수정 5: 전체 엣지 사용 (샘플링 제거)
        num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))
            
            # Positive 배치
            batch_pos_edge = pos_edge_index[:, start_idx:end_idx]
            batch_pos_weight = pos_edge_weight[start_idx:end_idx]
            
            # Negative 샘플링
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
                num_neg_samples=batch_pos_edge.size(1),  # 🔧 배치 크기만큼
                method='sparse'
            )
            
            optimizer.zero_grad()
            
            edge_label_index = torch.cat([batch_pos_edge, neg_edge_index], dim=-1)
            z_dict_new = model.encode(data, edge_index_dict, None)
            out = model.decode(z_dict_new, edge_label_index).view(-1)
            
            neg_edge_labels = torch.zeros(neg_edge_index.size(1), device=batch_pos_edge.device)
            edge_label = torch.cat([batch_pos_weight, neg_edge_labels], dim=0)
            
            loss = criterion(out, edge_label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 메모리 정리
            del out, edge_label, neg_edge_labels, edge_label_index
        
        if inner_epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 최종 z_dict
    with torch.no_grad():
        z_dict_final = model.encode(data, edge_index_dict, None)
    
    # ✅ 수정 6: Loss 계산 수정
    avg_loss = total_loss / (num_batches * num_inner_epochs)
    
    return torch.tensor(avg_loss), z_dict_final, pos_edge_index, pos_edge_weight


@torch.no_grad()
def test(data, model, split_edge_index, criterion):
    """ 테스트/검증 - 동일 """
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    pos_edge_index = split_edge_index['pos']
    neg_edge_index = split_edge_index['neg']
    
    # 배치 처리
    batch_size = 256
    
    pos_outs = []
    for i in range(0, pos_edge_index.size(1), batch_size):
        batch = pos_edge_index[:, i:i+batch_size]
        pos_outs.append(model.decode(z_dict, batch))
    pos_out = torch.cat(pos_outs)
    
    neg_outs = []
    for i in range(0, neg_edge_index.size(1), batch_size):
        batch = neg_edge_index[:, i:i+batch_size]
        neg_outs.append(model.decode(z_dict, batch))
    neg_out = torch.cat(neg_outs)
    
    out = torch.cat([pos_out, neg_out]).view(-1)
    edge_label = torch.cat([
        torch.ones(pos_out.size(0)),
        torch.zeros(neg_out.size(0))
    ], dim=0).to(out.device)
    
    loss = criterion(out, edge_label)
    auc = roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())
    
    return loss, auc


@torch.no_grad()
def get_drug_repurposing_candidates(data, model, num_candidates=20):
    """ 약물 재창출 후보 분석 """
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    compound_embeds = z_dict['Compound'].cpu()
    disease_embeds = z_dict['Disease'].cpu()
    
    # ✅ 수정 7: Sigmoid 제거, 음수 infinity로 제외
    prob_adj = compound_embeds @ disease_embeds.t()
    
    for split in ['train', 'val', 'test']:
        key_name = f'{split}_pos_edge_index'
        if key_name in data['Compound', 'treats', 'Disease']:
            edge_index = data['Compound', 'treats', 'Disease'][key_name].cpu()
            prob_adj[edge_index[0], edge_index[1]] = -float('inf')  # 🔧 0 → -inf
    
    flat_probs = prob_adj.flatten()
    top_k_indices = torch.topk(flat_probs, num_candidates).indices
    
    # ✅ 수정 8: 상위 결과만 Sigmoid 적용
    top_k_probs = torch.sigmoid(flat_probs[top_k_indices])  # 🔧 여기서만 sigmoid
    
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