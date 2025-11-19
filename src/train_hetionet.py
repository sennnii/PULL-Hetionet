from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import gc

def train(model, optimizer, data, train_data, criterion, epoch, z_dict=None):
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    
    # --- [설정] 논문 기반 하이퍼파라미터 + 안전장치 ---
    # 논문은 r=0.05를 쓰지만, 데이터가 작을 땐 0.02가 안전합니다.
    # L_C가 추가되었으므로 0.03으로 살짝 높여 학습 속도를 확보합니다.
    GROWTH_RATE = 0.03      
    MAX_EDGE_RATIO = 1.0    # Max Cap: 원본 대비 100%까지만 확장
    CONFIDENCE_THR = 0.85   # Threshold: 0.85 이상만 신뢰 (더 엄격하게)
    # ------------------------------------------------

    # 1. PULL: Expected Graph 구성 (Positive Edge 확장)
    if epoch == 1:
        # 첫 에폭은 원본 데이터만 사용
        pos_edge_index = train_data[edge_type_to_predict].edge_index
        pos_edge_weight = torch.ones(pos_edge_index.shape[1], device=pos_edge_index.device)
    else:
        print("  - PULL: 새로운 positive 엣지(Expected Graph) 탐색 중...")
        with torch.no_grad():
            z_src = z_dict['Compound']
            z_dst = z_dict['Disease']
            
            # Score 계산
            raw_scores = z_src @ z_dst.t()
            probs = torch.sigmoid(raw_scores)
            
            # 기존 엣지 마스킹
            train_edges = train_data[edge_type_to_predict].edge_index
            probs[train_edges[0], train_edges[1]] = 0.0
            
            # [안전장치 1] Bounded Growth (개수 제한)
            n_total_edges = train_edges.shape[1]
            target_n_add = int(n_total_edges * GROWTH_RATE * (epoch - 1))
            max_allowed = int(n_total_edges * MAX_EDGE_RATIO)
            n_add = min(target_n_add, max_allowed)
            
            if n_add > 0:
                flat_probs = probs.flatten()
                
                # [안전장치 2] Confidence Threshold (품질 보장)
                mask = flat_probs > CONFIDENCE_THR
                if mask.sum() < n_add:
                    n_add = mask.sum().item()
                
                if n_add > 0:
                    topk_values, topk_indices = torch.topk(flat_probs, n_add)
                    
                    row_idx = topk_indices // probs.shape[1]
                    col_idx = topk_indices % probs.shape[1]
                    
                    pos_edge_index_add = torch.stack([row_idx, col_idx], dim=0)
                    pos_edge_weight_add = topk_values 
                    
                    print(f"  - PULL: {n_add}개 엣지 추가 (Max Cap: {max_allowed}, Min Prob: {topk_values.min():.4f})")
                else:
                    print("  - PULL: 신뢰도 높은 엣지가 없어 추가하지 않음.")
                    pos_edge_index_add = torch.empty((2, 0), dtype=torch.long, device=raw_scores.device)
                    pos_edge_weight_add = torch.tensor([], device=raw_scores.device)
            else:
                pos_edge_index_add = torch.empty((2, 0), dtype=torch.long, device=raw_scores.device)
                pos_edge_weight_add = torch.tensor([], device=raw_scores.device)
        
        # 병합 (Expected Graph)
        original_edges = train_data[edge_type_to_predict].edge_index
        original_weights = torch.ones(original_edges.shape[1], device=original_edges.device)
        
        pos_edge_index = torch.cat([original_edges, pos_edge_index_add], dim=1)
        pos_edge_weight = torch.cat([original_weights, pos_edge_weight_add], dim=0)

    # 2. Edge Weight Dict 준비 (L_E용)
    edge_weight_dict = {}
    for etype in data.edge_types:
        if etype == edge_type_to_predict:
            edge_weight_dict[etype] = pos_edge_weight
        else:
            if etype in train_data.edge_index_dict:
                 edge_weight_dict[etype] = torch.ones(train_data[etype].edge_index.shape[1], device=pos_edge_index.device)

    edge_index_dict = train_data.edge_index_dict.copy()
    edge_index_dict[edge_type_to_predict] = pos_edge_index

    # [L_C 준비] 원본 엣지 (Ground Truth)
    orig_pos_edge_index = train_data[edge_type_to_predict].edge_index

    # 3. 내부 학습 루프 (Inner Loop)
    model.train()
    total_loss = 0
    num_inner_epochs = 50 
    
    for _ in range(num_inner_epochs):
        optimizer.zero_grad()
        
        # 임베딩 생성
        z_dict_new = model.encode(data, edge_index_dict, edge_weight_dict)

        # --- Loss 1: L_E (Expected Graph Loss) ---
        # 확장된 그래프(PULL 엣지 포함)에 대한 학습
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, 
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=pos_edge_index.shape[1],
            method='sparse'
        )
        
        pos_out = model.decode(z_dict_new, pos_edge_index)
        neg_out = model.decode(z_dict_new, neg_edge_index)
        
        out = torch.cat([pos_out, neg_out])
        label = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        weight = torch.cat([pos_edge_weight, torch.ones_like(neg_out)]) 
        
        loss_e = F.binary_cross_entropy_with_logits(out, label, weight=weight)
        
        # --- Loss 2: L_C (Correction Loss) - 논문 핵심 구현 ---
        # 오직 원본 데이터(Ground Truth)에 대해서만 학습하여 중심을 잡음
        # (PULL 엣지가 아무리 많아져도, 원본 데이터의 중요도는 유지됨)
        
        neg_edge_index_c = negative_sampling(
            edge_index=orig_pos_edge_index,
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=orig_pos_edge_index.shape[1], # 원본 크기만큼 샘플링
            method='sparse'
        )

        pos_out_c = model.decode(z_dict_new, orig_pos_edge_index)
        neg_out_c = model.decode(z_dict_new, neg_edge_index_c)

        out_c = torch.cat([pos_out_c, neg_out_c])
        label_c = torch.cat([torch.ones_like(pos_out_c), torch.zeros_like(neg_out_c)])
        
        loss_c = F.binary_cross_entropy_with_logits(out_c, label_c)

        # 최종 Loss 합산
        loss = loss_e + loss_c
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    with torch.no_grad():
        z_dict = model.encode(data, edge_index_dict, edge_weight_dict)
        
    return total_loss / num_inner_epochs, z_dict, None, None

@torch.no_grad()
def test(data, model, split_edge_index, criterion):
    model.eval()
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    pos_out = model.decode(z_dict, split_edge_index['pos'])
    neg_out = model.decode(z_dict, split_edge_index['neg'])
    
    out = torch.cat([pos_out, neg_out])
    label = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    
    loss = criterion(out, label)
    auc = roc_auc_score(label.cpu().numpy(), out.cpu().numpy())
    
    return loss.item(), auc

@torch.no_grad()
def get_drug_repurposing_candidates(data, model, num_candidates=20):
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    
    # Temperature Scaling: 1.0 (기본값)
    # L_C가 추가되어 모델이 더 보수적으로 변했을 것이므로 
    # Temperature를 너무 높이지 않아도 됨
    temperature = 1.0 
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    compound_embeds = z_dict['Compound'].cpu()
    disease_embeds = z_dict['Disease'].cpu()
    
    raw_scores = compound_embeds @ disease_embeds.t()

    print(f"[디버깅] Raw scores (마스킹 전):")
    valid_scores = raw_scores[raw_scores != -float('inf')]
    print(f"  - 최소: {valid_scores.min():.4f}, 최대: {valid_scores.max():.4f}")

    if ('Compound', 'treats', 'Disease') in data.edge_types:
        for split in ['train_pos_edge_index', 'val_pos_edge_index', 'test_pos_edge_index']:
            if split in data['Compound', 'treats', 'Disease']:
                edge_index = data['Compound', 'treats', 'Disease'][split].cpu()
                raw_scores[edge_index[0], edge_index[1]] = -float('inf')
    
    probs = torch.sigmoid(raw_scores / temperature)
    
    initial_candidates = num_candidates * 10
    flat_probs = probs.flatten()
    
    top_k_probs, top_k_indices = torch.topk(flat_probs, initial_candidates)
    
    row_indices = top_k_indices // probs.shape[1]
    col_indices = top_k_indices % probs.shape[1]
    
    compound_names = data.node_names['Compound']
    disease_names = data.node_names['Disease']
    
    disease_count = {}
    final_candidates = []
    max_per_disease = 3
    
    print("\n--- 새로운 약물 재창출 후보 Top 20 (Correction Loss Applied) ---")
    
    for i in range(initial_candidates):
        prob = top_k_probs[i].item()
        if prob < 0.5: continue 

        idx_c = row_indices[i].item()
        idx_d = col_indices[i].item()
        
        d_name = disease_names[idx_d]
        c_name = compound_names[idx_c]
        
        if disease_count.get(d_name, 0) >= max_per_disease:
            continue
            
        disease_count[d_name] = disease_count.get(d_name, 0) + 1
        
        final_candidates.append({
            'compound': c_name,
            'disease': d_name,
            'prob': prob
        })
        
        if len(final_candidates) >= num_candidates:
            break
            
    for i, item in enumerate(final_candidates):
        print(f"{i+1:02d}. [약물] {item['compound']} -> [질병] {item['disease']} (확률: {item['prob']:.4f})")
        
    return final_candidates