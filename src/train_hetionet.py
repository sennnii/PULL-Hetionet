from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import gc

def train(model, optimizer, data, train_data, criterion, epoch, z_dict=None):
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    
    # --- [설정] 논문 기반 하이퍼파라미터 조정 ---
    GROWTH_RATE = 0.02      # r: 에폭당 엣지 증가율 (기존 0.05 -> 0.02로 보수적 조정)
    MAX_EDGE_RATIO = 1.0    # 추가된 엣지가 원본 엣지의 100%를 넘지 않도록 제한 (Cap)
    CONFIDENCE_THR = 0.8    # 최소 신뢰도: 이 확률보다 낮으면 Top-K여도 추가 안 함
    # -----------------------------------------

    # 1. PULL: Expected Graph 구성 (Positive Edge 확장)
    if epoch == 1:
        # 첫 에폭은 원본 데이터만 사용
        pos_edge_index = train_data[edge_type_to_predict].edge_index
        pos_edge_weight = torch.ones(pos_edge_index.shape[1], device=pos_edge_index.device)
    else:
        print("  - PULL: 새로운 positive 엣지(Expected Graph) 탐색 중...")
        with torch.no_grad():
            # 임베딩 기반 스코어 계산
            z_src = z_dict['Compound']
            z_dst = z_dict['Disease']
            
            # 메모리 효율을 위해 간단히 계산
            raw_scores = z_src @ z_dst.t()
            probs = torch.sigmoid(raw_scores)
            
            # 이미 존재하는 엣지는 제외 (Masking)
            train_edges = train_data[edge_type_to_predict].edge_index
            probs[train_edges[0], train_edges[1]] = 0.0
            
            # [수정 1] 추가할 엣지 수 계산 및 제한 (Bounded Growth)
            n_total_edges = train_edges.shape[1]
            
            # 목표 추가 개수: 에폭에 비례하여 증가
            target_n_add = int(n_total_edges * GROWTH_RATE * (epoch - 1))
            
            # 최대 허용 개수 (Cap): 원본 데이터보다 너무 커지지 않게 제한
            max_allowed = int(n_total_edges * MAX_EDGE_RATIO)
            n_add = min(target_n_add, max_allowed)
            
            if n_add > 0:
                # Flatten 및 Top-K 추출
                flat_probs = probs.flatten()
                
                # [수정 2] Confidence Threshold 적용 (노이즈 필터링)
                # Top-K를 뽑되, 너무 낮은 확률값은 제외하기 위해 마스킹
                mask = flat_probs > CONFIDENCE_THR
                if mask.sum() < n_add:
                    n_add = mask.sum().item() # 신뢰할 수 있는 엣지가 부족하면 개수 줄임
                
                if n_add > 0:
                    topk_values, topk_indices = torch.topk(flat_probs, n_add)
                    
                    row_idx = topk_indices // probs.shape[1]
                    col_idx = topk_indices % probs.shape[1]
                    
                    pos_edge_index_add = torch.stack([row_idx, col_idx], dim=0)
                    pos_edge_weight_add = topk_values  # 확률값을 weight로 사용 (Soft Label)
                    
                    print(f"  - PULL: {n_add}개 엣지 추가 (Max Cap: {max_allowed}, Avg Weight: {pos_edge_weight_add.mean():.4f})")
                else:
                    print("  - PULL: 신뢰도 높은 엣지가 없어 추가하지 않음.")
                    pos_edge_index_add = torch.empty((2, 0), dtype=torch.long, device=raw_scores.device)
                    pos_edge_weight_add = torch.tensor([], device=raw_scores.device)
            else:
                pos_edge_index_add = torch.empty((2, 0), dtype=torch.long, device=raw_scores.device)
                pos_edge_weight_add = torch.tensor([], device=raw_scores.device)
        
        # 원본 엣지 + 추가된 엣지 병합
        original_edges = train_data[edge_type_to_predict].edge_index
        original_weights = torch.ones(original_edges.shape[1], device=original_edges.device)
        
        # 병합
        pos_edge_index = torch.cat([original_edges, pos_edge_index_add], dim=1)
        pos_edge_weight = torch.cat([original_weights, pos_edge_weight_add], dim=0)

    # 2. Edge Weight Dict 생성 (모델 전달용)
    edge_weight_dict = {}
    for etype in data.edge_types:
        if etype == edge_type_to_predict:
            edge_weight_dict[etype] = pos_edge_weight
        else:
            if etype in train_data.edge_index_dict:
                 edge_weight_dict[etype] = torch.ones(train_data[etype].edge_index.shape[1], device=pos_edge_index.device)

    # Edge Index Dict 업데이트
    edge_index_dict = train_data.edge_index_dict.copy()
    edge_index_dict[edge_type_to_predict] = pos_edge_index

    # 3. 내부 학습 루프 (Inner Loop)
    model.train()
    total_loss = 0
    num_inner_epochs = 50  # Inner Epoch
    
    for _ in range(num_inner_epochs):
        optimizer.zero_grad()
        
        # Negative Sampling
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, 
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=pos_edge_index.shape[1],
            method='sparse'
        )
        
        # Forward Pass (edge_weight_dict 전달!)
        z_dict_new = model.encode(data, edge_index_dict, edge_weight_dict)
        
        # Score 계산
        pos_out = model.decode(z_dict_new, pos_edge_index)
        neg_out = model.decode(z_dict_new, neg_edge_index)
        
        # Weighted BCE Loss (논문의 L_E에 해당)
        out = torch.cat([pos_out, neg_out])
        label = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        weight = torch.cat([pos_edge_weight, torch.ones_like(neg_out)]) 
        
        loss = F.binary_cross_entropy_with_logits(out, label, weight=weight)
        
        # [선택 사항] Correction Loss (L_C) - 원본 데이터 강조
        # 논문에서는 L_C를 별도로 계산하지만, 간단히 구현하기 위해 L2 정규화나 
        # 원본 엣지에 대한 가중치를 높이는 방식을 사용할 수도 있음.
        # 현재 구조에서는 'Bounded Growth'(위의 n_limit)가 가장 큰 효과를 줍니다.
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 4. 다음 Epoch을 위한 임베딩 저장
    with torch.no_grad():
        z_dict = model.encode(data, edge_index_dict, edge_weight_dict)
        
    return total_loss / num_inner_epochs, z_dict, None, None

@torch.no_grad()
def test(data, model, split_edge_index, criterion):
    model.eval()
    # Test 시에는 Weight 없이 Forward (None 전달)
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
    """ 약물 재창출 후보 분석 with diversity algorithm """
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    
    # Temperature Scaling 추가 (확률 분포 완화)
    temperature = 2.0 
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    compound_embeds = z_dict['Compound'].cpu()
    disease_embeds = z_dict['Disease'].cpu()
    
    raw_scores = compound_embeds @ disease_embeds.t()

    # 마스킹 전 통계
    print(f"[디버깅] Raw scores (마스킹 전):")
    valid_scores = raw_scores[raw_scores != -float('inf')]
    print(f"  - 최소: {valid_scores.min():.4f}, 최대: {valid_scores.max():.4f}")

    # 기존 엣지 마스킹
    if ('Compound', 'treats', 'Disease') in data.edge_types:
        for split in ['train_pos_edge_index', 'val_pos_edge_index', 'test_pos_edge_index']:
            if split in data['Compound', 'treats', 'Disease']:
                edge_index = data['Compound', 'treats', 'Disease'][split].cpu()
                raw_scores[edge_index[0], edge_index[1]] = -float('inf')
    
    # Temperature Scaling 적용 및 Sigmoid
    probs = torch.sigmoid(raw_scores / temperature)
    
    # Get Top-K Candidates
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
    
    print("\n--- 새로운 약물 재창출 후보 Top 20 (Temperature Adjusted) ---")
    
    for i in range(initial_candidates):
        prob = top_k_probs[i].item()
        # 임계값 완화
        if prob < 0.1: continue 

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