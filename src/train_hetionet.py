from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
import gc

def train(model, optimizer, data, train_data, criterion, epoch, z_dict=None):
    """ PULL 로직 - 최종 개선 버전 """
    
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    
    if epoch == 1:
        pos_edge_index = train_data[edge_type_to_predict].edge_index
        pos_edge_weight = torch.ones(pos_edge_index.shape[1], device=pos_edge_index.device)
        edge_index_dict = train_data.edge_index_dict
        
    else:
        print("  - PULL: 새로운 positive 엣지 탐색 중... (CPU)")
        
        with torch.no_grad():
            # GPU → CPU
            z_dict_cpu = {k: v.cpu() for k, v in z_dict.items()}
            train_edge_cpu = train_data[edge_type_to_predict].edge_index.cpu()
            
            compound_embeds = z_dict_cpu['Compound']
            disease_embeds = z_dict_cpu['Disease']
            
            # 🔥 L2 정규화 추가
            compound_embeds = F.normalize(compound_embeds, p=2, dim=-1)
            disease_embeds = F.normalize(disease_embeds, p=2, dim=-1)

            # Cosine similarity
            raw_scores = compound_embeds @ disease_embeds.t()
            raw_scores = raw_scores / 1.0  # Temperature 0.1

            # ✅ 핵심 수정: Raw scores 사용, -inf로 마스킹
            raw_scores = compound_embeds @ disease_embeds.t()
            
            # Apply temperature scaling (match model's temperature)
            raw_scores = raw_scores / 5.0
            
            raw_scores[train_edge_cpu[0], train_edge_cpu[1]] = -float('inf')
            
            n_edge = train_edge_cpu.shape[1]
            n_edge_add = int(n_edge * 0.02 * (epoch - 1)) 
            
            if n_edge_add > 0:
                flat_scores = raw_scores.flatten()
                top_k_indices = torch.topk(flat_scores, n_edge_add).indices
                
                row_indices = top_k_indices // raw_scores.shape[1]
                col_indices = top_k_indices % raw_scores.shape[1]
                
                device = train_data[edge_type_to_predict].edge_index.device
                pos_edge_index_add = torch.stack([row_indices, col_indices], dim=0).to(device)
                
                # Soft weight: 선택된 엣지에만 sigmoid with clipping
                selected_scores = raw_scores[row_indices, col_indices]
                pos_edge_weight_add = torch.sigmoid(selected_scores)
                # Clip weights to 0.5-0.95 range to prevent overconfidence
                pos_edge_weight_add = torch.clamp(pos_edge_weight_add, min=0.3, max=0.8).to(device)
                
                print(f"  - PULL: {n_edge_add}개 엣지 추가됨 (평균 weight: {pos_edge_weight_add.mean():.4f})")
            else:
                device = train_data[edge_type_to_predict].edge_index.device
                pos_edge_index_add = torch.tensor([[],[]], dtype=torch.long, device=device)
                pos_edge_weight_add = torch.tensor([], device=device)
        
        # 메모리 정리
        del z_dict_cpu, compound_embeds, disease_embeds, raw_scores, flat_scores
        gc.collect()
        torch.cuda.empty_cache()
        
        # 엣지 병합
        original_pos_edge_index = train_data[edge_type_to_predict].edge_index
        original_pos_edge_weight = torch.ones(original_pos_edge_index.shape[1], device=original_pos_edge_index.device)
        
        pos_edge_index = torch.cat((original_pos_edge_index, pos_edge_index_add), 1)
        pos_edge_weight = torch.cat((original_pos_edge_weight, pos_edge_weight_add))
        
        edge_index_dict = train_data.edge_index_dict.copy()
        edge_index_dict[edge_type_to_predict] = pos_edge_index

    # ✅ 학습 파라미터 최적화
    num_inner_epochs = 100  # 증가
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    for inner_epoch in range(1, num_inner_epochs + 1):
        # ✅ 전체 엣지 사용 (샘플링 제거)
        batch_size = min(300, pos_edge_index.size(1))  # 동적 배치
        perm = torch.randperm(pos_edge_index.size(1))
        
        for i in range(0, pos_edge_index.size(1), batch_size):
            batch_idx = perm[i:min(i+batch_size, pos_edge_index.size(1))]
            
            batch_pos_edge = pos_edge_index[:, batch_idx]
            batch_pos_weight = pos_edge_weight[batch_idx]
            
            # Negative 샘플링
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
                num_neg_samples=batch_pos_edge.size(1),
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
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
            # 메모리 정리
            del out, edge_label, neg_edge_labels, edge_label_index, z_dict_new
        
        # 주기적 메모리 정리
        if inner_epoch % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 최종 z_dict
    with torch.no_grad():
        z_dict_final = model.encode(data, edge_index_dict, None)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return torch.tensor(avg_loss), z_dict_final, pos_edge_index, pos_edge_weight


@torch.no_grad()
def test(data, model, split_edge_index, criterion):
    """ 테스트/검증 """
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    pos_edge_index = split_edge_index['pos']
    neg_edge_index = split_edge_index['neg']
    
    pos_out = model.decode(z_dict, pos_edge_index)
    neg_out = model.decode(z_dict, neg_edge_index)
    
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
    """ 약물 재창출 후보 분석 with diversity algorithm """
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    compound_embeds = z_dict['Compound'].cpu()
    disease_embeds = z_dict['Disease'].cpu()
    
    # 🔥 L2 정규화 추가 (중요!)
    compound_embeds = F.normalize(compound_embeds, p=2, dim=-1)
    disease_embeds = F.normalize(disease_embeds, p=2, dim=-1)
    
    # Cosine similarity로 변환 (범위: -1 ~ 1)
    raw_scores = compound_embeds @ disease_embeds.t()
    
    # 디버깅
    print(f"[디버깅] Raw scores (정규화 후, cosine similarity):")
    print(f"  - 최소: {raw_scores[raw_scores != -float('inf')].min():.4f}")
    print(f"  - 최대: {raw_scores[raw_scores != -float('inf')].max():.4f}")
    print(f"  - 평균: {raw_scores[raw_scores != -float('inf')].mean():.4f}")
    
    # Temperature scaling
    raw_scores = raw_scores / 1.0
    
    # 기존 엣지 제외
    for split in ['train', 'val', 'test']:
        key_name = f'{split}_pos_edge_index'
        if key_name in data['Compound', 'treats', 'Disease']:
            edge_index = data['Compound', 'treats', 'Disease'][key_name].cpu()
            raw_scores[edge_index[0], edge_index[1]] = -float('inf')
    
    # Get more candidates
    initial_candidates = num_candidates * 10
    flat_scores = raw_scores.flatten()
    top_k_indices = torch.topk(flat_scores, initial_candidates).indices
    
    # Calculate probabilities
    top_k_probs = torch.sigmoid(flat_scores[top_k_indices])
    
    # 디버깅: 확률 분포 확인
    print(f"[디버깅] 상위 200개 확률 통계:")
    print(f"  - 최소: {top_k_probs.min():.4f}")
    print(f"  - 최대: {top_k_probs.max():.4f}")
    print(f"  - 평균: {top_k_probs.mean():.4f}")
    print(f"  - 중간값: {top_k_probs.median():.4f}")
    print(f"  - 0.99 이상 개수: {(top_k_probs >= 0.99).sum().item()}/{len(top_k_probs)}")
    print(f"  - 0.95 이상 개수: {(top_k_probs >= 0.95).sum().item()}/{len(top_k_probs)}")
    print(f"  - 0.90 이상 개수: {(top_k_probs >= 0.90).sum().item()}/{len(top_k_probs)}")
    
    row_indices = top_k_indices // raw_scores.shape[1]
    col_indices = top_k_indices % raw_scores.shape[1]
    
    compound_names = data.node_names['Compound']
    disease_names = data.node_names['Disease']
    
    # ✅ Diversity algorithm (필터 완화)
    disease_count = {}
    final_candidates = []
    max_per_disease = 3
    
    filtered_count = 0
    
    for i in range(initial_candidates):
        compound_idx = row_indices[i].item()
        disease_idx = col_indices[i].item()
        prob = top_k_probs[i].item()
        
        if prob >= 0.90:
            filtered_count += 1
            continue
            
        disease_name = disease_names[disease_idx]
        
        # Check diversity constraint
        if disease_count.get(disease_name, 0) >= max_per_disease:
            continue
            
        disease_count[disease_name] = disease_count.get(disease_name, 0) + 1
        final_candidates.append({
            'compound_idx': compound_idx,
            'disease_idx': disease_idx,
            'compound_name': compound_names[compound_idx],
            'disease_name': disease_name,
            'prob': prob,
            'raw_score': flat_scores[top_k_indices[i]].item()
        })
        
        if len(final_candidates) >= num_candidates:
            break
    
    print(f"\n[필터링 통계]")
    print(f"  - 0.97 이상으로 필터링된 후보: {filtered_count}개")
    print(f"  - 다양성 필터 통과: {len(final_candidates)}개")
    
    # Sort by raw score (descending)
    final_candidates.sort(key=lambda x: x['raw_score'], reverse=True)
    
    print("\n--- 새로운 약물 재창출 후보 Top 20 (다양성 보장) ---")
    print(f"필터링: 예측 확률 >= 0.97 제외, 질병당 최대 {max_per_disease}개")
    
    if len(final_candidates) == 0:
        print("⚠️ 경고: 조건을 만족하는 후보가 없습니다!")
        print("대안: Raw score 기준 상위 20개 (확률 무시)")
        
        # Raw score로 정렬
        all_candidates = []
        for i in range(min(num_candidates * 5, len(top_k_probs))):
            compound_idx = row_indices[i].item()
            disease_idx = col_indices[i].item()
            prob = top_k_probs[i].item()
            raw_score = flat_scores[top_k_indices[i]].item()
            
            disease_name = disease_names[disease_idx]
            if disease_count.get(disease_name, 0) >= max_per_disease:
                continue
                
            disease_count[disease_name] = disease_count.get(disease_name, 0) + 1
            all_candidates.append({
                'compound_idx': compound_idx,
                'disease_idx': disease_idx,
                'compound_name': compound_names[compound_idx],
                'disease_name': disease_name,
                'prob': prob,
                'raw_score': raw_score
            })
            
            if len(all_candidates) >= num_candidates:
                break
        
        for i, candidate in enumerate(all_candidates[:num_candidates]):
            print(f"{i+1:02d}. [약물] {candidate['compound_name']} -> [질병] {candidate['disease_name']} (확률: {candidate['prob']:.4f}, raw: {candidate['raw_score']:.2f})")
    else:
        for i, candidate in enumerate(final_candidates[:num_candidates]):
            print(f"{i+1:02d}. [약물] {candidate['compound_name']} -> [질병] {candidate['disease_name']} (예측 확률: {candidate['prob']:.4f})")
    
    return final_candidates
    """ 약물 재창출 후보 분석 with diversity algorithm """
    print("\n[약물 재창출 후보 분석 시작]")
    model.eval()
    
    z_dict = model.encode(data, data.edge_index_dict, None)
    
    compound_embeds = z_dict['Compound'].cpu()
    disease_embeds = z_dict['Disease'].cpu()
    
    # ✅ Raw scores 사용 with temperature scaling
    raw_scores = compound_embeds @ disease_embeds.t()
    # 🔥 디버깅: raw score 분포 확인
    print(f"[디버깅] Raw scores (temperature 적용 전):")
    print(f"  - 최소: {raw_scores.max():.4f}")  # max가 중요 (inf 제외)
    print(f"  - 최대: {raw_scores[raw_scores != -float('inf')].max():.4f}")
    print(f"  - 평균: {raw_scores[raw_scores != -float('inf')].mean():.4f}")
    raw_scores = raw_scores / 5.0  # Temperature 5.0
    
    # 기존 엣지 제외
    for split in ['train', 'val', 'test']:
        key_name = f'{split}_pos_edge_index'
        if key_name in data['Compound', 'treats', 'Disease']:
            edge_index = data['Compound', 'treats', 'Disease'][key_name].cpu()
            raw_scores[edge_index[0], edge_index[1]] = -float('inf')
    
    # Get more candidates initially for diversity filtering
    initial_candidates = num_candidates * 10
    flat_scores = raw_scores.flatten()
    top_k_indices = torch.topk(flat_scores, initial_candidates).indices
    
    # ✅ Calculate probabilities
    top_k_probs = torch.sigmoid(flat_scores[top_k_indices])
    
    # 🔥 디버깅: 확률 분포 확인
    print(f"[디버깅] 상위 200개 확률 통계:")
    print(f"  - 최소: {top_k_probs.min():.4f}")
    print(f"  - 최대: {top_k_probs.max():.4f}")
    print(f"  - 평균: {top_k_probs.mean():.4f}")
    print(f"  - 중간값: {top_k_probs.median():.4f}")
    print(f"  - 0.99 이상 개수: {(top_k_probs >= 0.99).sum().item()}/{len(top_k_probs)}")
    print(f"  - 0.95 이상 개수: {(top_k_probs >= 0.95).sum().item()}/{len(top_k_probs)}")
    
    row_indices = top_k_indices // raw_scores.shape[1]
    col_indices = top_k_indices % raw_scores.shape[1]
    
    compound_names = data.node_names['Compound']
    disease_names = data.node_names['Disease']
    
    # ✅ Diversity algorithm: max 3 per disease, filter prob > 0.95 (0.99에서 변경)
    disease_count = {}
    final_candidates = []
    max_per_disease = 3
    
    filtered_count = 0  # 필터링된 개수 카운트
    
    for i in range(initial_candidates):
        compound_idx = row_indices[i].item()
        disease_idx = col_indices[i].item()
        prob = top_k_probs[i].item()
        
        # Filter out overconfident predictions (0.99 → 0.98로 완화)
        if prob >= 0.98:
            filtered_count += 1
            continue
            
        disease_name = disease_names[disease_idx]
        
        # Check diversity constraint
        if disease_count.get(disease_name, 0) >= max_per_disease:
            continue
            
        disease_count[disease_name] = disease_count.get(disease_name, 0) + 1
        final_candidates.append({
            'compound_idx': compound_idx,
            'disease_idx': disease_idx,
            'compound_name': compound_names[compound_idx],
            'disease_name': disease_name,
            'prob': prob,
            'raw_score': flat_scores[top_k_indices[i]].item()
        })
        
        if len(final_candidates) >= num_candidates:
            break
    
    print(f"\n[필터링 통계]")
    print(f"  - 0.98 이상으로 필터링된 후보: {filtered_count}개")
    print(f"  - 다양성 필터 통과: {len(final_candidates)}개")
    
    # Sort by raw score (descending)
    final_candidates.sort(key=lambda x: x['raw_score'], reverse=True)
    
    print("\n--- 새로운 약물 재창출 후보 Top 20 (다양성 보장) ---")
    print(f"필터링: 예측 확률 >= 0.98 제외, 질병당 최대 {max_per_disease}개")
    
    if len(final_candidates) == 0:
        print("⚠️ 경고: 조건을 만족하는 후보가 없습니다!")
        print("대안: 필터 없이 상위 20개 출력")
        
        # 필터 없이 출력
        for i in range(min(num_candidates, len(top_k_probs))):
            compound_idx = row_indices[i].item()
            disease_idx = col_indices[i].item()
            prob = top_k_probs[i].item()
            print(f"{i+1:02d}. [약물] {compound_names[compound_idx]} -> [질병] {disease_names[disease_idx]} (예측 확률: {prob:.4f})")
    else:
        for i, candidate in enumerate(final_candidates[:num_candidates]):
            print(f"{i+1:02d}. [약물] {candidate['compound_name']} -> [질병] {candidate['disease_name']} (예측 확률: {candidate['prob']:.4f})")
    
    return final_candidates