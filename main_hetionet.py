import os.path as osp
import argparse, random
import numpy as np
import time
import torch
import torch_geometric.transforms as T
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

# --- PULL-Hetionet 임포트 ---
from src.model_hetionet import HeteroPULLModel
from src.train_hetionet import train, test, get_drug_repurposing_candidates

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10, help="PULL의 Outer Epoch 수")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4, help="HGT의 Attention Head 수")
    parser.add_argument('--layers', type=int, default=2, help="HGT 레이어 수")
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--verbose', type=str, default="y")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 1. 전처리된 데이터 로드
    data_path = osp.join('data', 'hetionet_data.pt')
    if not osp.exists(data_path):
        print(f"오류: {data_path} 파일을 찾을 수 없습니다.")
        print("먼저 `python preprocess_hetionet.py`를 실행하여 데이터를 전처리하세요.")
        return
        
    print("전처리된 Hetionet 데이터 로드 중...")
    data = torch.load(data_path)
    
    # 2. 데이터 분할 (Train/Validation/Test)
    print("데이터 분할 중 (Train/Val/Test)...")
    edge_type_to_predict = ('Compound', 'treats', 'Disease')
    rev_edge_type_to_predict = ('Disease', 'rev_treats', 'Compound')
    
    # --- 수정: negative 샘플 추가 ---
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,  # 학습용은 PULL에서 자체 생성
        neg_sampling_ratio=1.0,  # validation/test용 negative 샘플 비율
        edge_types=[edge_type_to_predict],
        rev_edge_types=[rev_edge_type_to_predict]
    )
    
    train_data, val_data, test_data = transform(data)
    
    # 분할된 엣지 인덱스를 원본 'data' 객체에 저장
    data[edge_type_to_predict]['train_pos_edge_index'] = train_data[edge_type_to_predict].edge_index
    data[edge_type_to_predict]['val_pos_edge_index'] = val_data[edge_type_to_predict].edge_label_index
    data[edge_type_to_predict]['test_pos_edge_index'] = test_data[edge_type_to_predict].edge_label_index
    
    # --- 수정: negative 샘플 처리 ---
    val_edges = {
        'pos': val_data[edge_type_to_predict].edge_label_index,
        'neg': val_data[edge_type_to_predict].get('neg_edge_label_index', 
                torch.empty((2, 0), dtype=torch.long))  # 없으면 빈 텐서
    }
    test_edges = {
        'pos': test_data[edge_type_to_predict].edge_label_index,
        'neg': test_data[edge_type_to_predict].get('neg_edge_label_index',
                torch.empty((2, 0), dtype=torch.long))  # 없으면 빈 텐서
    }
    
    # negative 샘플이 없으면 수동 생성
    if val_edges['neg'].shape[1] == 0:
        print("  - Validation negative 샘플 생성 중...")
        from torch_geometric.utils import negative_sampling
        val_edges['neg'] = negative_sampling(
            edge_index=train_data[edge_type_to_predict].edge_index,
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=val_edges['pos'].size(1),
            method='sparse'
        )
    
    if test_edges['neg'].shape[1] == 0:
        print("  - Test negative 샘플 생성 중...")
        from torch_geometric.utils import negative_sampling
        test_edges['neg'] = negative_sampling(
            edge_index=train_data[edge_type_to_predict].edge_index,
            num_nodes=(data['Compound'].num_nodes, data['Disease'].num_nodes),
            num_neg_samples=test_edges['pos'].size(1),
            method='sparse'
        )
    
    # 데이터를 GPU/CPU로 이동
    data = data.to(device)
    train_data = train_data.to(device)
    val_edges['pos'] = val_edges['pos'].to(device)
    val_edges['neg'] = val_edges['neg'].to(device)
    test_edges['pos'] = test_edges['pos'].to(device)
    test_edges['neg'] = test_edges['neg'].to(device)

    print("\n[데이터 로드 완료]")
    print(f"학습용 'treats' 엣지: {train_data['Compound', 'treats', 'Disease'].edge_index.shape[1]}")
    print(f"검증용 'treats' 엣지 (Pos): {val_edges['pos'].shape[1]}")
    print(f"검증용 'treats' 엣지 (Neg): {val_edges['neg'].shape[1]}")
    print(f"테스트용 'treats' 엣지 (Pos): {test_edges['pos'].shape[1]}")
    print(f"테스트용 'treats' 엣지 (Neg): {test_edges['neg'].shape[1]}")

    # 3. 모델 초기화
    model = HeteroPULLModel(
        data=data,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_heads=args.heads,
        num_layers=args.layers,
    ).to(device)

    # [중요] Weight Decay 추가 확인됨
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = BCEWithLogitsLoss()

    # 4. PULL 학습 시작
    print("\n[PULL 모델 학습 시작]")
    best_val_auc = test_auc = 0
    best_epoch = 0
    z_dict = None 
    
    patience = 20  # 성능이 안 올라도 20번은 기다림
    patience_counter = 0 

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # PULL 학습
        loss, z_dict, _, _ = train(model, optimizer, data, train_data, criterion, epoch, z_dict)
        
        # 검증
        val_loss, val_auc = test(data, model, val_edges, criterion)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            # 테스트
            test_loss, test_auc = test(data, model, test_edges, criterion)
            patience_counter = 0 # 신기록 달성 시 카운터 초기화
        else:
            patience_counter += 1 # 실패 시 카운터 증가
            if patience_counter >= patience:
                print(f"Early Stopping: {patience} Epoch 동안 성능 향상이 없어 조기 종료합니다.")
                break

        epoch_time = time.time() - epoch_start_time
        if args.verbose == 'y':
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f} (Patience: {patience_counter}/{patience})')
        
    print("\n[학습 완료]")
    print(f'Best Epoch: {best_epoch:02d}, Val AUC: {best_val_auc:.4f}, Test AUC: {test_auc:.4f}')
    print(f'총 학습 시간: {(time.time() - start_time):.2f}s')

    # 5. 약물 재창출 후보 분석
    get_drug_repurposing_candidates(data, model, num_candidates=20)

if __name__ == '__main__':
    main()