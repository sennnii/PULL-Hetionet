import bz2
import json
import os
import os.path as osp
from collections import defaultdict

import torch
import pandas as pd
from torch_geometric.data import HeteroData
from tqdm import tqdm

def load_hetionet_json():
    """ hetionet-v1.0.json.bz2 파일을 로드합니다. """
    print("Hetionet JSON 파일 로드 중...")

    data_dir = 'data'
    if not osp.exists(data_dir):
        os.makedirs(data_dir)

    filepath = osp.join(data_dir, 'hetionet-v1.0.json.bz2')
    if not osp.exists(filepath):
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        print(f"`{data_dir}/` 폴더에 hetionet-v1.0.json.bz2 파일을 넣어주세요.")
        exit()

    with bz2.open(filepath, 'rt') as f:
        data = json.load(f)
    return data

def preprocess_hetionet(hetionet_data):
    """ Hetionet JSON을 torch_geometric.data.HeteroData 객체로 변환합니다. """
    print("HeteroData 객체 생성 중...")

    data = HeteroData()

    # 1. 노드(Node) 처리
    print("  - 1/3: 노드 매핑 중...")
    node_mapping = {} # {'NodeType': {'Compound::DB001': mapped_int}}

    # --- 수정 1: 'metanode_kinds'는 문자열의 배열입니다. ---
    # (예: "Compound", "Gene")
    for node_type_name in hetionet_data['metanode_kinds']:
        node_mapping[node_type_name] = {}
    # --- (수정 1 끝) ---

    node_idx_counter = defaultdict(int)

    for node in tqdm(hetionet_data['nodes']):
        node_type = node['kind']       # 예: "Gene"
        node_identifier = node['identifier'] # 예: 5345 또는 "GO:0031753"
        
        # --- 수정 2: 'edges' 형식과 일치하는 복합 ID 생성 ---
        # (예: "Gene::5345")
        compound_id = f"{node_type}::{str(node_identifier)}"
        # --- (수정 2 끝) ---

        # (이제 node_type이 올바른 문자열 키이므로 정상 동작)
        if compound_id not in node_mapping[node_type]:
            mapped_idx = node_idx_counter[node_type]
            node_mapping[node_type][compound_id] = mapped_idx
            node_idx_counter[node_type] += 1

    # 각 노드 타입별로 노드 수 저장
    for node_type, mapping in node_mapping.items():
        data[node_type].num_nodes = len(mapping)

    # 2. 엣지(Edge) 처리
    print("  - 2/3: 엣지 인덱스 생성 중...")
    edge_data = defaultdict(lambda: [[], []]) # { (src, type, dst): ([src_list], [dst_list]) }
    
    skipped_edges_count = 0 

    for edge in tqdm(hetionet_data['edges']):
        
        # 'source_id'는 ["Gene", 5345] 리스트입니다.
        src_id_list = edge['source_id']
        src_id_list_str = [str(item) for item in src_id_list] 
        src_id = '::'.join(src_id_list_str) # 예: "Gene::5345"
        src_type = src_id.split('::')[0]    # 예: "Gene"

        dst_id_list = edge['target_id']
        dst_id_list_str = [str(item) for item in dst_id_list]
        dst_id = '::'.join(dst_id_list_str) # 예: "Compound::DB001"
        dst_type = dst_id.split('::')[0]    # 예: "Compound"
        
        # --- (KeyError 방지 검증 로직) ---
        if src_type not in node_mapping or dst_type not in node_mapping:
             skipped_edges_count += 1
             continue
        
        # --- 수정 3: 복합 ID (src_id, dst_id)로 매핑 조회 ---
        if (src_id not in node_mapping[src_type]) or \
           (dst_id not in node_mapping[dst_type]):
            skipped_edges_count += 1
            continue
        # --- (수정 3 끝) ---

        # (이제 모든 ID가 검증되었으므로 안전하게 매핑된 인덱스를 가져옵니다)
        src_mapped_idx = node_mapping[src_type][src_id]
        dst_mapped_idx = node_mapping[dst_type][dst_id]

        edge_type = edge['kind'].replace(' ', '_') # 공백을 '_'로

        # (양방향 엣지 처리)
        # 1. 정방향 엣지 추가
        edge_data[(src_type, edge_type, dst_type)][0].append(src_mapped_idx)
        edge_data[(src_type, edge_type, dst_type)][1].append(dst_mapped_idx)

        # 2. 역방향 엣지 추가 ('rev_' 접두사 사용)
        rev_edge_type = f"rev_{edge_type}"
        edge_data[(dst_type, rev_edge_type, src_type)][0].append(dst_mapped_idx)
        edge_data[(dst_type, rev_edge_type, src_type)][1].append(src_mapped_idx)

    if skipped_edges_count > 0:
        print(f"   - 경고: 노드 리스트에 없는 노드를 참조하는 엣지 {skipped_edges_count}개를 스킵했습니다.")

    for edge_tuple, (src_list, dst_list) in edge_data.items():
        data[edge_tuple].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # 3. ID-이름 매핑 저장 (결과 분석용)
    print("  - 3/3: ID-이름 매핑 저장 중...")
    data.node_names = {} # {'Compound': {0: 'Aspirin', 1: ...}}
    
    for node_type in node_mapping.keys():
        data.node_names[node_type] = {}

    for node in hetionet_data['nodes']:
        node_type = node['kind']
        node_identifier = node['identifier']
        node_name = node['name']
        
        # --- 수정 4: 복합 ID로 매핑 조회 ---
        compound_id = f"{node_type}::{str(node_identifier)}"
        
        if compound_id in node_mapping[node_type]:
            mapped_idx = node_mapping[node_type][compound_id]
            data.node_names[node_type][mapped_idx] = node_name
        # --- (수정 4 끝) ---

    return data

if __name__ == "__main__":
    # 1. `data/hetionet-v1.0.json.bz2` 파일 로드
    hetionet_raw_data = load_hetionet_json()

    # 2. `HeteroData` 객체로 변환
    hetionet_hetero_data = preprocess_hetionet(hetionet_raw_data)

    print("\n[전처리 요약]")
    print(hetionet_hetero_data)

    # 3. 변환된 데이터를 .pt 파일로 저장
    save_path = osp.join('data', 'hetionet_data.pt')
    torch.save(hetionet_hetero_data, save_path)

    print(f"\n전처리 완료. 데이터가 {save_path} 에 저장되었습니다.")
    print("이 파일을 `main_hetionet.py`에서 불러와 학습을 진행할 수 있습니다.")
