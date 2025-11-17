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
        print("다운로드 링크: https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2")
        exit()

    with bz2.open(filepath, 'rt') as f:
        data = json.load(f)
    
    # 데이터 구조 검증
    print(f"  - 메타노드 종류: {len(data.get('metanode_kinds', []))}")
    print(f"  - 전체 노드 수: {len(data.get('nodes', []))}")
    print(f"  - 전체 엣지 수: {len(data.get('edges', []))}")
    
    return data

def validate_data_structure(hetionet_data):
    """ 데이터 구조를 검증하고 통계를 출력합니다. """
    print("\n[데이터 구조 검증]")
    
    # 노드 타입별 카운트
    node_type_counts = defaultdict(int)
    for node in hetionet_data['nodes']:
        node_type_counts[node['kind']] += 1
    
    print("노드 타입별 개수:")
    for node_type, count in sorted(node_type_counts.items()):
        print(f"  - {node_type}: {count}")
    
    # 엣지 타입별 카운트
    edge_type_counts = defaultdict(int)
    for edge in hetionet_data['edges']:
        edge_type_counts[edge['kind']] += 1
    
    print("\n엣지 타입별 개수:")
    for edge_type, count in sorted(edge_type_counts.items()):
        print(f"  - {edge_type}: {count}")
    
    # Compound-Disease 관계 확인
    treats_count = sum(1 for edge in hetionet_data['edges'] 
                      if edge['kind'] == 'treats')
    print(f"\n약물-질병 'treats' 관계: {treats_count}개")
    
    return node_type_counts, edge_type_counts

def preprocess_hetionet(hetionet_data):
    """ Hetionet JSON을 torch_geometric.data.HeteroData 객체로 변환합니다. """
    print("\n[HeteroData 객체 생성 중...]")

    data = HeteroData()

    # 1. 노드(Node) 처리
    print("  - 1/4: 노드 매핑 생성 중...")
    node_mapping = {}
    
    for node_type_name in hetionet_data['metanode_kinds']:
        node_mapping[node_type_name] = {}

    node_idx_counter = defaultdict(int)
    node_info = defaultdict(list)  # 디버깅용

    for node in tqdm(hetionet_data['nodes'], desc="노드 처리"):
        node_type = node['kind']
        node_identifier = node['identifier']
        
        # 복합 ID 생성
        compound_id = f"{node_type}::{str(node_identifier)}"
        
        if compound_id not in node_mapping[node_type]:
            mapped_idx = node_idx_counter[node_type]
            node_mapping[node_type][compound_id] = mapped_idx
            node_idx_counter[node_type] += 1
            
            # 디버깅 정보 저장
            node_info[node_type].append({
                'compound_id': compound_id,
                'mapped_idx': mapped_idx,
                'name': node.get('name', 'Unknown')
            })

    # 각 노드 타입별로 노드 수 저장
    for node_type, mapping in node_mapping.items():
        data[node_type].num_nodes = len(mapping)

    print(f"    총 {sum(len(m) for m in node_mapping.values())}개 노드 매핑 완료")

    # 2. 엣지(Edge) 처리
    print("  - 2/4: 엣지 인덱스 생성 중...")
    edge_data = defaultdict(lambda: [[], []])
    
    skipped_edges = []
    skipped_edges_count = 0

    for edge in tqdm(hetionet_data['edges'], desc="엣지 처리"):
        # Source 노드 처리
        src_id_list = edge['source_id']
        src_id = '::'.join([str(item) for item in src_id_list])
        src_type = src_id.split('::')[0]

        # Target 노드 처리
        dst_id_list = edge['target_id']
        dst_id = '::'.join([str(item) for item in dst_id_list])
        dst_type = dst_id.split('::')[0]
        
        # 노드 타입 검증
        if src_type not in node_mapping or dst_type not in node_mapping:
            skipped_edges.append({
                'reason': 'unknown_node_type',
                'src_type': src_type,
                'dst_type': dst_type,
                'edge_kind': edge['kind']
            })
            skipped_edges_count += 1
            continue
        
        # 노드 ID 검증
        if src_id not in node_mapping[src_type]:
            skipped_edges.append({
                'reason': 'missing_source_node',
                'src_id': src_id,
                'edge_kind': edge['kind']
            })
            skipped_edges_count += 1
            continue
            
        if dst_id not in node_mapping[dst_type]:
            skipped_edges.append({
                'reason': 'missing_target_node',
                'dst_id': dst_id,
                'edge_kind': edge['kind']
            })
            skipped_edges_count += 1
            continue

        # 매핑된 인덱스 가져오기
        src_mapped_idx = node_mapping[src_type][src_id]
        dst_mapped_idx = node_mapping[dst_type][dst_id]

        edge_type = edge['kind'].replace(' ', '_')

        # 정방향 엣지 추가
        edge_data[(src_type, edge_type, dst_type)][0].append(src_mapped_idx)
        edge_data[(src_type, edge_type, dst_type)][1].append(dst_mapped_idx)

        # 역방향 엣지 추가
        rev_edge_type = f"rev_{edge_type}"
        edge_data[(dst_type, rev_edge_type, src_type)][0].append(dst_mapped_idx)
        edge_data[(dst_type, rev_edge_type, src_type)][1].append(src_mapped_idx)

    if skipped_edges_count > 0:
        print(f"\n    경고: {skipped_edges_count}개 엣지 스킵됨")
        
        # 스킵 이유 분석
        skip_reasons = defaultdict(int)
        for skip_info in skipped_edges[:100]:  # 처음 100개만
            skip_reasons[skip_info['reason']] += 1
        
        print("    스킵 이유 통계:")
        for reason, count in skip_reasons.items():
            print(f"      - {reason}: {count}")

    # 엣지 인덱스를 텐서로 변환
    for edge_tuple, (src_list, dst_list) in edge_data.items():
        data[edge_tuple].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    print(f"    총 {len(edge_data)}개 엣지 타입 생성")

    # 3. ID-이름 매핑 저장
    print("  - 3/4: ID-이름 매핑 저장 중...")
    data.node_names = {}
    
    for node_type in node_mapping.keys():
        data.node_names[node_type] = {}

    for node in tqdm(hetionet_data['nodes'], desc="이름 매핑"):
        node_type = node['kind']
        node_identifier = node['identifier']
        node_name = node['name']
        
        compound_id = f"{node_type}::{str(node_identifier)}"
        
        if compound_id in node_mapping[node_type]:
            mapped_idx = node_mapping[node_type][compound_id]
            data.node_names[node_type][mapped_idx] = node_name

    # 4. 데이터 품질 검증
    print("  - 4/4: 데이터 품질 검증 중...")
    
    # Compound-Disease treats 엣지 확인
    if ('Compound', 'treats', 'Disease') in data.edge_types:
        treats_edges = data['Compound', 'treats', 'Disease'].edge_index
        print(f"    ✓ 'Compound-treats-Disease' 엣지: {treats_edges.shape[1]}개")
        
        # 역방향 엣지 확인
        if ('Disease', 'rev_treats', 'Compound') in data.edge_types:
            rev_treats_edges = data['Disease', 'rev_treats', 'Compound'].edge_index
            print(f"    ✓ 'Disease-rev_treats-Compound' 엣지: {rev_treats_edges.shape[1]}개")
    else:
        print("    ⚠ 경고: 'Compound-treats-Disease' 엣지가 없습니다!")

    return data, node_mapping, skipped_edges

if __name__ == "__main__":
    # 1. hetionet-v1.0.json.bz2 파일 로드
    hetionet_raw_data = load_hetionet_json()

    # 2. 데이터 구조 검증
    validate_data_structure(hetionet_raw_data)

    # 3. HeteroData 객체로 변환
    hetionet_hetero_data, node_mapping, skipped_edges = preprocess_hetionet(hetionet_raw_data)

    print("\n[전처리 요약]")
    print(hetionet_hetero_data)

    # 노드 타입별 정보
    print("\n노드 타입별 개수:")
    for node_type in hetionet_hetero_data.node_types:
        print(f"  - {node_type}: {hetionet_hetero_data[node_type].num_nodes}")

    # 엣지 타입별 정보
    print("\n엣지 타입별 개수:")
    for edge_type in hetionet_hetero_data.edge_types:
        edge_index = hetionet_hetero_data[edge_type].edge_index
        print(f"  - {edge_type}: {edge_index.shape[1]}")

    # 4. 변환된 데이터를 .pt 파일로 저장
    save_path = osp.join('data', 'hetionet_data.pt')
    torch.save(hetionet_hetero_data, save_path)
    
    # 5. 디버깅 정보 저장
    debug_path = osp.join('data', 'preprocessing_debug.json')
    with open(debug_path, 'w', encoding='utf-8') as f:
        json.dump({
            'skipped_edges_count': len(skipped_edges),
            'skipped_edges_sample': skipped_edges[:10],  # 처음 10개만
            'node_types': list(node_mapping.keys()),
            'node_counts': {k: len(v) for k, v in node_mapping.items()}
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 전처리 완료!")
    print(f"  - 데이터 저장: {save_path}")
    print(f"  - 디버깅 정보: {debug_path}")
    print("\n다음 단계: python main_hetionet.py --epochs 10")