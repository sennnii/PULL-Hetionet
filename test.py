import torch
data = torch.load('data/hetionet_data.pt')
print("--- data.pt 파일에 포함된 엣지 타입 ---")
for edge_type in data.edge_types:
    print(edge_type)
