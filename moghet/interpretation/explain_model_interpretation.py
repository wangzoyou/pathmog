import os
import os.path as osp
import sys
import torch
# --- 修正Python路径问题 ---
# 将项目根目录添加到sys.path，以支持绝对路径导入
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 路径修正结束 ---
import json
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Batch
import numpy as np
import gc

# ====== collate_fn 迁移自 train_hierarchical.py ======
def hierarchical_collate_fn(data_packages):
    if not data_packages:
        return None
    list_of_graph_lists = [pkg['intra_pathway_graphs'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
    if not list_of_graph_lists:
        return None
    clinical_features_list = [pkg['clinical_features'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
    time_list = [pkg['time'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
    event_list = [pkg['event'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
    if not clinical_features_list:
        return None
    all_pathway_graphs = [graph for sublist in list_of_graph_lists for graph in sublist]
    pathway_to_patient_map = []
    patient_idx_counter = 0
    for sublist in list_of_graph_lists:
        pathway_to_patient_map.extend([patient_idx_counter] * len(sublist))
        patient_idx_counter += 1
    graphs_batch = Batch.from_data_list(all_pathway_graphs)
    graphs_batch.pathway_to_patient_batch_map = torch.tensor(pathway_to_patient_map, dtype=torch.long)
    pathway_indices = [g.pathway_idx.item() for g in all_pathway_graphs]
    graphs_batch.pathway_idx = torch.tensor(pathway_indices, dtype=torch.long)
    # 手动补充 batch['gene']
    gene_batch = []
    for i, g in enumerate(all_pathway_graphs):
        n = g['gene'].num_nodes
        gene_batch.extend([i] * n)
    if not hasattr(graphs_batch, 'batch') or not isinstance(graphs_batch.batch, dict):
        graphs_batch.batch = {}
    graphs_batch.batch['gene'] = torch.tensor(gene_batch, dtype=torch.long)
    clinical_features = torch.cat(clinical_features_list, dim=0)
    time = torch.stack(time_list)
    event = torch.stack(event_list)
    return {
        'graphs_batch': graphs_batch,
        'clinical_features': clinical_features,
        'time': time,
        'event': event
    }


from models.hierarchical_model import HierarchicalGNNModel
from interpretation.explain_module import HierarchicalGNNExplainer

# ====== 配置区 ======
# 将所有路径都构建为绝对路径，以避免工作目录问题
FOLD_DIR = osp.join(PROJECT_ROOT, 'results', 'hierarchical_model_brca_fast', 'fold_1')  # 切换到BRCA数据集
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 自动推断路径 ---
print(f"正在使用 Fold 目录: {FOLD_DIR}")
# 修正：从路径中提取数据集名称，并强制转为大写，以匹配数据目录
DATASET = FOLD_DIR.split('/')[-2].replace('hierarchical_model_', '').replace('_fast', '').split('_')[0].upper()
print(f"自动检测到数据集为: {DATASET}")
# 修正：直接指向正确的数据子目录，处理不一致的目录结构
DATA_DIR = osp.join(PROJECT_ROOT, 'data', 'processed', DATASET, 'hierarchical_patient_data')
OUTPUT_DIR = osp.join(PROJECT_ROOT, 'results', f'explanations_{DATASET.lower()}')
SURVIVAL_CSV = osp.join(PROJECT_ROOT, 'data', 'processed', DATASET, 'patient_survival.csv')
# --- 路径推断结束 ---


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 1. 加载模型结构与权重 ======
print("加载模型...")
with open(osp.join(FOLD_DIR, 'model_params.json'), 'r') as f:
    model_params = json.load(f)

# 加载生存数据表
survival_df = pd.read_csv(SURVIVAL_CSV, index_col=0)

# 修正 edge_types 为 tuple
if 'metadata' in model_params:
    node_types, edge_types = model_params['metadata']
    edge_types = tuple(tuple(et) for et in edge_types)
    node_types = tuple(node_types)
    model_params['metadata'] = (node_types, edge_types)

model = HierarchicalGNNModel(
    gnn_hidden_channels=model_params['gnn_hidden_channels'],
    pathway_out_channels=model_params['pathway_out_channels'],
    metadata=model_params['metadata'],
    num_pathways=model_params['num_pathways'],
    pathway_embedding_dim=model_params['pathway_embedding_dim'],
    intra_attention_hidden_channels=model_params['intra_attention_hidden_channels'],
    inter_attention_hidden_channels=model_params['inter_attention_hidden_channels'],
    clinical_in_features=model_params['clinical_in_features'],
    clinical_hidden_channels=model_params['clinical_hidden_channels'],
    final_hidden_channels=model_params['final_hidden_channels'],
).to(DEVICE)

model.load_state_dict(torch.load(osp.join(FOLD_DIR, 'best_model.pt'), map_location=DEVICE))
model.eval()

explainer = HierarchicalGNNExplainer(model, dataset_name=DATASET, project_root=PROJECT_ROOT)

# ====== 2. 加载病人数据 ======
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pt')]
files = all_files

print(f"共找到 {len(files)} 个病人数据包，将逐一进行可解释性分析...")

# ====== 3. 批量推理与可解释性分析 ======
for idx, fname in tqdm(enumerate(files), total=len(files), desc="生成解释文件"):
    patient_id = fname.replace('.pt', '')
    
    try:
        data_package = torch.load(osp.join(DATA_DIR, fname), map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"无法加载病人 {patient_id} 的数据包，跳过。错误: {e}")
        continue

    # 补充time和event字段
    if patient_id in survival_df.index:
        survival_info = survival_df.loc[patient_id]
        data_package['time'] = torch.tensor(survival_info['OS.time'] / 365.0, dtype=torch.float)
        data_package['event'] = torch.tensor(survival_info['OS'], dtype=torch.float)
    else:
        data_package['time'] = torch.tensor(0.0, dtype=torch.float)
        data_package['event'] = torch.tensor(0.0, dtype=torch.float)
    batch = hierarchical_collate_fn([data_package])
    graphs_batch = batch['graphs_batch'].to(DEVICE)
    clinical_features = batch['clinical_features'].to(DEVICE)

    # 可解释性前向传播
    explain_out = explainer.explain_full_forward(graphs_batch, clinical_features)

    # 提取需要保存的数据
    risk_score = explain_out['risk_score'].squeeze().detach().cpu().item()
    
    pathway_indices = graphs_batch.pathway_idx.detach().cpu().numpy()
    pathway_attn = explain_out['pathway_attention_weights'].squeeze().detach().cpu().numpy()
    
    # 确保只取有效的部分
    if pathway_attn.ndim > 0:
        pathway_attn = pathway_attn[:len(pathway_indices)]

    gene_attn = explain_out['gene_attention_weights'].squeeze().detach().cpu().numpy()
    
    # 保存到一个字典
    save_data = {
        'risk_score': risk_score,
        'pathway_indices': pathway_indices.tolist(),
        'pathway_attention_weights': pathway_attn.tolist(),
        'gene_attention_weights': gene_attn.tolist(),
    }

    # 保存为.pt文件
    output_filepath = osp.join(OUTPUT_DIR, f'explanation_{patient_id}.pt')
    torch.save(save_data, output_filepath)

    # --- 手动清理内存 ---
    del data_package, batch, graphs_batch, clinical_features, explain_out, save_data, risk_score, pathway_indices, pathway_attn, gene_attn
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    # --- 内存清理结束 ---

print(f"分析完成，结果已保存到 {OUTPUT_DIR}") 