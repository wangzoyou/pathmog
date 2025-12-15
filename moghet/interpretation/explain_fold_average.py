import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
moghet_dir = os.path.join(project_root, "moghet")
if moghet_dir not in sys.path:
    sys.path.insert(0, moghet_dir)

import json
import numpy as np
import torch
from torch_geometric.data import Batch
from models.hierarchical_model import HierarchicalGNNModel
from interpretation.explain_module import HierarchicalGNNExplainer
import pandas as pd

# 只保留病人小图批量解释

# ====== collate_fn 迁移自 train_hierarchical.py ======
def hierarchical_collate_fn(data_packages):
    if not data_packages:
        return None
    list_of_graph_lists = [pkg['intra_pathway_graphs'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
    if not list_of_graph_lists:
        return None
    clinical_features_list = [pkg['clinical_features'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
    
    # 修复：处理缺少time和event字段的情况
    time_list = []
    event_list = []
    for pkg in data_packages:
        if pkg.get('intra_pathway_graphs'):
            if 'time' in pkg:
                time_list.append(pkg['time'])
            else:
                time_list.append(torch.tensor(0.0, dtype=torch.float))
            if 'event' in pkg:
                event_list.append(pkg['event'])
            else:
                event_list.append(torch.tensor(0.0, dtype=torch.float))
    
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


def get_fold_model_paths(base_dir):
    model_paths = []
    for i in range(1, 100):
        fold_dir = os.path.join(base_dir, f"fold_{i}")
        model_path = os.path.join(fold_dir, "best_model.pt")
        if os.path.exists(model_path):
            model_paths.append(model_path)
        else:
            break
    return model_paths


def analyze_one_fold(model_path, patient_graphs_dir):
    print(f"  加载模型: {model_path}")
    fold_dir = os.path.dirname(model_path)
    params_path = os.path.join(fold_dir, "model_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"找不到模型参数文件: {params_path}")
    with open(params_path, 'r') as f:
        fold_params = json.load(f)
    
    def list2tuple(obj):
        if isinstance(obj, list):
            return tuple(list2tuple(e) for e in obj)
        return obj
    fold_params['metadata'] = list2tuple(fold_params['metadata'])
    
    # 使用GPU如果可用，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ 使用设备: {device}")
    
    model = HierarchicalGNNModel(
        metadata=fold_params["metadata"],
        clinical_in_features=fold_params["clinical_in_features"],
        num_pathways=fold_params["num_pathways"],
        pathway_embedding_dim=fold_params["pathway_embedding_dim"],
        gnn_hidden_channels=fold_params["gnn_hidden_channels"],
        pathway_out_channels=fold_params["pathway_out_channels"],
        intra_attention_hidden_channels=fold_params["intra_attention_hidden_channels"],
        inter_attention_hidden_channels=fold_params["inter_attention_hidden_channels"],
        clinical_hidden_channels=fold_params["clinical_hidden_channels"],
        final_hidden_channels=fold_params["final_hidden_channels"]
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("  ✓ 模型加载成功")
    explainer = HierarchicalGNNExplainer(model)
    if patient_graphs_dir is None:
        raise ValueError('patient_graphs_dir不能为空')
    print(f"  遍历病人小图目录: {patient_graphs_dir}")
    import glob
    patient_files = sorted(glob.glob(os.path.join(patient_graphs_dir, '*.pt')))
    all_pathway_scores = []
    all_gene_scores = []
    
    # 处理全体病人
    print(f"  处理模式：处理全体 {len(patient_files)} 个病人")
    
    # 使用tqdm显示进度
    from tqdm import tqdm
    
    for pf in tqdm(patient_files, desc="处理患者"):
        try:
            patient_package = torch.load(pf, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"    尝试pickle.load加载: {os.path.basename(pf)}")
            import pickle
            with open(pf, 'rb') as f:
                patient_package = pickle.load(f)
        if not isinstance(patient_package, dict) or 'intra_pathway_graphs' not in patient_package:
            continue
        patient_id = patient_package['patient_id']
        intra_pathway_graphs = patient_package['intra_pathway_graphs']
        clinical_features = patient_package['clinical_features']
        
        # 修复：使用 hierarchical_collate_fn 组装batch，一次性处理所有通路
        try:
            batch = hierarchical_collate_fn([patient_package])
            if batch is None:
                continue
                
            graphs_batch = batch['graphs_batch']
            clinical_features_batch = batch['clinical_features']
            
            # 将数据移到设备上
            graphs_batch = graphs_batch.to(device)
            clinical_features_batch = clinical_features_batch.to(device)
            
            # 使用 explain_full_forward 一次性处理所有通路
            with torch.no_grad():  # 禁用梯度计算以加速
                explain_out = explainer.explain_full_forward(graphs_batch, clinical_features_batch)
            
            # 提取通路层注意力
            pathway_attn = explain_out['pathway_attention_weights'].squeeze(-1).cpu().numpy()  # [1, N_pathways]
            pathway_indices = graphs_batch.pathway_idx.cpu().numpy()
            
            # 提取基因层注意力
            gene_attn = explain_out['gene_attention_weights'].squeeze(-1).cpu().numpy()  # [1, N_genes]
            gene_batch_map = graphs_batch.batch['gene'].cpu().numpy()
            
            # 保存通路层分数
            patient_pathway_scores = pathway_attn[0, :len(pathway_indices)]
            
            # 按通路分组保存基因层分数
            patient_gene_scores = []
            
            for pidx in range(len(pathway_indices)):
                gene_indices = (gene_batch_map == pidx).nonzero()[0]
                
                # 过滤掉超出范围的索引
                gene_indices = gene_indices[gene_indices < gene_attn.shape[1]]
                
                if len(gene_indices) > 0:
                    # 使用局部基因索引
                    local_gene_indices = np.arange(len(gene_indices))
                    gene_attn_vals = gene_attn[pidx, local_gene_indices]
                    patient_gene_scores.append(gene_attn_vals)
                else:
                    patient_gene_scores.append(np.array([]))
            
            all_pathway_scores.append(patient_pathway_scores)
            all_gene_scores.append(patient_gene_scores)
            
            # 保存单个患者的解释结果
            patient_result = {
                'patient_id': patient_id,
                'pathway_importance': patient_pathway_scores.tolist(),
                'gene_importance': [scores.tolist() for scores in patient_gene_scores],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 保存到fold目录
            fold_dir = os.path.dirname(model_path)
            patient_result_file = os.path.join(fold_dir, f"patient_{patient_id}_explanation.pt")
            torch.save(patient_result, patient_result_file)
            
            print(f"    ✓ 病人 {patient_id} 完成")
            print(f"    ✓ 患者结果已保存到: {patient_result_file}")
            
        except Exception as e:
            print(f"    ✗ 病人 {patient_id} 解释失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    if not all_pathway_scores:
        raise RuntimeError('没有成功分析的病人小图')
    
    # 修复：汇总所有病人的通路分数
    avg_pathway = np.mean(np.stack(all_pathway_scores), axis=0)
    
    # 修复：汇总所有病人的基因分数（每个通路分别计算）
    # 找到最大通路数和最大基因数
    max_pathways = max(len(patient_scores) for patient_scores in all_gene_scores)
    max_genes_per_pathway = 0
    for patient_scores in all_gene_scores:
        for pathway_scores in patient_scores:
            if len(pathway_scores) > max_genes_per_pathway:
                max_genes_per_pathway = len(pathway_scores)
    
    # 初始化汇总数组
    avg_gene_by_pathway = np.zeros((max_pathways, max_genes_per_pathway))
    pathway_gene_counts = np.zeros((max_pathways, max_genes_per_pathway))
    
    # 累加所有病人的基因分数
    for patient_idx, patient_scores in enumerate(all_gene_scores):
        for pidx, pathway_scores in enumerate(patient_scores):
            if len(pathway_scores) > 0:
                avg_gene_by_pathway[pidx, :len(pathway_scores)] += pathway_scores
                pathway_gene_counts[pidx, :len(pathway_scores)] += 1
    
    # 计算平均值（避免除零）
    pathway_gene_counts[pathway_gene_counts == 0] = 1
    avg_gene_by_pathway /= pathway_gene_counts
    
    print(f"  ✓ 病人平均通路重要性: {avg_pathway.shape}, 平均基因重要性: {avg_gene_by_pathway.shape}")
    return avg_pathway, avg_gene_by_pathway


def load_id_mappings(data_dir):
    """
    加载ID映射文件，将索引映射回基因名和通路名
    """
    mappings_path = os.path.join(data_dir, 'id_mappings.json')
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"找不到ID映射文件: {mappings_path}")
    
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    # 创建反向映射
    gene_idx_to_id = {v: k for k, v in mappings['gene_id_to_idx'].items()}
    pathway_idx_to_id = {v: k for k, v in mappings['pathway_id_to_idx'].items()}
    
    # 加载通路信息以获取通路名称
    pathway_info_path = os.path.join(data_dir, 'kegg_pathway_info.csv')
    pathway_names = {}
    if os.path.exists(pathway_info_path):
        pathway_df = pd.read_csv(pathway_info_path)
        # 跳过标题行和无效行
        pathway_df = pathway_df[(pathway_df['id'] != 'ID') & (pathway_df['id'] != 'id')]
        for _, row in pathway_df.iterrows():
            pathway_id = row['id']
            pathway_name = row['name'].split(' - ')[0]  # 提取通路名称部分
            pathway_names[pathway_id] = pathway_name
    
    # 加载通路结构信息 - 这是关键！
    # 需要重新运行build_hetero_graph.py来生成通路结构，或者从保存的文件中加载
    pathway_structures = {}
    try:
        # 尝试从保存的JSON文件中加载通路结构
        structures_path = os.path.join(data_dir, 'pathway_structures.json')
        if os.path.exists(structures_path):
            with open(structures_path, 'r') as f:
                pathway_structures = json.load(f)
            print(f"✓ 从文件加载了 {len(pathway_structures)} 个通路结构")
        else:
            print("⚠ 未找到通路结构文件，需要重新生成")
            # 这里可以调用build_hetero_graph.py的逻辑来重新生成
    except Exception as e:
        print(f"⚠ 加载通路结构失败: {e}")
    
    return gene_idx_to_id, pathway_idx_to_id, pathway_names, pathway_structures


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_graphs_dir', type=str, required=True, help='病人小图目录')
    parser.add_argument('--data_dir', type=str, default=None, help='数据目录，用于加载ID映射')
    args = parser.parse_args()
    
    # 设置数据目录
    if args.data_dir is None:
        args.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    
    print("=== 批量分析所有fold模型，输出平均通路/基因重要性排名 ===")
    
    # 加载ID映射
    try:
        gene_idx_to_id, pathway_idx_to_id, pathway_names, pathway_structures = load_id_mappings(args.data_dir)
        print(f"✓ 加载了 {len(gene_idx_to_id)} 个基因映射, {len(pathway_idx_to_id)} 个通路映射")
        print(f"✓ 加载了 {len(pathway_names)} 个通路名称, {len(pathway_structures)} 个通路结构")
    except Exception as e:
        print(f"⚠ 加载ID映射失败: {e}")
        gene_idx_to_id, pathway_idx_to_id, pathway_names, pathway_structures = {}, {}, {}, {}
    
    # 尝试多个可能的模型目录位置
    possible_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "kirc_model")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "hierarchical_model_single_run")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "hierarchical_model_kirc")),
    ]
    
    base_dir = None
    for pdir in possible_dirs:
        if os.path.exists(pdir):
            base_dir = pdir
            break
    
    if base_dir is None:
        print("⚠ 未找到任何模型目录，尝试使用数据集名称推断")
        # 从patient_graphs_dir推断数据集名称
        dataset_name = os.path.basename(os.path.dirname(patient_graphs_dir))
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", f"{dataset_name.lower()}_model"))
        
    print(f"查找fold模型目录: {base_dir}")
    model_paths = get_fold_model_paths(base_dir)
    print(f"共检测到 {len(model_paths)} 个fold模型")
    for i, path in enumerate(model_paths):
        print(f"  fold {i+1}: {path}")
    if not model_paths:
        print("未找到fold模型，退出")
        return
    all_pathway_scores = []
    all_gene_scores = []
    for i, model_path in enumerate(model_paths):
        print(f"\n分析fold {i+1}: {os.path.basename(os.path.dirname(model_path))}")
        try:
            avg_pathway, avg_gene = analyze_one_fold(model_path, args.patient_graphs_dir)
            all_pathway_scores.append(avg_pathway)
            all_gene_scores.append(avg_gene)
            print(f"  ✓ fold {i+1} 分析完成")
        except Exception as e:
            print(f"  ✗ fold {i+1} 分析失败: {e}")
            continue
    if not all_pathway_scores:
        print("没有成功分析的fold，退出")
        return
    print(f"\n计算 {len(all_pathway_scores)} 个fold的平均值...")
    mean_pathway = np.mean(np.stack(all_pathway_scores), axis=0)
    mean_gene = np.mean(np.stack(all_gene_scores), axis=0)
    topk = 10
    print(f"\n=== 平均通路重要性TOP{topk} ===")
    top_pathway_idx = np.argsort(-mean_pathway)[:topk]
    for i, idx in enumerate(top_pathway_idx):
        pathway_id = pathway_idx_to_id.get(idx, f"pathway_{idx}")
        pathway_name = pathway_names.get(pathway_id, pathway_id)
        print(f"  {i+1:2d}. 通路 {idx:3d} ({pathway_id}): {pathway_name} - {mean_pathway[idx]:.4f}")
    
    print(f"\n=== 平均基因重要性TOP{topk} ===")
    # 修复：基因重要性现在是按通路分组的，需要展平后排序
    mean_gene_flat = mean_gene.flatten()
    top_gene_idx = np.argsort(-mean_gene_flat)[:topk]
    for i, idx in enumerate(top_gene_idx):
        pathway_idx = idx // mean_gene.shape[1]
        gene_idx = idx % mean_gene.shape[1]
        
        # 获取通路信息
        pathway_id = pathway_idx_to_id.get(pathway_idx, f"pathway_{pathway_idx}")
        pathway_name = pathway_names.get(pathway_id, pathway_id)
        
        # 获取基因信息
        gene_name = "unknown"
        if pathway_id in pathway_structures and gene_idx < len(pathway_structures[pathway_id]['genes']):
            gene_name = pathway_structures[pathway_id]['genes'][gene_idx]
        
        print(f"  {i+1:2d}. 通路{pathway_idx}({pathway_name}) 基因{gene_idx}({gene_name}): {mean_gene_flat[idx]:.4f}")
    
    # 生成总览统计
    print(f"\n=== 全体病人分析总览 ===")
    print(f"分析的患者数量: {len(all_pathway_scores)}")
    print(f"平均通路重要性范围: {mean_pathway.min():.4f} - {mean_pathway.max():.4f}")
    print(f"平均基因重要性范围: {mean_gene_flat.min():.4f} - {mean_gene_flat.max():.4f}")
    print(f"非零基因重要性数量: {(mean_gene_flat > 0).sum()}")
    print(f"非零通路重要性数量: {(mean_pathway > 0).sum()}")
    
    # 统计每个通路的重要性分布
    pathway_importance_stats = {
        'high': int((mean_pathway > mean_pathway.mean() + mean_pathway.std()).sum()),
        'medium': int(((mean_pathway <= mean_pathway.mean() + mean_pathway.std()) & 
                  (mean_pathway > mean_pathway.mean() - mean_pathway.std())).sum()),
        'low': int((mean_pathway <= mean_pathway.mean() - mean_pathway.std()).sum())
    }
    print(f"通路重要性分布:")
    print(f"  高重要性 (>均值+标准差): {pathway_importance_stats['high']} 个")
    print(f"  中等重要性: {pathway_importance_stats['medium']} 个")
    print(f"  低重要性 (<均值-标准差): {pathway_importance_stats['low']} 个")
    
    # 保存详细结果
    result = {
        "analysis_folds": len(all_pathway_scores),
        "mean_pathway_importance": mean_pathway.tolist(),
        "mean_gene_importance": mean_gene.tolist(),
        "top_pathways": top_pathway_idx.tolist(),
        "top_genes": top_gene_idx.tolist(),
        "pathway_mappings": {
            str(idx): {
                "id": pathway_idx_to_id.get(idx, f"pathway_{idx}"),
                "name": pathway_names.get(pathway_idx_to_id.get(idx, f"pathway_{idx}"), f"pathway_{idx}")
            } for idx in range(len(mean_pathway))
        },
        "gene_mappings": {
            str(pathway_idx): {
                str(gene_idx): pathway_structures.get(pathway_idx_to_id.get(pathway_idx, f"pathway_{pathway_idx}"), {}).get('genes', [])[gene_idx] if pathway_idx_to_id.get(pathway_idx, f"pathway_{pathway_idx}") in pathway_structures and gene_idx < len(pathway_structures[pathway_idx_to_id.get(pathway_idx, f"pathway_{pathway_idx}")].get('genes', [])) else f"gene_{gene_idx}"
                for gene_idx in range(mean_gene.shape[1])
            } for pathway_idx in range(mean_gene.shape[0])
        },
        "statistics": {
            "total_patients": len(all_pathway_scores),
            "pathway_importance_range": [float(mean_pathway.min()), float(mean_pathway.max())],
            "gene_importance_range": [float(mean_gene_flat.min()), float(mean_gene_flat.max())],
            "non_zero_genes": int((mean_gene_flat > 0).sum()),
            "non_zero_pathways": int((mean_pathway > 0).sum()),
            "pathway_importance_distribution": pathway_importance_stats
        }
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "results/fold_average_explainability.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(result, output_path)
    print(f"\n✓ 已保存平均可解释性结果到: {output_path}")
    
    # 生成可视化
    generate_visualizations(result, args.data_dir, output_path)

def generate_visualizations(result, data_dir, output_path):
    """
    生成可解释性结果的可视化图表
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import rcParams
        import pandas as pd
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建输出目录
        viz_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 通路重要性热力图
        pathway_importance = np.array(result['mean_pathway_importance'])
        pathway_names = []
        for i in range(len(pathway_importance)):
            pathway_id = result['pathway_mappings'][str(i)]['id']
            pathway_name = result['pathway_mappings'][str(i)]['name']
            pathway_names.append(f"{pathway_id}\n{pathway_name}")
        
        # 选择前20个最重要的通路
        top_20_idx = np.argsort(-pathway_importance)[:20]
        top_20_importance = pathway_importance[top_20_idx]
        top_20_names = [pathway_names[i] for i in top_20_idx]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_20_names)), top_20_importance)
        plt.yticks(range(len(top_20_names)), top_20_names)
        plt.xlabel('平均注意力权重')
        plt.title('Top 20 通路重要性排名')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'top_pathways_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 基因重要性分布
        gene_importance = np.array(result['mean_gene_importance'])
        gene_importance_flat = gene_importance.flatten()
        
        plt.figure(figsize=(10, 6))
        plt.hist(gene_importance_flat[gene_importance_flat > 0], bins=50, alpha=0.7, color='skyblue')
        plt.xlabel('基因注意力权重')
        plt.ylabel('频次')
        plt.title('基因重要性分布')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'gene_importance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 通路-基因重要性矩阵热力图
        # 选择前10个通路和前20个基因
        top_10_pathways = np.argsort(-pathway_importance)[:10]
        gene_importance_by_pathway = gene_importance[top_10_pathways]
        
        # 为每个通路找到最重要的基因
        max_genes_per_pathway = 20
        pathway_gene_matrix = np.zeros((len(top_10_pathways), max_genes_per_pathway))
        pathway_gene_names = []
        
        for i, pathway_idx in enumerate(top_10_pathways):
            pathway_id = result['pathway_mappings'][str(pathway_idx)]['id']
            pathway_name = result['pathway_mappings'][str(pathway_idx)]['name']
            pathway_gene_names.append(f"{pathway_id}\n{pathway_name}")
            
            # 获取该通路的所有基因重要性
            pathway_genes = gene_importance[pathway_idx]
            non_zero_genes = pathway_genes[pathway_genes > 0]
            
            if len(non_zero_genes) > 0:
                # 选择最重要的基因
                top_gene_indices = np.argsort(-non_zero_genes)[:max_genes_per_pathway]
                pathway_gene_matrix[i, :len(top_gene_indices)] = non_zero_genes[top_gene_indices]
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pathway_gene_matrix, 
                   xticklabels=[f"Gene_{i}" for i in range(max_genes_per_pathway)],
                   yticklabels=pathway_gene_names,
                   cmap='YlOrRd', 
                   cbar_kws={'label': '基因注意力权重'})
        plt.title('Top 10 通路中基因重要性热力图')
        plt.xlabel('基因排名')
        plt.ylabel('通路')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'pathway_gene_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 生成详细报告
        generate_detailed_report(result, viz_dir)
        
        print(f"✓ 可视化结果已保存到: {viz_dir}")
        
    except ImportError as e:
        print(f"⚠ 缺少可视化依赖包: {e}")
        print("请安装: pip install matplotlib seaborn")
    except Exception as e:
        print(f"⚠ 生成可视化失败: {e}")

def generate_detailed_report(result, viz_dir):
    """
    生成详细的分析报告
    """
    try:
        # 1. 通路重要性报告
        pathway_importance = np.array(result['mean_pathway_importance'])
        pathway_report = []
        
        for i in range(len(pathway_importance)):
            pathway_id = result['pathway_mappings'][str(i)]['id']
            pathway_name = result['pathway_mappings'][str(i)]['name']
            importance = pathway_importance[i]
            pathway_report.append({
                'rank': i + 1,
                'pathway_id': pathway_id,
                'pathway_name': pathway_name,
                'importance': importance
            })
        
        # 按重要性排序
        pathway_report.sort(key=lambda x: x['importance'], reverse=True)
        
        # 保存为CSV
        pathway_df = pd.DataFrame(pathway_report)
        pathway_df.to_csv(os.path.join(viz_dir, 'pathway_importance_report.csv'), index=False)
        
        # 2. 基因重要性报告
        gene_importance = np.array(result['mean_gene_importance'])
        gene_report = []
        
        for pathway_idx in range(gene_importance.shape[0]):
            pathway_id = result['pathway_mappings'][str(pathway_idx)]['id']
            pathway_name = result['pathway_mappings'][str(pathway_idx)]['name']
            
            pathway_genes = gene_importance[pathway_idx]
            non_zero_indices = np.where(pathway_genes > 0)[0]
            
            for gene_idx in non_zero_indices:
                gene_name = result['gene_mappings'][str(pathway_idx)][str(gene_idx)]
                importance = pathway_genes[gene_idx]
                gene_report.append({
                    'pathway_id': pathway_id,
                    'pathway_name': pathway_name,
                    'gene_name': gene_name,
                    'gene_idx': gene_idx,
                    'importance': importance
                })
        
        # 按重要性排序
        gene_report.sort(key=lambda x: x['importance'], reverse=True)
        
        # 保存为CSV
        gene_df = pd.DataFrame(gene_report)
        gene_df.to_csv(os.path.join(viz_dir, 'gene_importance_report.csv'), index=False)
        
        # 3. 生成文本报告
        with open(os.path.join(viz_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=== MOGHET 模型可解释性分析报告 ===\n\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析的fold数量: {result['analysis_folds']}\n")
            f.write(f"总通路数量: {len(pathway_importance)}\n")
            f.write(f"总基因数量: {len(gene_report)}\n\n")
            
            f.write("=== Top 10 通路重要性 ===\n")
            for i, pathway in enumerate(pathway_report[:10]):
                f.write(f"{i+1:2d}. {pathway['pathway_name']} ({pathway['pathway_id']}): {pathway['importance']:.4f}\n")
            
            f.write("\n=== Top 10 基因重要性 ===\n")
            for i, gene in enumerate(gene_report[:10]):
                f.write(f"{i+1:2d}. {gene['gene_name']} in {gene['pathway_name']} ({gene['pathway_id']}): {gene['importance']:.4f}\n")
            
            f.write(f"\n=== 统计信息 ===\n")
            f.write(f"通路重要性范围: {pathway_importance.min():.4f} - {pathway_importance.max():.4f}\n")
            f.write(f"基因重要性范围: {gene_df['importance'].min():.4f} - {gene_df['importance'].max():.4f}\n")
            f.write(f"平均通路重要性: {pathway_importance.mean():.4f}\n")
            f.write(f"平均基因重要性: {gene_df['importance'].mean():.4f}\n")
        
        print(f"✓ 详细报告已保存到: {viz_dir}")
        
    except Exception as e:
        print(f"⚠ 生成详细报告失败: {e}")

if __name__ == "__main__":
    main() 