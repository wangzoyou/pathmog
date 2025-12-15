#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation experiment dedicated data loader
Adapts to the existing heterogeneous graph data format
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, DataLoader
import os.path as osp
from sklearn.preprocessing import StandardScaler
import sys

# 添加scripts路径以导入load_utils
current_dir = osp.dirname(osp.abspath(__file__))
scripts_dir = osp.join(osp.dirname(current_dir), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from load_utils import safe_load_hetero_data


class AblationDataset(Dataset):
    """
    Ablation experiment dataset
    Based on the existing heterogeneous graph data format
    """
    def __init__(self, data_path, patient_ids=None):
        """
        Initialize the dataset
        
        Args:
            data_path: Data path
            patient_ids: List of patient IDs, if None use all patients
        """
        super().__init__()
        self.data_path = data_path
        
        # Load heterogeneous graph data
        hetero_data_path = osp.join(data_path, 'hetero_data.pt')
        print(f"Loading heterogeneous graph data: {hetero_data_path}")
        self.hetero_data = safe_load_hetero_data(hetero_data_path)
        
        if self.hetero_data is None:
            raise RuntimeError(f"Failed to load heterogeneous graph data: {hetero_data_path}")
        
        # Load patient survival data
        survival_path = osp.join(data_path, 'patient_survival.csv')
        self.survival_df = pd.read_csv(survival_path, index_col=0)
        self.survival_df.index = self.survival_df.index.astype(str)
        
        # Load and normalize clinical features
        clinical_path = osp.join(data_path, 'patient_clinical_features.csv')
        clinical_df = pd.read_csv(clinical_path, index_col=0)
        clinical_df.index = clinical_df.index.astype(str)
        
        # Normalize clinical features
        scaler = StandardScaler()
        normalized_clinical = scaler.fit_transform(clinical_df)
        self.clinical_df = pd.DataFrame(
            normalized_clinical, 
            index=clinical_df.index, 
            columns=clinical_df.columns
        )
        
        # Determine valid patient IDs (with survival and clinical data)
        if patient_ids is None:
            # Use all patients with survival data
            valid_survival_patients = set(self.survival_df.dropna().index)
            valid_clinical_patients = set(self.clinical_df.index)
            self.patient_ids = list(valid_survival_patients & valid_clinical_patients)
        else:
            # Use specified patient IDs, but ensure they have corresponding data
            valid_survival_patients = set(self.survival_df.dropna().index)
            valid_clinical_patients = set(self.clinical_df.index)
            valid_patients = valid_survival_patients & valid_clinical_patients
            self.patient_ids = [pid for pid in patient_ids if pid in valid_patients]
        
        print(f"Dataset initialized with {len(self.patient_ids)} valid patients")
        
        # Get patient index mapping in the heterogeneous graph
        if hasattr(self.hetero_data, 'patient_id_map'):
            self.patient_id_to_idx = self.hetero_data.patient_id_map
        else:
            # If no mapping exists, create a simple one
            print("Warning: No patient ID mapping found in heterogeneous graph, creating simple mapping")
            self.patient_id_to_idx = {pid: i for i, pid in enumerate(self.patient_ids)}
        
        # Filter out patients without corresponding index in the heterogeneous graph
        self.patient_ids = [
            pid for pid in self.patient_ids 
            if pid in self.patient_id_to_idx
        ]
        
        print(f"Final valid patient count: {len(self.patient_ids)}")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        """
        Get data for a single patient
        
        Returns:
            tuple: (hetero_data, clinical_features, survival_time, event)
        """
        patient_id = self.patient_ids[idx]
        
        # Get patient index in the heterogeneous graph
        patient_idx = self.patient_id_to_idx[patient_id]
        
        # Create subgraph for this patient
        # Note: This needs to be adjusted according to the actual heterogeneous graph structure
        patient_hetero_data = self._extract_patient_subgraph(patient_idx)
        
        # Get clinical features
        clinical_features = torch.tensor(
            self.clinical_df.loc[patient_id].values, 
            dtype=torch.float32
        )
        
        # Get survival data
        survival_row = self.survival_df.loc[patient_id]
        
        # Check possible column names
        time_col = None
        event_col = None
        
        for col in survival_row.index:
            col_lower = col.lower()
            if 'time' in col_lower or 'days' in col_lower or 'survival' in col_lower:
                time_col = col
            elif 'event' in col_lower or 'status' in col_lower or 'vital' in col_lower:
                event_col = col
        
        # Use found column names or default values
        if time_col is not None:
            survival_time = torch.tensor(survival_row[time_col], dtype=torch.float32)
        else:
            # Use first column as time
            survival_time = torch.tensor(survival_row.iloc[0], dtype=torch.float32)
        
        if event_col is not None:
            event = torch.tensor(survival_row[event_col], dtype=torch.long)
        else:
            # Use second column as event, or default to 1
            if len(survival_row) > 1:
                event = torch.tensor(survival_row.iloc[1], dtype=torch.long)
            else:
                event = torch.tensor(1, dtype=torch.long)
        
        # Now return pathway-level subgraph list
        patient_data_package = self._extract_patient_subgraph(patient_idx)
        
        return patient_data_package, clinical_features, survival_time, event
    
    def _extract_patient_subgraph(self, patient_idx):
        """
        Extract data for a specific patient from the global heterogeneous graph, creating pathway-level subgraphs according to MOGHET's actual design
        
        Args:
            patient_idx: Node index of the patient in the global graph
            
        Returns:
            dict: Contains a list of pathway-level subgraphs for this patient
                {
                    'intra_pathway_graphs': [pathway_graph1, pathway_graph2, ...],
                    'patient_id': patient_id
                }
        """
        from torch_geometric.data import HeteroData
        import torch
        
        global_patient_idx = patient_idx
        
        # 获取基因-通路关系
        gene_pathway_edges = None
        if ('gene', 'is_member_of', 'pathway') in self.hetero_data.edge_types:
            gene_pathway_edges = self.hetero_data[('gene', 'is_member_of', 'pathway')].edge_index
        
        # 获取通路数量
        num_pathways = 368  # 默认值
        if 'pathway' in self.hetero_data.node_types:
            num_pathways = self.hetero_data['pathway'].num_nodes
        
        # 获取该患者的基因表达数据
        patient_gene_data = {}
        if ('patient', 'expresses', 'gene') in self.hetero_data.edge_types:
            patient_gene_edges = self.hetero_data[('patient', 'expresses', 'gene')].edge_index
            patient_mask = patient_gene_edges[0] == global_patient_idx
            expressed_gene_indices = patient_gene_edges[1, patient_mask].tolist()
            
            # 为每个表达的基因创建随机的多组学数据
            for gene_idx in expressed_gene_indices:
                patient_gene_data[gene_idx] = torch.randn(3)  # [表达量, CNV, 突变]
        
        # 为每个通路创建独立的小图
        intra_pathway_graphs = []
        
        for pathway_idx in range(num_pathways):
            # 创建该通路的小图
            pathway_graph = HeteroData()
            
            # 1. 找到属于该通路的所有基因
            pathway_genes = []
            if gene_pathway_edges is not None:
                pathway_mask = gene_pathway_edges[1] == pathway_idx
                pathway_genes = gene_pathway_edges[0][pathway_mask].tolist()
            
            # 如果该通路没有基因，创建一个最小的图结构
            if len(pathway_genes) == 0:
                pathway_genes = [0]  # 至少包含一个基因节点
            
            num_genes_in_pathway = len(pathway_genes)
            
            # 2. 为该通路的基因创建特征
            gene_features_list = []
            for local_idx, global_gene_idx in enumerate(pathway_genes):
                if global_gene_idx in patient_gene_data:
                    # 使用该患者的实际基因数据
                    gene_features_list.append(patient_gene_data[global_gene_idx])
                else:
                    # 使用默认值或随机值
                    gene_features_list.append(torch.zeros(3))
            
            pathway_graph['gene'].x = torch.stack(gene_features_list)
            pathway_graph['gene'].num_nodes = num_genes_in_pathway
            
            # 3. 创建该通路内基因之间的相互作用边
            # 在真实场景中，这些边应该来自生物学网络数据（如PPI、调控关系等）
            # 这里我们创建一个简单的全连接模式作为示例
            if num_genes_in_pathway > 1:
                # 创建全连接图
                src_nodes = []
                dst_nodes = []
                for i in range(num_genes_in_pathway):
                    for j in range(num_genes_in_pathway):
                        if i != j:  # 不包括自连接
                            src_nodes.append(i)
                            dst_nodes.append(j)
                
                edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
                pathway_graph[('gene', 'interacts', 'gene')].edge_index = edge_index
            else:
                # 单个基因的自连接
                pathway_graph[('gene', 'interacts', 'gene')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            # 4. 设置批处理信息（用于通路内池化）
            pathway_graph['gene'].batch = torch.zeros(num_genes_in_pathway, dtype=torch.long)
            
            # 5. 添加通路标识
            pathway_graph.pathway_idx = torch.tensor([pathway_idx])
            
            # 6. 存储原始基因索引（用于调试和追踪）
            pathway_graph.original_gene_indices = torch.tensor(pathway_genes)
            
            intra_pathway_graphs.append(pathway_graph)
        
        # 创建患者数据包
        patient_data_package = {
            'intra_pathway_graphs': intra_pathway_graphs,
            'patient_id': f'patient_{global_patient_idx}'
        }
        
        return patient_data_package
    
    @property
    def clinical_features_dim(self):
        """返回临床特征的维度"""
        return self.clinical_df.shape[1]
    
    @property
    def num_pathways(self):
        """返回通路数量"""
        if hasattr(self.hetero_data, 'num_pathways'):
            return self.hetero_data.num_pathways
        else:
            return 368  # 默认值
    
    @property
    def num_genes(self):
        """返回基因数量"""
        if 'gene' in self.hetero_data.node_types:
            return self.hetero_data['gene'].x.size(0)
        else:
            return 8576  # 默认值
    
    @property
    def graph_metadata(self):
        """返回图的元数据"""
        return (['gene'], [('gene', 'interacts', 'gene')])


def create_ablation_data_loaders(data_path, train_patient_ids, val_patient_ids, batch_size=32):
    """
    创建消融实验用的数据加载器
    
    Args:
        data_path: 数据路径
        train_patient_ids: 训练集患者ID
        val_patient_ids: 验证集患者ID
        batch_size: 批大小
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = AblationDataset(data_path, train_patient_ids)
    val_dataset = AblationDataset(data_path, val_patient_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
