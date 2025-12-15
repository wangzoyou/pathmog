"""
Custom batching function for handling MOGHET's pathway-level subgraph data structure
"""

from typing import List, Tuple, Dict, Any
import torch
from torch_geometric.data import HeteroData


def custom_collate_fn(batch_list: List[Tuple]):
    """
    Custom collate function for processing MOGHET's pathway-level subgraph data
    
    Args:
        batch_list: Batch data list from DataLoader, each element contains:
                   (patient_data_package, clinical_features, survival_time, event)
                   where patient_data_package contains:
                   {
                       'intra_pathway_graphs': [pathway_graph1, pathway_graph2, ...],
                       'patient_id': patient_id
                   }
    
    Returns:
        batch_data: Batched data, containing:
        {
            'intra_pathway_graphs': [[patient1_pathway_graphs], [patient2_pathway_graphs], ...],
            'clinical_features': torch.tensor,
            'survival_time': torch.tensor,
            'event': torch.tensor,
            'batch_size': int
        }
    """
    
    # Separate different types of data
    patient_data_packages = []
    clinical_features_list = []
    survival_time_list = []
    event_list = []
    
    for patient_data_package, clinical_features, survival_time, event in batch_list:
        patient_data_packages.append(patient_data_package)
        clinical_features_list.append(clinical_features)
        survival_time_list.append(survival_time)
        event_list.append(event)
    
    batch_size = len(batch_list)
    
    try:
        # 批处理临床特征和生存数据
        batched_clinical_features = torch.stack(clinical_features_list)
        batched_survival_time = torch.stack(survival_time_list)  
        batched_event = torch.stack(event_list)
        
        # 收集所有患者的通路图
        all_intra_pathway_graphs = []
        for patient_data_package in patient_data_packages:
            all_intra_pathway_graphs.append(patient_data_package['intra_pathway_graphs'])
        
        # 创建批处理数据结构
        batch_data = {
            'intra_pathway_graphs': all_intra_pathway_graphs,  # [batch_size][num_pathways] 的通路图列表
            'clinical_features': batched_clinical_features,    # [batch_size, clinical_dim]
            'survival_time': batched_survival_time,            # [batch_size]
            'event': batched_event,                           # [batch_size]
            'batch_size': batch_size,
            'patient_ids': [pkg['patient_id'] for pkg in patient_data_packages]
        }
        
        return batch_data
            
    except Exception as e:
        print(f"批处理通路级别数据失败: {e}")
        print(f"临床特征形状: {[cf.shape for cf in clinical_features_list]}")
        print(f"生存时间形状: {[st.shape for st in survival_time_list]}")
        print(f"事件形状: {[ev.shape for ev in event_list]}")
        print(f"患者数据包数量: {len(patient_data_packages)}")
        if len(patient_data_packages) > 0:
            print(f"第一个患者的通路图数量: {len(patient_data_packages[0]['intra_pathway_graphs'])}")
        raise e