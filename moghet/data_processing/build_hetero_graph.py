#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
构建层次化图神经网络数据对象模块

本模块为 "HierarchicalGNNModel" 架构准备数据。
它将为数据集中的每一位患者生成一个独立的、层次化的数据包。

核心产物 (为每位患者生成一个 .pt 文件):
- **通路内基因子图 (Intra-Pathway Graphs)**: 一个包含多个独立图对象的列表。
  每个图对象 (torch_geometric.data.Data) 代表一个通路，包含:
    - 该通路内的基因节点及其多组学特征 (表达, CNV, 突变)。
    - 仅存在于这些基因之间的相互作用边。
- **临床特征**: 该患者的临床数据张量。

实现流程:
1. **保留数据加载与ID映射**: 完全重用成熟的数据处理和ID对齐逻辑。
2. **预计算通路结构**: 新增一个步骤，预先计算和缓存所有通路的静态拓扑结构
   (包含哪些基因，基因间如何连接)，此操作与患者无关，只需执行一次。
3. **生成并保存患者数据包**: 核心循环遍历所有患者，利用预计算的通路结构和
   每位患者特有的组学数据，动态生成上述的数据包并保存。
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData  # 修改: 使用HeteroData来支持多边类型
from tqdm import tqdm
import json
import os.path as osp
import shutil

class HierarchicalGraphBuilder:
    """
    层次化图数据构建器。
    为每个患者生成一个数据对象，其中包含：
    1. 一系列通路内基因子图 (Intra-Pathway Graphs)
    2. 患者的临床特征
    此架构旨在忠实地将原CNN+Transformer模型替换为GNN+GNN层次化模型。
    """
    
    def __init__(self, input_path, output_path=None, dataset_name="BRCA"):
        """
        初始化层次化图构建器
        
        参数:
            input_path (str): 输入数据目录路径，包含KEGGParser生成的CSV文件
            output_path (str): 输出目录路径，默认为None
            dataset_name (str): 数据集名称，用于选择相应的临床特征处理逻辑
        """
        self.input_path = input_path
        self.output_path = output_path if output_path else input_path
        self.dataset_name = dataset_name
        
        # 核心输出目录
        self.patient_data_path = osp.join(self.output_path, 'hierarchical_patient_data')
        os.makedirs(self.output_path, exist_ok=True)
        
        # 清理旧的数据文件目录
        if osp.exists(self.patient_data_path):
            print(f"正在清理旧的层次化数据文件目录: {self.patient_data_path}")
            shutil.rmtree(self.patient_data_path)
        os.makedirs(self.patient_data_path, exist_ok=True)
        
        # 数据存储 (与原版一致)
        self.gene_info = None
        self.pathway_info = None
        self.gene_gene_relations = None
        self.gene_pathway_relations = None
        self.patient_gene_expression = None
        self.patient_gene_mutation = None
        self.patient_gene_cnv = None
        self.patient_features = None
        self.patient_survival = None
        
        # ID映射 (与原版一致)
        self.patient_id_to_idx = {}
        self.gene_id_to_idx = {}
        self.pathway_id_to_idx = {}
        
        # 基因ID和符号映射
        self.gene_id_to_symbol = {}
        self.gene_symbol_to_id = {}
        
        # 新增: 预计算的通路静态结构
        self.pathway_structures = {}
    
    def load_data(self):
        """
        加载KEGGParser生成的CSV文件。
        此方法被完全保留，以确保数据加载的稳定性和一致性。
        """
        print("加载数据文件...")
        
        def load_csv_with_id_column(file_path, id_col_name, gene_col_name=None):
            """(最终重构版) 读取CSV文件，高效、鲁棒地处理ID列。"""
            try:
                # 1. 只读取一次文件
                df = pd.read_csv(file_path)

                # 2. 检查ID列是否存在
                if id_col_name not in df.columns:
                    # 如果不存在，检查是否有'Unnamed: 0'列，这通常是保存索引时产生的
                    if 'Unnamed: 0' in df.columns:
                        df.rename(columns={'Unnamed: 0': id_col_name}, inplace=True)
                    else:
                        # 如果仍然找不到，执行回退逻辑 (在同一个df上)
                        for col in df.columns:
                            if col.lower() in [id_col_name.lower(), 'id', 'sample', 'patient', 'gene_id', 'pathway_id', 'source_gene']:
                                df.rename(columns={col: id_col_name}, inplace=True)
                                break
                        # 如果还没有，则假定第一列是ID列
                        if id_col_name not in df.columns and len(df.columns) > 0:
                            df.rename(columns={df.columns[0]: id_col_name}, inplace=True)
                
                # 3. (可选) 重命名基因列
                if gene_col_name and gene_col_name not in df.columns and len(df.columns) > 1:
                    df.rename(columns={df.columns[1]: gene_col_name}, inplace=True)
                
                # 4. 过滤掉无效的ID行（如标题行）
                if id_col_name in df.columns:
                    df = df[~df[id_col_name].isin(['ID', 'id', 'Id'])]

            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                return None
            return df
        
        # 加载非患者特异性数据 (恢复使用原版的鲁棒加载方式)
        self.gene_info = load_csv_with_id_column(osp.join(self.input_path, 'kegg_gene_info.csv'), id_col_name='gene_id')
        self.pathway_info = load_csv_with_id_column(osp.join(self.input_path, 'kegg_pathway_info.csv'), id_col_name='pathway_id')
        
        # 修正: 不再重命名列，直接使用CSV文件头中的'source_gene'和'target_gene'
        self.gene_gene_relations = load_csv_with_id_column(
            osp.join(self.input_path, 'kegg_gene_gene_relations.csv'), 
            id_col_name='source_gene' # 指定第一列为source_gene，让加载函数正确识别
        )
        
        self.gene_pathway_relations = load_csv_with_id_column(osp.join(self.input_path, 'kegg_gene_pathway_relations.csv'), id_col_name='gene_id', gene_col_name='pathway_id')

        # 加载患者特异性数据 (保留宽表加载逻辑)
        def load_wide_format_csv(file_path, entity_name):
            """加载行为患者、列为基因的宽格式CSV，并处理重复项。"""
            if not os.path.exists(file_path):
                print(f"警告: 未找到{entity_name}文件 {file_path}")
                return None
            try:
                df = pd.read_csv(file_path, index_col=0)
                
                # 处理索引（患者）中的重复项
                if df.index.has_duplicates:
                    num_dupes = df.index.duplicated().sum()
                    print(f"警告: 在 {entity_name} 数据的患者ID中发现 {num_dupes} 个重复项，将保留第一个。")
                    df = df[~df.index.duplicated(keep='first')]
                
                # 处理列（基因）中的重复项
                if df.columns.has_duplicates:
                    num_dupes = df.columns.duplicated().sum()
                    print(f"警告: 在 {entity_name} 数据的基因列中发现 {num_dupes} 个重复项，将保留第一个。")
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]

                df.columns = df.columns.astype(str)
                df.index = df.index.astype(str)
                print(f"加载并清理后，有 {df.shape[0]} 个患者的 {df.shape[1]} 个基因的{entity_name}数据 (宽格式)")
                return df
            except Exception as e:
                print(f"加载或处理宽格式文件 {file_path} 时出错: {e}")
                return None

        self.patient_gene_expression = load_wide_format_csv(osp.join(self.input_path, 'patient_gene_expression.csv'), "基因表达")
        self.patient_gene_mutation = load_wide_format_csv(osp.join(self.input_path, 'patient_gene_mutation.csv'), "基因突变")
        self.patient_gene_cnv = load_wide_format_csv(osp.join(self.input_path, 'patient_gene_cnv.csv'), "CNV")
        
        # 加载患者临床特征
        patient_file = os.path.join(self.input_path, 'patient_clinical_features.csv')
        if os.path.exists(patient_file):
            self.patient_features = pd.read_csv(patient_file, index_col=0)
            if self.patient_features.index.has_duplicates:
                print(f"警告: 在临床特征数据的患者ID中发现重复项，将保留第一个。")
                self.patient_features = self.patient_features[~self.patient_features.index.duplicated(keep='first')]
            self.patient_features.index = self.patient_features.index.astype(str)
        else:
            print(f"警告: 未找到患者特征文件 {patient_file}")

        # 加载患者生存数据
        survival_file = os.path.join(self.input_path, 'patient_survival.csv')
        if os.path.exists(survival_file):
            self.patient_survival = pd.read_csv(survival_file, index_col=0)
            if self.patient_survival.index.has_duplicates:
                print(f"警告: 在生存数据的患者ID中发现重复项，将保留第一个。")
                self.patient_survival = self.patient_survival[~self.patient_survival.index.duplicated(keep='first')]
            self.patient_survival.index = self.patient_survival.index.astype(str)
        else:
            print(f"警告: 未找到生存数据文件 {survival_file}")

        return True

    def create_id_mappings(self):
        """
        创建ID映射。
        此方法被完全保留，以确保ID映射的稳定性和一致性。
        """
        print("创建ID映射...")
        
        # 1. 患者ID映射 (基于临床和生存数据的交集)
        if self.patient_features is not None and self.patient_survival is not None:
            clinical_patients = set(self.patient_features.index)
            survival_patients = set(self.patient_survival.index)
            patient_ids = sorted(list(clinical_patients.intersection(survival_patients)))
            print(f"基于临床和生存数据的交集，确定了 {len(patient_ids)} 位核心患者。")
        else:
            print("警告: 临床或生存数据缺失，将使用所有可用的患者ID进行联合。")
            patient_ids = set()
            if self.patient_features is not None: patient_ids.update(self.patient_features.index)
            if self.patient_survival is not None: patient_ids.update(self.patient_survival.index)
            if self.patient_gene_expression is not None: patient_ids.update(self.patient_gene_expression.index)
            patient_ids = sorted(list(patient_ids))

        self.patient_id_to_idx = {pid: idx for idx, pid in enumerate(patient_ids)}
        print(f"创建了 {len(self.patient_id_to_idx)} 个患者ID映射")

        # 2. 基因ID映射 (使用通路中的基因)
        gene_symbols = set(self.gene_pathway_relations['gene_id'].astype(str))
        gene_symbol_list = sorted(list(gene_symbols))
        self.gene_id_to_idx = {gs: idx for idx, gs in enumerate(gene_symbol_list)}
        print(f"创建了 {len(self.gene_id_to_idx)} 个KEGG通路基因符号ID映射")

        # 3. 通路ID映射
        pathway_ids = set(self.pathway_info['pathway_id'].astype(str))
        # 过滤掉无效的通路ID
        pathway_ids = {pid for pid in pathway_ids if pid not in ['ID', 'id']}
        pathway_id_list = sorted(list(pathway_ids))
        self.pathway_id_to_idx = {pid: idx for idx, pid in enumerate(pathway_id_list)}
        print(f"创建了 {len(self.pathway_id_to_idx)} 个通路ID映射")
        return True

    def precompute_pathway_structures(self):
        """
        (新增) 预计算所有通路的静态拓扑结构。
        对于每个通路，计算其包含的基因子集以及这些基因之间的相互作用边。
        这个操作独立于任何患者数据，因此只需要执行一次。
        """
        print("预计算所有通路的静态拓扑结构...")
        
        if self.gene_pathway_relations is None or self.gene_gene_relations is None:
            print("错误: 基因-通路或基因-基因关系数据未加载，无法预计算。")
            return False

        all_pathway_ids = list(self.pathway_id_to_idx.keys())
        
        for pathway_id in tqdm(all_pathway_ids, desc="预计算通路结构"):
            # 1. 找到属于该通路的所有基因 (基于我们的基因 universe)
            pathway_genes_df = self.gene_pathway_relations[self.gene_pathway_relations['pathway_id'] == pathway_id]
            pathway_gene_symbols = set(pathway_genes_df['gene_id'].astype(str))
            
            # 只保留存在于总基因映射中的基因
            valid_genes = sorted(list(pathway_gene_symbols.intersection(self.gene_id_to_idx.keys())))
            
            if len(valid_genes) < 2: # 如果通路内少于2个基因，无法构成图，跳过
                continue
            
            # 2. 为通路内的基因创建局部索引
            local_gene_to_idx = {gene_symbol: i for i, gene_symbol in enumerate(valid_genes)}
            
            # 3. 找到这些基因之间的内部连接边 (新逻辑: 按关系类型分组)
            gg_relations = self.gene_gene_relations
            pathway_edges_df = gg_relations[
                gg_relations['source_gene'].isin(valid_genes) & 
                gg_relations['target_gene'].isin(valid_genes)
            ].copy() # 修正: 使用 .copy() 避免 SettingWithCopyWarning
            
            edge_indices_by_type = {}
            if not pathway_edges_df.empty:
                # 清理和规范化关系类型名称
                pathway_edges_df['relation_type'] = pathway_edges_df['relation_type'].str.replace(' ', '_').str.replace('/', '_')
                
                # 按关系类型分组
                for rel_type, group in pathway_edges_df.groupby('relation_type'):
                    edges = []
                    for _, row in group.iterrows():
                        # 确保源和目标基因都在局部索引中
                        source_symbol = str(row['source_gene'])
                        target_symbol = str(row['target_gene'])
                        if source_symbol in local_gene_to_idx and target_symbol in local_gene_to_idx:
                            g1_local_idx = local_gene_to_idx[source_symbol]
                            g2_local_idx = local_gene_to_idx[target_symbol]
                            edges.append([g1_local_idx, g2_local_idx])
                    
                    if edges:
                        # 创建有向边索引
                        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                        # 定义异构图中的边类型三元组
                        edge_type_tuple = ('gene', rel_type, 'gene')
                        edge_indices_by_type[edge_type_tuple] = edge_index

            # 4. 存储该通路的结构信息 (包含多种边类型)
            self.pathway_structures[pathway_id] = {
                'genes': valid_genes, # 基因符号列表，顺序与局部索引一致
                'edge_indices': edge_indices_by_type, # 存储按类型分的边
                'num_genes': len(valid_genes)
            }
            
        print(f"预计算完成，成功处理了 {len(self.pathway_structures)} 个有效通路。")
        return True

    def generate_and_save_hierarchical_data(self):
        """
        (核心重写) 为每个患者生成并保存其层次化的图数据包。
        """
        if not self.pathway_structures:
            print("错误: 通路结构未预计算。请先运行 precompute_pathway_structures。")
            return

        # 对齐所有组学数据 (与之前版本逻辑一致)
        print("对齐所有患者的组学数据...")
        patient_list = list(self.patient_id_to_idx.keys())
        # 使用完整的基因列表进行对齐，以保证后续索引的正确性
        omics_gene_list = sorted(list(self.gene_id_to_idx.keys()))
        
        aligned_omics = {}
        all_omics_dfs = {'expr': self.patient_gene_expression, 'cnv': self.patient_gene_cnv, 'mut': self.patient_gene_mutation}
        
        for name, df in all_omics_dfs.items():
            if df is not None:
                # 使用 reindex 保证所有患者和基因都在DF中，缺失值为NaN
                aligned_df = df.reindex(index=patient_list, columns=omics_gene_list, fill_value=np.nan)
                aligned_omics[name] = aligned_df
            else:
                print(f"警告: 组学数据 '{name}' 未加载，将使用 NaN 填充。")
                aligned_omics[name] = pd.DataFrame(np.nan, index=patient_list, columns=omics_gene_list)
        
        simplified_clinical_df = self.simplify_clinical_features()
        aligned_clinical = simplified_clinical_df.reindex(patient_list)
        
        print(f"将为 {len(patient_list)} 位患者生成层次化数据包...")

        saved_patient_ids = []
        for patient_id in tqdm(patient_list, desc="生成患者数据包"):
            intra_pathway_graphs = []
            
            # 遍历所有预计算好的、有效的通路
            for pathway_id, structure in self.pathway_structures.items():

                # 1. 提取该通路在该患者上的基因特征
                pathway_genes = structure['genes']
                
                # 从已对齐的DataFrame中高效提取数据
                expr_vals = aligned_omics['expr'].loc[patient_id, pathway_genes].values
                cnv_vals = aligned_omics['cnv'].loc[patient_id, pathway_genes].values
                mut_vals = aligned_omics['mut'].loc[patient_id, pathway_genes].values

                # 如果一个通路的所有基因在本患者上都没有任何数据，则跳过该通路
                if np.isnan(expr_vals).all() and np.isnan(cnv_vals).all() and np.isnan(mut_vals).all():
                    continue

                # (修正) 将以下代码块缩进到 for pathway_id 循环内部
                gene_features = np.stack([expr_vals, cnv_vals, mut_vals], axis=1)
                gene_features_tensor = torch.tensor(np.nan_to_num(gene_features, nan=0.0), dtype=torch.float)
            
                # 2. 创建通路子图 HeteroData 对象
                pathway_graph = HeteroData()

                # 2.1 添加基因节点特征
                pathway_graph['gene'].x = gene_features_tensor
                pathway_graph['gene'].num_nodes = structure['num_genes']
                
                # 2.2 添加所有类型的边
                for edge_type, edge_index in structure['edge_indices'].items():
                    pathway_graph[edge_type].edge_index = edge_index
                
                # 2.3 添加辅助信息
                pathway_graph.pathway_idx = torch.tensor([self.pathway_id_to_idx[pathway_id]], dtype=torch.long)
                
                intra_pathway_graphs.append(pathway_graph)
                
            # 如果患者没有任何有效的通路图，则跳过该患者
            if not intra_pathway_graphs:
                print(f"警告: 患者 {patient_id} 未能生成任何有效的通路子图，已跳过。")
                continue

            # 3. 提取临床特征
            clinical_vals = aligned_clinical.loc[patient_id].values
            
            # 确保所有值都是数值类型
            clinical_vals_clean = []
            for val in clinical_vals:
                if isinstance(val, (int, float, np.integer, np.floating)):
                    clinical_vals_clean.append(float(val))
                elif isinstance(val, str):
                    # 如果是字符串，尝试转换为数字，失败则为0
                    try:
                        clinical_vals_clean.append(float(val))
                    except:
                        clinical_vals_clean.append(0.0)
                else:
                    clinical_vals_clean.append(0.0)
            
            clinical_tensor = torch.tensor(clinical_vals_clean, dtype=torch.float).unsqueeze(0)

            # 4. 组装最终的数据包
            patient_data_package = {
                'patient_id': patient_id,
                'intra_pathway_graphs': intra_pathway_graphs,
                'clinical_features': clinical_tensor
            }
            
            # 5. 保存这个数据包
            output_file = osp.join(self.patient_data_path, f"{patient_id}.pt")
            torch.save(patient_data_package, output_file)
            saved_patient_ids.append(patient_id)

        print(f"\n所有有效的患者数据包已成功生成并保存在: {self.patient_data_path}")
        print(f"共为 {len(saved_patient_ids)} / {len(patient_list)} 位患者成功生成了数据。")
            
        # 6. 保存辅助文件 (只传递成功保存的患者ID)
        self.save_auxiliary_files(saved_patient_ids)

    def save_auxiliary_files(self, patient_ids_with_graphs):
        """(简化版) 保存辅助文件，如ID映射和对齐后的临床数据。"""
        print("保存辅助文件...")

        # 1. 保存ID映射 (包含三类ID)
        id_mappings = {
            'patient_id_to_idx': self.patient_id_to_idx,
            'gene_id_to_idx': self.gene_id_to_idx,
            'pathway_id_to_idx': self.pathway_id_to_idx
        }
        mappings_path = osp.join(self.output_path, 'id_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump(id_mappings, f, indent=2)
        print(f"ID映射已保存到: {mappings_path}")
        
        # 2. 保存通路结构信息 (新增)
        pathway_structures_simple = {}
        for pathway_id, structure in self.pathway_structures.items():
            pathway_structures_simple[pathway_id] = {
                'genes': structure['genes'],  # 基因符号列表，顺序与局部索引一致
                'num_genes': structure['num_genes']
            }
        structures_path = osp.join(self.output_path, 'pathway_structures.json')
        with open(structures_path, 'w') as f:
            json.dump(pathway_structures_simple, f, indent=2)
        print(f"通路结构信息已保存到: {structures_path}")
        
        # 3. 保存对齐后的临床数据
        if self.patient_features is not None:
            # 只保留那些成功生成了图的患者的临床数据
            simplified_clinical_df = self.simplify_clinical_features()
            # 确保即使没有生成任何图，也不会出错
            if patient_ids_with_graphs:
                aligned_clinical = simplified_clinical_df.reindex(patient_ids_with_graphs)
                clinical_path = osp.join(self.output_path, 'patient_clinical_features.csv')
                aligned_clinical.to_csv(clinical_path)
                print(f"对齐后的【简化】临床特征已保存到: {clinical_path}")
            else:
                print("警告: 没有为任何患者生成数据，跳过保存对齐的临床特征。")

    def simplify_clinical_features(self):
        """
        对临床特征进行简化和特征工程。根据数据集类型选择不同的处理逻辑。
        """
        print(f"简化临床特征 (数据集: {self.dataset_name})...")
        df = self.patient_features.copy()

        if self.dataset_name == "BRCA":
            # BRCA数据集的临床特征简化逻辑
            simplified_df = self._simplify_brca_features(df)
        elif self.dataset_name == "LUAD":
            # LUAD数据集的临床特征简化逻辑
            simplified_df = self._simplify_luad_features(df)
        elif self.dataset_name == "COAD":
            # COAD（结肠腺癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_coad_features(df)
        elif self.dataset_name == "GBM":
            # GBM（胶质母细胞瘤）数据集的临床特征简化逻辑
            simplified_df = self._simplify_gbm_features(df)
        elif self.dataset_name == "KIRC":
            # KIRC（肾透明细胞癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_kirc_features(df)
        elif self.dataset_name == "LUNG":
            # LUNG（肺癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_lung_features(df)
        elif self.dataset_name == "OV":
            # OV（卵巢癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_ov_features(df)
        elif self.dataset_name == "SKCM":
            # SKCM（皮肤黑色素瘤）数据集的临床特征简化逻辑
            simplified_df = self._simplify_skcm_features(df)
        elif self.dataset_name == "LIHC":
            # LIHC（肝细胞癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_lihc_features(df)
        elif self.dataset_name == "LUSC":
            # LUSC（肺鳞癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_lusc_features(df)
        elif self.dataset_name == "STAD":
            # STAD（胃腺癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_stad_features(df)
        elif self.dataset_name == "UCEC":
            # UCEC（子宫内膜癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_ucec_features(df)
        elif self.dataset_name == "HNSC":
            # HNSC（头颈鳞癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_hnsc_features(df)
        elif self.dataset_name == "PAAD":
            # PAAD（胰腺腺癌）数据集的临床特征简化逻辑
            simplified_df = self._simplify_paad_features(df)
        elif self.dataset_name == "LGG":
            # LGG（低级别胶质瘤）数据集的临床特征简化逻辑
            simplified_df = self._simplify_lgg_features(df)
        else:
            # 默认的通用特征简化逻辑
            simplified_df = self._simplify_general_features(df)

        print(f"临床特征简化完成。特征数量从 {len(df.columns)} 减少到 {len(simplified_df.columns)}。")
        print("新特征:", simplified_df.columns.tolist())
        return simplified_df

    def _simplify_brca_features(self, df):
        """BRCA数据集的临床特征简化逻辑"""
        # 1. 年龄和性别保持不变
        # 处理原始的gender列，转换为二进制特征
        simplified_df = df[['age_at_initial_pathologic_diagnosis', 'gender']].copy()
        simplified_df['is_female'] = (simplified_df['gender'] == 'FEMALE').astype(int)
        simplified_df = simplified_df.drop('gender', axis=1)

        # 2. 聚合病理分期 (Pathologic Stage)
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0 # 0 代表未知或无法分类
        }
        # 直接处理原始的pathologic_stage列
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0

        # 3. 聚合TNM分期
        # T (Tumor)
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({f'T{i}c': i for i in range(1, 5)})
        t_map.update({f'T{i}d': i for i in range(1, 5)})
        t_map['TX'] = 0
        # 直接处理原始的pathologic_T列
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0

        # N (Node)
        n_map = {
            'N0': 0, 'N0 (i+)': 0, 'N0 (i-)': 0, 'N0 (mol+)': 0,
            'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
            'N2': 2, 'N2a': 2,
            'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
            'NX': 0 # 视为未知
        }
        # 直接处理原始的pathologic_N列
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0

        # M (Metastasis)
        m_map = {'M0': 0, 'cM0 (i+)': 0, 'M1': 1, 'MX': 0}
        # 直接处理原始的pathologic_M列
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
        
        # 4. 聚合激素受体状态 (ER, PR, HER2)
        # 0: Indeterminate, 1: Negative, 2: Positive
        if 'ER_Status_nature2012' in df.columns:
            simplified_df['ER_status'] = df['ER_Status_nature2012'].apply(
                lambda x: 1 if x == 'Negative' else (2 if x == 'Positive' else 0)
            )
        else:
            simplified_df['ER_status'] = 0

        if 'PR_Status_nature2012' in df.columns:
            simplified_df['PR_status'] = df['PR_Status_nature2012'].apply(
                lambda x: 1 if x == 'Negative' else (2 if x == 'Positive' else 0)
            )
        else:
            simplified_df['PR_status'] = 0

        if 'HER2_Final_Status_nature2012' in df.columns:
            simplified_df['HER2_status'] = df['HER2_Final_Status_nature2012'].apply(
                lambda x: 1 if x == 'Negative' else (2 if x == 'Positive' else (0 if x == 'Equivocal' else 0))
            )
        else:
            simplified_df['HER2_status'] = 0
        
        # 5. 使用PAM50分型
        pam50_map = {
            'LumA': 1,
            'LumB': 2,
            'Her2': 3,
            'Basal': 4,
            'Normal': 5,
        }
        if 'PAM50Call_RNAseq' in df.columns:
            simplified_df['PAM50'] = df['PAM50Call_RNAseq'].map(pam50_map).fillna(0).astype(int)
        else:
            simplified_df['PAM50'] = 0

        return simplified_df

    def _simplify_luad_features(self, df):
        """精简LUAD临床特征，仅保留核心特征，编码方式与BRCA一致"""
        simplified_df = pd.DataFrame(index=df.index)
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce')
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        # 2. 性别二值化
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
        # 3. 病理分期（聚合为1-4级）
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0
        # 4. T分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({f'T{i}c': i for i in range(1, 5)})
        t_map.update({f'T{i}d': i for i in range(1, 5)})
        t_map['TX'] = 0
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
        # 5. N分期
        n_map = {
            'N0': 0, 'N0 (i+)': 0, 'N0 (i-)': 0, 'N0 (mol+)': 0,
            'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
            'N2': 2, 'N2a': 2,
            'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
            'NX': 0
        }
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0
        # 6. M分期
        m_map = {'M0': 0, 'cM0 (i+)': 0, 'M1': 1, 'MX': 0}
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
        # 7. KPS评分
        if 'karnofsky_performance_score' in df.columns:
            simplified_df['karnofsky_performance_score'] = pd.to_numeric(df['karnofsky_performance_score'], errors='coerce')
        else:
            simplified_df['karnofsky_performance_score'] = 0
        
        # 8. 组织学类型（关键预后特征）
        if 'histological_type' in df.columns:
            # 编码为类别值
            histology_map = {
                'Lung Adenocarcinoma- Not Otherwise Specified (NOS)': 1,
                'Lung Adenocarcinoma Mixed Subtype': 2,
                'Lung Papillary Adenocarcinoma': 3,
                'Lung Bronchioloalveolar Carcinoma Nonmucinous': 4,
                'Lung Acinar Adenocarcinoma': 5,
                'Mucinous (Colloid) Carcinoma': 6,
                'Lung Bronchioloalveolar Carcinoma Mucinous': 7,
                'Lung Solid Pattern Predominant Adenocarcinoma': 8,
                'Lung Mucinous Adenocarcinoma': 9,
                'Lung Clear Cell Adenocarcinoma': 10,
            }
            simplified_df['histological_type'] = df['histological_type'].map(histology_map).fillna(0).astype(int)
        else:
            simplified_df['histological_type'] = 0
        
        # 9. 吸烟史（关键预后特征）
        if 'tobacco_smoking_history' in df.columns:
            # 吸烟史编码：1=从不吸烟, 2=戒烟<=15年, 3=戒烟>15年, 4=当前吸烟, 5=未知
            simplified_df['tobacco_smoking_history'] = pd.to_numeric(df['tobacco_smoking_history'], errors='coerce').fillna(0).astype(int)
        else:
            simplified_df['tobacco_smoking_history'] = 0
        
        return simplified_df

    def _simplify_coad_features(self, df):
        """COAD（结肠腺癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄 - 所有数据集都有
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
            
        # 2. 性别 - 大部分数据集都有
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
            
        # 3. 病理分期 - 通用特征
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        elif 'clinical_stage' in df.columns:  # OV数据集使用clinical_stage
            simplified_df['pathologic_stage'] = df['clinical_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0
            
        # 4. TNM分期
        # T分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({f'T{i}c': i for i in range(1, 5)})
        t_map.update({f'T{i}d': i for i in range(1, 5)})
        t_map['TX'] = 0
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
            
        # N分期
        n_map = {
            'N0': 0, 'N0 (i+)': 0, 'N0 (i-)': 0, 'N0 (mol+)': 0,
            'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
            'N2': 2, 'N2a': 2,
            'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
            'NX': 0
        }
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0
            
        # M分期
        m_map = {'M0': 0, 'cM0 (i+)': 0, 'M1': 1, 'MX': 0}
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
            
        # 填充缺失值
        simplified_df = simplified_df.fillna(0)
        
        return simplified_df

    def _simplify_gbm_features(self, df):
        """GBM（胶质母细胞瘤）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 年龄特征
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce')
        
        # 性别特征
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        
        # 组织学类型特征 - 简化为二进制特征
        if 'histological_type' in df.columns:
            # 大多数GBM患者都是"Untreated primary (de novo) GBM"，创建一个二进制特征
            simplified_df['is_primary_gbm'] = df['histological_type'].str.contains('primary', case=False, na=False).astype(int)
        
        # Karnofsky评分
        if 'karnofsky_performance_score' in df.columns:
            simplified_df['karnofsky_performance_score'] = pd.to_numeric(df['karnofsky_performance_score'], errors='coerce')
        
        # ECOG评分
        if 'eastern_cancer_oncology_group' in df.columns:
            simplified_df['eastern_cancer_oncology_group'] = pd.to_numeric(df['eastern_cancer_oncology_group'], errors='coerce')
        
        # 治疗相关特征 - 转换为二进制
        treatment_cols = ['radiation_therapy', 'chemo_therapy', 'additional_chemo_therapy', 
                         'additional_drug_therapy', 'additional_pharmaceutical_therapy', 
                         'additional_radiation_therapy', 'targeted_molecular_therapy']
        
        for col in treatment_cols:
            if col in df.columns:
                simplified_df[col] = (df[col] == 'YES').astype(int)
        
        # 填充缺失值
        simplified_df = simplified_df.fillna(0)
        
        # 确保所有列都是数值类型
        for col in simplified_df.columns:
            simplified_df[col] = pd.to_numeric(simplified_df[col], errors='coerce').fillna(0)
        
        return simplified_df
    
    def _simplify_kirc_features(self, df):
        """KIRC（肾透明细胞癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)

        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
            
        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
            
        # 3. 病理分期 (Pathologic Stage)
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0, '[Not Available]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0

        # 4. TNM分期
        # T (Tumor)
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({f'T{i}c': i for i in range(1, 5)})
        t_map.update({f'T{i}d': i for i in range(1, 5)})
        t_map['TX'] = 0
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0

        # N (Node)
        n_map = {
            'N0': 0, 'N0 (i+)': 0, 'N0 (i-)': 0, 'N0 (mol+)': 0,
            'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
            'N2': 2, 'N2a': 2,
            'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
            'NX': 0
        }
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0

        # M (Metastasis)
        m_map = {'M0': 0, 'cM0 (i+)': 0, 'M1': 1, 'MX': 0}
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0

        # 5. 肿瘤组织学分级 (Neoplasm Histologic Grade) - KIRC特有
        grade_map = {
            'G1': 1,
            'G2': 2,
            'G3': 3,
            'G4': 4,
            'GX': 0, '[Discrepancy]': 0, '[Not Available]': 0
        }
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0

        # 填充所有可能的NaN值
        simplified_df = simplified_df.fillna(0)

        return simplified_df
    
    def _simplify_lung_features(self, df):
        """LUNG（肺癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)

        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0

        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0

        # 3. 病理分期
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0, '[Not Available]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0

        # 4. TNM分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({'TX': 0, 'T0': 0})
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0

        n_map = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'NX': 0}
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0

        # 注意：M_stage覆盖率太低（3.1%），已删除
        # 注意：LUNG原始数据没有neoplasm_histologic_grade，已删除此特征
        
        # 5. 吸烟史（详细编码，与LUAD一致）
        if 'tobacco_smoking_history' in df.columns:
            # 使用数字编码保留更多信息
            simplified_df['tobacco_smoking_history'] = pd.to_numeric(df['tobacco_smoking_history'], errors='coerce').fillna(0).astype(int)
        else:
            simplified_df['tobacco_smoking_history'] = 0

        # 6. 组织学类型（重要预后特征）
        if 'histological_type' in df.columns:
            # LUNG包含多种肺癌亚型，编码为类别
            histology_map = {
                'Lung Adenocarcinoma- Not Otherwise Specified (NOS)': 1,
                'Lung Adenocarcinoma Mixed Subtype': 2,
                'Lung Squamous Cell Carcinoma- Not Otherwise Specified (NOS)': 3,
                'Lung Large Cell Carcinoma': 4,
                'Lung Bronchioloalveolar Carcinoma Nonmucinous': 5,
                'Lung Acinar Adenocarcinoma': 6,
                'Other, specify': 7,
            }
            simplified_df['histological_type'] = df['histological_type'].map(histology_map).fillna(0).astype(int)
        else:
            simplified_df['histological_type'] = 0

        # 7. 卡氏功能状态评分（重要预后特征）
        if 'karnofsky_performance_score' in df.columns:
            simplified_df['karnofsky_performance_score'] = pd.to_numeric(df['karnofsky_performance_score'], errors='coerce').fillna(0)
        else:
            simplified_df['karnofsky_performance_score'] = 0

        # 填充所有可能的NaN值
        simplified_df = simplified_df.fillna(0)
        
        return simplified_df
    
    def _simplify_ov_features(self, df):
        """OV（卵巢癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)

        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0

        # 2. 临床分期 (Clinical Stage)
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage IC': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage IIC': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Not Available]': 0
        }
        if 'clinical_stage' in df.columns:
            simplified_df['clinical_stage'] = df['clinical_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['clinical_stage'] = 0

        # 3. 肿瘤组织学分级 (Neoplasm Histologic Grade)
        grade_map = {
            'G1': 1,
            'G2': 2,
            'G3': 3,
            'G4': 4,
            'GX': 0, 'GB': 0, '[Not Available]': 0
        }
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 4. 残留病灶 (Tumor Residual Disease) - OV特有的重要预后因素
        residual_map = {
            '1-10 mm': 1,
            '11-20 mm': 2, 
            '>20 mm': 3,
            'Macroscopic': 3,
            'Microscopic': 0,
            'No Macroscopic disease': 0
        }
        if 'tumor_residual_disease' in df.columns:
            simplified_df['tumor_residual_disease'] = df['tumor_residual_disease'].map(residual_map).fillna(0).astype(int)
        else:
             simplified_df['tumor_residual_disease'] = 0
             
        # 5. 残留肿瘤 (Residual Tumor)
        residual_tumor_map = {
            'R0': 0,  # 无残留
            'R1': 1,  # 镜下残留
            'R2': 2,  # 肉眼残留
            'RX': 0   # 未评估
        }
        if 'residual_tumor' in df.columns:
            simplified_df['residual_tumor'] = df['residual_tumor'].map(residual_tumor_map).fillna(0).astype(int)
        else:
            simplified_df['residual_tumor'] = 0
        
        # 6. Karnofsky评分（数值型）
        if 'karnofsky_performance_score' in df.columns:
            simplified_df['karnofsky_performance_score'] = pd.to_numeric(df['karnofsky_performance_score'], errors='coerce').fillna(0)
        else:
            simplified_df['karnofsky_performance_score'] = 0
        
        # 7. ECOG评分（数值型）
        if 'eastern_cancer_oncology_group' in df.columns:
            simplified_df['eastern_cancer_oncology_group'] = pd.to_numeric(df['eastern_cancer_oncology_group'], errors='coerce').fillna(0)
        else:
            simplified_df['eastern_cancer_oncology_group'] = 0

        # 8. 二值化特征 (YES/NO)
        binary_features = [
            'lymphatic_invasion', 'venous_invasion', 'radiation_therapy', 
            'additional_pharmaceutical_therapy', 'additional_radiation_therapy'
        ]
        for feature in binary_features:
            if feature in df.columns:
                simplified_df[feature] = (df[feature] == 'YES').astype(int)
            else:
                simplified_df[feature] = 0

        # 填充所有可能的NaN值
        simplified_df = simplified_df.fillna(0)
        
        return simplified_df
    
    def _simplify_skcm_features(self, df):
        """SKCM（皮肤黑色素瘤）数据集的临床特征简化逻辑"""
        # 暂时使用通用逻辑，后续可根据SKCM特异性特征进行优化
        return self._simplify_general_features(df)

    def _simplify_lihc_features(self, df):
        """LIHC（肝细胞癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)

        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age'] = 0

        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
            
        # 3. 病理分期 (Pathologic Stage)
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4,
            'Stage X': 0, '[Discrepancy]': 0, '[Not Available]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0

        # 4. T分期 (保留T_stage，删除N和M因为数据太少)
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({'TX': 0})
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
        
        # 注意：删除了N_stage和M_stage，因为数据覆盖率太低（<1%）会导致过拟合

        # 5. 肝功能分级（Child-Pugh）
        if 'child_pugh_classification_grade' in df.columns:
            cp_map = {'A': 1, 'B': 2, 'C': 3}
            simplified_df['child_pugh'] = df['child_pugh_classification_grade'].map(cp_map).fillna(0).astype(int)
        else:
            simplified_df['child_pugh'] = 0
        
        # 注意：删除了ishak_score，因为所有患者数据都为0（无信息量）

        # 6. 病毒性肝炎
        if 'viral_hepatitis_serology' in df.columns:
            df['viral_hepatitis_serology'] = df['viral_hepatitis_serology'].astype(str)
            simplified_df['hep_b_positive'] = df['viral_hepatitis_serology'].str.contains('Hepatitis B', case=False, na=False).astype(int)
            simplified_df['hep_c_positive'] = df['viral_hepatitis_serology'].str.contains('Hepatitis C', case=False, na=False).astype(int)
        else:
            simplified_df['hep_b_positive'] = 0
            simplified_df['hep_c_positive'] = 0

        # 7. 实验室指标
        lab_features = [
            'albumin_result_specified_value',
            'bilirubin_upper_limit',
            'creatinine_value_in_mg_dl',
            'prothrombin_time_result_value',
            'platelet_result_count',
            'fetoprotein_outcome_value'
        ]
        for col in lab_features:
            if col in df.columns:
                simplified_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                simplified_df[col] = 0

        return simplified_df

    def _simplify_lusc_features(self, df):
        """LUSC（肺鳞癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        
        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
        
        # 3. 病理分期
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0
        
        # 4. TNM分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({'TX': 0})
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
        
        n_map = {
            'N0': 0, 'N1': 1, 'N1a': 1, 'N1b': 1,
            'N2': 2, 'N2a': 2, 'N2b': 2,
            'N3': 3, 'NX': 0
        }
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0
        
        m_map = {'M0': 0, 'M1': 1, 'M1a': 1, 'M1b': 1, 'MX': 0}
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
        
        # 5. 组织学分级
        grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'GX': 0}
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 6. 吸烟史（关键预后特征）
        if 'tobacco_smoking_history' in df.columns:
            # 吸烟史编码：1=从不吸烟, 2=戒烟<=15年, 3=戒烟>15年, 4=当前吸烟, 5=未知
            simplified_df['tobacco_smoking_history'] = pd.to_numeric(df['tobacco_smoking_history'], errors='coerce').fillna(0).astype(int)
        else:
            simplified_df['tobacco_smoking_history'] = 0
        
        return simplified_df

    def _simplify_stad_features(self, df):
        """STAD（胃腺癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        
        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
        
        # 3. 病理分期
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0
        
        # 4. TNM分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({'TX': 0})
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
        
        n_map = {
            'N0': 0, 'N1': 1, 'N1a': 1, 'N1b': 1,
            'N2': 2, 'N2a': 2, 'N2b': 2,
            'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
            'NX': 0
        }
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0
        
        m_map = {'M0': 0, 'M1': 1, 'MX': 0}
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
        
        # 5. 组织学分级
        grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'GX': 0}
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 6. 幽门螺杆菌感染
        if 'h_pylori_infection' in df.columns:
            simplified_df['h_pylori_positive'] = (df['h_pylori_infection'] == 'YES').astype(int)
        else:
            simplified_df['h_pylori_positive'] = 0
        
        return simplified_df

    def _simplify_ucec_features(self, df):
        """UCEC（子宫内膜癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        
        # 2. 性别（UCEC都是女性，但保留）
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 1
        
        # 3. 临床分期（UCEC用clinical_stage）
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage IC': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'clinical_stage' in df.columns:
            simplified_df['clinical_stage'] = df['clinical_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['clinical_stage'] = 0
        
        # 4. 组织学分级
        grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'GX': 0}
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 5. 绝经状态
        if 'menopause_status' in df.columns:
            menopause_map = {
                'Post (prior bilateral ovariectomy OR >12 mo since LMP with no prior hysterectomy)': 1,
                'Pre (<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)': 0,
                'Peri (6-12 months since last menstrual period)': 0,
                'Indeterminate (neither Pre or Postmenopausal)': 0
            }
            simplified_df['is_postmenopausal'] = df['menopause_status'].map(menopause_map).fillna(0).astype(int)
        else:
            simplified_df['is_postmenopausal'] = 0
        
        return simplified_df

    def _simplify_hnsc_features(self, df):
        """HNSC（头颈鳞癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        
        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
        
        # 3. 病理分期
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4, 'Stage IVC': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0
        
        # 4. TNM分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({'TX': 0, 'Tis': 0})
        if 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
        
        n_map = {
            'N0': 0, 'N1': 1, 'N2': 2, 'N2a': 2, 'N2b': 2, 'N2c': 2,
            'N3': 3, 'NX': 0
        }
        if 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0
        
        m_map = {'M0': 0, 'M1': 1, 'MX': 0}
        if 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
        
        # 5. 组织学分级
        grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'GX': 0}
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 6. HPV状态（合并两种检测方法）
        hpv_positive = 0
        if 'hpv_status_by_ish_testing' in df.columns:
            hpv_positive = (df['hpv_status_by_ish_testing'] == 'Positive').astype(int)
        if 'hpv_status_by_p16_testing' in df.columns:
            hpv_positive = hpv_positive | (df['hpv_status_by_p16_testing'] == 'Positive').astype(int)
        simplified_df['hpv_positive'] = hpv_positive
        
        # 7. 吸烟史（关键预后特征）
        if 'tobacco_smoking_history' in df.columns:
            # 吸烟史编码：1=从不吸烟, 2=戒烟<=15年, 3=戒烟>15年, 4=当前吸烟, 5=未知
            simplified_df['tobacco_smoking_history'] = pd.to_numeric(df['tobacco_smoking_history'], errors='coerce').fillna(0).astype(int)
        else:
            simplified_df['tobacco_smoking_history'] = 0
        
        # 8. 饮酒史（关键预后特征）
        if 'alcohol_history_documented' in df.columns:
            # 饮酒史：YES=1, NO=0
            alcohol_map = {'YES': 1, 'Yes': 1, 'NO': 0, 'No': 0}
            simplified_df['alcohol_history'] = df['alcohol_history_documented'].map(alcohol_map).fillna(0).astype(int)
        else:
            simplified_df['alcohol_history'] = 0
        
        return simplified_df

    def _simplify_paad_features(self, df):
        """PAAD（胰腺腺癌）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        
        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
        
        # 3. 病理分期
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        if 'pathologic_stage' in df.columns:
            simplified_df['pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(0).astype(int)
        else:
            simplified_df['pathologic_stage'] = 0
        
        # 4. TNM分期
        t_map = {f'T{i}': i for i in range(1, 5)}
        t_map.update({f'T{i}a': i for i in range(1, 5)})
        t_map.update({f'T{i}b': i for i in range(1, 5)})
        t_map.update({'TX': 0})
        if 'T_stage' in df.columns:  # 修正：使用T_stage而不是pathologic_T
            simplified_df['T_stage'] = pd.to_numeric(df['T_stage'], errors='coerce').fillna(0).astype(int)
        elif 'pathologic_T' in df.columns:
            simplified_df['T_stage'] = df['pathologic_T'].map(t_map).fillna(0).astype(int)
        else:
            simplified_df['T_stage'] = 0
        
        n_map = {'N0': 0, 'N1': 1, 'N1a': 1, 'N1b': 1, 'NX': 0}
        if 'N_stage' in df.columns:  # 修正：使用N_stage而不是pathologic_N
            simplified_df['N_stage'] = pd.to_numeric(df['N_stage'], errors='coerce').fillna(0).astype(int)
        elif 'pathologic_N' in df.columns:
            simplified_df['N_stage'] = df['pathologic_N'].map(n_map).fillna(0).astype(int)
        else:
            simplified_df['N_stage'] = 0
        
        m_map = {'M0': 0, 'M1': 1, 'MX': 0}
        if 'M_stage' in df.columns:  # 修正：使用M_stage而不是pathologic_M
            simplified_df['M_stage'] = pd.to_numeric(df['M_stage'], errors='coerce').fillna(0).astype(int)
        elif 'pathologic_M' in df.columns:
            simplified_df['M_stage'] = df['pathologic_M'].map(m_map).fillna(0).astype(int)
        else:
            simplified_df['M_stage'] = 0
        
        # 5. 组织学分级
        grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'GX': 0}
        if 'histologic_grade' in df.columns:  # 修正：使用histologic_grade而不是neoplasm_histologic_grade
            simplified_df['histologic_grade'] = pd.to_numeric(df['histologic_grade'], errors='coerce').fillna(0).astype(int)
        elif 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 6. 糖尿病史
        if 'has_diabetes' in df.columns:  # 修正：使用has_diabetes而不是history_of_diabetes
            simplified_df['has_diabetes'] = pd.to_numeric(df['has_diabetes'], errors='coerce').fillna(0).astype(int)
        elif 'history_of_diabetes' in df.columns:
            simplified_df['has_diabetes'] = (df['history_of_diabetes'] == 'YES').astype(int)
        else:
            simplified_df['has_diabetes'] = 0
        
        # 7. 慢性胰腺炎史
        if 'has_pancreatitis' in df.columns:  # 修正：使用has_pancreatitis
            simplified_df['has_pancreatitis'] = pd.to_numeric(df['has_pancreatitis'], errors='coerce').fillna(0).astype(int)
        elif 'history_of_chronic_pancreatitis' in df.columns:
            simplified_df['has_pancreatitis'] = (df['history_of_chronic_pancreatitis'] == 'YES').astype(int)
        else:
            simplified_df['has_pancreatitis'] = 0
        
        # 8. 吸烟史（关键预后特征）
        if 'tobacco_smoking_history' in df.columns:
            simplified_df['tobacco_smoking_history'] = pd.to_numeric(df['tobacco_smoking_history'], errors='coerce').fillna(0).astype(int)
        else:
            simplified_df['tobacco_smoking_history'] = 0
        
        # 9. 饮酒史（关键预后特征）
        if 'alcohol_history_documented' in df.columns:
            alcohol_map = {'YES': 1, 'Yes': 1, 'NO': 0, 'No': 0}
            simplified_df['alcohol_history'] = df['alcohol_history_documented'].map(alcohol_map).fillna(0).astype(int)
        else:
            simplified_df['alcohol_history'] = 0
        
        # 10. 放疗史（关键预后特征）
        if 'radiation_therapy' in df.columns:
            radiation_map = {'YES': 1, 'Yes': 1, 'NO': 0, 'No': 0}
            simplified_df['radiation_therapy'] = df['radiation_therapy'].map(radiation_map).fillna(0).astype(int)
        else:
            simplified_df['radiation_therapy'] = 0
        
        return simplified_df

    def _simplify_lgg_features(self, df):
        """LGG（低级别胶质瘤）数据集的临床特征简化逻辑"""
        simplified_df = pd.DataFrame(index=df.index)
        
        # 1. 年龄
        if 'age_at_initial_pathologic_diagnosis' in df.columns:
            simplified_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(0)
        else:
            simplified_df['age_at_initial_pathologic_diagnosis'] = 0
        
        # 2. 性别
        if 'gender' in df.columns:
            simplified_df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
        else:
            simplified_df['is_female'] = 0
        
        # 3. 组织学分级
        grade_map = {
            'G2': 2, 'G3': 3,
            'GX': 0, '[Not Available]': 0
        }
        if 'neoplasm_histologic_grade' in df.columns:
            simplified_df['histologic_grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(0).astype(int)
        else:
            simplified_df['histologic_grade'] = 0
        
        # 4. 卡氏评分
        if 'karnofsky_performance_score' in df.columns:
            simplified_df['karnofsky_performance_score'] = pd.to_numeric(df['karnofsky_performance_score'], errors='coerce').fillna(0)
        else:
            simplified_df['karnofsky_performance_score'] = 0
        
        # 5. 癫痫史
        if 'seizure_history' in df.columns:
            simplified_df['has_seizure'] = (df['seizure_history'] == 'YES').astype(int)
        else:
            simplified_df['has_seizure'] = 0
        
        # 6. 运动功能变化
        if 'motor_movement_changes' in df.columns:
            simplified_df['has_motor_changes'] = (df['motor_movement_changes'] == 'YES').astype(int)
        else:
            simplified_df['has_motor_changes'] = 0
        
        return simplified_df

    def _simplify_general_features(self, df):
        """通用数据集的临床特征简化逻辑"""
        # 选择基本的临床特征
        basic_features = ['age_at_initial_pathologic_diagnosis', 'gender', 'pathologic_stage', 
                         'pathologic_T', 'pathologic_N', 'pathologic_M', 'histological_type']
        
        available_features = [col for col in basic_features if col in df.columns]
        simplified_df = df[available_features].copy()
        
        # 性别处理
        if 'gender' in simplified_df.columns:
            simplified_df['is_female'] = (simplified_df['gender'] == 'FEMALE').astype(int)
            simplified_df.drop('gender', axis=1, inplace=True)
        
        # 处理分类变量 - 使用factorize转换为数值
        for col in simplified_df.columns:
            if simplified_df[col].dtype == 'object':
                # factorize会将分类变量转换为整数编码
                simplified_df[col] = pd.factorize(simplified_df[col])[0]
        
        # 确保所有列都是数值类型
        for col in simplified_df.columns:
            simplified_df[col] = pd.to_numeric(simplified_df[col], errors='coerce').fillna(0)
        
        return simplified_df

def main():
    """主函数，演示如何使用新的 HierarchicalGraphBuilder 类"""
    import argparse
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="构建层次化图神经网络数据")
    parser.add_argument('--dataset', type=str, default='BRCA', 
                       choices=['BRCA', 'LUAD', 'COAD', 'GBM', 'KIRC', 'LUNG', 'OV', 'SKCM', 'LIHC',
                               'LUSC', 'STAD', 'UCEC', 'HNSC', 'PAAD', 'LGG'], 
                       help='数据集名称 (默认: BRCA)')
    parser.add_argument('--input_path', type=str, default=None,
                       help='输入数据路径 (默认: moghet/data/processed/{dataset})')
    parser.add_argument('--output_path', type=str, default=None,
                       help='输出数据路径 (默认: moghet/data/processed/{dataset})')
    
    args = parser.parse_args()
    
    try:
        # 设置路径
        current_dir = osp.dirname(osp.abspath(__file__))
        project_root = osp.dirname(current_dir)
        
        # 根据数据集设置默认路径
        if args.input_path is None:
            input_path = osp.join(project_root, "data", "processed", args.dataset)
        else:
            input_path = args.input_path
            
        if args.output_path is None:
            output_path = osp.join(project_root, "data", "processed", args.dataset)
        else:
            output_path = args.output_path
        
        print(f"数据集: {args.dataset}")
        print(f"输入数据路径: {input_path}")
        print(f"输出路径: {output_path}")
        
        builder = HierarchicalGraphBuilder(input_path, output_path, dataset_name=args.dataset)
        
        # 1. 加载数据
        if not builder.load_data():
            print("数据加载失败，程序终止。")
            return
        
        # 2. 创建ID映射
        if not builder.create_id_mappings():
            print("ID映射创建失败，程序终止。")
            return
        
        # 3. (新) 预计算通路静态结构
        if not builder.precompute_pathway_structures():
            print("预计算通路结构失败，程序终止。")
            return

        # 4. (新) 生成并保存所有患者的层次化数据包
        builder.generate_and_save_hierarchical_data()
        
        print("\n处理完成！")
        print("产物清单:")
        print(f"  - ID映射: {osp.join(output_path, 'id_mappings.json')}")
        if builder.patient_features is not None:
            print(f"  - 临床数据: {osp.join(output_path, 'patient_clinical_features.csv')}")
        print(f"  - 患者数据包目录: {osp.join(output_path, 'hierarchical_patient_data/')}")

    except Exception as e:
        print(f"执行过程中出现未捕获的错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()