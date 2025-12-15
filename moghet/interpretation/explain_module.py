import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
import os
import os.path as osp
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class HierarchicalGNNExplainer:
    """
    基于注意力机制的分层GNN可解释性模块。
    支持提取通路层和基因层的注意力权重。
    所有变量命名、shape、来源严格对齐HierarchicalGNNModel。
    注意：HeteroData的batch属性应通过data.batch['gene']访问。
    
    新增多数据集支持功能：
    - 支持通过命令行参数指定数据集
    - 自动加载对应数据集的模型和数据
    - 将解释结果保存到数据集特定的目录
    - 支持批量处理多个数据集
    """
    def __init__(self, model, dataset_name: Optional[str] = None, project_root: Optional[str] = None):
        """
        初始化解释器
        
        Args:
            model: 训练好的HierarchicalGNNModel模型
            dataset_name: 数据集名称 (BRCA, LUAD等)
            project_root: 项目根目录路径
        """
        self.model = model
        self.dataset_name = dataset_name
        self.project_root = project_root
        
        # 如果指定了数据集，设置相关路径
        if dataset_name and project_root:
            self._setup_dataset_paths()
    
    def _setup_dataset_paths(self):
        """设置数据集相关的路径"""
        self.processed_data_dir = osp.join(self.project_root, "moghet", "data", "processed", self.dataset_name)
        self.results_dir = osp.join(self.project_root, "moghet", "results", f"hierarchical_model_{self.dataset_name.lower()}")
        self.explanation_dir = osp.join(self.project_root, "moghet", "results", f"explanations_{self.dataset_name.lower()}")
        
        # 创建解释结果目录
        os.makedirs(self.explanation_dir, exist_ok=True)
        
        # 加载ID映射
        self.id_mappings_path = osp.join(self.processed_data_dir, "id_mappings.json")
        if osp.exists(self.id_mappings_path):
            with open(self.id_mappings_path, 'r') as f:
                self.id_mappings = json.load(f)
            
            # 创建反向映射（索引到名称）
            self.idx_to_patient_id = {v: k for k, v in self.id_mappings.get('patient_id_to_idx', {}).items()}
            self.idx_to_gene_id = {v: k for k, v in self.id_mappings.get('gene_id_to_idx', {}).items()}
            self.idx_to_pathway_id = {v: k for k, v in self.id_mappings.get('pathway_id_to_idx', {}).items()}
        else:
            self.id_mappings = {}
            self.idx_to_patient_id = {}
            self.idx_to_gene_id = {}
            self.idx_to_pathway_id = {}
            print(f"警告: 未找到ID映射文件: {self.id_mappings_path}")
    
    @classmethod
    def from_dataset(cls, dataset_name: str, project_root: str, fold_idx: Optional[int] = None, device: str = 'auto'):
        """
        从数据集创建解释器实例
        
        Args:
            dataset_name: 数据集名称 (BRCA, LUAD等)
            project_root: 项目根目录路径
            fold_idx: 交叉验证折索引 (None表示使用所有折的平均)
            device: 设备 ('auto', 'cpu', 'cuda')
            
        Returns:
            HierarchicalGNNExplainer实例
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 设置路径
        import os.path as osp
        
        # 修正：智能地查找模型目录，支持新的命名格式
        base_results_dir = osp.join(project_root, "moghet", "results")
        dataset_name_lower = dataset_name.lower()
        
        # 候选目录列表（优先使用新格式）
        candidate_dirs = [
            osp.join(base_results_dir, f"{dataset_name_lower}_model"),
            osp.join(base_results_dir, f"hierarchical_model_{dataset_name_lower}"),
            osp.join(base_results_dir, f"hierarchical_model_{dataset_name_lower}_fast")
        ]
        
        results_dir = None
        for cand_dir in candidate_dirs:
            # 检查fold_1是否存在且包含必要文件，作为目录有效性的判断依据
            check_path = osp.join(cand_dir, "fold_1" if fold_idx is None else f"fold_{fold_idx}")
            if osp.exists(check_path) and osp.exists(osp.join(check_path, "best_model.pt")):
                results_dir = cand_dir
                print(f"找到有效模型目录: {results_dir}")
                break
        
        if results_dir is None:
            raise FileNotFoundError(f"在以下位置均未找到有效模型目录: {candidate_dirs}")
        
        # 如果指定了特定折，加载该折的模型
        if fold_idx is not None:
            fold_dir = osp.join(results_dir, f"fold_{fold_idx}")
            model_path = osp.join(fold_dir, "best_model.pt")
            params_path = osp.join(fold_dir, "model_params.json")
            preprocessor_path = osp.join(fold_dir, "preprocessor.pkl")
            
            if not osp.exists(model_path):
                raise FileNotFoundError(f"找不到模型文件: {model_path}")
            if not osp.exists(params_path):
                raise FileNotFoundError(f"找不到模型参数文件: {params_path}")
            
            # 加载模型参数
            with open(params_path, 'r') as f:
                model_params = json.load(f)
            
            # 修复metadata格式：将列表转换为元组
            if 'metadata' in model_params:
                metadata = model_params['metadata']
                # 将列表转换为元组
                def list2tuple(obj):
                    if isinstance(obj, list):
                        return tuple(list2tuple(e) for e in obj)
                    return obj
                model_params['metadata'] = list2tuple(metadata)
            
            # 加载preprocessor，如果失败则优雅降级
            preprocessor = None
            if osp.exists(preprocessor_path):
                try:
                    import pickle
                    with open(preprocessor_path, 'rb') as f:
                        preprocessor = pickle.load(f)
                except Exception as e:
                    print(f"警告: 加载 preprocessor.pkl 失败 (错误: {e}). 将使用备用预处理方案。")
            
            # 创建模型
            import sys
            import os.path as osp
            # 添加moghet目录到Python路径
            moghet_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
            if moghet_dir not in sys.path:
                sys.path.insert(0, moghet_dir)
            from models.hierarchical_model import HierarchicalGNNModel
            model = HierarchicalGNNModel(**model_params)
            
            # 加载模型权重
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except RuntimeError as e:
                if "size mismatch" in str(e) and "clinical_encoder" in str(e):
                    print(f"警告: 模型权重与当前参数不匹配，尝试适配...")
                    # 获取模型权重的实际输入维度
                    state_dict = torch.load(model_path, map_location=device)
                    actual_input_dim = state_dict['clinical_encoder.0.weight'].shape[1]
                    expected_input_dim = model_params['clinical_in_features']
                    
                    print(f"模型权重期望输入维度: {actual_input_dim}")
                    print(f"当前参数输入维度: {expected_input_dim}")
                    
                    # 更新模型参数以匹配权重
                    model_params['clinical_in_features'] = actual_input_dim
                    
                    # 重新创建模型
                    model = HierarchicalGNNModel(**model_params)
                    model.load_state_dict(state_dict)
                    print(f"✅ 模型已适配到输入维度: {actual_input_dim}")
                else:
                    raise e
            
            model.to(device)
            model.eval()
            
            # 创建解释器实例并保存preprocessor
            explainer = cls(model, dataset_name, project_root)
            if preprocessor is not None:
                explainer.preprocessor = preprocessor
            explainer.fold_idx = fold_idx
            
            return explainer
        
        else:
            # 加载所有折的模型
            models = []
            for fold in range(1, 100):  # 最多检查100个折
                fold_dir = osp.join(results_dir, f"fold_{fold}")
                model_path = osp.join(fold_dir, "best_model.pt")
                params_path = osp.join(fold_dir, "model_params.json")
                
                if not osp.exists(model_path) or not osp.exists(params_path):
                    break
                
                # 加载模型参数
                with open(params_path, 'r') as f:
                    model_params = json.load(f)
                
                # 修复metadata格式：将列表转换为元组
                if 'metadata' in model_params:
                    metadata = model_params['metadata']
                    # 将列表转换为元组
                    def list2tuple(obj):
                        if isinstance(obj, list):
                            return tuple(list2tuple(e) for e in obj)
                        return obj
                    model_params['metadata'] = list2tuple(metadata)
                
                # 创建模型
                import sys
                import os.path as osp
                # 添加moghet目录到Python路径
                moghet_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
                if moghet_dir not in sys.path:
                    sys.path.insert(0, moghet_dir)
                from src.hierarchical_model import HierarchicalGNNModel
                model = HierarchicalGNNModel(**model_params)
                
                # 加载模型权重
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                
                models.append(model)
            
            if not models:
                raise FileNotFoundError(f"在 {results_dir} 中找不到任何训练好的模型")
            
            # 创建多模型解释器
            return cls(MultiModelExplainer(models), dataset_name, project_root)
    
    def load_patient_data(self, patient_id: str) -> Dict:
        """
        加载指定患者的数据
        
        Args:
            patient_id: 患者ID
            
        Returns:
            包含患者数据的字典
        """
        if not hasattr(self, 'processed_data_dir'):
            raise ValueError("请先调用 from_dataset 或设置 dataset_name 和 project_root")
        
        patient_file = osp.join(self.processed_data_dir, "hierarchical_patient_data", f"{patient_id}.pt")
        if not osp.exists(patient_file):
            raise FileNotFoundError(f"找不到患者数据文件: {patient_file}")
        
        # 加载患者数据
        data_package = torch.load(patient_file, weights_only=False)
        
        # 加载临床特征
        clinical_df = pd.read_csv(osp.join(self.processed_data_dir, "patient_clinical_features.csv"), index_col=0)
        if patient_id in clinical_df.index:
            # 使用保存的preprocessor进行特征预处理
            if hasattr(self, 'preprocessor'):
                # 使用训练时保存的preprocessor
                patient_data = clinical_df.loc[[patient_id]]
                patient_features = self.preprocessor.transform(patient_data)
                
                # 检查特征维度是否匹配模型
                if hasattr(self.model, 'clinical_encoder'):
                    expected_dim = self.model.clinical_encoder[0].in_features
                    actual_dim = patient_features.shape[1]
                    
                    if actual_dim != expected_dim:
                        print(f"特征维度不匹配: 实际{actual_dim} vs 期望{expected_dim}")
                        if actual_dim > expected_dim:
                            # 截断多余的特征
                            patient_features = patient_features[:, :expected_dim]
                            print(f"已截断特征到 {expected_dim} 维")
                        else:
                            # 填充缺失的特征
                            padding = np.zeros((1, expected_dim - actual_dim))
                            patient_features = np.concatenate([patient_features, padding], axis=1)
                            print(f"已填充特征到 {expected_dim} 维")
                
                data_package['clinical_features'] = torch.tensor(patient_features, dtype=torch.float)
            else:
                # 如果没有preprocessor，使用默认方法（兼容性）
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                
                # 分离数值型和分类型特征
                numeric_features = clinical_df.select_dtypes(include=np.number).columns.tolist()
                categorical_features = clinical_df.select_dtypes(include='object').columns.tolist()
                
                # 创建预处理器
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 
                         Pipeline([
                             ('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler())
                         ]), 
                         numeric_features),
                        ('cat', 
                         Pipeline([
                             ('imputer', SimpleImputer(strategy='most_frequent')),
                             ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                         ]), 
                         categorical_features)
                    ],
                    remainder='passthrough'
                )
                
                # 使用所有数据进行fit_transform（简化处理）
                clinical_features_processed = preprocessor.fit_transform(clinical_df)
                
                # 找到当前患者的索引
                patient_idx = clinical_df.index.get_loc(patient_id)
                patient_features = clinical_features_processed[patient_idx]
                
                data_package['clinical_features'] = torch.tensor(patient_features, dtype=torch.float).unsqueeze(0)
        else:
            raise ValueError(f"患者 {patient_id} 在临床特征文件中不存在")
        
        return data_package
    
    def explain_patient(self, patient_id: str, save_results: bool = True) -> Dict:
        """
        解释指定患者的预测结果
        
        Args:
            patient_id: 患者ID
            save_results: 是否保存结果到文件
            
        Returns:
            解释结果字典
        """
        # 加载患者数据
        data_package = self.load_patient_data(patient_id)
        
        # 使用现有的解释方法
        # 需要将图列表转换为批处理格式
        from torch_geometric.data import Batch
        
        # 定义collate函数
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
        
        # 创建单个患者的数据包列表
        data_packages = [data_package]
        
        # 使用collate函数处理数据
        batch_data = hierarchical_collate_fn(data_packages)
        
        if batch_data is None:
            raise ValueError("数据批处理失败")
        
        # 修正：将所有输入张量移动到模型所在的设备
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            # 模型没有参数，使用 'cpu' 作为默认设备
            device = 'cpu'
            
        graphs_batch = batch_data['graphs_batch'].to(device)
        clinical_features = batch_data['clinical_features'].to(device)
        
        # 使用批处理后的数据进行解释
        results = self.explain_full_forward(graphs_batch, clinical_features)
        
        # 添加患者ID和数据集信息
        results['patient_id'] = patient_id
        results['dataset'] = self.dataset_name
        
        # 保存结果
        if save_results and hasattr(self, 'explanation_dir'):
            self._save_explanation_results(patient_id, results)
        
        return results
    
    def explain_multiple_patients(self, patient_ids: List[str], save_results: bool = True) -> Dict[str, Dict]:
        """
        解释多个患者的预测结果
        
        Args:
            patient_ids: 患者ID列表
            save_results: 是否保存结果到文件
            
        Returns:
            患者ID到解释结果的映射
        """
        results = {}
        
        for patient_id in patient_ids:
            try:
                print(f"正在解释患者 {patient_id}...")
                patient_results = self.explain_patient(patient_id, save_results)
                results[patient_id] = patient_results
            except Exception as e:
                print(f"解释患者 {patient_id} 时出错: {e}")
                results[patient_id] = {'error': str(e)}
        
        # 保存批量结果摘要
        if save_results and hasattr(self, 'explanation_dir'):
            self._save_batch_summary(results)
        
        return results
    
    def _save_explanation_results(self, patient_id: str, results: Dict):
        """保存单个患者的解释结果"""
        output_file = osp.join(self.explanation_dir, f"explanation_{patient_id}.json")
        
        # 转换tensor为可序列化的格式
        serializable_results = self._convert_tensors_to_lists(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"解释结果已保存到: {output_file}")
    
    def _save_batch_summary(self, results: Dict[str, Dict]):
        """保存批量解释结果摘要"""
        summary = {
            'dataset': self.dataset_name,
            'total_patients': len(results),
            'successful_explanations': len([r for r in results.values() if 'error' not in r]),
            'failed_explanations': len([r for r in results.values() if 'error' in r]),
            'patient_ids': list(results.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        summary_file = osp.join(self.explanation_dir, "batch_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"批量解释摘要已保存到: {summary_file}")
    
    def _convert_tensors_to_lists(self, obj):
        """递归转换tensor为可序列化的列表，并处理tuple key。"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            # 修正：检查键是否为元组，如果是，则转换为字符串
            return {str(k) if isinstance(k, tuple) else k: self._convert_tensors_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_lists(item) for item in obj]
        else:
            return obj

    @torch.no_grad()
    def explain_pathway_gene_importance(self, x, batch_map):
        """
        返回每个通路内每个基因的注意力权重（基因对通路向量的贡献）。
        Args:
            x: 所有基因的特征 [N_genes, feat_dim]，应为x_dict['gene']
            batch_map: 每个基因对应的通路索引 [N_genes]，应为data.batch['gene']
        Returns:
            gene_attention_weights: [batch_size, max_genes, 1]
            mask: [batch_size, max_genes]
        """
        if x is None or batch_map is None:
            raise ValueError("x和batch_map不能为空，需分别为x_dict['gene']和data.batch['gene']")
        
        # 修复：明确指定max_num_nodes，确保所有基因都被处理
        max_num_nodes = x.shape[0]  # 使用实际的基因总数
        x_dense, mask = to_dense_batch(x, batch_map, max_num_nodes=max_num_nodes)
        
        attention_scores = self.model.intra_pathway_pooling.attention_net(x_dense)
        # ====== 调试输出 attention_scores ======
        print('attention_scores shape:', attention_scores.shape)
        for i in range(min(attention_scores.shape[0], 5)):  # 只显示前5个通路
            pathway_name = self.idx_to_pathway_id.get(i, f"通路{i}")
            print(f'{pathway_name} attention_scores: min={attention_scores[i].min().item()}, max={attention_scores[i].max().item()}, mean={attention_scores[i].mean().item()}')
        attention_scores[~mask] = -1e9
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights, mask

    @torch.no_grad()
    def explain_patient_pathway_importance(self, path_vectors, batch_map):
        """
        返回每个病人每个通路的注意力权重（通路对病人风险预测的贡献）。
        Args:
            path_vectors: 所有通路的向量 [N_pathways, feat_dim]，应为intra_pathway_pooling输出
            batch_map: 每个通路对应的病人索引 [N_pathways]，应为data.pathway_to_patient_batch_map
        Returns:
            pathway_attention_weights: [batch_size, max_pathways, 1]
            mask: [batch_size, max_pathways]
        """
        if path_vectors is None or batch_map is None:
            raise ValueError("path_vectors和batch_map不能为空，需分别为intra_pathway_pooling输出和data.pathway_to_patient_batch_map")
        
        # 修复：明确指定max_num_nodes，确保所有通路都被处理
        max_num_nodes = path_vectors.shape[0]  # 使用实际的通路总数
        x_dense, mask = to_dense_batch(path_vectors, batch_map, max_num_nodes=max_num_nodes)
        
        attention_scores = self.model.inter_pathway_pooling.attention_net(x_dense)
        attention_scores[~mask] = -1e9
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights, mask

    @torch.no_grad()
    def explain_full_forward(self, data, clinical_features):
        """
        完整复现模型前向传播，返回关键中间特征和注意力权重，便于解释性分析。
        Args:
            data: HeteroData batch，包含所有通路子图，需包含：
                - x_dict, edge_index_dict
                - data.batch['gene']
                - data.pathway_to_patient_batch_map
                - data.pathway_idx
            clinical_features: [batch_size, clinical_dim]
        Returns:
            dict，包含：
                - gene_fused_features
                - intra_gnn_out
                - path_vectors
                - gene_attention_weights, gene_mask
                - patient_omics_vector
                - pathway_attention_weights, pathway_mask
                - risk_score
        """
        self.model.eval()
        # 检查关键属性
        for attr in ['x_dict', 'edge_index_dict', 'pathway_to_patient_batch_map', 'pathway_idx']:
            if not hasattr(data, attr):
                raise AttributeError(f"输入data缺少关键属性: {attr}")
        if not hasattr(data, 'batch') or 'gene' not in data.batch:
            raise AttributeError("输入data.batch缺少'gene'属性")
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        gene_batch_map = data.batch['gene']
        pathway_to_patient_batch_map = data.pathway_to_patient_batch_map
        pathway_indices_in_batch = data.pathway_idx

        # 临床特征编码
        patient_clinical_vector = self.model.clinical_encoder(clinical_features)
        clinical_context_per_pathway = patient_clinical_vector[pathway_to_patient_batch_map]
        clinical_context_per_gene = clinical_context_per_pathway[gene_batch_map]

        # 通路身份与状态
        pathway_identity_vec = self.model.pathway_embedding(pathway_indices_in_batch)
        modulator_features = x_dict['gene'][:, 1:3]
        pathway_state_vec = global_mean_pool(modulator_features, gene_batch_map)
        full_pathway_context = torch.cat([pathway_identity_vec, pathway_state_vec], dim=1)
        pathway_context_encoded = self.model.pathway_encoder(full_pathway_context)
        pathway_context_per_gene = pathway_context_encoded[gene_batch_map]

        # HOM调制
        modulated_expression, gamma, beta = self.model.hom_modulation(
            x_dict['gene'], pathway_context_per_gene, clinical_context_per_gene)

        # 高维特征拼接
        fused_features = torch.cat([
            modulated_expression,                 # [N_genes, 1]
            x_dict['gene'][:, 0:1],               # [N_genes, 1] raw_expression
            x_dict['gene'][:, 1:3],               # [N_genes, 2] cnv, mut
            gamma,                                # [N_genes, 1]
            beta,                                 # [N_genes, 1]
            pathway_identity_vec[gene_batch_map], # [N_genes, pathway_emb_dim]
            pathway_state_vec[gene_batch_map]     # [N_genes, 2]
        ], dim=1)
        x_dict['gene'] = fused_features

        # 通路内GNN
        x_dict = self.model.intra_pathway_gnn1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = self.model.intra_pathway_gnn2(x_dict, edge_index_dict)
        x = x_dict['gene']

        # 通路内注意力池化
        path_vectors = self.model.intra_pathway_pooling(x, gene_batch_map)
        gene_attention_weights, gene_mask = self.explain_pathway_gene_importance(x, gene_batch_map)

        # 通路间注意力池化
        patient_omics_vector = self.model.inter_pathway_pooling(path_vectors, pathway_to_patient_batch_map)
        pathway_attention_weights, pathway_mask = self.explain_patient_pathway_importance(path_vectors, pathway_to_patient_batch_map)

        # 最终风险预测
        final_vector = torch.cat([patient_omics_vector, patient_clinical_vector], dim=1)
        risk_score = self.model.survival_head(final_vector)

        # 获取通路和基因的标准名称
        pathway_names = []
        for i in range(path_vectors.shape[0]):
            pathway_name = self.idx_to_pathway_id.get(i, f"通路{i}")
            pathway_names.append(pathway_name)
        
        return {
            'gene_fused_features': fused_features,
            'intra_gnn_out': x,
            'path_vectors': path_vectors,
            'gene_attention_weights': gene_attention_weights,
            'gene_mask': gene_mask,
            'patient_omics_vector': patient_omics_vector,
            'pathway_attention_weights': pathway_attention_weights,
            'pathway_mask': pathway_mask,
            'risk_score': risk_score,
            'pathway_names': pathway_names,
            'pathway_indices': pathway_indices_in_batch.tolist(),
            'edge_index_dict': data.edge_index_dict  # 新增：返回边索引字典
        }

    @torch.no_grad()
    def _mean_pool_by_index(self, vectors: torch.Tensor, group_indices: torch.Tensor) -> torch.Tensor:
        """按索引进行均值池化。vectors [N, D], group_indices [N] -> [num_groups, D]"""
        if vectors.numel() == 0:
            return vectors
        num_groups = int(group_indices.max().item()) + 1 if group_indices.numel() > 0 else 0
        device = vectors.device
        dim = vectors.size(1) if vectors.dim() > 1 else 1
        sums = torch.zeros((num_groups, dim), device=device, dtype=vectors.dtype)
        sums.index_add_(0, group_indices, vectors)
        counts = torch.bincount(group_indices, minlength=num_groups).clamp(min=1).to(device).unsqueeze(1)
        means = sums / counts
        return means

    @torch.no_grad()
    def _forward_with_module_toggle(self, data, clinical_features, disable: str = None) -> torch.Tensor:
        """
        复现前向传播，并可选择关闭某个模块。
        disable 可选: 'hom', 'pathway_identity', 'pathway_state', 'intra_attention', 'inter_attention', 'clinical'
        返回: 风险分数张量 shape [batch_size, 1]
        """
        model = self.model
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        gene_batch_map = data.batch['gene']

        # 临床特征编码（可关闭）
        if disable == 'clinical':
            patient_clinical_vector = torch.zeros(
                (clinical_features.size(0), model.clinical_encoder[0].out_features),
                device=clinical_features.device, dtype=clinical_features.dtype
            )
        else:
            patient_clinical_vector = model.clinical_encoder(clinical_features)

        clinical_context_per_pathway = patient_clinical_vector[data.pathway_to_patient_batch_map]
        clinical_context_per_gene = clinical_context_per_pathway[gene_batch_map]

        # 通路身份与状态（可关闭identity/state）
        pathway_indices_in_batch = data.pathway_idx
        if disable == 'pathway_identity':
            pathway_identity_vec = torch.zeros(
                (pathway_indices_in_batch.size(0), model.pathway_embedding.embedding_dim),
                device=clinical_features.device, dtype=clinical_features.dtype
            )
        else:
            pathway_identity_vec = model.pathway_embedding(pathway_indices_in_batch)

        modulator_features = x_dict['gene'][:, 1:3]
        if disable == 'pathway_state':
            pathway_state_vec = torch.zeros(
                (pathway_indices_in_batch.size(0), modulator_features.size(1)),
                device=clinical_features.device, dtype=clinical_features.dtype
            )
        else:
            pathway_state_vec = global_mean_pool(modulator_features, gene_batch_map)

        full_pathway_context = torch.cat([pathway_identity_vec, pathway_state_vec], dim=1)
        pathway_context_encoded = model.pathway_encoder(full_pathway_context)
        pathway_context_per_gene = pathway_context_encoded[gene_batch_map]

        # HOM 调制（可关闭）
        if disable == 'hom':
            modulated_expression = x_dict['gene'][:, 0:1]
            gamma = torch.zeros_like(modulated_expression)
            beta = torch.zeros_like(modulated_expression)
        else:
            modulated_expression, gamma, beta = model.hom_modulation(
                x_dict['gene'], pathway_context_per_gene, clinical_context_per_gene
            )

        # 高维特征拼接
        fused_features = torch.cat([
            modulated_expression,
            x_dict['gene'][:, 0:1],
            x_dict['gene'][:, 1:3],
            gamma,
            beta,
            pathway_identity_vec[gene_batch_map],
            pathway_state_vec[gene_batch_map]
        ], dim=1)
        x_dict_local = {'gene': fused_features}

        # Intra-Pathway GNN
        x_dict_local = model.intra_pathway_gnn1({'gene': x_dict_local['gene']}, edge_index_dict)
        x_dict_local = {key: F.elu(x) for key, x in x_dict_local.items()}
        x_dict_local = model.intra_pathway_gnn2(x_dict_local, edge_index_dict)
        x_gene = x_dict_local['gene']

        # Intra pooling（可替换为均值）
        if disable == 'intra_attention':
            path_vectors = self._mean_pool_by_index(x_gene, gene_batch_map)
        else:
            path_vectors = model.intra_pathway_pooling(x_gene, gene_batch_map)

        # Inter pooling（可替换为均值）
        if disable == 'inter_attention':
            patient_omics_vector = self._mean_pool_by_index(path_vectors, data.pathway_to_patient_batch_map)
        else:
            patient_omics_vector = model.inter_pathway_pooling(path_vectors, data.pathway_to_patient_batch_map)

        # 最终风险
        final_vector = torch.cat([patient_omics_vector, patient_clinical_vector], dim=1)
        risk_score = model.survival_head(final_vector)
        return risk_score

    @torch.no_grad()
    def compute_patient_module_importance_from_batch(self, graphs_batch, clinical_features) -> dict:
        """基于一个已准备好的batch，计算各模块的重要性(输出变化量)。"""
        base_out = self.explain_full_forward(graphs_batch, clinical_features)
        base_risk = base_out['risk_score']  # [B,1]
        module_list = ['hom', 'pathway_identity', 'pathway_state', 'intra_attention', 'inter_attention', 'clinical']
        contributions = {}
        for m in module_list:
            toggled_risk = self._forward_with_module_toggle(graphs_batch, clinical_features, disable=m)
            # 使用绝对变化量作为重要性
            delta = (base_risk - toggled_risk).abs().squeeze().cpu().numpy()
            contributions[m] = delta
        return {
            'baseline_risk': base_risk.squeeze().cpu().numpy(),
            'module_contributions': contributions
        }

    def explain_patient_module_importance(self, patient_id: str) -> dict:
        """加载单个患者，返回各模块重要性及基线风险。"""
        data_package = self.load_patient_data(patient_id)
        # 复用本文件中的collate函数
        from torch_geometric.data import Batch
        def hierarchical_collate_fn(data_packages):
            if not data_packages:
                return None
            list_of_graph_lists = [pkg['intra_pathway_graphs'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
            if not list_of_graph_lists:
                return None
            clinical_features_list = [pkg['clinical_features'] for pkg in data_packages if pkg.get('intra_pathway_graphs')]
            # 时间/事件占位
            time_list = [torch.tensor(0.0, dtype=torch.float) for _ in list_of_graph_lists]
            event_list = [torch.tensor(0.0, dtype=torch.float) for _ in list_of_graph_lists]
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

        batch = hierarchical_collate_fn([data_package])
        if batch is None:
            raise ValueError('无法构建患者批次数据')
        device = next(self.model.parameters()).device
        graphs_batch = batch['graphs_batch'].to(device)
        clinical_features = batch['clinical_features'].to(device)
        return self.compute_patient_module_importance_from_batch(graphs_batch, clinical_features)


class MultiModelExplainer:
    """
    多模型解释器，用于处理多个交叉验证折的模型
    """
    def __init__(self, models: List):
        self.models = models
        self.num_models = len(models)
    
    def __getattr__(self, name):
        """代理到第一个模型，用于兼容性"""
        if hasattr(self.models[0], name):
            return getattr(self.models[0], name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def explain_full_forward(self, data, clinical_features):
        """使用所有模型进行解释，返回平均结果"""
        all_results = []
        
        for i, model in enumerate(self.models):
            explainer = HierarchicalGNNExplainer(model)
            result = explainer.explain_full_forward(data, clinical_features)
            all_results.append(result)
        
        # 计算平均结果
        avg_result = {}
        for key in all_results[0].keys():
            if isinstance(all_results[0][key], torch.Tensor):
                # 对tensor求平均
                stacked = torch.stack([r[key] for r in all_results])
                avg_result[key] = stacked.mean(dim=0)
            else:
                # 对于非tensor，使用第一个结果
                avg_result[key] = all_results[0][key]
        
        return avg_result


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='MOGHET模型解释器')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['BRCA', 'LUAD'], 
                       help='要解释的数据集')
    parser.add_argument('--project_root', type=str, default=None,
                       help='项目根目录路径 (默认: 当前目录)')
    parser.add_argument('--fold', type=int, default=None,
                       help='交叉验证折索引 (默认: 使用所有折的平均)')
    parser.add_argument('--patient_id', type=str, default=None,
                       help='要解释的患者ID (默认: 解释所有患者)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='是否保存解释结果到文件')
    
    args = parser.parse_args()
    
    # 设置项目根目录
    if args.project_root is None:
        args.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    
    print(f"开始解释数据集: {args.dataset}")
    print(f"项目根目录: {args.project_root}")
    print(f"设备: {args.device}")
    
    try:
        # 创建解释器
        explainer = HierarchicalGNNExplainer.from_dataset(
            dataset_name=args.dataset,
            project_root=args.project_root,
            fold_idx=args.fold,
            device=args.device
        )
        
        # 加载患者列表
        clinical_df = pd.read_csv(
            osp.join(args.project_root, "moghet", "data", "processed", args.dataset, "patient_clinical_features.csv"), 
            index_col=0
        )
        patient_ids = clinical_df.index.tolist()
        
        if args.patient_id:
            # 解释单个患者
            if args.patient_id not in patient_ids:
                print(f"错误: 患者 {args.patient_id} 不在数据集中")
                return
            
            print(f"正在解释患者: {args.patient_id}")
            results = explainer.explain_patient(args.patient_id, args.save_results)
            print(f"解释完成，风险分数: {results['risk_score'].item():.4f}")
            
        else:
            # 解释所有患者
            print(f"正在解释 {len(patient_ids)} 个患者...")
            results = explainer.explain_multiple_patients(patient_ids, args.save_results)
            
            successful = len([r for r in results.values() if 'error' not in r])
            print(f"解释完成: {successful}/{len(patient_ids)} 个患者成功")
            
            # 显示一些统计信息
            if successful > 0:
                risk_scores = [r['risk_score'].item() for r in results.values() if 'error' not in r]
                print(f"风险分数统计: 平均={np.mean(risk_scores):.4f}, 标准差={np.std(risk_scores):.4f}")
        
    except Exception as e:
        print(f"解释过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 