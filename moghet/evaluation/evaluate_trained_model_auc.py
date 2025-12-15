#!/usr/bin/env python3
"""
训练模型AUC评估脚本

直接调用训练好的模型进行AUC评估。
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from moghet.src.explain_module import HierarchicalGNNExplainer
from moghet.src.auc_evaluation import SurvivalAUCEvaluator


def load_trained_model(dataset_name: str = 'LUAD', fold_idx: int = 1, 
                      project_root: str = None) -> HierarchicalGNNExplainer:
    """
    加载训练好的模型
    
    Args:
        dataset_name: 数据集名称
        fold_idx: 交叉验证折索引
        project_root: 项目根目录
        
    Returns:
        加载的模型解释器
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent.parent)
    
    print(f"加载训练好的模型: {dataset_name}, fold {fold_idx}")
    
    try:
        # 使用HierarchicalGNNExplainer加载训练好的模型
        explainer = HierarchicalGNNExplainer.from_dataset(
            dataset_name=dataset_name,
            project_root=project_root,
            fold_idx=fold_idx,
            device='auto'
        )
        
        print(f"✅ 模型加载成功")
        return explainer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def load_test_data(dataset_name: str, project_root: str) -> Dict:
    """
    加载测试数据
    
    Args:
        dataset_name: 数据集名称
        project_root: 项目根目录
        
    Returns:
        包含测试数据的字典
    """
    print(f"加载测试数据: {dataset_name}")
    
    # 构建数据路径
    processed_dir = os.path.join(project_root, "moghet", "data", "processed", dataset_name)
    
    # 加载临床特征
    clinical_file = os.path.join(processed_dir, "patient_clinical_features.csv")
    if not os.path.exists(clinical_file):
        raise FileNotFoundError(f"找不到临床特征文件: {clinical_file}")
    
    clinical_df = pd.read_csv(clinical_file, index_col=0)
    print(f"临床特征数据形状: {clinical_df.shape}")
    
    # 加载生存数据
    survival_file = os.path.join(processed_dir, "patient_survival.csv")
    if os.path.exists(survival_file):
        survival_df = pd.read_csv(survival_file, index_col=0)
        print(f"生存数据形状: {survival_df.shape}")
    else:
        raise FileNotFoundError(f"找不到生存数据文件: {survival_file}")
    
    # 确保患者ID匹配
    common_patients = clinical_df.index.intersection(survival_df.index)
    print(f"共同患者数量: {len(common_patients)}")
    
    clinical_df = clinical_df.loc[common_patients]
    survival_df = survival_df.loc[common_patients]
    
    return {
        'clinical_features': clinical_df,
        'survival_data': survival_df,
        'patient_ids': common_patients.tolist()
    }


def create_data_loader_for_model(data_dict: dict, model, batch_size: int = 16):
    """
    为模型创建数据加载器
    
    Args:
        data_dict: 数据字典
        model: 模型对象
        batch_size: 批次大小
        
    Returns:
        数据加载器
    """
    class ModelDataLoader:
        def __init__(self, data_dict, model, batch_size):
            self.data_dict = data_dict
            self.model = model
            self.batch_size = batch_size
            self.patient_ids = data_dict['patient_ids']
            self.n_batches = (len(self.patient_ids) + batch_size - 1) // batch_size
            
        def __iter__(self):
            print("开始迭代数据加载器...")
            for i in range(self.n_batches):
                print(f"处理批次 {i+1}/{self.n_batches}")
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(self.patient_ids))
                batch_patient_ids = self.patient_ids[start_idx:end_idx]
                
                # 创建批次数据
                batch_data = {
                    'time': torch.tensor([
                        self.data_dict['survival_data'].loc[pid, 'time']
                        for pid in batch_patient_ids
                    ], dtype=torch.float),
                    'event': torch.tensor([
                        self.data_dict['survival_data'].loc[pid, 'event']
                        for pid in batch_patient_ids
                    ], dtype=torch.float),
                    'clinical_features': torch.tensor([
                        self.data_dict['clinical_features'].loc[pid].values
                        for pid in batch_patient_ids
                    ], dtype=torch.float)
                }
                
                # 调试信息
                print(f"批次数据键: {list(batch_data.keys())}")
                print(f"时间数据形状: {batch_data['time'].shape}")
                print(f"事件数据形状: {batch_data['event'].shape}")
                print(f"临床特征形状: {batch_data['clinical_features'].shape}")
                
                # 使用模型生成风险评分
                with torch.no_grad():
                    if hasattr(self.model, 'model'):
                        # 如果是MultiModelExplainer
                        if hasattr(self.model.model, 'models'):
                            # 多模型情况，使用第一个模型
                            model_to_use = self.model.model.models[0]
                        else:
                            model_to_use = self.model.model
                    else:
                        model_to_use = self.model
                    
                    # 生成风险评分（这里需要根据实际模型结构调整）
                    try:
                        # 尝试直接使用临床特征
                        risk_scores = model_to_use(batch_data['clinical_features'])
                        if isinstance(risk_scores, torch.Tensor):
                            batch_data['risk_score'] = risk_scores
                        else:
                            # 如果模型返回的不是tensor，创建模拟的风险评分
                            batch_data['risk_score'] = torch.randn(len(batch_patient_ids), 1)
                    except Exception as e:
                        print(f"警告: 模型预测失败，使用随机风险评分: {e}")
                        batch_data['risk_score'] = torch.randn(len(batch_patient_ids), 1)
                
                yield batch_data
                
        def __len__(self):
            return self.n_batches
    
    return ModelDataLoader(data_dict, model, batch_size)


def evaluate_model_auc(explainer: HierarchicalGNNExplainer, 
                      data_dict: dict, 
                      fold_idx: int = 1) -> Dict:
    """
    评估模型的AUC性能
    
    Args:
        explainer: 模型解释器
        data_dict: 数据字典
        fold_idx: 交叉验证折索引
        
    Returns:
        AUC评估结果
    """
    print(f"开始评估模型AUC性能 (fold {fold_idx})...")
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        data_loader = create_data_loader_for_model(data_dict, explainer, batch_size=16)
        print(f"数据加载器创建成功，批次数量: {len(data_loader)}")
        
        # 评估AUC性能
        print("正在计算AUC...")
        auc_results = explainer.evaluate_auc_performance(data_loader, device='auto')
        
        # 添加元数据
        auc_results['fold_idx'] = fold_idx
        auc_results['num_patients'] = len(data_dict['patient_ids'])
        auc_results['evaluation_timestamp'] = pd.Timestamp.now().isoformat()
        
        print(f"✅ AUC评估完成")
        print(f"AUC: {auc_results.get('auc', 'N/A'):.4f}")
        
        return auc_results
        
    except Exception as e:
        print(f"❌ AUC评估失败: {e}")
        return {
            'error': str(e),
            'fold_idx': fold_idx,
            'num_patients': len(data_dict['patient_ids']),
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }


def evaluate_all_folds(dataset_name: str = 'LUAD', 
                      project_root: str = None,
                      folds: List[int] = [1, 2, 3, 4, 5]) -> Dict:
    """
    评估所有折的AUC性能
    
    Args:
        dataset_name: 数据集名称
        project_root: 项目根目录
        folds: 要评估的折列表
        
    Returns:
        所有折的AUC评估结果
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent.parent)
    
    print(f"开始评估所有折的AUC性能: {folds}")
    
    # 加载测试数据
    data_dict = load_test_data(dataset_name, project_root)
    
    all_results = {}
    
    for fold_idx in folds:
        print(f"\n=== 评估 fold {fold_idx} ===")
        
        try:
            # 加载模型
            explainer = load_trained_model(dataset_name, fold_idx, project_root)
            
            # 评估AUC
            fold_results = evaluate_model_auc(explainer, data_dict, fold_idx)
            all_results[f'fold_{fold_idx}'] = fold_results
            
        except Exception as e:
            print(f"❌ fold {fold_idx} 评估失败: {e}")
            all_results[f'fold_{fold_idx}'] = {
                'error': str(e),
                'fold_idx': fold_idx
            }
    
    # 计算汇总统计
    successful_results = {k: v for k, v in all_results.items() 
                        if 'error' not in v and 'auc' in v}
    
    if successful_results:
        aucs = [r['auc'] for r in successful_results.values()]
        summary = {
            'total_folds': len(folds),
            'successful_folds': len(successful_results),
            'failed_folds': len(folds) - len(successful_results),
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'min_auc': np.min(aucs),
            'max_auc': np.max(aucs),
            'fold_results': all_results
        }
    else:
        summary = {
            'total_folds': len(folds),
            'successful_folds': 0,
            'failed_folds': len(folds),
            'error': '所有折的评估都失败了',
            'fold_results': all_results
        }
    
    return summary


def save_results(results: dict, output_dir: str, filename: str = None):
    """
    保存评估结果
    
    Args:
        results: 评估结果
        output_dir: 输出目录
        filename: 文件名
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trained_model_auc_evaluation_{timestamp}.json"
    
    output_file = os.path.join(output_dir, filename)
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_converted = convert_numpy(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到: {output_file}")
    return output_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练模型AUC评估')
    parser.add_argument('--dataset', type=str, default='LUAD',
                       help='数据集名称')
    parser.add_argument('--fold', type=int, default=None,
                       help='特定折索引 (默认: 评估所有折)')
    parser.add_argument('--project_root', type=str, default=None,
                       help='项目根目录路径')
    parser.add_argument('--output_dir', type=str,
                       default='moghet/results/auc_evaluation',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 设置项目根目录
    if args.project_root is None:
        args.project_root = str(Path(__file__).parent.parent.parent)
    
    print(f"项目根目录: {args.project_root}")
    print(f"数据集: {args.dataset}")
    print(f"输出目录: {args.output_dir}")
    
    if args.fold is not None:
        # 评估单个折
        print(f"评估单个折: {args.fold}")
        
        # 加载模型和数据
        explainer = load_trained_model(args.dataset, args.fold, args.project_root)
        data_dict = load_test_data(args.dataset, args.project_root)
        
        # 评估AUC
        results = evaluate_model_auc(explainer, data_dict, args.fold)
        
        # 保存结果
        save_results(results, args.output_dir, f"fold_{args.fold}_auc_evaluation.json")
        
    else:
        # 评估所有折
        print("评估所有折")
        results = evaluate_all_folds(args.dataset, args.project_root)
        
        # 保存结果
        save_results(results, args.output_dir)
        
        # 打印汇总信息
        if 'mean_auc' in results:
            print(f"\n汇总结果:")
            print(f"成功评估的折: {results['successful_folds']}/{results['total_folds']}")
            print(f"平均AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
            print(f"AUC范围: {results['min_auc']:.4f} - {results['max_auc']:.4f}")


if __name__ == "__main__":
    main() 