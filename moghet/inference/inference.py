import os
import os.path as osp
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -- 项目路径设置 --
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# -- 完成路径设置 --

from src.hierarchical_model import HierarchicalGNNModel
from src.auc_evaluation import calculate_auc_from_predictions
from train_hierarchical import HierarchicalDataset, hierarchical_collate_fn

def main():
    parser = argparse.ArgumentParser(description='使用已训练的模型进行推理并评估性能')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (例如 LUAD)')
    parser.add_argument('--fold', type=int, required=True, help='交叉验证的折数 (例如 1)')
    parser.add_argument('--results_dir', type=str, default=osp.join(project_root, "moghet", "results"),
                        help='包含所有模型训练结果的根目录')
    parser.add_argument('--data_root_dir', type=str, default=osp.join(project_root, "moghet", "data", "processed"),
                        help='处理后数据的根目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='推理设备')
    parser.add_argument('--smoke-test', action='store_true', help='使用冒烟测试生成的模型进行推理')

    args = parser.parse_args()
    
    DEVICE = torch.device(args.device)
    print(f"--- 开始推理评估 ---")
    print(f"数据集: {args.dataset}, 折数: {args.fold}, 设备: {DEVICE}")

    # --- 1. 构建路径 ---
    model_dir_suffix = "smoke_test" if args.smoke_test else "fast"
    model_dir_name = f"hierarchical_model_{args.dataset.lower()}_{model_dir_suffix}"
    fold_dir = osp.join(args.results_dir, model_dir_name, f"fold_{args.fold}")
    
    print(f"正在从以下目录加载模型和数据: {fold_dir}")
    
    model_path = osp.join(fold_dir, "best_model.pt")
    params_path = osp.join(fold_dir, "model_params.json")
    split_info_path = osp.join(fold_dir, "split_info.json")
    
    processed_data_dir = osp.join(args.data_root_dir, args.dataset)
    data_path = osp.join(processed_data_dir, "hierarchical_patient_data")
    clinical_data_path = osp.join(processed_data_dir, "patient_clinical_features.csv")
    survival_data_path = osp.join(processed_data_dir, "patient_survival.csv")

    # 检查所有文件是否存在
    for path in [model_path, params_path, split_info_path, data_path, clinical_data_path, survival_data_path]:
        if not osp.exists(path):
            print(f"错误: 必需文件或目录不存在: {path}")
            sys.exit(1)

    # --- 2. 加载模型配置和数据划分信息 ---
    with open(params_path, 'r') as f:
        model_params = json.load(f)
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)
        
    train_patient_ids = split_info['train_patient_ids']
    test_patient_ids = split_info['test_patient_ids']

    # --- 3. 加载并预处理数据 ---
    print("\n--- 正在加载和预处理数据... ---")
    survival_df = pd.read_csv(survival_data_path, index_col=0)
    clinical_df = pd.read_csv(clinical_data_path, index_col=0)

    # 为了正确地对测试集进行预处理，我们需要在对应的训练集上fit预处理器
    train_clinical_df = clinical_df.loc[train_patient_ids]
    test_clinical_df = clinical_df.loc[test_patient_ids]

    numeric_features = train_clinical_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = train_clinical_df.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
        ],
        remainder='passthrough'
    )

    # 在训练集上fit，然后在测试集上transform
    preprocessor.fit(train_clinical_df)
    test_features_processed = preprocessor.transform(test_clinical_df)
    test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features_processed)}

    print(f"测试集数据预处理完成。特征维度: {test_features_processed.shape}")

    # --- 4. 创建数据集和数据加载器 ---
    test_dataset = HierarchicalDataset(data_path, test_patient_ids, survival_df, test_features_dict)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)
    print(f"为 {len(test_dataset)} 个测试样本创建了DataLoader。")

    # --- 5. 加载模型 ---
    print("\n--- 正在加载模型... ---")
    # metadata 可能以列表形式存储在json中，需要转换为元组
    if 'metadata' in model_params and isinstance(model_params['metadata'], list):
        model_params['metadata'] = tuple(model_params['metadata'])
        
    model = HierarchicalGNNModel(**model_params).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("模型加载成功。")

    # --- 6. 执行推理和评估 ---
    print("\n--- 开始在测试集上进行推理和评估... ---")
    results = calculate_auc_from_predictions(model, test_loader, device=args.device)
    
    print("\n--- 评估结果 ---")
    print(f"C-Index (一致性指数): {results.get('c_index', 'N/A'):.4f}")
    print(f"时间依赖性 AUC (基于公式15): {results.get('time_dependent_auc', 'N/A'):.4f}")
    print(f"标准 AUC (ROC): {results.get('standard_auc', 'N/A'):.4f}")
    print(f"集成 AUC: {results.get('integrated_auc', 'N/A'):.4f}")
    print("\n各时间点AUC:")
    for t, auc in results.get('time_specific_auc', {}).items():
        print(f"  - {t} 年: {auc:.4f}")
    print("\n------------------")
    print(f"在 {results.get('num_samples', 0)} 个样本上评估, 其中包含 {results.get('num_events', 0)} 个事件。")

if __name__ == "__main__":
    main()
