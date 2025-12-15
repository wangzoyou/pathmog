import os
import os.path as osp
import sys
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines.utils import concordance_index
import pandas as pd
from tqdm import tqdm
import argparse

# 项目路径设置
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.hierarchical_model import HierarchicalGNNModel
from src.auc_evaluation import SurvivalAUCEvaluator, calculate_auc_from_predictions

# 复用原有的数据集类和其他函数
from train_hierarchical import HierarchicalDataset, hierarchical_collate_fn, cox_loss, train_one_epoch, evaluate, monitor_tensor_stats, check_model_parameters

def set_seed(seed):
    """设置所有随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='训练层次化GNN模型 (极速版本)')
    parser.add_argument('--dataset', type=str, default='LUAD', 
                       choices=['BRCA', 'LUAD', 'COAD', 'GBM', 'KIRC', 'LUNG', 'OV', 'SKCM', 'LIHC'], 
                       help='要使用的数据集 (默认: LUAD)')
    parser.add_argument('--seed', type=int, default=123, help='设置随机种子 (默认: 123)')
    parser.add_argument('--smoke-test', action='store_true', help='运行冒烟测试，使用少量数据和周期')
    args = parser.parse_args()
    
    # --- 设置全局随机种子 ---
    SEED = args.seed
    set_seed(SEED)
    
    dataset_name = args.dataset
    print(f"--- 开始训练 HierarchicalGNNModel (数据集: {dataset_name}, 极速版本) ---")
    
    # 设置路径
    processed_data_dir = osp.join(project_root, "moghet", "data", "processed", dataset_name)
    id_mappings_path = osp.join(processed_data_dir, "id_mappings.json")
    
    with open(id_mappings_path, 'r') as f:
        id_mappings = json.load(f)
    num_pathways = len(id_mappings['pathway_id_to_idx'])
    
    # 完全照抄原版配置，只改batch_size
    config = {
        "data_path": osp.join(processed_data_dir, "hierarchical_patient_data"),
        "clinical_data_path": osp.join(processed_data_dir, "patient_clinical_features.csv"),
        "survival_data_path": osp.join(processed_data_dir, "patient_survival.csv"),
        "output_dir": osp.join(project_root, "moghet", "results", f"hierarchical_model_{dataset_name.lower()}_fast"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 200,
        "batch_size": 32,  # 只改这里，从4到16
        "lr": 1e-4,
        "n_splits": 5,
        "test_size": 0.2,
        "random_state": SEED,
        "accumulation_steps": 4,
        "model_params": {
            "num_pathways": num_pathways,
            "pathway_embedding_dim": 8,
            "gnn_hidden_channels": 64,
            "pathway_out_channels": 128,
            "intra_attention_hidden_channels": 128,
            "inter_attention_hidden_channels": 128,
            "clinical_hidden_channels": 32,
            "final_hidden_channels": 64
        },
        "early_stopping": {
            "patience": 20,
            "min_delta": 0.001,
        }
    }
    
    # 如果是冒烟测试，覆盖部分配置
    if args.smoke_test:
        print("\n--- [警告] 正在以冒烟测试模式运行 ---")
        config["epochs"] = 2
        config["n_splits"] = 2 # 使用2折以加快速度
        config["output_dir"] = osp.join(project_root, "moghet", "results", f"hierarchical_model_{dataset_name.lower()}_smoke_test")

    os.makedirs(config["output_dir"], exist_ok=True)
    DEVICE = torch.device(config["device"])
    
    print(f"使用设备: {DEVICE}")
    print(f"数据集: {dataset_name}")
    print(f"输出目录: {config['output_dir']}")
    print(f"批次大小: {config['batch_size']} (原版4，提升4倍)")
    
    # 数据加载和预处理
    print("\n--- 数据加载和预处理 ---")
    
    survival_df = pd.read_csv(config["survival_data_path"], index_col=0)
    clinical_df = pd.read_csv(config["clinical_data_path"], index_col=0)
    
    print(f"生存数据患者数量: {len(survival_df)}")
    print(f"临床数据患者数量: {len(clinical_df)}")
    
    # 数据对齐
    survival_patients = set(survival_df.index)
    clinical_patients = set(clinical_df.index)
    common_patients = survival_patients.intersection(clinical_patients)
    all_patient_ids = np.array(sorted(list(common_patients)))
    
    # 调试：打印从CSV加载的ID
    print(f"--- [调试] 从CSV中找到 {len(all_patient_ids)} 个共同患者. 示例: {all_patient_ids[:3].tolist() if len(all_patient_ids) > 0 else '无'}")

    # 验证数据文件存在
    data_files_path = config["data_path"]
    if not osp.exists(data_files_path):
        print(f"--- [错误] 数据路径不存在: {data_files_path}")
        sys.exit(1)
    
    existing_patient_files = os.listdir(data_files_path)
    existing_patient_ids = {f.replace('.pt', '') for f in existing_patient_files if f.endswith('.pt')}
    
    # 调试：打印从文件名加载的ID
    print(f"--- [调试] 从 {data_files_path} 找到 {len(existing_patient_ids)} 个 .pt 文件. 示例: {list(existing_patient_ids)[:3] if len(existing_patient_ids) > 0 else '无'}")

    all_patient_ids_before_filter = all_patient_ids.copy()
    all_patient_ids = np.array([pid for pid in all_patient_ids if pid in existing_patient_ids])

    # 如果直接匹配失败，尝试转换ID格式进行匹配
    if len(all_patient_ids) == 0 and len(all_patient_ids_before_filter) > 0 and len(existing_patient_ids) > 0:
        print("--- [警告] 直接ID匹配失败。尝试转换ID格式 (例如 TCGA-XX-YYYY -> TCGA.XX.YYYY)...")
        # 转换CSV中的ID格式
        all_patient_ids_transformed = np.array([pid.replace('-', '.') for pid in all_patient_ids_before_filter])
        
        # 使用转换后的ID进行过滤，但保留原始ID
        all_patient_ids = np.array([
            original_pid
            for original_pid, transformed_pid in zip(all_patient_ids_before_filter, all_patient_ids_transformed)
            if transformed_pid in existing_patient_ids
        ])
        
        if len(all_patient_ids) > 0:
            print(f"--- [成功] 转换ID格式后，成功匹配 {len(all_patient_ids)} 个患者。")
        else:
            print("--- [失败] 转换ID格式后仍然无法匹配。请手动检查ID格式。")

    # 如果最终患者数量为0，则退出
    if len(all_patient_ids) == 0:
        print("--- [错误] 数据对齐后无可用患者，程序终止。请检查数据文件和ID。")
        sys.exit(1)
    
    survival_df = survival_df.loc[all_patient_ids]
    clinical_df = clinical_df.loc[all_patient_ids]
    
    # 如果是冒烟测试，只取一小部分数据
    if args.smoke_test:
        sample_size = min(len(all_patient_ids), 64)
        all_patient_ids = all_patient_ids[:sample_size]
        survival_df = survival_df.iloc[:sample_size]
        clinical_df = clinical_df.iloc[:sample_size]
        print(f"--- [冒烟测试] 数据集已被缩减为 {len(all_patient_ids)} 个样本 ---")

    print(f"✅ 最终数据规模: {len(survival_df)} 患者")
    
    # 数据集划分
    print("\n--- 开始数据集划分 ---")
    
    train_val_ids, test_patient_ids = train_test_split(
        all_patient_ids,
        test_size=config["test_size"],
        random_state=config["random_state"],
        shuffle=True
    )
    
    print(f"总数据集划分为: 训练/验证集 {len(train_val_ids)} | 最终测试集 {len(test_patient_ids)}")
    
    # 交叉验证
    kf = KFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    
    test_c_indices = []
    test_aucs = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_ids)):
        print(f"\n--- 开始第 {fold + 1}/{config['n_splits']} 折交叉验证 ---")
        
        train_patient_ids = train_val_ids[train_idx]
        val_patient_ids = train_val_ids[val_idx]
        
        print(f"当前折划分: 训练集 {len(train_patient_ids)} | 验证集 {len(val_patient_ids)} | 测试集 {len(test_patient_ids)}")
        
        # 创建输出目录
        fold_output_dir = osp.join(config["output_dir"], f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # --- 新增代码：保存数据划分信息 ---
        split_info = {
            "fold": fold + 1,
            "train_patient_ids": train_patient_ids.tolist(),
            "val_patient_ids": val_patient_ids.tolist(),
            "test_patient_ids": test_patient_ids.tolist(),
        }
        with open(osp.join(fold_output_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"[OK] 已保存fold {fold+1}的数据划分信息到: {osp.join(fold_output_dir, 'split_info.json')}")
        # --- 新增代码结束 ---
        
        # 特征预处理
        print(f"\n--- [Fold {fold + 1}] 开始特征预处理 ---")
        
        # === 添加缺失值检查和预处理 ===
        # 获取当前折的训练、验证、测试临床特征
        train_clinical_df = clinical_df.loc[train_patient_ids]
        val_clinical_df = clinical_df.loc[val_patient_ids]
        test_clinical_df = clinical_df.loc[test_patient_ids]
        
        # 检查原始特征中的缺失值
        print(f"原始特征缺失值统计:")
        print(f"  训练集: {train_clinical_df.isnull().sum().sum()} 个缺失值")
        print(f"  验证集: {val_clinical_df.isnull().sum().sum()} 个缺失值")
        print(f"  测试集: {test_clinical_df.isnull().sum().sum()} 个缺失值")
        
        # 分离数值型和分类型特征
        numeric_features = train_clinical_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = train_clinical_df.select_dtypes(include='object').columns.tolist()
        
        print(f"数值型特征: {len(numeric_features)} 个")
        print(f"分类型特征: {len(categorical_features)} 个")
        
        # 创建预处理器（仅在训练集上fit）
        from sklearn.impute import SimpleImputer
        
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
        
        # 核心：仅在训练数据上进行 fit_transform
        train_features_processed = preprocessor.fit_transform(train_clinical_df)
        # 核心：在验证和测试数据上只进行 transform
        val_features_processed = preprocessor.transform(val_clinical_df)
        test_features_processed = preprocessor.transform(test_clinical_df)
        
        # === 添加缺失值断言 ===
        assert not np.isnan(train_features_processed).any(), f"训练集特征仍有NaN，位置: {np.where(np.isnan(train_features_processed))}"
        assert not np.isnan(val_features_processed).any(), f"验证集特征仍有NaN，位置: {np.where(np.isnan(val_features_processed))}"
        assert not np.isnan(test_features_processed).any(), f"测试集特征仍有NaN，位置: {np.where(np.isnan(test_features_processed))}"
        
        print(f"✅ 特征预处理完成，无NaN值")
        print(f"  训练集特征维度: {train_features_processed.shape}")
        print(f"  验证集特征维度: {val_features_processed.shape}")
        print(f"  测试集特征维度: {test_features_processed.shape}")
        
        # 将处理后的特征转换为字典格式
        train_features_dict = {pid: f for pid, f in zip(train_patient_ids, train_features_processed)}
        val_features_dict = {pid: f for pid, f in zip(val_patient_ids, val_features_processed)}
        test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features_processed)}
        
        train_dataset = HierarchicalDataset(config["data_path"], train_patient_ids, survival_df, train_features_dict)
        val_dataset = HierarchicalDataset(config["data_path"], val_patient_ids, survival_df, val_features_dict)
        test_dataset = HierarchicalDataset(config["data_path"], test_patient_ids, survival_df, test_features_dict)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=hierarchical_collate_fn, num_workers=128)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=128)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=128)
        
        # 创建模型
        first_graph = train_dataset[0]['intra_pathway_graphs'][0]
        metadata = first_graph.metadata()
        clinical_in_features = train_dataset[0]['clinical_features'].shape[-1]
        
        model = HierarchicalGNNModel(
            metadata=metadata,
            clinical_in_features=clinical_in_features,
            **config["model_params"]
        ).to(DEVICE)
        
        # --- 新增代码：保存每个fold的模型参数 ---
        print(f"\n--- [Fold {fold + 1}] 保存模型参数 ---")
        fold_params = {
            "num_pathways": num_pathways,
            "metadata": metadata,
            "clinical_in_features": clinical_in_features,
            **config["model_params"]
        }
        with open(osp.join(fold_output_dir, "model_params.json"), "w") as f:
            json.dump(fold_params, f, indent=2)
        print(f"[OK] 已保存模型参数到: {osp.join(fold_output_dir, 'model_params.json')}")
        # --- 新增代码结束 ---
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        
        # 训练循环
        print(f"\n--- [Fold {fold + 1}] 开始训练 ---")
        
        best_val_c_index = 0
        best_val_auc = 0
        best_epoch = 0
        early_stopping_counter = 0
        patience = config["early_stopping"]["patience"]
        min_delta = config["early_stopping"]["min_delta"]
        stopped_early = False
        
        for epoch in range(config["epochs"]):
            # === 训练前参数监控 ===
            if (epoch % 5 == 0) or (epoch == config["epochs"] - 1):
                print(f"[监控] Epoch {epoch+1} 开始前模型参数统计：")
                check_model_parameters(model, verbose=True)

            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, config["accumulation_steps"])
            val_results = evaluate(model, val_loader, DEVICE)
            
            val_c_index = val_results['c_index']
            val_auc = val_results['auc']
            
            # === 训练后参数监控 ===
            if (epoch % 5 == 0) or (epoch == config["epochs"] - 1):
                print(f"[监控] Epoch {epoch+1} 结束后模型参数统计：")
                check_model_parameters(model, verbose=True)
                print(f"[监控] 验证集C-index/AUC分布：")
                monitor_tensor_stats(torch.tensor([val_c_index]), name="val_c_index", verbose=True)
                monitor_tensor_stats(torch.tensor([val_auc]), name="val_auc", verbose=True)

            print(f"Epoch {epoch+1:02d}/{config['epochs']:02d} | 训练损失: {train_loss:.4f} | 验证 C-Index: {val_c_index:.4f} | 验证 AUC: {val_auc:.4f}")
            
            if val_c_index > best_val_c_index + min_delta:
                best_val_c_index = val_c_index
                best_val_auc = val_auc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), osp.join(fold_output_dir, "best_model.pt"))
                print(f"  -> 新的最佳模型已保存 (Epoch: {best_epoch}, C-Index: {best_val_c_index:.4f}, AUC: {best_val_auc:.4f})")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"  -> 验证性能未提升，早停计数: {early_stopping_counter}/{patience}")
                
                if early_stopping_counter >= patience:
                    print(f"\n早停触发！")
                    stopped_early = True
                    break
        
        if stopped_early:
            print(f"--- [Fold {fold + 1}] 训练因早停而终止 ---")
        else:
            print(f"--- [Fold {fold + 1}] 训练完成 ---")
        
        print(f"当前折最佳验证 C-Index: {best_val_c_index:.4f} (在第 {best_epoch} 个 epoch 达到)")

        # 在固定测试集上评估当前折的最佳模型
        print(f"\n--- [Fold {fold + 1}] 在最终测试集上评估 ---")
        best_model_path = osp.join(fold_output_dir, "best_model.pt")
        if not osp.exists(best_model_path):
            print("警告: 找不到当前折叠的最佳模型，可能是因为训练从未改善。跳过测试评估。")
            test_c_index = 0
            test_auc = 0.5
        else:
            model.load_state_dict(torch.load(best_model_path))
            test_results = evaluate(model, test_loader, DEVICE)
            test_c_index = test_results['c_index']
            test_auc = test_results['auc']
        
        print(f"当前折测试集 C-Index: {test_c_index:.4f} | AUC: {test_auc:.4f}")
        test_c_indices.append(test_c_index)
        test_aucs.append(test_auc)
        
        # 记录当前折的结果
        fold_results.append({
            "fold": fold + 1,
            "best_validation_c_index": best_val_c_index,
            "best_validation_auc": best_val_auc,
            "test_c_index": test_c_index,
            "test_auc": test_auc,
            "best_epoch": best_epoch,
            "total_epochs_trained": epoch + 1,
            "early_stopped": stopped_early,
        })
    
    # 交叉验证总结
    print("\n\n--- 交叉验证完成 ---")
    
    mean_test_c_index = np.mean(test_c_indices)
    std_test_c_index = np.std(test_c_indices)
    mean_test_auc = np.mean(test_aucs)
    std_test_auc = np.std(test_aucs)
    
    print(f"所有折叠的测试 C-Index: {test_c_indices}")
    print(f"平均测试 C-Index: {mean_test_c_index:.4f} ± {std_test_c_index:.4f}")
    print(f"所有折叠的测试 AUC: {test_aucs}")
    print(f"平均测试 AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    
    # 总结与保存
    summary = {
        "dataset": dataset_name,
        "version": "fast",
        "cross_validation_summary": {
            "mean_test_c_index": mean_test_c_index,
            "std_test_c_index": std_test_c_index,
            "all_test_c_indices": test_c_indices,
            "mean_test_auc": mean_test_auc,
            "std_test_auc": std_test_auc,
            "all_test_aucs": test_aucs,
        },
        "fold_details": fold_results,
        "config": {
            "seed": SEED,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "lr": config["lr"],
            "n_splits": config["n_splits"],
            "test_size": config["test_size"],
            "accumulation_steps": config["accumulation_steps"],
            "early_stopping": config["early_stopping"]
        }
    }
    with open(osp.join(config["output_dir"], "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"训练和评估完成。所有结果保存在: {config['output_dir']}")

if __name__ == "__main__":
    main() 