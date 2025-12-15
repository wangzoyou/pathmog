import os
import os.path as osp
import sys
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # (修复) 使用标准 DataLoader
from torch_geometric.data import Batch, Data
from sklearn.model_selection import train_test_split, KFold # (新增) 导入 KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines.utils import concordance_index
import pandas as pd
from tqdm import tqdm # (核心修改) 导入tqdm
import argparse  # (新增) 导入argparse用于命令行参数
import argparse  # (新增) 导入argparse用于命令行参数

# -- 项目路径设置 --
# 将项目根目录添加到Python路径，以便能够导入moghet模块
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -- 完成路径设置 --

# 导入模型和评估工具（使用新的导入路径）
from models import HierarchicalGNNModel
from evaluation.auc_evaluation import SurvivalAUCEvaluator, calculate_auc_from_predictions

# --- 1. 数据集定义 ---
from core.data_loader import HierarchicalDataset, hierarchical_collate_fn

# --- 2. 自定义 Collate 函数 (恢复正确的生存分析) ---
# (已移动到 data_loader.py)

# --- 3. 损失函数 ---
from training.loss import cox_loss

# --- 4. 训练与评估函数 ---
from training.engine import train_one_epoch, evaluate


# --- 5. 主训练流程 ---
def main():
    # (新增) 解析命令行参数
    parser = argparse.ArgumentParser(description='训练层次化GNN模型')
    parser.add_argument('--dataset', type=str, default='LUAD', 
                       choices=['BRCA', 'LUAD', 'COAD', 'GBM', 'KIRC', 'LUNG', 'OV', 'SKCM'], 
                       help='要使用的数据集 (默认: LUAD)')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    print(f"--- 开始训练 HierarchicalGNNModel (数据集: {dataset_name}) ---")
    
    # --- 配置 ---
    # (修改) 使用数据集特定的路径
    processed_data_dir = osp.join(project_root, "moghet", "data", "processed", dataset_name)
    # (新增) 读取ID映射文件以获取通路数量
    id_mappings_path = osp.join(processed_data_dir, "id_mappings.json")
    with open(id_mappings_path, 'r') as f:
        id_mappings = json.load(f)
    num_pathways = len(id_mappings['pathway_id_to_idx'])

    config = {
        "data_path": osp.join(processed_data_dir, "hierarchical_patient_data"),
        "clinical_data_path": osp.join(processed_data_dir, "patient_clinical_features.csv"),
        "survival_data_path": osp.join(processed_data_dir, "patient_survival.csv"),
        # (修改) 输出目录按数据集分开
        "output_dir": osp.join(project_root, "moghet", "results", f"hierarchical_model_{dataset_name.lower()}"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 200,
        "batch_size": 32, # 优化后的批次大小，提升训练效率       
        "lr": 1e-4,
        "n_splits": 5,  # (修改) 使用5折交叉验证
        "test_size": 0.2, # (新增) 为最终测试集保留20%的数据
        "random_state": 24,
        "accumulation_steps": 4, # 增加累积步数
        # (新增) 模型参数
        "model_params": {
            "num_pathways": num_pathways,
            "pathway_embedding_dim": 8, # 示例维度
            "gnn_hidden_channels": 64,
            "pathway_out_channels": 128,
            "intra_attention_hidden_channels": 128,
            "inter_attention_hidden_channels": 128,
            "clinical_hidden_channels": 32,
            "final_hidden_channels": 64
        },
        # (新增) 早停参数
        "early_stopping": {
            "patience": 10,  # 连续10个epoch性能未提升则停止
            "min_delta": 0.001,  # C-Index最小提升幅度
        }
    }
    os.makedirs(config["output_dir"], exist_ok=True)
    DEVICE = torch.device(config["device"])
    print(f"使用设备: {DEVICE}")
    print(f"数据集: {dataset_name}")
    print(f"输出目录: {config['output_dir']}")

    # --- 数据加载和预处理 ---
    print("\n--- 数据加载和预处理 ---")
    
    # 1. 首先加载生存数据，以生存数据为主
    survival_df = pd.read_csv(config["survival_data_path"], index_col=0)
    print(f"生存数据患者数量: {len(survival_df)}")
    
    # 2. 加载临床数据
    clinical_df = pd.read_csv(config["clinical_data_path"], index_col=0)
    print(f"临床数据患者数量: {len(clinical_df)}")
    
    # 3. 以生存数据为主，只保留在生存数据中存在的患者
    survival_patients = set(survival_df.index)
    clinical_patients = set(clinical_df.index)
    
    # 找出在生存数据中存在但在临床数据中缺失的患者
    missing_in_clinical = survival_patients - clinical_patients
    if missing_in_clinical:
        print(f"⚠️  警告: {len(missing_in_clinical)} 个患者在生存数据中存在但在临床数据中缺失")
        print(f"缺失的患者ID: {list(missing_in_clinical)[:5]}...")  # 只显示前5个
    
    # 找出在临床数据中存在但在生存数据中缺失的患者
    missing_in_survival = clinical_patients - survival_patients
    if missing_in_survival:
        print(f"⚠️  警告: {len(missing_in_survival)} 个患者在临床数据中存在但在生存数据中缺失")
        print(f"多余的患者ID: {list(missing_in_survival)[:5]}...")  # 只显示前5个
    
    # 4. 以生存数据为主，筛选出共同的患者
    common_patients = survival_patients.intersection(clinical_patients)
    all_patient_ids = np.array(sorted(list(common_patients)))
    
    print(f"✅ 以生存数据为主，共找到 {len(all_patient_ids)} 位具有完整临床和生存数据的患者。")

    # 5. 过滤患者列表，只保留那些真实存在 .pt 文件的患者
    print("正在扫描数据目录以确认有效的患者数据文件...")
    existing_patient_files = os.listdir(config["data_path"])
    existing_patient_ids = {f.replace('.pt', '') for f in existing_patient_files if f.endswith('.pt')}
    
    original_count = len(all_patient_ids)
    all_patient_ids = np.array([pid for pid in all_patient_ids if pid in existing_patient_ids])
    final_count = len(all_patient_ids)
    
    print(f"扫描完成。有效的患者数量从 {original_count} 减少到 {final_count}。")
    if final_count == 0:
        print("错误: 没有任何患者数据文件与临床/生存数据匹配。无法继续训练。")
        return

    # 6. 更新数据框，只保留有效的患者
    survival_df = survival_df.loc[all_patient_ids]
    clinical_df = clinical_df.loc[all_patient_ids]
    
    print(f"✅ 最终数据规模: 生存数据 {len(survival_df)} 患者, 临床数据 {len(clinical_df)} 患者")

    # --- 数据集划分 (Train/Validation/Test) ---
    print("\n--- 开始数据集划分 ---")
    
    # 1. 首先，分出一个固定的测试集 (hold-out test set)
    train_val_ids, test_patient_ids = train_test_split(
        all_patient_ids,
        test_size=config["test_size"],
        random_state=config["random_state"],
        shuffle=True
    )

    print(f"总数据集划分为: "
          f"训练/验证集 {len(train_val_ids)} | "
          f"最终测试集 {len(test_patient_ids)}")

    # --- K-Fold 交叉验证循环 ---
    kf = KFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    
    test_c_indices = []
    test_aucs = []
    fold_results = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(train_val_ids)):
        print(f"\n\n--- 开始第 {fold + 1}/{config['n_splits']} 折交叉验证 ---")
        
        # --- 获取当前折的患者ID ---
        train_patient_ids = train_val_ids[train_index]
        val_patient_ids = train_val_ids[val_index]
        
        print(f"当前折划分: "
              f"训练集 {len(train_patient_ids)} | "
              f"验证集 {len(val_patient_ids)} | "
              f"测试集 {len(test_patient_ids)}")

        # --- 特征预处理 (在每个折叠内部进行以防数据泄露) ---
        print(f"\n--- [Fold {fold + 1}] 开始特征预处理 ---")
        
        # 从原始DataFrame中获取当前折的数据
        train_clinical_df = clinical_df.loc[train_patient_ids]
        val_clinical_df = clinical_df.loc[val_patient_ids]
        test_clinical_df = clinical_df.loc[test_patient_ids]

        # 检查缺失值
        print(f"训练集缺失值统计:")
        train_missing = train_clinical_df.isnull().sum()
        if train_missing.sum() > 0:
            print(train_missing[train_missing > 0])
        else:
            print("✅ 训练集无缺失值")
            
        print(f"验证集缺失值统计:")
        val_missing = val_clinical_df.isnull().sum()
        if val_missing.sum() > 0:
            print(val_missing[val_missing > 0])
        else:
            print("✅ 验证集无缺失值")
            
        print(f"测试集缺失值统计:")
        test_missing = test_clinical_df.isnull().sum()
        if test_missing.sum() > 0:
            print(test_missing[test_missing > 0])
        else:
            print("✅ 测试集无缺失值")

        # 分离数值型和分类型特征
        numeric_features = train_clinical_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = train_clinical_df.select_dtypes(include='object').columns.tolist()
        
        print(f"数值型特征: {len(numeric_features)} 个")
        print(f"分类型特征: {len(categorical_features)} 个")

        # 创建预处理器，使用中位数填补数值型特征的缺失值
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
        
        # 将处理后的特征（numpy数组）转换为字典格式
        train_features_dict = {pid: f for pid, f in zip(train_patient_ids, train_features_processed)}
        val_features_dict = {pid: f for pid, f in zip(val_patient_ids, val_features_processed)}
        test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features_processed)}

        # 将处理好的特征传递给Dataset
        train_dataset = HierarchicalDataset(config["data_path"], train_patient_ids, survival_df, train_features_dict)
        val_dataset = HierarchicalDataset(config["data_path"], val_patient_ids, survival_df, val_features_dict)
        test_dataset = HierarchicalDataset(config["data_path"], test_patient_ids, survival_df, test_features_dict)

        # 使用标准 DataLoader 和自定义 collate_fn
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=hierarchical_collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)
        
        # --- 获取图元数据 ---
        print(f"\n--- [Fold {fold + 1}] 获取图元数据 ---")
        try:
            # 从训练集中取一个样本来获取元数据
            sample_data = train_dataset[0]['intra_pathway_graphs'][0]
            metadata = sample_data.metadata()
            print(f"成功提取元数据: {metadata}")
        except (IndexError, KeyError) as e:
            print(f"错误: 无法从数据中提取元数据，折叠将跳过。错误: {e}")
            continue # 跳到下一个折叠

        # 获取处理后的特征维度
        processed_feature_dim = train_features_processed.shape[1]
        print(f"处理后的临床特征维度: {processed_feature_dim}")
        
        # 创建当前折叠的输出目录
        fold_output_dir = osp.join(config["output_dir"], f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # 保存当前fold的模型参数，用于推理时正确加载
        fold_params = {
            "metadata": metadata,
            "clinical_in_features": processed_feature_dim,
            **config["model_params"]
        }
        with open(osp.join(fold_output_dir, "model_params.json"), "w") as f:
            json.dump(fold_params, f, indent=2)
        print(f"[OK] 已保存fold {fold+1}的模型参数到: {osp.join(fold_output_dir, 'model_params.json')}")
        
        # 保存当前fold的数据划分信息
        split_info = {
            "fold": fold + 1,
            "train_patient_ids": train_patient_ids.tolist(),
            "val_patient_ids": val_patient_ids.tolist(),
            "test_patient_ids": test_patient_ids.tolist(),
            "total_patients": len(all_patient_ids),
            "train_size": len(train_patient_ids),
            "val_size": len(val_patient_ids),
            "test_size": len(test_patient_ids),
            "index_to_patient_id_mapping": {
                "all_patient_ids": all_patient_ids.tolist(),
                "train_val_indices": train_val_ids.tolist(),
                "test_indices": test_patient_ids.tolist()
            }
        }
        with open(osp.join(fold_output_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"[OK] 已保存fold {fold+1}的数据划分信息到: {osp.join(fold_output_dir, 'split_info.json')}")

        # --- 模型初始化 (为每个折叠重新初始化) ---
        model = HierarchicalGNNModel(
            metadata=metadata,
            clinical_in_features=processed_feature_dim,
            **config["model_params"]
        ).to(DEVICE)
        
        # 初始化模型权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        
        model.apply(init_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        
        # 添加梯度裁剪来防止梯度爆炸
        max_grad_norm = 1.0
        
        # --- 训练循环 ---
        print(f"\n--- [Fold {fold + 1}] 开始训练 ---")
        best_val_c_index = 0
        best_val_auc = 0.5
        best_epoch = -1
        early_stopping_counter = 0
        patience = config["early_stopping"]["patience"]
        min_delta = config["early_stopping"]["min_delta"]
        stopped_early = False
        
        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, config["accumulation_steps"])
            val_results = evaluate(model, val_loader, DEVICE) # 传递 survival_df
            
            val_c_index = val_results['c_index']
            val_auc = val_results['auc']
            
            print(f"Epoch {epoch+1:02d}/{config['epochs']:02d} | 训练损失: {train_loss:.4f} | 验证 C-Index: {val_c_index:.4f} | 验证 AUC: {val_auc:.4f} | 验证集成AUC: {val_results.get('integrated_auc', 0):.4f}")
            
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

        # --- 在固定测试集上评估当前折的最佳模型 ---
        print(f"\n--- [Fold {fold + 1}] 在最终测试集上评估 ---")
        # 加载当前折叠的最佳模型
        best_model_path = osp.join(fold_output_dir, "best_model.pt")
        if not osp.exists(best_model_path):
            print("警告: 找不到当前折叠的最佳模型，可能是因为训练从未改善。跳过测试评估。")
            test_c_index = 0
            test_auc = 0.5
        else:
            model.load_state_dict(torch.load(best_model_path))
            test_results = evaluate(model, test_loader, DEVICE, return_predictions=True)  # 返回详细预测
            test_c_index = test_results['c_index']
            test_auc = test_results['auc']
            
            # 保存测试集的预测结果（每个患者的风险得分）
            if 'predictions' in test_results:
                predictions_df = pd.DataFrame({
                    'patient_id': test_patient_ids,  # 使用真实的患者ID列表
                    'risk_score': test_results['predictions']['risk_scores'],
                    'survival_time': test_results['predictions']['survival_times'],
                    'event': test_results['predictions']['events']
                })
                predictions_path = osp.join(fold_output_dir, "test_predictions.csv")
                predictions_df.to_csv(predictions_path, index=False)
                print(f"[OK] 已保存测试集预测结果到: {predictions_path}")
                print(f"     共保存 {len(predictions_df)} 个患者的风险得分")
        
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
    
    # --- 交叉验证总结 ---
    print("\n\n--- 交叉验证完成 ---")
    
    mean_test_c_index = np.mean(test_c_indices)
    std_test_c_index = np.std(test_c_indices)
    mean_test_auc = np.mean(test_aucs)
    std_test_auc = np.std(test_aucs)
    
    print(f"所有折叠的测试 C-Index: {test_c_indices}")
    print(f"平均测试 C-Index: {mean_test_c_index:.4f} ± {std_test_c_index:.4f}")
    print(f"所有折叠的测试 AUC: {test_aucs}")
    print(f"平均测试 AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    
    # 收集所有折的划分信息
    all_splits_info = {}
    for fold in range(config["n_splits"]):
        fold_output_dir = osp.join(config["output_dir"], f"fold_{fold+1}")
        split_info_path = osp.join(fold_output_dir, "split_info.json")
        if osp.exists(split_info_path):
            with open(split_info_path, 'r') as f:
                all_splits_info[f"fold_{fold+1}"] = json.load(f)

    # --- 总结与保存 ---
    summary = {
        "dataset": dataset_name,  # (新增) 记录数据集信息
        "cross_validation_summary": {
            "mean_test_c_index": mean_test_c_index,
            "std_test_c_index": std_test_c_index,
            "all_test_c_indices": test_c_indices,
            "mean_test_auc": mean_test_auc,
            "std_test_auc": std_test_auc,
            "all_test_aucs": test_aucs,
        },
        "fold_details": fold_results,
        "data_splits": all_splits_info,  # (新增) 所有折的数据划分信息
        "config": {
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
    
    # --- 汇总所有折的预测结果 ---
    print("\n--- 汇总所有折的预测结果 ---")
    all_predictions = []
    for fold_idx in range(config["n_splits"]):
        fold_num = fold_idx + 1
        predictions_path = osp.join(config["output_dir"], f"fold_{fold_num}", "test_predictions.csv")
        if osp.exists(predictions_path):
            fold_predictions = pd.read_csv(predictions_path)
            fold_predictions['fold'] = fold_num
            all_predictions.append(fold_predictions)
    
    if all_predictions:
        # 合并所有折的预测
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # 计算每个患者的平均风险得分（如果同一患者在多个折中被预测）
        avg_predictions = combined_predictions.groupby('patient_id').agg({
            'risk_score': 'mean',
            'survival_time': 'first',  # 生存时间应该是一样的
            'event': 'first',  # 事件状态应该是一样的
            'fold': lambda x: ','.join(map(str, sorted(x)))  # 记录在哪些折中出现
        }).reset_index()
        
        # 保存汇总结果
        combined_path = osp.join(config["output_dir"], "all_test_predictions.csv")
        combined_predictions.to_csv(combined_path, index=False)
        print(f"[OK] 所有折的预测结果保存到: {combined_path}")
        
        avg_path = osp.join(config["output_dir"], "averaged_test_predictions.csv")
        avg_predictions.to_csv(avg_path, index=False)
        print(f"[OK] 平均预测结果保存到: {avg_path}")
        print(f"     共 {len(avg_predictions)} 个独立患者")
        
        # 打印汇总统计
        print(f"\n预测结果统计:")
        print(f"  总预测数: {len(combined_predictions)}")
        print(f"  独立患者数: {len(avg_predictions)}")
        print(f"  风险得分范围: [{avg_predictions['risk_score'].min():.4f}, {avg_predictions['risk_score'].max():.4f}]")
        print(f"  平均风险得分: {avg_predictions['risk_score'].mean():.4f}")
    else:
        print("警告: 没有找到任何预测结果文件")


if __name__ == "__main__":
    main() 