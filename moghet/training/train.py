#!/usr/bin/env python
"""
MOGHET Unified Training Script
Integrates all the advantages of the original train_hierarchical.py and train_hierarchical_fast.py

Features:
- ✅ Complete 5-fold cross-validation
- ✅ Automatically save risk scores for each patient
- ✅ Random seed setting (reproducible)
- ✅ Smoke test mode (quick testing)
- ✅ Early stopping mechanism
- ✅ Gradient accumulation
- ✅ Use new modular import paths
"""

import os
import os.path as osp
import sys
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from tqdm import tqdm
import argparse

# 项目路径设置
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用新的导入路径
from models import HierarchicalGNNModel
from core.data_loader import HierarchicalDataset, hierarchical_collate_fn
from training.engine import train_one_epoch, evaluate
from training.loss import cox_loss


def set_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='MOGHET Unified Training Script')
    parser.add_argument('--dataset', type=str, default='BRCA', 
                       choices=['BRCA', 'LUAD', 'COAD', 'GBM', 'KIRC', 'LIHC', 'LUNG', 'OV', 'SKCM', 
                                'LUSC', 'STAD', 'UCEC', 'HNSC', 'PAAD', 'LGG'], 
                       help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--smoke-test', action='store_true', help='Quick test mode (2 folds, 2 epochs)')
    parser.add_argument('--use-existing-splits', type=str, default=None, 
                       help='Use existing data split directory (e.g.: results/hierarchical_model_brca_fast)')
    parser.add_argument('--resume-fold', type=int, default=None,
                       help='Resume training from specified fold (1-5)')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    dataset_name = args.dataset
    print("=" * 70)
    print(f"  MOGHET Training Script - {dataset_name}")
    print("=" * 70)
    print(f"🔢 Random Seed: {args.seed}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔄 Cross-validation: {args.n_folds} folds")
    print(f"📈 Maximum Epochs: {args.epochs}")
    print(f"🎯 Batch Size: {args.batch_size}")
    if args.smoke_test:
        print("⚡ Smoke Test Mode: ON")
    print("=" * 70)
    
    # Path settings
    processed_data_dir = osp.join(project_root, "data", "processed", dataset_name)
    id_mappings_path = osp.join(processed_data_dir, "id_mappings.json")
    
    with open(id_mappings_path, 'r') as f:
        id_mappings = json.load(f)
    num_pathways = len(id_mappings['pathway_id_to_idx'])
    
    # Configuration
    config = {
        "data_path": osp.join(processed_data_dir, "hierarchical_patient_data"),
        "clinical_data_path": osp.join(processed_data_dir, "patient_clinical_features.csv"),
        "survival_data_path": osp.join(processed_data_dir, "patient_survival.csv"),
        "output_dir": osp.join(project_root, "results", f"{dataset_name.lower()}_model"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 2 if args.smoke_test else args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_splits": 2 if args.smoke_test else args.n_folds,
        "test_size": 0.2,
        "random_state": args.seed,
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
            "patience": args.patience,
            "min_delta": 0.001,
        }
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    DEVICE = torch.device(config["device"])
    
    print(f"\n✅ Using device: {DEVICE}")
    print(f"✅ Output directory: {config['output_dir']}\n")
    
    # Data loading
    print("--- Data Loading ---")
    survival_df = pd.read_csv(config["survival_data_path"], index_col=0)
    clinical_df = pd.read_csv(config["clinical_data_path"], index_col=0)
    
    # Match patient IDs
    common_patients = sorted(list(set(survival_df.index) & set(clinical_df.index)))
    data_files_path = config["data_path"]
    existing_patient_ids = {f.replace('.pt', '') for f in os.listdir(data_files_path) if f.endswith('.pt')}
    all_patient_ids = np.array([pid for pid in common_patients if pid in existing_patient_ids])
    
    print(f"✅ Valid patients: {len(all_patient_ids)}")
    
    # Smoke test: use only a small amount of data
    if args.smoke_test:
        all_patient_ids = all_patient_ids[:min(50, len(all_patient_ids))]
        print(f"⚡ Smoke Test: Using {len(all_patient_ids)} patients")
    
    # Data splitting
    if args.use_existing_splits:
        # Use existing data splits
        print(f"\n📂 Using existing data splits: {args.use_existing_splits}")
        existing_splits = []
        for fold_idx in range(1, config["n_splits"] + 1):
            split_file = osp.join(args.use_existing_splits, f"fold_{fold_idx}", "split_info.json")
            if osp.exists(split_file):
                with open(split_file, 'r') as f:
                    split_info = json.load(f)
                existing_splits.append(split_info)
            else:
                print(f"⚠️  Warning: Split file for fold {fold_idx} not found")
                # Continue with other folds
                pass
        
        if existing_splits:
            # Use test set from the first fold (all folds should share the same test set)
            test_patient_ids = np.array(existing_splits[0]['test_patient_ids'])
            train_val_ids = np.array(existing_splits[0]['train_patient_ids'] + existing_splits[0]['val_patient_ids'])
            print(f"✅ Loaded {len(existing_splits)} fold splits (Test set: {len(test_patient_ids)})")
        else:
            # If loading fails, use new random split
            print("❌ Loading failed, using new random split")
            train_val_ids, test_patient_ids = train_test_split(
                all_patient_ids, test_size=config["test_size"], 
                random_state=config["random_state"], shuffle=True
            )
            existing_splits = None
    else:
        train_val_ids, test_patient_ids = train_test_split(
            all_patient_ids, test_size=config["test_size"], 
            random_state=config["random_state"], shuffle=True
        )
        existing_splits = None
    
    print(f"✅ Train/Validation: {len(train_val_ids)} | Test: {len(test_patient_ids)}\n")
    
    # Cross-validation
    test_c_indices, test_aucs = [], []
    fold_results = []
    
    # Initialize KFold for fallback
    kf = KFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
    kf_splits = list(kf.split(train_val_ids))
    
    for fold in range(config["n_splits"]):
        if existing_splits and fold < len(existing_splits):
            # Use existing fold split
            train_patient_ids = np.array(existing_splits[fold]['train_patient_ids'])
            val_patient_ids = np.array(existing_splits[fold]['val_patient_ids'])
            print(f"Using existing Fold {fold+1} split")
        else:
            # Use new random split
            train_idx, val_idx = kf_splits[fold]
            train_patient_ids = train_val_ids[train_idx]
            val_patient_ids = train_val_ids[val_idx]
            print(f"Using newly generated Fold {fold+1} split")
        print(f"\n{'='*70}")
        print(f"  Fold {fold+1}/{config['n_splits']} Cross-validation")
        print(f"{'='*70}")
        
        print(f"Train set: {len(train_patient_ids)} | Validation set: {len(val_patient_ids)} | Test set: {len(test_patient_ids)}")
        
        # Feature preprocessing
        train_clinical_df = clinical_df.loc[train_patient_ids]
        val_clinical_df = clinical_df.loc[val_patient_ids]
        test_clinical_df = clinical_df.loc[test_patient_ids]
        
        numeric_features = train_clinical_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = train_clinical_df.select_dtypes(include='object').columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ],
            remainder='passthrough'
        )
        
        train_features = preprocessor.fit_transform(train_clinical_df)
        val_features = preprocessor.transform(val_clinical_df)
        test_features = preprocessor.transform(test_clinical_df)
        
        train_features_dict = {pid: f for pid, f in zip(train_patient_ids, train_features)}
        val_features_dict = {pid: f for pid, f in zip(val_patient_ids, val_features)}
        test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features)}
        
        # Create datasets
        train_dataset = HierarchicalDataset(config["data_path"], train_patient_ids, survival_df, train_features_dict)
        val_dataset = HierarchicalDataset(config["data_path"], val_patient_ids, survival_df, val_features_dict)
        test_dataset = HierarchicalDataset(config["data_path"], test_patient_ids, survival_df, test_features_dict)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                                 shuffle=True, collate_fn=hierarchical_collate_fn, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], 
                               shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], 
                                shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)
        
        # Get metadata
        sample_data = train_dataset[0]['intra_pathway_graphs'][0]
        metadata = sample_data.metadata()
        clinical_in_features = train_features.shape[1]
        
        # Create fold output directory
        fold_output_dir = osp.join(config["output_dir"], f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Save configuration
        fold_params = {
            "metadata": metadata,
            "clinical_in_features": clinical_in_features,
            **config["model_params"]
        }
        with open(osp.join(fold_output_dir, "model_params.json"), "w") as f:
            json.dump(fold_params, f, indent=2)
        
        # Save data split
        split_info = {
            "fold": fold + 1,
            "train_patient_ids": train_patient_ids.tolist(),
            "val_patient_ids": val_patient_ids.tolist(),
            "test_patient_ids": test_patient_ids.tolist()
        }
        with open(osp.join(fold_output_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        
        # Create model
        model = HierarchicalGNNModel(
            metadata=metadata,
            clinical_in_features=clinical_in_features,
            **config["model_params"]
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        
        # Check if need to resume training
        start_epoch = 0
        best_val_c_index = 0
        best_val_auc = 0.5
        best_epoch = -1
        early_stopping_counter = 0
        
        checkpoint_file = osp.join(fold_output_dir, "checkpoint.pt")
        best_model_file = osp.join(fold_output_dir, "best_model.pt")
        
        if args.resume_fold == fold + 1 and osp.exists(checkpoint_file):
            print(f"\n🔄 Resuming training from checkpoint...")
            checkpoint = torch.load(checkpoint_file, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Optimizer state loaded")
            start_epoch = checkpoint['epoch']
            best_val_c_index = checkpoint['best_val_c_index']
            best_val_auc = checkpoint['best_val_auc']
            best_epoch = checkpoint['best_epoch']
            early_stopping_counter = checkpoint['early_stopping_counter']
            print(f"✅ Restored to Epoch {start_epoch}, Best C-Index: {best_val_c_index:.4f} (Epoch {best_epoch})")
            print(f"   Early stopping counter: {early_stopping_counter}/{config['early_stopping']['patience']}")
        
        # Training loop
        print(f"\n--- Training Started ---")
        
        for epoch in range(start_epoch, config["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, 
                                        config["accumulation_steps"])
            val_results = evaluate(model, val_loader, DEVICE)
            
            val_c_index = val_results['c_index']
            val_auc = val_results['auc']
            
            print(f"Epoch {epoch+1:03d}/{config['epochs']:03d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val C-Index: {val_c_index:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
            
            if val_c_index > best_val_c_index + config["early_stopping"]["min_delta"]:
                best_val_c_index = val_c_index
                best_val_auc = val_auc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), osp.join(fold_output_dir, "best_model.pt"))
                print(f"  ✅ Saved best model (Epoch {best_epoch}, C-Index: {best_val_c_index:.4f})")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config["early_stopping"]["patience"]:
                    print(f"\n⏹️  Early stopping triggered (Epoch {epoch+1})")
                    break
            
            # Save checkpoint (each epoch)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_c_index': best_val_c_index,
                'best_val_auc': best_val_auc,
                'best_epoch': best_epoch,
                'early_stopping_counter': early_stopping_counter
            }
            torch.save(checkpoint, checkpoint_file)
        
        # Test evaluation
        print(f"\n--- Test Set Evaluation ---")
        model.load_state_dict(torch.load(osp.join(fold_output_dir, "best_model.pt"), weights_only=False))
        test_results = evaluate(model, test_loader, DEVICE, return_predictions=True)
        
        test_c_index = test_results['c_index']
        test_auc = test_results['auc']
        
        # Save predictions
        if 'predictions' in test_results:
            predictions_df = pd.DataFrame({
                'patient_id': test_patient_ids,
                'risk_score': test_results['predictions']['risk_scores'],
                'survival_time': test_results['predictions']['survival_times'],
                'event': test_results['predictions']['events']
            })
            predictions_path = osp.join(fold_output_dir, "test_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            print(f"✅ Saved predictions for {len(predictions_df)} patients")
        
        print(f"📊 Test C-Index: {test_c_index:.4f} | AUC: {test_auc:.4f}")
        
        test_c_indices.append(test_c_index)
        test_aucs.append(test_auc)
        
        fold_results.append({
            "fold": fold + 1,
            "best_validation_c_index": best_val_c_index,
            "best_validation_auc": best_val_auc,
            "test_c_index": test_c_index,
            "test_auc": test_auc,
            "best_epoch": best_epoch
        })
    
    # Cross-validation summary
    print(f"\n{'='*70}")
    print("  Cross-validation Completed")
    print(f"{'='*70}")
    
    mean_c = np.mean(test_c_indices)
    std_c = np.std(test_c_indices)
    mean_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs)
    
    print(f"📊 Average Test C-Index: {mean_c:.4f} ± {std_c:.4f}")
    print(f"📊 Average Test AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"📊 All Fold C-Indices: {[f'{c:.4f}' for c in test_c_indices]}")
    print(f"📊 All Fold AUCs: {[f'{a:.4f}' for a in test_aucs]}")
    
    # Save summary
    summary = {
        "dataset": dataset_name,
        "seed": args.seed,
        "cross_validation_summary": {
            "mean_test_c_index": mean_c,
            "std_test_c_index": std_c,
            "all_test_c_indices": test_c_indices,
            "mean_test_auc": mean_auc,
            "std_test_auc": std_auc,
            "all_test_aucs": test_aucs
        },
        "fold_details": fold_results,
        "config": config
    }
    
    with open(osp.join(config["output_dir"], "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Aggregate predictions
    print(f"\n--- Aggregating All Predictions ---")
    all_predictions = []
    for fold_idx in range(config["n_splits"]):
        pred_path = osp.join(config["output_dir"], f"fold_{fold_idx+1}", "test_predictions.csv")
        if osp.exists(pred_path):
            fold_preds = pd.read_csv(pred_path)
            fold_preds['fold'] = fold_idx + 1
            all_predictions.append(fold_preds)
    
    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(osp.join(config["output_dir"], "all_test_predictions.csv"), index=False)
        
        avg = combined.groupby('patient_id').agg({
            'risk_score': 'mean',
            'survival_time': 'first',
            'event': 'first',
            'fold': lambda x: ','.join(map(str, sorted(x)))
        }).reset_index()
        avg.to_csv(osp.join(config["output_dir"], "averaged_test_predictions.csv"), index=False)
        
        # Also save as all_patients_predictions.csv for compatibility with old visualization scripts
        avg[['patient_id', 'risk_score']].to_csv(osp.join(config["output_dir"], "all_patients_predictions.csv"), index=False)
        
        print(f"✅ Aggregated predictions saved")
        print(f"   - all_test_predictions.csv ({len(combined)} records)")
        print(f"   - averaged_test_predictions.csv ({len(avg)} patients)")
        print(f"   - all_patients_predictions.csv ({len(avg)} patients, compatibility format)")
    
    print(f"\n{'='*70}")
    print(f"  Training Completed! Results saved at: {config['output_dir']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

