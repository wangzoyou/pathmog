import os
import os.path as osp
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from tqdm import tqdm
import argparse

# Project path setup
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir) # -> gnn/moghet
gnn_dir = osp.dirname(project_root) # -> gnn
if gnn_dir not in sys.path:
    sys.path.insert(0, gnn_dir)

# Import the necessary functions and classes from the project
from moghet.inference.predict_fast import generate_predictions
from moghet.src.data_loader import HierarchicalDataset, hierarchical_collate_fn
from moghet.src.hierarchical_model import HierarchicalGNNModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from moghet.src.auc_evaluation import SurvivalAUCEvaluator

def main():
    parser = argparse.ArgumentParser(description='Aggregate predictions from all CV folds for a dataset.')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to process. If not provided, all datasets will be processed.')
    args = parser.parse_args()

    datasets = []
    if args.dataset:
        datasets.append(args.dataset)
    else:
        # Automatically find all datasets with trained models in the results directory
        results_root = osp.join(project_root, "results")
        for item in os.listdir(results_root):
            if item.startswith("hierarchical_model_") and item.endswith("_fast"):
                dataset_name = item.replace("hierarchical_model_", "").replace("_fast", "").upper()
                datasets.append(dataset_name)
    
    print(f"--- Found {len(datasets)} datasets to process: {datasets} ---")

    for dataset_name in datasets:
        print(f"\n\n--- Aggregating predictions for the entire {dataset_name} dataset ---")
        
        # --- Basic Config ---
        SEED = 105
        N_SPLITS = 5
        TEST_SIZE = 0.2
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        processed_data_dir = osp.join(project_root, "data", "processed", dataset_name)
        results_dir = osp.join(project_root, "results", f"hierarchical_model_{dataset_name.lower()}_fast")
        output_path = osp.join(results_dir, "all_patients_predictions.csv")

        # --- Load Data and Patient IDs (mirroring train/predict scripts) ---
        print("\n--- Loading and aligning all patient data ---")
        survival_df = pd.read_csv(osp.join(processed_data_dir, "patient_survival.csv"), index_col=0)
        clinical_df = pd.read_csv(osp.join(processed_data_dir, "patient_clinical_features.csv"), index_col=0)
        
        common_patients = sorted(list(set(survival_df.index).intersection(set(clinical_df.index))))
        data_files_path = osp.join(processed_data_dir, "hierarchical_patient_data")
        existing_patient_ids = {f.replace('.pt', '') for f in os.listdir(data_files_path) if f.endswith('.pt')}
        all_patient_ids = np.array([pid for pid in common_patients if pid in existing_patient_ids])

        print(f"Found {len(all_patient_ids)} total patients with complete data.")

        # --- Determine splitting strategy ---
        use_json_splits = all(osp.exists(osp.join(results_dir, f"fold_{f+1}", "split_info.json")) for f in range(N_SPLITS))

        if use_json_splits:
            print("\n--- Strategy: Loading data splits from pre-saved split_info.json files ---")
            # Load test patient IDs from the first fold's split file (assumed to be the same for all folds)
            with open(osp.join(results_dir, "fold_1", "split_info.json"), 'r') as f:
                split_data = json.load(f)
                test_patient_ids = np.array(split_data['test_patient_ids'])
            print(f"Loaded {len(test_patient_ids)} test patient IDs from JSON.")

        else:
            print(f"\n--- Strategy: Recreating data splits using random_state={SEED} ---")
            train_val_ids, test_patient_ids = train_test_split(
                all_patient_ids, test_size=TEST_SIZE, random_state=SEED, shuffle=True
            )
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            # Store the fold splits to be used in both loops
            cv_splits = list(kf.split(train_val_ids))


        # --- Preprocess clinical features ---
        numeric_features = clinical_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = clinical_df.select_dtypes(include='object').columns.tolist()

        # --- Storage for predictions ---
        all_predictions = {}

        # --- Loop 1: Generate predictions for each fold's validation set ---
        print("\n--- Generating predictions for validation sets across all folds ---")
        for fold_idx in range(N_SPLITS):
            fold_num = fold_idx + 1
            
            # --- Get patient IDs for this fold based on the chosen strategy ---
            if use_json_splits:
                split_info_path = osp.join(results_dir, f"fold_{fold_num}", "split_info.json")
                with open(split_info_path, 'r') as f:
                    split_data = json.load(f)
                    train_patient_ids_fold = np.array(split_data['train_patient_ids'])
                    val_patient_ids = np.array(split_data['val_patient_ids'])
            else:
                train_indices, val_indices = cv_splits[fold_idx]
                train_patient_ids_fold = train_val_ids[train_indices]
                val_patient_ids = train_val_ids[val_indices]
            
            print(f"\nProcessing Fold {fold_num} validation set ({len(val_patient_ids)} patients)...")
            
            # --- Correct Preprocessing Step ---
            # Create and fit the preprocessor ONLY on the training data for this fold
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
                    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
                ], remainder='passthrough'
            )
            preprocessor.fit(clinical_df.loc[train_patient_ids_fold])
            
            model_path = osp.join(results_dir, f"fold_{fold_num}", "best_model.pt")
            if not osp.exists(model_path):
                print(f"Warning: Model for fold {fold_num} of dataset {dataset_name} not found. Skipping.")
                continue

            val_features_processed = preprocessor.transform(clinical_df.loc[val_patient_ids])
            val_features_dict = {pid: f for pid, f in zip(val_patient_ids, val_features_processed)}
            
            val_dataset = HierarchicalDataset(data_files_path, val_patient_ids, survival_df, val_features_dict)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)
            
            # Load model for this fold
            first_valid_item = next(item for item in val_dataset if item['intra_pathway_graphs'])
            metadata = first_valid_item['intra_pathway_graphs'][0].metadata()
            clinical_in_features = first_valid_item['clinical_features'].shape[-1]
            with open(osp.join(processed_data_dir, "id_mappings.json"), 'r') as f:
                num_pathways = len(json.load(f)['pathway_id_to_idx'])

            model_params = {
                "num_pathways": num_pathways, "pathway_embedding_dim": 8, "gnn_hidden_channels": 64, 
                "pathway_out_channels": 128, "intra_attention_hidden_channels": 128, 
                "inter_attention_hidden_channels": 128, "clinical_hidden_channels": 32, 
                "final_hidden_channels": 64
            }
            model = HierarchicalGNNModel(metadata=metadata, clinical_in_features=clinical_in_features, **model_params).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            # Generate predictions
            risk_scores = generate_predictions(model, val_loader, DEVICE)
            if risk_scores is not None:
                for pid, score in zip(val_patient_ids, risk_scores):
                    all_predictions[pid] = {'scores': [score], 'source': f'validation_fold_{fold_num}'}

        # --- Loop 2: Generate predictions for the hold-out test set using all models ---
        print(f"\n--- Generating predictions for the {dataset_name} hold-out test set (ensembling) ---")
        
        test_patient_predictions_by_fold = {pid: [] for pid in test_patient_ids}

        for fold_idx in range(N_SPLITS):
            fold_num = fold_idx + 1
            
            # --- Get training patient IDs for this fold for preprocessor fitting ---
            if use_json_splits:
                split_info_path = osp.join(results_dir, f"fold_{fold_num}", "split_info.json")
                with open(split_info_path, 'r') as f:
                    split_data = json.load(f)
                train_patient_ids_fold = np.array(split_data['train_patient_ids'])
            else:
                train_indices, _ = cv_splits[fold_idx]
                train_patient_ids_fold = train_val_ids[train_indices]

            # Fit preprocessor on this fold's training data
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
                    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
                ], remainder='passthrough'
            )
            preprocessor.fit(clinical_df.loc[train_patient_ids_fold])

            # Transform the test set using this fold-specific preprocessor
            test_features_processed = preprocessor.transform(clinical_df.loc[test_patient_ids])
            test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features_processed)}
            test_dataset = HierarchicalDataset(data_files_path, test_patient_ids, survival_df, test_features_dict)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)

            print(f"Using model from Fold {fold_num} to predict test set...")
            model_path = osp.join(results_dir, f"fold_{fold_num}", "best_model.pt")
            
            # Re-initialize the model to be safe, as its architecture might depend on preprocessed feature dimensions
            first_valid_item = next(item for item in test_dataset if item['intra_pathway_graphs'])
            metadata = first_valid_item['intra_pathway_graphs'][0].metadata()
            clinical_in_features = first_valid_item['clinical_features'].shape[-1]
            with open(osp.join(processed_data_dir, "id_mappings.json"), 'r') as f:
                num_pathways = len(json.load(f)['pathway_id_to_idx'])
            model_params = {
                "num_pathways": num_pathways, "pathway_embedding_dim": 8, "gnn_hidden_channels": 64, 
                "pathway_out_channels": 128, "intra_attention_hidden_channels": 128, 
                "inter_attention_hidden_channels": 128, "clinical_hidden_channels": 32, 
                "final_hidden_channels": 64
            }
            model = HierarchicalGNNModel(metadata=metadata, clinical_in_features=clinical_in_features, **model_params).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            risk_scores = generate_predictions(model, test_loader, DEVICE)
            
            if risk_scores is not None:
                for pid, score in zip(test_patient_ids, risk_scores):
                    test_patient_predictions_by_fold[pid].append(score)

        # Now, populate the all_predictions dictionary for the test set by averaging
        for pid, scores in test_patient_predictions_by_fold.items():
            if scores:
                all_predictions[pid] = {'scores': scores, 'source': 'holdout_test_set'}


        # --- Finalize and Save ---
        print("\n--- Finalizing predictions and saving results ---")
        final_data = []
        for pid, data in all_predictions.items():
            final_score = np.mean(data['scores']) # Average the scores (for test set, this is ensembling)
            final_data.append({'patient_id': pid, 'risk_score': final_score, 'source': data['source']})
            
        final_df = pd.DataFrame(final_data)
        final_df.to_csv(output_path, index=False)

        print(f"\nAggregation complete. Full predictions saved to: {output_path}")
        print(f"Total patients predicted: {len(final_df)} / {len(all_patient_ids)}")

        # --- Final Evaluation on the Test Set ---
        print(f"\n--- Final Performance Evaluation on the {dataset_name} Hold-out Test Set ---")
        test_set_df = final_df[final_df['source'] == 'holdout_test_set']
        if not test_set_df.empty:
            merged_df = pd.merge(test_set_df, survival_df.reset_index().rename(columns={'index': 'patient_id'}), on='patient_id', how='inner')
            
            risk_scores = merged_df['risk_score'].to_numpy()
            times = merged_df['OS.time'].to_numpy() / 365.0  # Convert days to years
            events = merged_df['OS'].to_numpy()

            evaluator = SurvivalAUCEvaluator()
            performance_results = evaluator.evaluate_model_performance({
                'risk_scores': risk_scores,
                'times': times,
                'events': events
            })

            print(f"C-index: {performance_results['c_index']:.4f}")
            print(f"Time-dependent AUC: {performance_results['time_dependent_auc']:.4f}")
            for time_point, auc in performance_results['time_specific_auc'].items():
                print(f"AUC at {int(time_point)} year(s): {auc:.4f}")
        else:
            print("No predictions for the hold-out test set were found. Skipping final evaluation.")


if __name__ == "__main__":
    main()
