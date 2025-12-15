import os
import os.path as osp
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from tqdm import tqdm
import argparse

# --- Project Path Setup ---
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir) # -> gnn/moghet
gnn_dir = osp.dirname(project_root) # -> gnn
if gnn_dir not in sys.path:
    sys.path.insert(0, gnn_dir)
# --- End Path Setup ---

# (修正) 从正确的位置导入
from moghet.src.data_loader import HierarchicalDataset, hierarchical_collate_fn
from moghet.src.hierarchical_model import HierarchicalGNNModel
from sklearn.impute import SimpleImputer

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_test_patient_ids_for_fold(all_patient_ids, n_splits=5, test_size=0.2, random_state=42):
    """
    Reproduces the exact train/validation/test split from the training script
    to get the correct list of patient IDs for the final test set.
    """
    train_val_ids, test_ids = train_test_split(
        all_patient_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return test_ids

def generate_predictions(model, data_loader, device):
    """Runs the model on the data and returns predictions."""
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Predictions"):
            if batch is None:
                continue
            
            graphs_batch = batch['graphs_batch'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            
            predictions = model(graphs_batch, clinical_features)
            all_predictions.append(predictions.cpu().numpy())
            
    return np.concatenate(all_predictions, axis=0).flatten()

def main():
    parser = argparse.ArgumentParser(description='Generate predictions using a pre-trained Hierarchical GNN model.')
    parser.add_argument('--dataset', type=str, default='LUAD', 
                       choices=['BRCA', 'LUAD', 'COAD', 'GBM', 'KIRC', 'LUNG', 'OV', 'SKCM', 'LIHC'], 
                       help='Dataset to use (default: LUAD)')
    parser.add_argument('--fold', type=int, default=1, help='The fold number of the pre-trained model to use.')
    args = parser.parse_args()
    
    SEED = 42
    set_seed(SEED)
    
    dataset_name = args.dataset
    fold = args.fold
    print(f"--- Generating predictions for dataset: {dataset_name}, Fold: {fold} ---")
    
    # --- Paths (mirroring the training script) ---
    processed_data_dir = osp.join(project_root, "moghet", "data", "processed", dataset_name)
    results_dir = osp.join(project_root, "moghet", "results", f"hierarchical_model_{dataset_name.lower()}_fast")
    model_path = osp.join(results_dir, f"fold_{fold}", "best_model.pt")
    output_path = osp.join(results_dir, f"fold_{fold}", "predictions.csv")
    
    if not osp.exists(model_path):
        print(f"--- [ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Loading and Preprocessing (identical to training script) ---
    print("\n--- Loading and preprocessing data ---")
    survival_df = pd.read_csv(osp.join(processed_data_dir, "patient_survival.csv"), index_col=0)
    clinical_df = pd.read_csv(osp.join(processed_data_dir, "patient_clinical_features.csv"), index_col=0)
    
    # Align patients across files
    common_patients = set(survival_df.index).intersection(set(clinical_df.index))
    data_files_path = osp.join(processed_data_dir, "hierarchical_patient_data")
    existing_patient_ids = {f.replace('.pt', '') for f in os.listdir(data_files_path) if f.endswith('.pt')}
    all_patient_ids = np.array(sorted([pid for pid in common_patients if pid in existing_patient_ids]))
    
    if len(all_patient_ids) == 0:
        print("--- [ERROR] No matched patient data found. Check patient IDs.")
        sys.exit(1)

    print(f"Found {len(all_patient_ids)} total patients with complete data.")

    # --- Recreate the test set for the specified fold ---
    test_patient_ids = get_test_patient_ids_for_fold(
        all_patient_ids, 
        n_splits=5, 
        test_size=0.2, 
        random_state=SEED
    )
    print(f"Recreated test set for Fold {fold} with {len(test_patient_ids)} patients.")

    # --- Clinical Feature Preprocessing (must be identical to training) ---
    numeric_features = clinical_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = clinical_df.select_dtypes(include='object').columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor on ALL data to ensure consistency, same as in training
    preprocessor.fit(clinical_df.loc[all_patient_ids])
    test_features_processed = preprocessor.transform(clinical_df.loc[test_patient_ids])
    test_features_dict = {pid: f for pid, f in zip(test_patient_ids, test_features_processed)}

    # --- Dataset and DataLoader ---
    test_dataset = HierarchicalDataset(data_files_path, test_patient_ids, survival_df, test_features_dict)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=hierarchical_collate_fn, num_workers=4)

    # --- Model Definition and Loading ---
    print("\n--- Loading pre-trained model ---")
    with open(osp.join(processed_data_dir, "id_mappings.json"), 'r') as f:
        id_mappings = json.load(f)
    num_pathways = len(id_mappings['pathway_id_to_idx'])

    model_params = {
        "num_pathways": num_pathways, "pathway_embedding_dim": 8, "gnn_hidden_channels": 64, 
        "pathway_out_channels": 128, "intra_attention_hidden_channels": 128, 
        "inter_attention_hidden_channels": 128, "clinical_hidden_channels": 32, 
        "final_hidden_channels": 64
    }
    
    # These must be determined from the data itself
    # Handle cases where the test set might be empty
    if not test_dataset:
        print("Test dataset is empty. Cannot proceed.")
        return
        
    first_data_point = test_dataset[0]
    if not first_data_point['intra_pathway_graphs']:
        print("First data point has no pathway graphs. Cannot determine metadata.")
        # Attempt to find a valid data point
        found_valid = False
        for item in test_dataset:
            if item['intra_pathway_graphs']:
                first_data_point = item
                found_valid = True
                break
        if not found_valid:
            print("No valid data points with pathway graphs found in the test set. Exiting.")
            return

    first_graph = first_data_point['intra_pathway_graphs'][0]
    metadata = first_graph.metadata()
    clinical_in_features = first_data_point['clinical_features'].shape[-1]
    
    model = HierarchicalGNNModel(
        metadata=metadata,
        clinical_in_features=clinical_in_features,
        **model_params
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Model state loaded successfully.")

    # --- Prediction ---
    risk_scores = generate_predictions(model, test_loader, DEVICE)

    # --- Save Results ---
    # The DataLoader preserves the order of the test_patient_ids
    predictions_df = pd.DataFrame({'patient_id': test_patient_ids, 'risk_score': risk_scores})
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\n--- Predictions generated successfully ---")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 