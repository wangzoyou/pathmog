import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, global_mean_pool # 修改: 导入 HGTConv
from torch_geometric.utils import to_dense_batch

# (新增) 导入HOM模块
from models.hom_module import HierarchicalOmicsModulation

class AttentionPooling(nn.Module):
    """
    An attention-based pooling layer. It computes a weighted sum of all pathway vectors for each patient,
    where the weights are automatically learned by the model.
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # A simple feedforward network to compute attention scores for each pathway vector
        self.attention_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, batch_map):
        """
        Args:
            x (Tensor): All pathway vectors for all patients in a batch.
                        Shape: [total number of pathways in batch, pathway vector dimension]
            batch_map (Tensor): Maps each pathway to its patient index in the batch.
                                Shape: [total number of pathways in batch]
        """
        # === Input Numerical Stability Check ===
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: AttentionPooling input x contains NaN or infinite values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Convert batched, variable-length pathway vector list into a dense tensor for easier patient-wise operations.
        # x_dense shape: [batch size, max pathways per patient, pathway vector dimension]
        # mask shape: [batch size, max pathways per patient] (marks which are real pathways vs padding)
        x_dense, mask = to_dense_batch(x, batch_map)
        
        # Compute attention scores for each pathway vector
        # attention_scores shape: [batch size, max pathways per patient, 1]
        attention_scores = self.attention_net(x_dense)
        
        # === Numerical Stability Handling ===
        # Limit attention scores range to prevent numerical overflow
        attention_scores = torch.clamp(attention_scores, min=-10.0, max=10.0)
        
        # Before calculating softmax, set attention scores for padding positions to a very small value to mask their impact
        attention_scores[~mask] = -1e9
        
        # Apply softmax across the pathway dimension for each patient to get normalized attention weights
        # Use more stable softmax implementation
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # === Numerical Stability Check ===
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            print("Warning: attention_weights in AttentionPooling contains NaN or infinite values, using uniform weights")
            # Use uniform weights as fallback
            attention_weights = torch.ones_like(attention_weights)
            attention_weights[~mask] = 0.0
            # Re-normalize
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Use attention weights to compute weighted sum of pathway vectors, resulting in a comprehensive omics vector for each patient
        # (x_dense * attention_weights) -> Element-wise multiplication using broadcasting
        # .sum(dim=1) -> Sum over the patient's pathway dimension
        # Return shape: [batch size, pathway vector dimension]
        patient_omics_vector = (x_dense * attention_weights).sum(dim=1)
        
        # === Final Numerical Stability Check ===
        if torch.isnan(patient_omics_vector).any() or torch.isinf(patient_omics_vector).any():
            print("Warning: AttentionPooling output contains NaN or infinite values, using mean pooling")
            # Use mean pooling as fallback
            patient_omics_vector = x_dense.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            # Check again
            if torch.isnan(patient_omics_vector).any() or torch.isinf(patient_omics_vector).any():
                print("Warning: Mean pooling also failed, using zero vectors")
                patient_omics_vector = torch.zeros_like(patient_omics_vector)
        
        return patient_omics_vector

class HierarchicalGNNModel(nn.Module):
    """
    (Updated) GNN model with Hierarchical Omics Modulation (HOM) for high-dimensional feature fusion.
    """
    def __init__(self, gnn_hidden_channels, pathway_out_channels, 
                 metadata, 
                 # (Added) Receive number of pathways and embedding dimension
                 num_pathways, pathway_embedding_dim,
                 intra_attention_hidden_channels, 
                 inter_attention_hidden_channels,
                 clinical_in_features, clinical_hidden_channels, 
                 final_hidden_channels, dropout_rate=0.3):
        super().__init__()

        # --- 1. Clinical and Pathway Context Encoder ---
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_in_features, clinical_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Pathway identity embedding layer
        self.pathway_embedding = nn.Embedding(
            num_embeddings=num_pathways, 
            embedding_dim=pathway_embedding_dim
        )
        
        # Pathway status encoder, input dimension is identity + status(CNV_mean, Mut_mean)
        pathway_context_dim = pathway_embedding_dim + 2 
        self.pathway_encoder = nn.Linear(pathway_context_dim, 16) # Example dimension: encode to 16 dimensions

        # --- 2. Hierarchical Omics Modulation (HOM) Module ---
        self.hom_modulation = HierarchicalOmicsModulation(
            modulator_dim=2, # CNV, mutation
            clinical_dim=clinical_hidden_channels,
            pathway_dim=16, # Encoded pathway context dimension
            hidden_dim=32
        )

        # (Core modification) Calculate the final high-dimensional feature dimension input to GNN
        # 1(mod_expr) + 1(raw_expr) + 2(cnv,mut) + 1(gamma) + 1(beta) + 8(path_ident) + 2(path_state) = 16
        gnn_in_channels = 1 + 1 + 2 + 1 + 1 + pathway_embedding_dim + 2
        
        # --- 3. Intra-Pathway Heterogeneous GNN ---
        # Input dimension is now the calculated high-dimensional feature dimension
        self.intra_pathway_gnn1 = HGTConv(
            in_channels=gnn_in_channels, 
            out_channels=gnn_hidden_channels, 
            metadata=metadata, 
            heads=4
        )
        self.intra_pathway_gnn2 = HGTConv(
            in_channels=gnn_hidden_channels, 
            out_channels=pathway_out_channels, 
            metadata=metadata, 
            heads=1
        )

        # --- 4. Intra-Pathway Attention Pooling ---
        self.intra_pathway_pooling = AttentionPooling(
            in_channels=pathway_out_channels,
            hidden_channels=intra_attention_hidden_channels
        )
        
        # --- 5. Inter-Pathway Attention Pooling ---
        self.inter_pathway_pooling = AttentionPooling(
            in_channels=pathway_out_channels,
            hidden_channels=inter_attention_hidden_channels
        )

        # --- 6. Clinical Feature Encoder (no longer directly used before prediction head in this model) ---
        # self.clinical_encoder ... already defined earlier

        # --- 7. Final Fusion and Prediction Head ---
        self.survival_head = nn.Sequential(
            nn.Linear(pathway_out_channels + clinical_hidden_channels, final_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden_channels, 1) # Output single risk score
        )

    def forward(self, data, clinical_features):
        """
        Model forward pass (updated to implement HOM and high-dimensional feature fusion)
        """
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        gene_batch_map = data['gene'].batch # gene -> pathway mapping
        
        # === Input Data Numerical Stability Check ===
        if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
            print("Warning: Model input clinical_features contains NaN or infinite values")
            clinical_features = torch.nan_to_num(clinical_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        for key, x in x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: Model input x_dict[{key}] contains NaN or infinite values")
                x_dict[key] = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # === Step 1: Generate and Distribute Various Contexts ===
        
        # 1a. Encode patient clinical features and distribute
        patient_clinical_vector = self.clinical_encoder(clinical_features)
        
        # Numerical stability check
        if torch.isnan(patient_clinical_vector).any() or torch.isinf(patient_clinical_vector).any():
            print("Warning: clinical_encoder output contains NaN or infinite values")
            patient_clinical_vector = torch.nan_to_num(patient_clinical_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        clinical_context_per_pathway = patient_clinical_vector[data.pathway_to_patient_batch_map]
        clinical_context_per_gene = clinical_context_per_pathway[gene_batch_map]
        
        # 1b. Generate and encode pathway context
        #   i. Extract pathway "identity"
        #   Note: Need to directly get unique index for each pathway from batch data 'data'
        pathway_indices_in_batch = data.pathway_idx 
        pathway_identity_vec = self.pathway_embedding(pathway_indices_in_batch)
        
        # Numerical stability check
        if torch.isnan(pathway_identity_vec).any() or torch.isinf(pathway_identity_vec).any():
            print("Warning: pathway_embedding output contains NaN or infinite values")
            pathway_identity_vec = torch.nan_to_num(pathway_identity_vec, nan=0.0, posinf=1e6, neginf=-1e6)
        
        #   ii. Extract pathway "status"
        modulator_features = x_dict['gene'][:, 1:3] # CNV, mutation
        pathway_state_vec = global_mean_pool(modulator_features, gene_batch_map)
        
        # Numerical stability check
        if torch.isnan(pathway_state_vec).any() or torch.isinf(pathway_state_vec).any():
            print("Warning: pathway_state_vec contains NaN or infinite values")
            pathway_state_vec = torch.nan_to_num(pathway_state_vec, nan=0.0, posinf=1e6, neginf=-1e6)
        
        #   iii. Concatenate identity and status, then encode
        full_pathway_context = torch.cat([pathway_identity_vec, pathway_state_vec], dim=1)
        pathway_context_encoded = self.pathway_encoder(full_pathway_context)
        
        # Numerical stability check
        if torch.isnan(pathway_context_encoded).any() or torch.isinf(pathway_context_encoded).any():
            print("Warning: pathway_encoder output contains NaN or infinite values")
            pathway_context_encoded = torch.nan_to_num(pathway_context_encoded, nan=0.0, posinf=1e6, neginf=-1e6)
        
        pathway_context_per_gene = pathway_context_encoded[gene_batch_map]

        # === Step 2: HOM Intelligent Modulation of Gene Expression ===
        modulated_expression, gamma, beta = self.hom_modulation(
            x_dict['gene'],
            pathway_context_per_gene,
            clinical_context_per_gene
        )
        
        # === Step 3: Concatenate into High-Dimensional Feature Vector ===
        fused_features = torch.cat([
            modulated_expression,                 # [N_genes, 1]
            x_dict['gene'][:, 0:1],               # [N_genes, 1] raw_expression
            x_dict['gene'][:, 1:3],               # [N_genes, 2] cnv, mut
            gamma,                                # [N_genes, 1]
            beta,                                 # [N_genes, 1]
            pathway_identity_vec[gene_batch_map], # [N_genes, 8] pathway_identity
            pathway_state_vec[gene_batch_map]     # [N_genes, 2] pathway_state
        ], dim=1)
        
        # Numerical stability check
        if torch.isnan(fused_features).any() or torch.isinf(fused_features).any():
            print("Warning: fused_features contains NaN or infinite values")
            fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Replace original 'gene' features with high-dimensional fused features
        x_dict['gene'] = fused_features

        # === Step 4: Intra-Pathway GNN Processing ===
        x_dict = self.intra_pathway_gnn1(x_dict, edge_index_dict)
        
        # Numerical stability check
        for key, x in x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: intra_pathway_gnn1 output x_dict[{key}] contains NaN or infinite values")
                x_dict[key] = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = self.intra_pathway_gnn2(x_dict, edge_index_dict)
        
        # Numerical stability check
        for key, x in x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: intra_pathway_gnn2 output x_dict[{key}] contains NaN or infinite values")
                x_dict[key] = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x = x_dict['gene']

        # === Step 5: Intra-Pathway Attention Pooling (Core Modification) ===
        path_vectors = self.intra_pathway_pooling(x, data['gene'].batch)
        
        # Numerical stability check
        if torch.isnan(path_vectors).any() or torch.isinf(path_vectors).any():
            print("Warning: intra_pathway_pooling output contains NaN or infinite values")
            path_vectors = torch.nan_to_num(path_vectors, nan=0.0, posinf=1e6, neginf=-1e6)

        # === Step 6: Inter-Pathway Attention Pooling ===
        patient_omics_vector = self.inter_pathway_pooling(
            path_vectors, 
            data.pathway_to_patient_batch_map
        )
        
        # Numerical stability check
        if torch.isnan(patient_omics_vector).any() or torch.isinf(patient_omics_vector).any():
            print("Warning: inter_pathway_pooling output contains NaN or infinite values")
            patient_omics_vector = torch.nan_to_num(patient_omics_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # === Step 7: Final Aggregation and Prediction ===
        final_vector = torch.cat([patient_omics_vector, patient_clinical_vector], dim=1)
        
        # Numerical stability check
        if torch.isnan(final_vector).any() or torch.isinf(final_vector).any():
            print("Warning: final_vector contains NaN or infinite values")
            final_vector = torch.nan_to_num(final_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        risk_score = self.survival_head(final_vector)
        
        # Final output numerical stability check
        if torch.isnan(risk_score).any() or torch.isinf(risk_score).any():
            print("Warning: Model final output risk_score contains NaN or infinite values, using default values")
            risk_score = torch.zeros_like(risk_score)
        
        return risk_score 