import torch
from tqdm import tqdm
import numpy as np
import sys
import os.path as osp

# 添加项目路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.auc_evaluation import SurvivalAUCEvaluator

def train_one_epoch(model, loader, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    valid_batches = 0
    optimizer.zero_grad() # Clear gradients before the loop

    # (Core modification) Wrap loader with tqdm to display progress bar
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for i, batch_data in enumerate(progress_bar):
        # Skip if batch is invalid
        if batch_data is None:
            continue
            
        # Get data and move to device
        graphs_batch = batch_data['graphs_batch'].to(device)
        clinical_features = batch_data['clinical_features'].to(device)
        time = batch_data['time'].to(device)
        event = batch_data['event'].to(device)
        
        # === Input Data Numerical Stability Check ===
        if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
            print(f"Warning: Batch {i} clinical_features contains NaN or infinite values")
            clinical_features = torch.nan_to_num(clinical_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(time).any() or torch.isinf(time).any():
            print(f"Warning: Batch {i} time contains NaN or infinite values")
            time = torch.nan_to_num(time, nan=1.0, posinf=1e6, neginf=1.0)
        
        if torch.isnan(event).any() or torch.isinf(event).any():
            print(f"Warning: Batch {i} event contains NaN or infinite values")
            event = torch.nan_to_num(event, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Forward pass
        risk_scores = model(graphs_batch, clinical_features)
        
        # Add NaN detection
        if torch.isnan(risk_scores).any():
            print(f"NaN values detected in risk_scores during training! Batch: {i}")
            print(f"risk_scores shape: {risk_scores.shape}")
            print(f"NaN positions: {torch.isnan(risk_scores).nonzero()}")
            print(f"risk_scores values: {risk_scores}")
            # Skip this batch
            continue
        
        # (Correct) Use Cox loss function, standard method for survival analysis
        from .loss import cox_loss
        loss = cox_loss(risk_scores, time, event)
        
        # === Loss Value Numerical Stability Check ===
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Batch {i} loss is NaN or infinite: {loss.item()}")
            continue
        
        if loss.item() <= 0:
            print(f"Warning: Batch {i} loss is non-positive: {loss.item()}")
            continue
        
        # Only process if loss is valid
        if loss.item() > 0 and not torch.isnan(loss) and not torch.isinf(loss):
            # Normalize loss according to accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass, accumulate gradients
            loss.backward()
            
            # === Gradient Monitoring and Clipping ===
            # Check if gradients contain NaN or infinite values
            total_norm = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Check gradients for individual parameters
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: Parameter gradients contain NaN or infinite values, setting to zero")
                        param.grad.data.zero_()
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                # Gradient clipping
                max_grad_norm = 1.0
                if total_norm > max_grad_norm:
                    print(f"Batch {i}: Gradient clipped, original norm: {total_norm:.4f}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Record gradient information
                if i % 10 == 0:  # Record every 10 batches
                    print(f"Batch {i}: Loss={loss.item():.4f}, Gradient norm={total_norm:.4f}")

        # Update weights every N steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # Check if gradients are valid
            valid_gradients = True
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: Invalid gradients detected before update, skipping this update")
                        valid_gradients = False
                        break
            
            if valid_gradients:
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                continue
        
        # Accumulate loss
        total_loss += loss.item()
        valid_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / max(valid_batches, 1):.4f}'
        })
    
    # Return average loss
    avg_loss = total_loss / max(valid_batches, 1)
    print(f"Training completed: Valid batches={valid_batches}, Average loss={avg_loss:.4f}")
    return avg_loss

def evaluate(model, loader, device, return_predictions=False):
    """
    Evaluate model performance
    (Correct) Survival data is used to calculate C-Index and AUC, with comprehensive numerical stability checks
    
    Args:
        model: Model
        loader: Data loader
        device: Device
        return_predictions: Whether to return detailed prediction results (including patient IDs)
    """
    model.eval()
    all_risk_scores, all_times, all_events = [], [], []
    all_patient_ids = []  # Collect patient IDs
    
    with torch.no_grad():
        # (Core modification) Wrap loader with tqdm to display progress bar
        for batch_data in tqdm(loader, desc="Evaluating", leave=False):
            # Skip if batch is invalid
            if batch_data is None:
                continue

            # Get data
            graphs_batch = batch_data['graphs_batch'].to(device)
            clinical_features = batch_data['clinical_features'].to(device)
            
            # Survival data remains on CPU for final calculations
            time = batch_data['time']
            event = batch_data['event']

            # === Input Data Numerical Stability Check ===
            if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
                print("Warning: clinical_features contains NaN or infinite values during evaluation")
                clinical_features = torch.nan_to_num(clinical_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if torch.isnan(time).any() or torch.isinf(time).any():
                print("Warning: time contains NaN or infinite values during evaluation")
                time = torch.nan_to_num(time, nan=1.0, posinf=1e6, neginf=1.0)
            
            if torch.isnan(event).any() or torch.isinf(event).any():
                print("Warning: event contains NaN or infinite values during evaluation")
                event = torch.nan_to_num(event, nan=0.0, posinf=1.0, neginf=0.0)

            risk_scores = model(graphs_batch, clinical_features)
            
            # Add NaN detection and debugging information
            if torch.isnan(risk_scores).any():
                print(f"Warning: NaN values detected in risk_scores!")
                print(f"risk_scores shape: {risk_scores.shape}")
                print(f"NaN positions: {torch.isnan(risk_scores).nonzero()}")
                print(f"risk_scores values: {risk_scores}")
                # Replace NaN with 0
                risk_scores = torch.nan_to_num(risk_scores, nan=0.0)
            
            # Check for infinite values
            if torch.isinf(risk_scores).any():
                print(f"Warning: Infinite values detected in risk_scores!")
                risk_scores = torch.nan_to_num(risk_scores, posinf=10.0, neginf=-10.0)
            
            # Limit risk score range
            risk_scores = torch.clamp(risk_scores, min=-10.0, max=10.0)
            
            all_risk_scores.append(risk_scores.cpu().numpy())
            all_times.append(time.cpu().numpy())
            all_events.append(event.cpu().numpy())
            
            # Collect patient IDs if detailed predictions are needed
            if return_predictions and 'patient_ids' in batch_data:
                all_patient_ids.extend(batch_data['patient_ids'])
            
    if not all_risk_scores:
        print("Warning: No valid prediction scores were generated during evaluation, cannot calculate metrics. Returning default values.")
        return {'c_index': 0, 'auc': 0.5}

    all_risk_scores = np.concatenate(all_risk_scores).squeeze()
    all_times = np.concatenate(all_times).squeeze()
    all_events = np.concatenate(all_events).squeeze()
    
    # === Detailed Data Quality Check ===
    print(f"Evaluation data statistics:")
    print(f"  Risk scores: shape={all_risk_scores.shape}, range=[{all_risk_scores.min():.4f}, {all_risk_scores.max():.4f}]")
    print(f"  Survival times: shape={all_times.shape}, range=[{all_times.min():.4f}, {all_times.max():.4f}]")
    print(f"  Event status: shape={all_events.shape}, events={np.sum(all_events)}, censored={len(all_events)-np.sum(all_events)}")
    
    # Calculate all metrics using the refactored evaluator
    try:
        evaluator = SurvivalAUCEvaluator()
        results = evaluator.evaluate_model_performance({
            'risk_scores': all_risk_scores,
            'times': all_times,
            'events': all_events
        })
        
        print(f"  Evaluation metrics calculated successfully:")
        print(f"    C-Index: {results.get('c_index', 0.5):.4f}")
        print(f"    Time-Dependent AUC (Formula 15): {results.get('time_dependent_auc', 0.5):.4f}")
        
        # For backward compatibility, ensure 'auc' key exists and points to time-dependent AUC
        if 'time_dependent_auc' in results and 'auc' not in results:
            results['auc'] = results['time_dependent_auc']
        
        # If detailed predictions are needed, add raw data
        if return_predictions:
            results['predictions'] = {
                'patient_ids': all_patient_ids,
                'risk_scores': all_risk_scores,
                'survival_times': all_times,
                'events': all_events
            }

        return results
    except Exception as e:
        print(f"  All evaluation metrics calculation failed: {e}")
        return {
            'c_index': 0.5,
            'auc': 0.5,
            'time_dependent_auc': 0.5,
            'integrated_auc': 0.5,
            'standard_auc': 0.5,
            'time_specific_auc': {},
            'num_samples': len(all_risk_scores),
            'num_events': int(np.sum(all_events))
        }
