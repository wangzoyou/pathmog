import torch
import torch.nn as nn

def cox_loss(risk_scores, time, events):
    """
    Calculate the correct Cox proportional hazards loss.
    This version correctly constructs the risk set based on survival time and includes numerical stability checks.

    Args:
        risk_scores (Tensor): Model's risk predictions, shape [batch_size, 1].
        time (Tensor): Survival times, shape [batch_size].
        events (Tensor): Event indicators (1=event, 0=censored), shape [batch_size].
    """
    # === Input Numerical Stability Check ===
    if torch.isnan(risk_scores).any() or torch.isinf(risk_scores).any():
        print("Warning: cox_loss input risk_scores contains NaN or infinite values")
        risk_scores = torch.nan_to_num(risk_scores, nan=0.0, posinf=10.0, neginf=-10.0)
    
    if torch.isnan(time).any() or torch.isinf(time).any():
        print("Warning: cox_loss input time contains NaN or infinite values")
        time = torch.nan_to_num(time, nan=0.0, posinf=1e6, neginf=0.0)
    
    if torch.isnan(events).any() or torch.isinf(events).any():
        print("Warning: cox_loss input events contains NaN or infinite values")
        events = torch.nan_to_num(events, nan=0.0, posinf=1.0, neginf=0.0)
    
    risk_scores = risk_scores.squeeze(1)
    time = time.squeeze()
    events = events.squeeze()

    # === Numerical Stability Handling ===
    # Limit risk score range to prevent numerical overflow
    risk_scores = torch.clamp(risk_scores, min=-10.0, max=10.0)
    
    # Ensure inputs are 1D tensors
    if risk_scores.dim() == 0:
        risk_scores = risk_scores.unsqueeze(0)
    if time.dim() == 0:
        time = time.unsqueeze(0)
    if events.dim() == 0:
        events = events.unsqueeze(0)
    
    # Ensure time values are positive
    time = torch.clamp(time, min=1e-6)

    # Sort samples by survival time in descending order
    # This makes calculating risk sets easy: at time t, the risk set consists of all samples with index >= t
    sorted_time, sort_idx = torch.sort(time, descending=True)
    sorted_risk_scores = risk_scores[sort_idx]
    sorted_events = events[sort_idx]

    # === Numerically Stable Exponential Calculation ===
    # Use log-sum-exp trick to avoid numerical overflow
    max_score = torch.max(sorted_risk_scores)
    exp_scores = torch.exp(sorted_risk_scores - max_score)
    
    # Check if exp_scores contains NaN or infinite values
    if torch.isnan(exp_scores).any() or torch.isinf(exp_scores).any():
        print("Warning: exp_scores contains NaN or infinite values, using default values")
        exp_scores = torch.ones_like(exp_scores)
    
    # Calculate log-sum of risk sets
    # cumsum from back to front simulates that at each time point, the risk set includes all "future" samples
    cumsum_exp_scores = torch.cumsum(exp_scores, dim=0)
    
    # Avoid log(0) cases
    cumsum_exp_scores = torch.clamp(cumsum_exp_scores, min=1e-8)
    log_risk_set = torch.log(cumsum_exp_scores) + max_score

    # Only select samples where events actually occurred to calculate loss (event == 1)
    # For these samples, the numerator is their own risk score, denominator is total risk in their risk set
    # In log space, this becomes log(numerator) - log(denominator)
    event_indices = torch.where(sorted_events == 1)[0]
    
    # If there are no events in the batch, loss is 0
    if len(event_indices) == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
    
    # Extract risk scores and log risk sets for samples with events
    event_risk_scores = sorted_risk_scores[event_indices]
    event_log_risk_set = log_risk_set[event_indices]
    
    # Calculate log likelihood for each event
    log_likelihood = event_risk_scores - event_log_risk_set
    
    # === Numerical Stability Check ===
    if torch.isnan(log_likelihood).any() or torch.isinf(log_likelihood).any():
        print("Warning: log_likelihood contains NaN or infinite values")
        log_likelihood = torch.nan_to_num(log_likelihood, nan=0.0, posinf=0.0, neginf=-10.0)
    
    # Loss is negative mean log likelihood
    loss = -torch.mean(log_likelihood)
    
    # === Final Numerical Stability Check ===
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: cox_loss final output contains NaN or infinite values, using default loss")
        loss = torch.tensor(1.0, device=risk_scores.device, requires_grad=True)
    
    return loss
