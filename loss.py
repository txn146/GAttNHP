import torch
import torch.nn.functional as F

def hawkes_nll_loss(lambda_t, targets, t_prev, t_next, mask, eps=1e-9):
    """
    Args:
        lambda_t: [B, L, N_entity] 
        targets:  [B, L] 
        t_prev:   [B, L] 
        t_next:   [B, L] 
        mask:     [B, L] 
    """
    # 1. Log-Likelihood of the target event
    lambda_target = lambda_t.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    log_intensity = torch.log(lambda_target + eps)

    # 2. Integral of total intensity 
    dt = (t_next - t_prev).clamp(min=0)
    total_lambda = lambda_t.sum(dim=-1)
    integral = dt * total_lambda

    # 3. NLL = - (Log_Likelihood - Integral)
    nll = -(log_intensity - integral)

    # Apply mask and average
    return (nll * mask).sum() / mask.sum().clamp(min=1)

def quantile_loss_multi(quantile_preds, delta_true, quantiles, mask):
    """
    Args:
        quantile_preds: [B, L, n_quantiles] 
        delta_true:     [B, L] 
        quantiles:      List or Tensor [q1, q2, ...] (e.g., [0.1, 0.5, 0.9])
    """
    B, L, n_q = quantile_preds.shape
    device = quantile_preds.device

    delta_true = delta_true.unsqueeze(-1)

    if not isinstance(quantiles, torch.Tensor):
        quantiles = torch.tensor(quantiles, device=device)
    quantiles = quantiles.view(1, 1, n_q)

    diff = delta_true - quantile_preds

    # Pinball Loss 
    loss = torch.where(diff >= 0, quantiles * diff, (quantiles - 1) * diff)

    masked_loss = loss * mask.unsqueeze(-1)

    return masked_loss.sum() / (mask.sum() * n_q).clamp(min=1)

def mse_loss_time(dt_pred, dt_true, mask):
    """
    Args:
        dt_pred: [B, L] 
        dt_true: [B, L] 
    """
    error = (dt_pred - dt_true) ** 2
    return (error * mask).sum() / mask.sum().clamp(min=1)