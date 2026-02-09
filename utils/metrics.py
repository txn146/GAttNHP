import numpy as np
import torch
import re


def calculate_rank(scores: torch.Tensor,
                   target_index: torch.Tensor):
    device = scores.device   
    x = torch.Tensor(scores) 
    _, sindex = torch.sort(-x, dim=1)  
    res = torch.zeros_like(x, dtype=sindex.dtype, device=device) 
    index = torch.arange(x.shape[1], device=device).expand(x.shape)
    t_rank = res.scatter(dim=1, index=sindex, src=index).float() + 1
    return t_rank[torch.arange(scores.shape[0]), target_index]


def calculate_hist(k: int, ranks: torch.Tensor):
    return float((ranks <= k).sum() / len(ranks))


def calculate_mrr(ranks: torch.Tensor):
    return float((1. / ranks).sum() / ranks.shape[0])


def calculate_mr(ranks: torch.Tensor):
    return float(ranks.mean())


def ranks_to_metrics(metric_list: list,
                     ranks,
                     filter_out=False):
    metrics = {}
    prefix = ""
    if filter_out:
        prefix = "filter "
    for metric in metric_list:
        if re.match(r'hits@\d+', metric):
            n = int(re.findall(r'\d+', metric)[0])
            metrics[prefix + metric] = calculate_hist(n, ranks)
        elif metric == 'mr':
            metrics[prefix + 'mr'] = calculate_mr(ranks)
        elif metric == 'mrr':
            metrics[prefix + 'mrr'] = calculate_mrr(ranks)
    return metrics


@torch.no_grad()
def evaluate_full(model, dataloader, device, time_scale=1.0, ks=[1, 3, 10], calc_time=True):
    model.eval()
    all_ranks = []
    total_mae = 0.0
    total_count = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mask = batch['mask'][:, 1:] 
        
        if mask.sum() == 0: continue
        
        output = model(
            batch['subs'], batch['marks'], batch['objs'], 
            batch['times'], batch['dt'], batch['mask']
        )
        
        if isinstance(output, tuple):
            lambda_t = output[0]
            time_output = output[1]
        else:
            lambda_t = output
            time_output = None
        
        valid_scores = lambda_t[mask]
        valid_targets = batch['objs'][:, 1:][mask]
        
        if len(valid_targets) > 0:
            batch_ranks = calculate_rank(valid_scores, valid_targets)
            all_ranks.append(batch_ranks.cpu())

        if calc_time and time_output is not None:
            if time_output.dim() == 3: 
                median_idx = time_output.shape[-1] // 2 
                dt_pred = time_output[..., median_idx] 
            else:
                dt_pred = time_output
            
            dt_true = batch['dt'][:, 1:] 
            
            if dt_pred.shape[1] != dt_true.shape[1]:
                min_len = min(dt_pred.shape[1], dt_true.shape[1])
                dt_pred = dt_pred[:, :min_len]
                dt_true = dt_true[:, :min_len]
                mask = mask[:, :min_len]

            error = torch.abs(dt_pred - dt_true)
            total_mae += (error * mask).sum().item() 
            total_count += mask.sum().item()

    results = {}
    
    # MRR / Hits
    if len(all_ranks) > 0:
        all_ranks = torch.cat(all_ranks)
        results["MRR"] = calculate_mrr(all_ranks)
        for k in ks:
            results[f"Hits@{k}"] = calculate_hist(k, all_ranks)
    else:
        results["MRR"] = 0.0
        for k in ks: results[f"Hits@{k}"] = 0.0
            
    # MAE
    if calc_time:
        results["MAE"] = total_mae / total_count if total_count > 0 else 0.0
    else:
        results["MAE"] = 0.0 
        
    return results