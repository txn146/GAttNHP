import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
from collections import Counter

# 1. Force setting environment variables to support deterministic scatter_add
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Import Loss and Utils
from loss import quantile_loss_multi, hawkes_nll_loss
from utils import metrics
from preprocess import SequenceManager
#from preprocess import SequenceManager

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def mse_loss_time(dt_pred, dt_true, mask):
    """
    Simple MSE Loss for ablation study
    """
    error = (dt_pred - dt_true) ** 2
    return (error * mask).sum() / mask.sum().clamp(min=1)

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Hawkes Process Ablation Study")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ICEWS14s', help='Dataset name')
    parser.add_argument('--root_path', type=str, default='C:/code/TKG-25-07-06/data/temporal/extrapolation', help='Data root path')
    
    # Model Selection (5 variants)
    parser.add_argument('--model', type=str, default='GAttNHP', 
                        choices=[
                            'AttNHP',                # 1. Baseline (No Group, No Time Prediction)
                            'GAttNHP',               # 2. Group Only (No Time Prediction)
                            'GAttNHP_KANNCQ',   # 3. Full Model (Group + KAN Time Prediction)
                            'GAttNHP_MLPNCQ',   # 4. Variant (Group + MLP Quantile)
                            'GAttNHP_MLPMSE'    # 5. Variant (Group + MLP MSE)
                        ], help='Model variant for ablation study')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16 )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1234)
    
    # Model Config
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--time_emb_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_groups', type=int, default=3)
    parser.add_argument('--time_weight', type=float, default=0.05, help='Weight for the time loss component')
    parser.add_argument('--max_len', type=int, default=256, help='Max sequence length for slicing GDELT')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Model save path')

    return parser.parse_args()

def get_model_class(model_name):
    """Dynamically import the corresponding model class based on name"""
    if model_name == 'AttNHP':
        from models.AttNHP import AttNHP_Model
        return AttNHP_Model
    elif model_name == 'GAttNHP':
        from models.GAttNHP import GAttNHP_Model
        return GAttNHP_Model
    elif model_name == 'GAttNHP_MLPNCQ':
        from models.GAttNHP_MLPNCQ import AttNHPDyad_GroupTime
        return AttNHPDyad_GroupTime
    elif model_name == 'GAttNHP_MLPMSE':
        from models.GAttNHP_MLPMSE import AttNHPDyad_GroupTime
        return AttNHPDyad_GroupTime
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    # 1. Initialization
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_save_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.model}_best_train_loss.pth")

    print(f">>> Loading dataset: {args.dataset}")
    print(f">>> Selected Model Variant: {args.model}")
    

    data_manager = SequenceManager(
        dataset_name=args.dataset, 
        root_path=args.root_path, 
        batch_size=args.batch_size,
        seed=args.seed,
        max_seq_len=args.max_len  
    )
    train_dl, valid_dl, test_dl = data_manager.load_and_process()
    
    n_entity = data_manager.n_entity
    n_rel = data_manager.n_rel
    
    # Group Map Logic
    train_raw = data_manager.train_data
    dyad_list = [(int(s), int(r)) for s, r, o, t in train_raw]
    counts = Counter(dyad_list)
    freqs = np.array(list(counts.values()))

    hi_p = 99
    lo_p = 90
    hi_thresh = np.percentile(freqs, hi_p)
    lo_thresh = np.percentile(freqs, lo_p)
    print("\n" + "="*40)
    print(f"DEBUG: Frequency Stats -> Max: {freqs.max()}, Min: {freqs.min()}, Mean: {freqs.mean():.2f}")
    print(f"DEBUG: Percentiles -> High({hi_p}%): {hi_thresh} | Low({lo_p}%): {lo_thresh}")
    
    group_map_tensor = torch.ones(n_entity * n_rel, dtype=torch.long) * 1 
    for (s, r), c in counts.items():
        if c >= hi_p: g_id = 0
        elif c <= lo_p: g_id = 2
        else: g_id = 1 
        idx = s * n_rel + r
        if idx < len(group_map_tensor):
            group_map_tensor[idx] = g_id
    group_map_tensor = group_map_tensor.to(device)

    # 3. Build Model
    model_config = {
        "hidden_size": args.hidden_size,
        "time_emb_size": args.time_emb_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_event_types_no_pad": n_entity
    }

    ModelClass = get_model_class(args.model)
    model = ModelClass(
        n_entity=n_entity, n_rel=n_rel, n_groups=args.n_groups,
        group_map_tensor=group_map_tensor, config=model_config
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    quantiles_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    
    # Define models that do NOT predict time
    NO_TIME_MODELS = ['AttNHP', 'GAttNHP']
    is_time_model = args.model not in NO_TIME_MODELS

    # 4. Training Loop
    # [CHANGE] Initialize best_train_loss
    best_train_loss = float('inf') 
    print(f">>> Starting training on {device}. Best model (Min Train Loss): {model_save_path}")

    for epoch in range(1, args.epochs + 1):
        # === Training Phase ===
        model.train()
        epoch_loss = 0
        total_batches = 0
        
        train_pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [Train]", unit="batch")
        
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            t_prev = batch['times'][:, :-1]
            t_next = batch['times'][:, 1:]
            targets = batch['objs'][:, 1:]
            mask_seq = batch['mask'][:, 1:]

            optimizer.zero_grad()
            
            if not is_time_model:
                lambda_t = model(
                    batch['subs'], batch['marks'], batch['objs'], 
                    batch['times'], batch['dt'], batch['mask']
                )
                loss = hawkes_nll_loss(lambda_t, targets, t_prev, t_next, mask_seq)
            else:
                lambda_t, time_output = model(
                    batch['subs'], batch['marks'], batch['objs'], 
                    batch['times'], batch['dt'], batch['mask']
                )
                loss_event = hawkes_nll_loss(lambda_t, targets, t_prev, t_next, mask_seq)
                dt_true = batch['dt'][:, 1:]
                if 'MSE' in args.model:
                    loss_time = mse_loss_time(time_output, dt_true, mask_seq)
                else:
                    loss_time = quantile_loss_multi(time_output, dt_true, quantiles_list, mask_seq)
                
                loss = loss_event + args.time_weight *loss_time

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_batches += 1
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate Average Training Loss
        avg_train_loss = epoch_loss / total_batches if total_batches > 0 else 0.0

        # [CHANGE] Save Model based on Minimum Training Loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_train_loss': best_train_loss,
                'model_name': args.model
            }, model_save_path)
            tqdm.write(f"[*] New Best Train Loss: {best_train_loss:.4f} (Saved)")

        # === Validation Phase (Only for Monitoring, NOT for saving) ===
        if epoch % 1 == 0:
            model.eval()
            val_loss_sum = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in valid_dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    t_prev = batch['times'][:, :-1]
                    t_next = batch['times'][:, 1:]
                    targets = batch['objs'][:, 1:]
                    mask_seq = batch['mask'][:, 1:]
                    
                    if not is_time_model:
                        lambda_t = model(
                            batch['subs'], batch['marks'], batch['objs'], 
                            batch['times'], batch['dt'], batch['mask']
                        )
                        loss = hawkes_nll_loss(lambda_t, targets, t_prev, t_next, mask_seq)
                    else:
                        lambda_t, time_output = model(
                            batch['subs'], batch['marks'], batch['objs'], 
                            batch['times'], batch['dt'], batch['mask']
                        )
                        loss_event = hawkes_nll_loss(lambda_t, targets, t_prev, t_next, mask_seq)
                        dt_true = batch['dt'][:, 1:]
                        if 'MSE' in args.model:
                            loss_time = mse_loss_time(time_output, dt_true, mask_seq)
                        else:
                            loss_time = quantile_loss_multi(time_output, dt_true, quantiles_list, mask_seq)
                        loss = loss_event + args.time_weight *loss_time
                    
                    val_loss_sum += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else float('inf')

            val_res = metrics.evaluate_full(
                model, valid_dl, device, time_scale=1.0, 
                calc_time=is_time_model
            )
            
            current_mrr = val_res['MRR']
            current_mae = val_res.get('MAE', 0.0) 

            msg = f"\n[Val Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MRR: {current_mrr:.4f}|Hits@1: {val_res.get('Hits@1',0):.4f}|Hits@3: {val_res.get('Hits@3',0):.4f} | Hits@10: {val_res.get('Hits@10',0):.4f}"
            if is_time_model:
                msg += f" | MAE: {current_mae:.4f}"
            tqdm.write(msg)

    # 5. Final Test
    print("\n" + "="*30)
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded BEST TRAIN LOSS model from epoch {checkpoint['epoch']} with Loss {checkpoint['best_train_loss']:.4f}")
    
    test_res = metrics.evaluate_full(
        model, test_dl, device, time_scale=1.0, 
        calc_time=is_time_model
    )
    
    print("\n" + "="*10 + f" FINAL TEST RESULTS ({args.model}) " + "="*10)
    print(f"MRR     : {test_res['MRR']:.4f}")
    print(f"Hits@1  : {test_res.get('Hits@1', 0.0):.4f}")
    print(f"Hits@3  : {test_res.get('Hits@3', 0.0):.4f}")
    print(f"Hits@10 : {test_res.get('Hits@10', 0.0):.4f}")
    if is_time_model:
        print(f"MAE     : {test_res['MAE']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()