import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from data.data_loader import DataLoader as TKGDataLoader

class DyadSeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]

def pad_collate(batch):
    B = len(batch)
    L = max(len(x['objs']) for x in batch)
    res = {}
    for key in ['subs', 'marks', 'objs']:
        tensor = torch.zeros((B, L), dtype=torch.long)
        for i, b in enumerate(batch):
            tensor[i, :len(b[key])] = torch.from_numpy(b[key])
        res[key] = tensor
    
    times = torch.zeros((B, L), dtype=torch.float32)
    dt = torch.zeros((B, L), dtype=torch.float32)
    mask = torch.zeros((B, L), dtype=torch.bool)
    
    for i, b in enumerate(batch):
        seq_len = len(b['times'])
        times[i, :seq_len] = torch.from_numpy(b['times'])
        dt[i, :seq_len] = torch.from_numpy(b['dt']) 
        mask[i, :seq_len] = True
        
    res['times'], res['dt'], res['mask'] = times, dt, mask
    return res

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SequenceManager:
    def __init__(self, dataset_name, root_path, batch_size=32, seed=1234, max_seq_len=256):
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.seed = seed 
        
        self.max_seq_len = max_seq_len  
        
        self.t_min = 0
        self.time_scale = 1.0 
        self.train_data = None 

    def load_and_process(self):
        data_manager = TKGDataLoader(dataset=self.dataset_name, root_path=self.root_path)
        data_manager.load()
        
        train_raw = data_manager.train
        self.train_data = train_raw 
        valid_raw = data_manager.valid
        test_raw = data_manager.test

        all_data = np.concatenate([train_raw, valid_raw, test_raw], axis=0)
        self.n_entity = int(all_data[:, [0, 2]].max()) + 1
        self.n_rel = int(all_data[:, 1].max()) + 1
        self.t_min = float(all_data[:, 3].min())
        t_max = float(all_data[:, 3].max())
        
        self.time_scale = t_max - self.t_min + 1e-8
        print(f"Time Scale (Global): {self.time_scale} (Normalized for MRR)")
        
        train_seqs = self._build_sequences(train_raw, is_train=True)
        valid_seqs = self._build_sequences(valid_raw, is_train=False)
        test_seqs = self._build_sequences(test_raw, is_train=False)
        
        print(f"Generated Sequences: Train {len(train_seqs)}, Valid {len(valid_seqs)}, Test {len(test_seqs)}")

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_dl = DataLoader(DyadSeqDataset(train_seqs), self.batch_size, shuffle=True, collate_fn=pad_collate, worker_init_fn=seed_worker, generator=g)
        valid_dl = DataLoader(DyadSeqDataset(valid_seqs), self.batch_size, shuffle=False, collate_fn=pad_collate, worker_init_fn=seed_worker, generator=g)
        test_dl = DataLoader(DyadSeqDataset(test_seqs), self.batch_size, shuffle=False, collate_fn=pad_collate, worker_init_fn=seed_worker, generator=g)

        return train_dl, valid_dl, test_dl

    def _build_sequences(self, array: np.ndarray, min_len: int = 2, is_train: bool = False) -> List[Dict]:
        array = array[np.argsort(array[:, 3])]
        buckets = {}

        for s, r, o, t in array:
            key = (int(s), int(r))
            buckets.setdefault(key, []).append((int(s), int(r), int(o), float(t)))
        
        seqs = []
        
        for g in buckets.values():
            if len(g) < min_len: continue
            g.sort(key=lambda x: x[3]) 
            
            total_len = len(g)
            
            stride = self.max_seq_len // 2 if is_train else self.max_seq_len
            
            if total_len <= self.max_seq_len:
                slices = [g]
            else:
                slices = []
                for i in range(0, total_len, stride):
                    chunk = g[i : i + self.max_seq_len]
                    if len(chunk) >= min_len: 
                        slices.append(chunk)
            
            for chunk in slices:

                chunk_subs = np.array([x[0] for x in chunk], dtype=np.int64)
                chunk_marks = np.array([x[1] for x in chunk], dtype=np.int64)
                chunk_objs = np.array([x[2] for x in chunk], dtype=np.int64)
                raw_times = np.array([x[3] for x in chunk], dtype=np.float32)
                
                base_time = raw_times[0]
                local_times = raw_times - base_time 
                
                norm_times = local_times / self.time_scale 
                
                dt = np.zeros_like(raw_times)
                if len(raw_times) > 1:
                    dt[1:] = raw_times[1:] - raw_times[:-1]
                
                seqs.append({
                    'subs': chunk_subs,
                    'marks': chunk_marks,
                    'objs': chunk_objs,
                    'times': norm_times.astype(np.float32), 
                    'dt': dt.astype(np.float32),            
                    'mask': np.ones(len(chunk), dtype=bool)
                })
                
        return seqs