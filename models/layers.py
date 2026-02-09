import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPNonCrossingQuantile(nn.Module):
    def __init__(self, d_context, quantiles=[0.1, 0.5, 0.9], hidden_size=64):
        super().__init__()
        self.quantiles = sorted(quantiles)
        self.n_quantiles = len(quantiles)
        self.d_context = d_context
        
        self.net = nn.Sequential(
            nn.Linear(d_context, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_quantiles)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        with torch.no_grad():
            last_layer = self.net[-1]
            last_layer.bias[1:].fill_(-5.0) 
            last_layer.bias[0].fill_(0.0)

    def forward(self, context):
        B, L, _ = context.shape
        x = context.reshape(B * L, self.d_context)
        raw_output = self.net(x)
        base = raw_output[:, 0:1]
        steps = F.softplus(raw_output[:, 1:])
        
        # CPU Cumsum fix for determinism
        steps_cpu = steps.cpu()
        cumsum_cpu = torch.cumsum(steps_cpu, dim=1)
        cumsum_gpu = cumsum_cpu.to(steps.device)
        
        quantiles_flat = torch.cat([base, base + cumsum_gpu], dim=1)
        return quantiles_flat.reshape(B, L, self.n_quantiles)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, L, D = q.size()
        q = self.q_linear(q).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3: mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_linear(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, d_model, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        return x + self.dropout(self.attn(x, x, x, mask))

class AttNHP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.d_model = model_config['hidden_size']
        self.d_time = model_config['time_emb_size']
        self.dt_time = model_config['time_emb_size']
        
        self.d_total = self.d_model + self.d_time + self.dt_time  
        
        self.layer_event_emb = nn.Embedding(model_config['num_event_types_no_pad'] + 1, self.d_model, padding_idx=0)
        
        self.dt_encoder = nn.Sequential(
            nn.Linear(1, self.dt_time),
            nn.Tanh()
        )
        
        div_term = torch.exp(torch.arange(0, self.d_time, 2).float() * -(math.log(10000.0) / self.d_time))
        self.register_buffer('div_term', div_term)
        
        self.layers = nn.ModuleList([EncoderLayer(self.d_total, model_config['num_heads'], model_config['dropout']) for _ in range(model_config['num_layers'])])
        self.norm = nn.LayerNorm(self.d_total)

    def compute_temporal_embedding(self, time):
        B, L = time.size()
        pe = torch.zeros(B, L, self.d_time, device=time.device)
        _time = time.unsqueeze(-1)
        pe[..., 0::2] = torch.sin(_time * self.div_term)
        pe[..., 1::2] = torch.cos(_time * self.div_term)
        return pe

    def forward_along_seqs(self, event_seqs, time_seqs, dt_seqs, attn_mask, query_times):
        """
        Args:
            event_seqs
            time_seqs
            dt_seqs
            query_times
        """
        # 1. History Encoding
        hist_emb = torch.tanh(self.layer_event_emb(event_seqs))
        hist_time_emb = self.compute_temporal_embedding(time_seqs)
        
        if dt_seqs is not None:
            log_dt = torch.log(dt_seqs.clamp(min=1e-6)).unsqueeze(-1)
            hist_dt_emb = self.dt_encoder(log_dt)
        else:
            hist_dt_emb = torch.zeros(hist_emb.size(0), hist_emb.size(1), self.d_model, device=hist_emb.device)
        
        # Concat: [Entity, AbsTime, Rhythm]
        enc_input = torch.cat([hist_emb, hist_time_emb, hist_dt_emb], dim=-1)
        
        # 2. Query Encoding
        curr_time_emb = self.compute_temporal_embedding(query_times)
        

        curr_dt_emb = torch.zeros(curr_time_emb.size(0), curr_time_emb.size(1), self.dt_time, device=curr_time_emb.device)
        curr_input = torch.cat([torch.zeros_like(hist_emb), curr_time_emb, curr_dt_emb], dim=-1)
        
        # 3. Transformer
        x = torch.cat([enc_input, curr_input], dim=1)
        
        B, L = event_seqs.size(0), event_seqs.size(1)
        full_mask = torch.zeros(B, 2*L, 2*L, device=event_seqs.device)
        full_mask[:, :L, :L] = attn_mask
        full_mask[:, L:, :L] = attn_mask
        full_mask[:, L:, L:] = torch.eye(L, device=event_seqs.device).unsqueeze(0)
        
        for layer in self.layers: x = layer(x, full_mask)
        return self.norm(x)[:, L:, :]