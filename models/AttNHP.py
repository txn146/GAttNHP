import torch
import torch.nn as nn
from .layers import AttNHP

class AttNHP_Model(nn.Module):
    def __init__(self, n_entity, n_rel, n_groups, group_map_tensor, config):
        super().__init__()
    
        self.core = AttNHP(config)
        
        self.sub_embed = nn.Embedding(n_entity, config["hidden_size"])
        self.rel_embed = nn.Embedding(n_rel, config["hidden_size"])

        self.d_feature = self.core.d_total + config["hidden_size"] * 2
        
        self.intensity_layer = nn.Sequential(
            nn.Linear(self.d_feature, n_entity),
            nn.Softplus()
        )
        
        
        self.eps = 1e-9

    def forward(self, subs, marks, objs, times, dt, mask):
        """
        No Group Interaction Version
        """
        B, L = objs.size()
        
        hist_objs = objs[:, :-1]
        hist_times = times[:, :-1]
        hist_dt = dt[:, :-1] 
        
        query_times = times[:, 1:]
        
        attn_mask = torch.tril(torch.ones(L-1, L-1, device=subs.device)).unsqueeze(0)
        
        enc_out = self.core.forward_along_seqs(
            hist_objs, hist_times, hist_dt, attn_mask, query_times
        )

        s_emb = self.sub_embed(subs[:, 0]).unsqueeze(1).expand(-1, L-1, -1)
        r_emb = self.rel_embed(marks[:, 0]).unsqueeze(1).expand(-1, L-1, -1)
        
        base_features = torch.cat([enc_out, s_emb, r_emb], dim=-1)

        features = base_features 

        lambda_t = self.intensity_layer(features)
        
        
        return lambda_t