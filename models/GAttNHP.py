import torch
import torch.nn as nn
from .layers import AttNHP

class GAttNHP_Model(nn.Module):
    def __init__(self, n_entity, n_rel, n_groups, group_map_tensor, config):
        super().__init__()
        # 1. Core Transformer Encoder
        self.core = AttNHP(config)
        
        self.n_groups = n_groups
        self.n_rel = n_rel
        self.register_buffer('group_map_tensor', group_map_tensor)
        
        # 2. Static Embeddings
        self.sub_embed = nn.Embedding(n_entity, config["hidden_size"])
        self.rel_embed = nn.Embedding(n_rel, config["hidden_size"])

        # 3. Dimension Definitions
        # Base Feature = Encoder Output (d_total) + Subject Emb + Relation Emb
        self.d_feature = self.core.d_total + config["hidden_size"] * 2
        
        # === Group Interaction Components ===
        self.group_proj_dim = 64
        self.group_proj = nn.Linear(self.d_feature, self.group_proj_dim)
        self.group_attn = nn.MultiheadAttention(embed_dim=self.group_proj_dim, num_heads=2, batch_first=True)
        
        self.group_ffn = nn.Sequential(
            nn.Linear(self.group_proj_dim, self.group_proj_dim),
            nn.ReLU(),
            nn.Linear(self.group_proj_dim, self.group_proj_dim)
        )
        self.group_norm1 = nn.LayerNorm(self.group_proj_dim)
        self.group_norm2 = nn.LayerNorm(self.group_proj_dim)
        self.merge_gate = nn.Linear(self.d_feature + self.group_proj_dim, self.d_feature)

        # 4. Prediction Heads
        
        # A. Event Intensity Prediction Head
        self.intensity_layer = nn.Sequential(
            nn.Linear(self.d_feature, n_entity),
            nn.Softplus()
        )
        
        self.eps = 1e-9

    def _apply_group_interaction(self, features, g_ids, mask):
        """
        Group Interaction Logic: Aggregation -> Interaction -> Broadcast -> Merge
        """
        B, L, D = features.shape
        flat_features = features.reshape(-1, D)    
        flat_g_ids = g_ids.reshape(-1)              
        flat_mask = mask.reshape(-1)                
        
        # 1. Aggregation (Scatter Mean)
        batch_ids = torch.arange(B, device=features.device).unsqueeze(1).expand(-1, L).reshape(-1)
        group_keys = batch_ids * self.n_groups + flat_g_ids 

        valid_indices = torch.nonzero(flat_mask).squeeze()
        if valid_indices.numel() == 0: return features

        valid_keys = group_keys[valid_indices]
        valid_feats = flat_features[valid_indices]
        
        max_key = B * self.n_groups
        group_sum = torch.zeros(max_key, D, device=features.device)
        group_count = torch.zeros(max_key, 1, device=features.device)
        
        group_sum.scatter_add_(0, valid_keys.unsqueeze(1).expand(-1, D), valid_feats)
        group_count.scatter_add_(0, valid_keys.unsqueeze(1), torch.ones_like(valid_feats[:, :1]))
        
        group_rep = group_sum / group_count.clamp(min=1) 
        group_rep = group_rep.reshape(B, self.n_groups, D) 

        # 2. Interaction (Self-Attention within groups)
        G_proj = self.group_proj(group_rep) 
        G_att, _ = self.group_attn(G_proj, G_proj, G_proj) 
        G_norm = self.group_norm1(G_proj + G_att)
        G_out = self.group_norm2(G_norm + self.group_ffn(G_norm)) 
        
        # 3. Broadcast (Gather back to individual events)
        G_flat = G_out.reshape(-1, self.group_proj_dim)
        group_enhanced_flat = torch.zeros(flat_features.size(0), self.group_proj_dim, device=features.device)
        
        gathered_feats = torch.gather(G_flat, 0, valid_keys.unsqueeze(1).expand(-1, self.group_proj_dim))
        group_enhanced_flat[valid_indices] = gathered_feats
        
        # 4. Merge (Gate)
        merged_flat = torch.cat([flat_features, group_enhanced_flat], dim=-1)
        enhanced_flat = self.merge_gate(merged_flat)
        
        return enhanced_flat.reshape(B, L, D)

    def forward(self, subs, marks, objs, times, dt, mask):
        B, L = objs.size()
        g_ids = self.group_map_tensor[subs * self.n_rel + marks]
        
        # 1. Data Slicing
        hist_objs = objs[:, :-1]
        hist_times = times[:, :-1]
        hist_dt = dt[:, :-1] # Historical Rhythm (Real Days)
        
        query_times = times[:, 1:]
        
        attn_mask = torch.tril(torch.ones(L-1, L-1, device=subs.device)).unsqueeze(0)
        
        # 2. Core Encoding
        # Pass hist_dt to let Transformer learn the rhythm
        enc_out = self.core.forward_along_seqs(
            hist_objs, hist_times, hist_dt, attn_mask, query_times
        )

        # 3. Feature Construction
        s_emb = self.sub_embed(subs[:, 0]).unsqueeze(1).expand(-1, L-1, -1)
        r_emb = self.rel_embed(marks[:, 0]).unsqueeze(1).expand(-1, L-1, -1)
        
        base_features = torch.cat([enc_out, s_emb, r_emb], dim=-1)

        # 4. Group Interaction Enhancement
        # The mask must also be sliced to correspond to the history part
        enhanced_features = self._apply_group_interaction(
            base_features, 
            g_ids[:, :-1], 
            mask[:, :-1]
        )

        # 5. Prediction
        lambda_t = self.intensity_layer(enhanced_features)
        
        return lambda_t