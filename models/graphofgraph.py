# aat_da_module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------
def sinusoidal_positional_encoding(length: int, d_model: int, device=None):
    device = device or torch.device("cpu")
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (length, d_model)

def bbox_centers(boxes):
    # boxes: (...,4) -> returns (...,2) centers
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return torch.stack([cx, cy], dim=-1)

# -------------------------
# AAT-DA module (batch, sequence)
# -------------------------
class AAT_DA_FullSeq(nn.Module):
    """
    AAT-DA (batch + sequence version) matching the paper.
    Input:
      - img_feat: (B, T, D)
      - obj_feats: (B, T, N, D)
      - obj_boxes: (B, T, N, 4) (normalized 0..1 or pixel coords)
      - driver_attn_map (optional): (B, T, H, W)
      - driver_attn_per_obj (optional): (B, T, N)
    Output:
      - logits: (B, T, 2)
      - probs:  (B, T, 2)
    """
    def __init__(
        self,
        in_dim: int = 4096,
        d_model: int = 1024,
        num_heads: int = 8,
        max_objects: int = 19,
        spatial_layers: int = 4,
        temporal_layers: int = 2,
        dropout_spatial: float = 0.3,
        dropout_temporal: float = 0.1,
        fc_dropout: float = 0.5,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.D = in_dim
        self.d = d_model
        self.N_max = max_objects

        # Input encoder (shared for image and object tokens): D -> d
        self.input_linear = nn.Linear(self.D, self.d)
        self.input_ln = nn.LayerNorm(self.d)
        self.input_act = nn.GELU()

        # Learnable class tokens for spatial & temporal
        self.cls_spatial = nn.Parameter(torch.randn(1, 1, self.d))
        self.cls_temporal = nn.Parameter(torch.randn(1, 1, self.d))

        # Learnable positional encodings for spatial tokens (max length N+2)
        self.spatial_pos_enc = nn.Parameter(torch.randn(self.N_max + 2, self.d))

        # Object self-attention (single-layer transformer)
        osa_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=num_heads,
                                               dim_feedforward=self.d * 4,
                                               dropout=dropout_spatial,
                                               activation='gelu',
                                               batch_first=True)
        self.object_self_attn = nn.TransformerEncoder(osa_layer, num_layers=1)

        # Spatial transformer (4 layers)
        spatial_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=num_heads,
                                                   dim_feedforward=self.d * 4,
                                                   dropout=dropout_spatial,
                                                   activation='gelu',
                                                   batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)

        # Temporal transformer (2 layers)
        temporal_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=num_heads,
                                                    dim_feedforward=self.d * 4,
                                                    dropout=dropout_temporal,
                                                    activation='gelu',
                                                    batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        # Final classification head (LayerNorm + FC -> 2 logits)
        self.final_ln = nn.LayerNorm(self.d)
        self.final_fc = nn.Sequential(
            nn.Linear(self.d, self.d // 2),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(self.d // 2, 2)
        )

        # Attention fusion for position + driver attention (learnable) -> scalar per object
        # We take 2 inputs [alpha_pw, alpha_da] and output a logit
        self.attn_fuse = nn.Linear(2, 1, bias=True)

        # token dropout
        self.token_dropout = nn.Dropout(dropout_spatial)

    # -------------------------
    # Spatial attention helpers (vectorized where feasible)
    # -------------------------
    def compute_position_weight_batch(self, boxes):
        """
        boxes: (B, T, N, 4)
        returns alpha_pw: (B, T, N) normalized s.t. sum over objects = 1 (per sample/time)
        Implementation of Eq (1)-(2).
        """
        B, T, N, _ = boxes.shape
        device = boxes.device
        centers = bbox_centers(boxes)  # (B,T,N,2)

        # create ego point per (B,T)
        # define ego_x as midpoint between min x1 and max x2 in that frame (robust)
        x1 = boxes[..., 0]  # (B,T,N)
        x2 = boxes[..., 2]  # (B,T,N)
        x_min = x1.min(dim=-1).values  # (B,T)
        x_max = x2.max(dim=-1).values  # (B,T)
        ego_x = (x_min + x_max) / 2.0  # (B,T)
        ego_y = torch.zeros_like(ego_x)  # center-top -> y=0
        ego = torch.stack([ego_x, ego_y], dim=-1)  # (B,T,2)

        # Build all_pts per frame: [ego, obj1, obj2, ...] -> (B,T,N+1,2)
        ego_exp = ego.unsqueeze(2).expand(B, T, N, 2)  # (B,T,N,2) to concat below
        all_pts = torch.cat([ego.unsqueeze(2), centers], dim=2)  # (B,T,N+1,2)

        # compute pairwise distances between centers and all_pts
        # centers: (B,T,N,2); all_pts: (B,T,N+1,2)
        # use broadcasting: compute (B,T,N,N+1)
        c = centers.unsqueeze(3)  # (B,T,N,1,2)
        a = all_pts.unsqueeze(2)  # (B,T,1,N+1,2)
        dif = c - a  # (B,T,N,N+1,2)
        dists = torch.norm(dif, dim=-1)  # (B,T,N,N+1)

        # Set self-distances (object i to itself at column i+1) to inf
        idx = torch.arange(N, device=device) + 1  # columns to ignore
        # expand idx for B,T dims
        # mask shape (N+1) with True for self columns? We will set by indexing
        for i in range(N):
            dists[:, :, i, i+1] = float('inf')

        Lt = dists.min(dim=-1).values  # (B,T,N) min distance to any other (or ego)

        # compute alpha_pw = 1 - Lt / sum(Lt) per (B,T)
        sumLt = Lt.sum(dim=-1, keepdim=True)  # (B,T,1)
        # avoid division by zero
        eps = 1e-9
        alpha_pw = 1.0 - Lt / (sumLt + eps)  # (B,T,N)
        alpha_pw = torch.clamp(alpha_pw, min=0.0)

        # normalize to sum 1 per (B,T)
        s = alpha_pw.sum(dim=-1, keepdim=True)  # (B,T,1)
        # if s==0 -> uniform
        uniform = torch.ones_like(alpha_pw) / max(1, N)
        alpha_pw = torch.where(s > 0, alpha_pw / (s + eps), uniform)
        return alpha_pw  # (B,T,N)

    def compute_driver_attention_weight_batch(self, boxes, driver_attn_map=None, driver_attn_per_obj=None):
        """
        Returns alpha_da: (B,T,N) normalized per (B,T).
        - if driver_attn_per_obj provided -> use it.
        - else if driver_attn_map provided -> compute mean per bbox (this loops over boxes; acceptable).
        - else uniform.
        """
        B, T, N, _ = boxes.shape
        device = boxes.device
        if driver_attn_per_obj is not None:
            alpha_da = driver_attn_per_obj.to(device).float()  # (B,T,N)
            # clamp and normalize
            alpha_da = torch.clamp(alpha_da, min=0.0)
            s = alpha_da.sum(dim=-1, keepdim=True)
            eps = 1e-9
            uniform = torch.ones_like(alpha_da) / max(1, N)
            alpha_da = torch.where(s > 0, alpha_da / (s + eps), uniform)
            return alpha_da

        if driver_attn_map is None:
            # uniform
            return torch.ones(B, T, N, device=device) / max(1, N)

        # driver_attn_map: (B,T,H,W)
        # compute mean within each bbox; this is somewhat loopy but straightforward
        B_t, T_t, H, W = driver_attn_map.shape
        assert B_t == B and T_t == T, "driver_attn_map batch/time mismatch"
        # Convert boxes to pixel coords if normalized
        boxes_px = boxes.clone()
        if boxes.max() <= 1.0 + 1e-6:
            # normalized -> pixel indices
            boxes_px[..., 0] = (boxes[..., 0] * (W - 1)).clamp(0, W - 1)
            boxes_px[..., 2] = (boxes[..., 2] * (W - 1)).clamp(0, W - 1)
            boxes_px[..., 1] = (boxes[..., 1] * (H - 1)).clamp(0, H - 1)
            boxes_px[..., 3] = (boxes[..., 3] * (H - 1)).clamp(0, H - 1)

        alpha_da = torch.zeros(B, T, N, device=device, dtype=torch.float32)
        for b in range(B):
            for t in range(T):
                att_map = driver_attn_map[b, t]  # (H,W)
                for n in range(N):
                    x1, y1, x2, y2 = boxes_px[b, t, n]
                    x1i = int(max(0, math.floor(x1.item())))
                    x2i = int(min(W - 1, math.ceil(x2.item())))
                    y1i = int(max(0, math.floor(y1.item())))
                    y2i = int(min(H - 1, math.ceil(y2.item())))
                    if x2i < x1i or y2i < y1i:
                        mean_val = 0.0
                    else:
                        region = att_map[y1i:y2i+1, x1i:x2i+1]
                        mean_val = float(region.mean().item()) if region.numel() > 0 else 0.0
                    alpha_da[b, t, n] = mean_val
        # normalize per (b,t)
        s = alpha_da.sum(dim=-1, keepdim=True)
        eps = 1e-9
        uniform = torch.ones_like(alpha_da) / max(1, N)
        alpha_da = torch.where(s > 0, alpha_da / (s + eps), uniform)
        return alpha_da

    # -------------------------
    # Helper to build causal attention mask for temporal transformer
    # -------------------------
    def _build_causal_mask(self, L):
        # attention mask for TransformerEncoder: shape (L, L) with True where masked (float('-inf') when later)
        # We'll build a float mask to use with attn_mask param
        # For causal (each pos can attend to <= current pos)
        # attn_mask[i,j] == float('-inf') if j > i else 0
        mask = torch.triu(torch.ones(L, L, device=self.device) * float('-inf'), diagonal=1)
        return mask  # (L,L)

    # -------------------------
    # Forward: batch processing
    # -------------------------
    def forward(self, img_feat, obj_feats, obj_boxes, driver_attn_map=None, driver_attn_per_obj=None):
        """
        img_feat: (B, T, D)
        obj_feats: (B, T, N, D)
        obj_boxes: (B, T, N, 4)
        driver_attn_map: optional (B, T, H, W)
        driver_attn_per_obj: optional (B, T, N)
        Returns:
          logits: (B, T, 2)
          probs:  (B, T, 2)
        """
        # device = img_feat.device
        device = next(self.parameters()).device   # <- model’s actual device

        img_feat = img_feat.to(device)
        obj_feats = obj_feats.to(device)
        obj_boxes = obj_boxes.to(device)
        if driver_attn_map is not None:
            driver_attn_map = driver_attn_map.to(device)
        if driver_attn_per_obj is not None:
            driver_attn_per_obj = driver_attn_per_obj.to(device)

        B, T, D = img_feat.shape
        _, T2, N, D2 = obj_feats.shape
        assert T == T2 and D == D2, "time/feature mismatch"

        # ------- Spatial attention module per frame (vectorized) -------
        # Compute alpha_pw (B,T,N) and alpha_da (B,T,N)
        alpha_pw = self.compute_position_weight_batch(obj_boxes)  # (B,T,N)
        alpha_da = self.compute_driver_attention_weight_batch(obj_boxes, driver_attn_map=driver_attn_map, driver_attn_per_obj=driver_attn_per_obj)  # (B,T,N)

        # Fuse into scalar logits per object
        fuse_input = torch.stack([alpha_pw, alpha_da], dim=-1)  # (B,T,N,2)
        # reshape to (B*T*N, 2) -> apply linear -> (B*T*N,1) -> reshape
        fuse_in_flat = fuse_input.view(-1, 2)
        alpha_logits_flat = self.attn_fuse(fuse_in_flat).squeeze(-1)  # (B*T*N,)
        alpha_logits = alpha_logits_flat.view(B, T, N)  # (B,T,N)
        # softmax over objects per (B,T)
        alpha_coeff = F.softmax(alpha_logits, dim=-1)  # (B,T,N)

        # Apply weights to object features
        # obj_feats: (B,T,N,D) * alpha_coeff (B,T,N,1) -> (B,T,N,D)
        alpha_coeff_exp = alpha_coeff.unsqueeze(-1)  # (B,T,N,1)
        obj_feats_weighted = obj_feats * alpha_coeff_exp  # (B,T,N,D)

        # ------- Input encoder: project to d -------
        # reshape for projection: (B*T, D) for images and (B*T*N, D) for objects
        img_flat = img_feat.view(B * T, D)  # (B*T, D)
        img_tok_flat = self.input_linear(img_flat)  # (B*T, d)
        img_tok_flat = self.input_ln(img_tok_flat)
        img_tok_flat = self.input_act(img_tok_flat)
        img_toks = img_tok_flat.view(B, T, self.d)  # (B,T,d)

        obj_flat = obj_feats_weighted.view(B * T * N, D)  # (B*T*N, D)
        obj_tok_flat = self.input_linear(obj_flat)  # (B*T*N, d)
        obj_tok_flat = self.input_ln(obj_tok_flat)
        obj_tok_flat = self.input_act(obj_tok_flat)
        obj_toks = obj_tok_flat.view(B, T, N, self.d)  # (B,T,N,d)

        # ------- Object self-attention per frame -------
        # We want to run object_self_attn on sequences of length N, batch size B*T
        obj_batched = obj_toks.view(B * T, N, self.d)  # (B*T, N, d)
        obj_batched = self.token_dropout(obj_batched)
        # pass through TransformerEncoder (batch_first=True)
        obj_batched_out = self.object_self_attn(obj_batched)  # (B*T, N, d)
        obj_out = obj_batched_out.view(B, T, N, self.d)  # (B,T,N,d)

        # ------- Spatial transformer per frame -------
        # Build tokens [CLS_spatial, img_tok, obj_tokens] -> seq_len = N+2
        # cls token expand to (B*T,1,d)
        cls_sp = self.cls_spatial.expand(B * T, -1, -1).to(device)  # (B*T,1,d)
        img_for_seq = img_toks.view(B * T, 1, self.d)  # (B*T,1,d)
        obj_for_seq = obj_out.view(B * T, N, self.d)  # (B*T,N,d)
        seq_tokens = torch.cat([cls_sp, img_for_seq, obj_for_seq], dim=1)  # (B*T, N+2, d)

        # add learnable positional encoding (slice first N+2 rows)
        seq_len = N + 2
        if seq_len <= (self.N_max + 2):
            pos_slice = self.spatial_pos_enc[:seq_len, :].unsqueeze(0).expand(B * T, -1, -1).to(device)  # (B*T, seq_len, d)
        else:
            # fallback: interpolate or repeat (unlikely if N<=N_max)
            pos_slice = self.spatial_pos_enc[:seq_len, :].unsqueeze(0).expand(B * T, -1, -1).to(device)
        seq_tokens = seq_tokens + pos_slice
        seq_tokens = self.token_dropout(seq_tokens)
        # run spatial transformer (process all frames in batch B*T)
        spatial_out = self.spatial_transformer(seq_tokens)  # (B*T, seq_len, d)
        cls_spatial_final = spatial_out[:, 0, :]  # (B*T, d)
        # reshape into (B, T, d) -> h_t for each frame
        Ht = cls_spatial_final.view(B, T, self.d)  # (B,T,d)

        # ------- Temporal transformer with causal mask to produce framewise outputs -------
        # We want logits at each t using features up to t. We'll implement this by feeding the entire sequence
        # of spatial features (per video) with a *temporal CLS token prepended* and using an attention mask
        # so token at position p cannot attend to future tokens.
        # Prepare per-video sequences: for each batch b, form seq = [CLS_temp, h1, h2, ..., hT] -> length L = T+1
        # We'll run the transformer per batch element (but we can batch them together).
        cls_temp = self.cls_temporal.expand(B, 1, self.d).to(device)  # (B,1,d)
        seq_feats = Ht  # (B,T,d)
        seq_with_cls = torch.cat([cls_temp, seq_feats], dim=1)  # (B, T+1, d)

        # Add sinusoidal positional encoding (non-learnable) for temporal tokens (length T+1)
        pos_enc = sinusoidal_positional_encoding(T + 1, self.d, device=device).unsqueeze(0).expand(B, -1, -1)  # (B,T+1,d)
        seq_with_cls = seq_with_cls + pos_enc

        seq_with_cls = self.token_dropout(seq_with_cls)

        # Build causal mask so temporal token i cannot attend to j>i (including cls at pos 0 -> cls can attend only to cls+past?)
        # TransformerEncoder (PyTorch) accepts attn_mask of shape (S,S) where masked positions are -inf.
        L = T + 1
        # causal mask: upper triangular (excluding diag) should be -inf
        attn_mask = torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)  # (L,L)

        # The TransformerEncoder expects input (batch, seq, d)
        # Apply temporal transformer to entire batch; it respects attn_mask (causal)
        # Note: PyTorch's TransformerEncoderLayer uses attn_mask where float('-inf') blocks attention.
        temporal_out = self.temporal_transformer(seq_with_cls, mask=attn_mask)  # (B, T+1, d)

        # temporal_out[:,0,:] is CLS after attending to all positions (past+present) for each time token? 
        # But we need logits per frame t. The paper takes class token of final layer at time t after feeding h1..ht.
        # To get framewise outputs we extract the **CLS after prefix up to t**. We can obtain them by reading
        # attention outputs corresponding to each prefix's CLS. However, since we used a single CLS (pos 0)
        # and full sequence with causal masking, the CLS token at pos 0 ended up attending to whole sequence,
        # but we need CLS per prefix. To replicate per-t behavior we instead will read outputs at positions 1..T,
        # i.e., the representation of each h_t after temporal transformer where token at position p has seen only
        # positions <= p because of causal mask. The paper uses class token of temporal transformer, but for
        # per-frame outputs extracting transformed h_t (pos 1..T) and applying final head achieves same behavior
        # because at time t those tokens have seen only h<=t. So we compute logits from temporal_out[:,1:,:].
        temporal_tokens = temporal_out[:, 1:, :]  # (B, T, d) - each position t has access to <= t due to mask

        # Apply final LN + FC to each temporal token -> logits per frame
        x = self.final_ln(temporal_tokens)  # (B,T,d)
        logits = self.final_fc(x)  # (B,T,2)
        probs = F.softmax(logits, dim=-1)

        return logits, probs, Ht  # Ht is optional: per-frame spatial features

# -------------------------
# Example usage notes:
# -------------------------
# model = AAT_DA_FullSeq().to(device)
# logits, probs, Ht = model(img_feat, obj_feats, obj_boxes, driver_attn_map=att_map)
# logits: (B,T,2)
# probs:  (B,T,2)
# Ht:     (B,T,d)  (spatial CLS tokens per frame)
