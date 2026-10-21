"""
Enhanced Commercial Biomechanical Gated-GNN & Multi-Scale MIL Transformer
Architecture:
1. Dual-Backbone Visual Tokens: 1792-dim (ConvNeXt-Small + DenseNet121)
2. Spatial & Contrast Positional Encodings: Plane (3), Anatomical Sub-ROI (5), Depth z, Fluid Contrast
3. 12 Target Queries with Dual-Pooling Cross-Attention:
   - Mean Cross-Attention Representation
   - Max Cross-Attention Slice Representation
4. High-Capacity Tabular Radiomics Feature Projection (46 normalized DICOM & bright-fluid radiomics)
5. Gated Biomechanical Graph Attention Network:
   - Learnable Gating: gate = sigmoid(W_g * [h_node, message])
   - Adaptive Residual Co-Occurrence: prevents feature dilution on isolated injuries (e.g. isolated ACL or MCL)
6. Masked Loss with Commercial Confidence Masks M_{i, c} in {0, 1}
7. Evaluated with 5-Fold Stratified Cross-Validation on held-out Gold-58
8. Hardware & Thermal Governor: MAX_TEMP = 78C, 45s breathers, atomic checkpointing
"""

import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from thermal_governor import check_gpu_thermals, thermal_cooldown_breather, atomic_torch_save, update_state

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_biomechanical_prior():
    """
    Biomechanical co-occurrence coupling:
    0: ACL, 1: MCL, 2: Medial Meniscus, 3: Lateral Meniscus
    4: Medial OA, 5: Lateral OA, 6: PF OA
    7: Effusion, 8: Synovitis, 9: Baker's
    10: Contusion, 11: Fracture
    """
    A = torch.eye(12) * 1.5
    edges = [
        # Unhappy Triad
        (0, 1), (0, 2), (1, 2),
        # Lateral joint coupling
        (0, 3), (3, 5),
        # Tricompartmental OA
        (4, 5), (4, 6), (5, 6),
        # Synovial fluid / Baker's / Synovitis
        (7, 8), (8, 9), (7, 9),
        # Secondary OA joint effusion
        (4, 7), (6, 7), (5, 7),
        # Traumatic bone impaction & pivot shift contusion
        (10, 11), (0, 10), (7, 11)
    ]
    for u, v in edges:
        A[u, v] = 0.8
        A[v, u] = 0.8
    return A

class GatedBiomechanicalGNNLayer(nn.Module):
    """
    Gated Graph Attention Layer with adaptive residual gating.
    Prevents over-smoothing and preserves isolated pathology features.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features
        
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        
        self.alpha_prior = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_features, out_features)
        
        # Adaptive Gating Mechanism: sigmoid(W_g * [h, message])
        self.gate_fc = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(self, h, A_prior):
        B, N, C = h.shape
        Q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.alpha_prior * A_prior.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        message = torch.matmul(attn, V)
        message = message.transpose(1, 2).contiguous().view(B, N, self.out_features)
        message = self.out_proj(message)
        
        # Gated fusion
        gate_input = torch.cat([h, message], dim=-1)
        gate = self.gate_fc(gate_input)
        
        out = h + gate * message
        return self.norm(out)

class EnhancedCommercialGNNTransformer(nn.Module):
    def __init__(
        self,
        visual_dim=1792,
        tab_dim=46,
        hidden_dim=256,
        num_classes=12,
        num_heads=4,
        dropout=0.15
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. Visual Token Projections
        self.token_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.plane_emb = nn.Embedding(3, hidden_dim)
        self.roi_emb = nn.Embedding(5, hidden_dim)
        self.depth_proj = nn.Linear(1, hidden_dim)
        self.contrast_proj = nn.Linear(3, hidden_dim)
        
        # 2. 12 Learnable Target Queries
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        
        # 3. Cross-Attention MIL with Attention Pooling
        self.mha_cross = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Tabular Radiomics MLP
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 5. Gated Biomechanical Anatomical Graph Attention
        self.register_buffer('A_prior', build_biomechanical_prior())
        self.gat = GatedBiomechanicalGNNLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # 6. Combined Classifier: Mean Visual (hidden) + Max Visual (hidden) + GNN Rep (hidden) + Tabular (hidden//2)
        clf_in_dim = hidden_dim * 3 + (hidden_dim // 2)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(clf_in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2.0),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def load_siglip_visual_backbone(self, checkpoint_path):
        """Warm-starts visual token embeddings from pretrained Biomedical SigLIP model."""
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location='cpu')
            model_dict = self.state_dict()
            transferred = 0
            for k, v in state.items():
                if k.startswith(('token_proj.', 'plane_emb.', 'roi_emb.', 'depth_proj.', 'contrast_proj.')):
                    if k in model_dict and model_dict[k].shape == v.shape:
                        model_dict[k] = v
                        transferred += 1
            self.load_state_dict(model_dict)
            print(f"[WARM-START] Transferred {transferred} visual parameter tensors from {checkpoint_path}", flush=True)

    def forward(self, tokens, planes, rois, depths, contrast, x_tab):
        B, K, _ = tokens.shape
        
        # Visual encoding
        h_vis = self.token_proj(tokens)
        h_vis = h_vis + self.plane_emb(planes) + self.roi_emb(rois) + self.depth_proj(depths) + self.contrast_proj(contrast)
        
        # Cross-Attention: 12 queries attend over K slice tokens
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1) # (B, 12, hidden_dim)
        attn_out, attn_weights = self.mha_cross(query=queries, key=h_vis, value=h_vis)
        node_vis = self.cross_norm(queries + attn_out) # (B, 12, hidden_dim)
        
        # Attention-Weighted Max pooling across slices
        weighted_vis = attn_weights.unsqueeze(-1) * h_vis.unsqueeze(1) # (B, 12, K, hidden_dim)
        attn_max, _ = torch.max(weighted_vis, dim=2) # (B, 12, hidden_dim)
        
        # Gated Biomechanical GNN message passing
        gnn_rep = self.gat(node_vis, self.A_prior) # (B, 12, hidden_dim)
        
        # Tabular radiomics features
        tab_h = self.tab_encoder(x_tab) # (B, hidden_dim // 2)
        tab_exp = tab_h.unsqueeze(1).expand(-1, self.num_classes, -1) # (B, 12, hidden_dim // 2)
        
        # Fuse all 4 representation streams per target node
        combined = torch.cat([node_vis, attn_max, gnn_rep, tab_exp], dim=-1) # (B, 12, clf_in_dim)
        logits = torch.cat([self.classifiers[c](combined[:, c, :]) for c in range(self.num_classes)], dim=-1)
        return logits

class EnhancedCommercialDataset(Dataset):
    def __init__(self, df, tokens_dict, feature_cols, max_tokens=20):
        self.uids = []
        for uid in df['StudyInstanceUID'].values:
            if uid in tokens_dict:
                self.uids.append(uid)
                
        self.df = df.set_index('StudyInstanceUID').loc[self.uids].reset_index()
        self.tokens_dict = tokens_dict
        self.feature_cols = feature_cols
        self.tab_features = self.df[feature_cols].values.astype(np.float32)
        self.labels = self.df[target_cols].values.astype(np.float32)
        self.masks = self.df[[f"{c}_mask" for c in target_cols]].values.astype(np.float32)
        self.weights = self.df['sample_weight'].values.astype(np.float32)
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        tok_data = self.tokens_dict[uid]
        
        tokens = tok_data['tokens'].float()
        planes = tok_data['planes'].long()
        rois = tok_data.get('rois', torch.zeros_like(planes)).long()
        depths = tok_data['depths'].float().unsqueeze(-1)
        contrast = tok_data.get('contrast', torch.zeros((len(planes), 3))).float()
        
        K = tokens.shape[0]
        if K > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            planes = planes[:self.max_tokens]
            rois = rois[:self.max_tokens]
            depths = depths[:self.max_tokens]
            contrast = contrast[:self.max_tokens]
        elif K < self.max_tokens:
            pad = self.max_tokens - K
            tokens = torch.cat([tokens, torch.zeros((pad, 1792), dtype=torch.float32)], dim=0)
            planes = torch.cat([planes, torch.zeros(pad, dtype=torch.long)], dim=0)
            rois = torch.cat([rois, torch.zeros(pad, dtype=torch.long)], dim=0)
            depths = torch.cat([depths, torch.zeros((pad, 1), dtype=torch.float32)], dim=0)
            contrast = torch.cat([contrast, torch.zeros((pad, 3), dtype=torch.float32)], dim=0)
            
        return (
            tokens,
            planes,
            rois,
            depths,
            contrast,
            torch.tensor(self.tab_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
            uid
        )

def train_enhanced_gnn():
    print("=" * 95, flush=True)
    print("STAGE 2B / 3: TRAINING ENHANCED GATED-GNN & MULTI-SCALE MIL TRANSFORMER (5 FOLDS)", flush=True)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 95, flush=True)

    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    rad_df = pd.read_csv('dicom_radiomics_phase7_features.csv')
    report_df = pd.read_csv('report_extracted_labels_4349_commercial.csv')

    df = pd.merge(master_df, rad_df, on='StudyInstanceUID', how='left')
    df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
    df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
    df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)

    mask_cols = [f"{c}_mask" for c in target_cols]
    df = df.merge(report_df[['StudyInstanceUID'] + mask_cols], on='StudyInstanceUID', how='left')

    # Higher weight on Gold studies (10.0x for precision grounding)
    df['sample_weight'] = np.where(df['is_gold'] == 1, 10.0, 1.0)
    for c in mask_cols:
        df[c] = df[c].fillna(1.0)

    exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report', 'sample_weight'] + target_cols + mask_cols
    feature_cols = [c for c in df.columns if c not in exclude]

    for col in feature_cols:
        df[col] = df[col].fillna(0.0)
        m, s = df[col].mean(), df[col].std()
        df[col] = (df[col] - m) / (s + 1e-5)

    print(f"Normalized {len(feature_cols)} Tabular & Radiomics features.", flush=True)

    print("Loading visual token cache (phase7_tokens_dual_backbone.pt)...", flush=True)
    tokens_cache = torch.load('phase7_tokens_dual_backbone.pt', map_location='cpu', weights_only=False)
    print(f"Loaded visual token cache for {len(tokens_cache)} studies.", flush=True)

    gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
    N_gold = len(gold_df)
    oof_enhanced = np.zeros((N_gold, 12))

    checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    oof_dir = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
    os.makedirs(oof_dir, exist_ok=True)

    for fold in range(5):
        print(f"\n{'='*75}\nTRAINING ENHANCED GATED GNN - FOLD {fold} / 4\n{'='*75}", flush=True)
        val_gold = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        val_gold_idx = gold_df[gold_df['fold'] == fold].index

        train_data = df[df['fold'] != fold].copy().reset_index(drop=True)

        train_ds = EnhancedCommercialDataset(train_data, tokens_cache, feature_cols, max_tokens=20)
        val_ds = EnhancedCommercialDataset(val_gold, tokens_cache, feature_cols, max_tokens=20)

        train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        model = EnhancedCommercialGNNTransformer(
            visual_dim=1792,
            tab_dim=len(feature_cols),
            hidden_dim=256,
            num_classes=12,
            num_heads=4,
            dropout=0.15
        ).to(DEVICE)

        # Warm-start visual encoder from SigLIP pretraining
        siglip_path = "checkpoints/biomedical_siglip_visual_encoder_best.pt"
        if os.path.exists(siglip_path):
            model.load_siglip_visual_backbone(siglip_path)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=9, eta_min=5e-6)
        scaler = torch.amp.GradScaler('cuda')

        best_fold_auc = 0.0
        best_fold_preds = None

        for epoch in range(1, 10):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            n_batches = 0

            for b_idx, (tokens, planes, rois, depths, contrast, x_tab, labels, masks, weights, _) in enumerate(train_loader):
                check_gpu_thermals(max_temp=78, resume_temp=70, check_interval=25, batch_idx=b_idx)

                tokens = tokens.to(DEVICE)
                planes = planes.to(DEVICE)
                rois = rois.to(DEVICE)
                depths = depths.to(DEVICE)
                contrast = contrast.to(DEVICE)
                x_tab = x_tab.to(DEVICE)
                labels = labels.to(DEVICE)
                masks = masks.to(DEVICE)
                weights = weights.to(DEVICE)

                optimizer.zero_grad()

                try:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits = model(tokens, planes, rois, depths, contrast, x_tab)
                        bce_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                        weighted_loss = (bce_raw * masks).sum(dim=-1) * weights
                        loss = weighted_loss.mean()

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()
                    n_batches += 1

                except torch.cuda.OutOfMemoryError:
                    print(f"[OOM] Batch {b_idx} caught. Clearing cache.", flush=True)
                    torch.cuda.empty_cache()
                    continue

            scheduler.step()

            # Validation on held-out Gold fold
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for tokens, planes, rois, depths, contrast, x_tab, labels, masks, weights, _ in val_loader:
                    tokens = tokens.to(DEVICE)
                    planes = planes.to(DEVICE)
                    rois = rois.to(DEVICE)
                    depths = depths.to(DEVICE)
                    contrast = contrast.to(DEVICE)
                    x_tab = x_tab.to(DEVICE)

                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        logits = model(tokens, planes, rois, depths, contrast, x_tab)
                        probs = torch.sigmoid(logits).cpu().numpy()

                    val_preds.append(probs)
                    val_targets.append(labels.numpy())

            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)

            fold_aucs = []
            for c in range(12):
                y_c = val_targets[:, c]
                p_c = val_preds[:, c]
                if len(np.unique(y_c)) > 1:
                    fold_aucs.append(roc_auc_score(y_c, p_c))
                else:
                    fold_aucs.append(0.5)

            macro_fold_auc = np.mean(fold_aucs)
            elapsed = time.time() - t0
            print(f"Fold {fold} | Epoch {epoch}/9 ({elapsed:.1f}s) | Loss: {train_loss/max(1, n_batches):.4f} | Gold AUC: {macro_fold_auc:.4f}", flush=True)

            if macro_fold_auc > best_fold_auc:
                best_fold_auc = macro_fold_auc
                best_fold_preds = val_preds
                ckpt_path = os.path.join(checkpoints_dir, f"enhanced_gated_gnn_fold{fold}_best.pt")
                atomic_torch_save(model.state_dict(), ckpt_path)

        oof_enhanced[val_gold_idx] = best_fold_preds
        print(f">>> Fold {fold} Best Gold Macro-AUC: {best_fold_auc:.4f} <<<", flush=True)

        thermal_cooldown_breather(seconds=45, reason=f"Breather after Fold {fold}")

    # Compute overall 5-fold Macro-AUC
    overall_aucs = [roc_auc_score(gold_df[target_cols[c]].values, oof_enhanced[:, c]) for c in range(12)]
    overall_macro = float(np.mean(overall_aucs))
    print("\n" + "=" * 95, flush=True)
    print(f"ENHANCED GATED-GNN 5-FOLD MACRO-AUC: {overall_macro:.4f}", flush=True)
    for c, t in enumerate(target_cols):
        print(f"  {t:20s}: {overall_aucs[c]:.4f}", flush=True)
    print("=" * 95, flush=True)

    # Save OOF
    oof_df = gold_df[['StudyInstanceUID']].copy()
    for c, t in enumerate(target_cols):
        oof_df[t] = oof_enhanced[:, c]
    oof_df.to_csv(os.path.join(oof_dir, "enhanced_gated_gnn_oof.csv"), index=False)
    print(f"Saved OOF predictions to {os.path.join(oof_dir, 'enhanced_gated_gnn_oof.csv')}", flush=True)

    update_state(
        phase_name="Enhanced Gated-GNN Transformer",
        status="COMPLETED",
        metrics={
            "macro_auc": overall_macro,
            "target_aucs": dict(zip(target_cols, [round(x, 4) for x in overall_aucs]))
        },
        notes=f"Enhanced Gated-GNN reached {overall_macro:.4f} 5-Fold Gold Macro-AUC with dual-pooling attention and adaptive residual gating."
    )

if __name__ == '__main__':
    train_enhanced_gnn()
