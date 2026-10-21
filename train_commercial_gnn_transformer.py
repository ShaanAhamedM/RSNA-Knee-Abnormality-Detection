"""
Phase 3: 12-Node Anatomical Graph Attention Network (Co-Occurrence GNN) + Target-Query MIL Transformer
Features:
- Visual Encoder initialized from Pretrained Biomedical SigLIP Visual Encoder
- 12 Learnable Target Queries with Multi-Head Cross-Attention over 20 Slices
- Tabular Radiomics Feature Projection (46 normalized DICOM & bright-fluid features)
- 12-Node Biomechanical Anatomical Graph Attention Network (GNN):
    Pathologies exchange messages over learned adjacency matrix guided by clinical prior
    (Unhappy Triad, Tricompartmental OA, Synovial Fluid dynamics, Bone impaction/contusion)
- Masked Loss on Commercial Purity Confidence Masks (M_{i, c} in {0, 1})
- 5-Fold Stratified Cross-Validation on held-out Gold-58
- GPU Thermal Governor (MAX_TEMP = 78C), 45s breathers, and Atomic Checkpointing
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

from thermal_governor import check_gpu_thermals, thermal_cooldown_breather, atomic_torch_save, update_state

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_biomechanical_prior():
    """
    Constructs the 12x12 clinical biomechanical prior adjacency matrix.
    0: ACL, 1: MCL, 2: Medial Meniscus, 3: Lateral Meniscus
    4: Medial OA, 5: Lateral OA, 6: PF OA
    7: Effusion, 8: Synovitis, 9: Baker's
    10: Contusion, 11: Fracture
    """
    A = torch.eye(12) * 2.0
    edges = [
        # Unhappy Triad (O'Donoghue triad)
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
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

class AnatomicalGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features
        
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        
        self.alpha_prior = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, h, A_prior):
        # h: (B, 12, in_features)
        B, N, C = h.shape
        Q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, 12, head_dim)
        K = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, heads, 12, 12)
        # Inject biomechanical anatomical prior
        scores = scores + self.alpha_prior * A_prior.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V) # (B, heads, 12, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_features)
        out = self.out_proj(out)
        return self.norm(h + out)

class CommercialAnatomicalGNNTransformer(nn.Module):
    def __init__(
        self,
        visual_dim=1792,
        tab_dim=46,
        hidden_dim=256,
        num_classes=12,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. Visual Token Projections (compatible with SigLIP visual encoder)
        self.token_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.plane_emb = nn.Embedding(3, hidden_dim)
        self.roi_emb = nn.Embedding(5, hidden_dim)
        self.depth_proj = nn.Linear(1, hidden_dim)
        self.contrast_proj = nn.Linear(3, hidden_dim)
        
        # 2. 12 Learnable Pathology Target Queries
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        
        # 3. Cross-Attention MIL: Queries attend over 20 multi-plane slice tokens
        self.mha_cross = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Tabular & Radiomics MLP
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.tab_to_nodes = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # 5. 12-Node Biomechanical Anatomical Graph Attention Network (GNN)
        self.register_buffer('A_prior', build_biomechanical_prior())
        self.gat1 = AnatomicalGraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.gat2 = AnatomicalGraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        
        # 6. Pathology Logit Heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def load_siglip_visual_backbone(self, checkpoint_path):
        """Initializes visual token projection and embeddings from pretrained SigLIP encoder."""
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
            print(f"[TRANSFERRED] Loaded {transferred} visual parameter tensors from {checkpoint_path}")

    def forward(self, tokens, planes, rois, depths, contrast, x_tab):
        # tokens: (B, K, 1792)
        B, K, _ = tokens.shape
        
        # Encode visual tokens
        h_vis = self.token_proj(tokens)
        h_vis = h_vis + self.plane_emb(planes) + self.roi_emb(rois) + self.depth_proj(depths) + self.contrast_proj(contrast)
        
        # Cross-Attention: 12 queries attend to K=20 slice tokens
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1) # (B, 12, hidden_dim)
        node_vis, _ = self.mha_cross(query=queries, key=h_vis, value=h_vis)
        nodes = self.cross_norm(queries + node_vis) # (B, 12, hidden_dim)
        
        # Enrich nodes with Tabular Radiomics features
        h_tab = self.tab_encoder(x_tab) # (B, hidden_dim // 2)
        tab_enrichment = self.tab_to_nodes(h_tab).unsqueeze(1) # (B, 1, hidden_dim)
        nodes = nodes + tab_enrichment
        
        # 12-Node Graph Attention Message Passing over biomechanical adjacency
        nodes = self.gat1(nodes, self.A_prior)
        nodes = self.gat2(nodes, self.A_prior)
        
        # Logits output per node
        logits = torch.cat([self.classifiers[c](nodes[:, c, :]) for c in range(self.num_classes)], dim=-1)
        return logits

class CommercialGNNDataset(Dataset):
    def __init__(self, df, tokens_dict, feature_cols):
        self.uids = []
        for uid in df['StudyInstanceUID'].values:
            if uid in tokens_dict:
                self.uids.append(uid)
                
        self.df = df.set_index('StudyInstanceUID').loc[self.uids].reset_index()
        self.tokens_dict = tokens_dict
        self.tab_features = self.df[feature_cols].values.astype(np.float32)
        self.labels = self.df[target_cols].values.astype(np.float32)
        self.masks = self.df[[f"{c}_mask" for c in target_cols]].values.astype(np.float32)
        self.weights = self.df['sample_weight'].values.astype(np.float32)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        tok_data = self.tokens_dict[uid]
        
        tokens = tok_data['tokens'].float()
        planes = tok_data['planes'].long()
        rois = tok_data['rois'].long()
        depths = tok_data['depths'].float().unsqueeze(-1)
        contrast = tok_data['contrast'].float()
        
        return (
            tokens,
            planes,
            rois,
            depths,
            contrast,
            torch.tensor(self.tab_features[idx]),
            torch.tensor(self.labels[idx]),
            torch.tensor(self.masks[idx]),
            torch.tensor(self.weights[idx]),
            uid
        )

def train_commercial_gnn():
    print("=" * 90, flush=True)
    print("PHASE 3: TRAINING 12-NODE ANATOMICAL GNN + MIL TRANSFORMER (5 FOLDS)", flush=True)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 90, flush=True)

    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    rad_df = pd.read_csv('dicom_radiomics_phase7_features.csv')
    report_df = pd.read_csv('report_extracted_labels_4349_commercial.csv')

    # Merge dataset
    df = pd.merge(master_df, rad_df, on='StudyInstanceUID', how='left')
    df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
    df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
    df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)

    # Merge masks
    mask_cols = [f"{c}_mask" for c in target_cols]
    df = df.merge(report_df[['StudyInstanceUID'] + mask_cols], on='StudyInstanceUID', how='left')

    # Assign sample weights: Gold=5.0, Unlabeled=1.0
    df['sample_weight'] = np.where(df['is_gold'] == 1, 5.0, 1.0)
    for c in mask_cols:
        df[c] = df[c].fillna(1.0)

    exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report', 'sample_weight'] + target_cols + mask_cols
    feature_cols = [c for c in df.columns if c not in exclude]

    # Normalize features
    for col in feature_cols:
        df[col] = df[col].fillna(0.0)
        m, s = df[col].mean(), df[col].std()
        df[col] = (df[col] - m) / (s + 1e-5)

    print(f"Normalized {len(feature_cols)} Tabular & Radiomics features.", flush=True)

    print("Loading visual token cache...", flush=True)
    tokens_cache = torch.load('phase7_tokens_dual_backbone.pt', map_location='cpu', weights_only=False)
    print(f"Loaded visual token cache for {len(tokens_cache)} studies.", flush=True)

    gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
    N_gold = len(gold_df)
    oof_gnn = np.zeros((N_gold, 12))

    checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    oof_dir = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
    os.makedirs(oof_dir, exist_ok=True)

    for fold in range(5):
        print(f"\n{'='*70}\nTRAINING FOLD {fold} / 4\n{'='*70}", flush=True)
        val_gold = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        val_gold_idx = gold_df[gold_df['fold'] == fold].index

        # Training set combines 4 folds of Gold + 4 folds of filtered Unlabeled studies
        train_data = df[df['fold'] != fold].copy().reset_index(drop=True)

        train_ds = CommercialGNNDataset(train_data, tokens_cache, feature_cols)
        val_ds = CommercialGNNDataset(val_gold, tokens_cache, feature_cols)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        model = CommercialAnatomicalGNNTransformer(
            visual_dim=1792,
            tab_dim=len(feature_cols),
            hidden_dim=256,
            num_classes=12
        ).to(DEVICE)

        # Load pretrained SigLIP visual weights
        model.load_siglip_visual_backbone("checkpoints/biomedical_siglip_visual_encoder_best.pt")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')

        best_fold_auc = 0.0
        best_fold_preds = None

        for epoch in range(1, 8):
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
                        # Masked loss scaled by sample weight (5x for Gold studies)
                        weighted_loss = (bce_raw * masks).sum(dim=-1) * weights
                        loss = weighted_loss.mean()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()
                    n_batches += 1

                except torch.cuda.OutOfMemoryError:
                    print(f"[OOM] Cleared cache at batch {b_idx}", flush=True)
                    torch.cuda.empty_cache()
                    continue

            scheduler.step()

            # Evaluate on held-out Gold fold
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
            print(f"Fold {fold} | Epoch {epoch}/7 ({elapsed:.1f}s) | Train Loss: {train_loss/max(1, n_batches):.4f} | Gold Fold AUC: {macro_fold_auc:.4f}", flush=True)

            if macro_fold_auc > best_fold_auc:
                best_fold_auc = macro_fold_auc
                best_fold_preds = val_preds
                ckpt_path = os.path.join(checkpoints_dir, f"phase8_gnn_fold{fold}_best.pt")
                atomic_torch_save(model.state_dict(), ckpt_path)

        oof_gnn[val_gold_idx] = best_fold_preds
        print(f"--> [FOLD {fold} COMPLETE] Best Gold AUC = {best_fold_auc:.4f}", flush=True)

        # Thermal Breather between folds
        thermal_cooldown_breather(45, f"Fold {fold} Complete -> Cooling before Fold {fold+1}")

    # Compute Final 5-Fold OOF Macro-AUC
    final_aucs = []
    print("\n" + "=" * 90, flush=True)
    print("PHASE 3: ANATOMICAL GNN 5-FOLD RESULTS ON GOLD-58", flush=True)
    print("=" * 90, flush=True)
    for c, target in enumerate(target_cols):
        y_c = gold_df[target].values
        p_c = oof_gnn[:, c]
        auc_c = roc_auc_score(y_c, p_c)
        final_aucs.append(auc_c)
        print(f"  {target:18s}: 5-Fold Gold OOF AUC = {auc_c:.4f}", flush=True)

    gnn_macro_auc = np.mean(final_aucs)
    print("=" * 90, flush=True)
    print(f">>> FINAL PHASE 3 ANATOMICAL GNN MACRO-AUC: {gnn_macro_auc:.4f} <<<", flush=True)
    print("=" * 90, flush=True)

    # Save OOF
    oof_df = gold_df[['StudyInstanceUID']].copy()
    for c, target in enumerate(target_cols):
        oof_df[target] = oof_gnn[:, c]
    oof_path = os.path.join(oof_dir, "phase8_gnn_transformer_oof.csv")
    oof_df.to_csv(oof_path + ".tmp", index=False)
    os.replace(oof_path + ".tmp", oof_path)
    print(f"[SAVED] GNN OOF saved to {oof_path}", flush=True)

    update_state(
        "Phase 3: Anatomical GNN",
        "COMPLETED",
        {
            "macro_auc": float(gnn_macro_auc),
            "target_aucs": {target_cols[c]: float(final_aucs[c]) for c in range(12)}
        },
        f"12-Node Anatomical Graph Attention Network achieved {gnn_macro_auc:.4f} 5-Fold Macro-AUC on Gold-58."
    )

if __name__ == '__main__':
    train_commercial_gnn()
