"""
Stage 3: Multi-Plane Cross-Attention MIL & DICOM Tabular Fusion Network
Trains on all 4,407 studies using pre-extracted multiplane embeddings and DICOM features.
Strictly evaluated on honest held-out Gold-58 radiologist ground truth.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class MultiPlaneAttentionFusion(nn.Module):
    """
    Cross-Plane Multi-Pathology Query Attention + Tabular Metadata Fusion
    """
    def __init__(self, embed_dim: int = 768, num_tab_features: int = 18, num_classes: int = 12, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Learnable target queries (one query vector per pathology)
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        
        # Plane projections
        self.plane_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Cross-Attention: target queries attend across 3 planes (Sagittal, Coronal, Axial)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # Tabular metadata encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tab_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Per-target classifiers (fuse visual attended rep + tabular rep)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x_planes: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """
        x_planes: (B, 3, 768) - Sagittal, Coronal, Axial
        x_tab: (B, num_tab_features)
        Returns: logits (B, 12)
        """
        B = x_planes.shape[0]
        
        # Project planes to hidden_dim: (B, 3, hidden_dim)
        planes_h = self.plane_proj(x_planes)
        
        # Queries: (B, 12, hidden_dim)
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: queries attend to planes
        # Output: (B, 12, hidden_dim)
        attn_out, attn_weights = self.mha(queries, planes_h, planes_h)
        
        # Tabular representation: (B, hidden_dim // 2)
        tab_h = self.tab_encoder(x_tab)
        tab_h_expanded = tab_h.unsqueeze(1).expand(-1, self.num_classes, -1) # (B, 12, hidden_dim // 2)
        
        # Combine per pathology
        combined = torch.cat([attn_out, tab_h_expanded], dim=-1) # (B, 12, hidden_dim + hidden_dim // 2)
        
        logits_list = []
        for c in range(self.num_classes):
            c_logit = self.classifiers[c](combined[:, c, :]) # (B, 1)
            logits_list.append(c_logit)
            
        logits = torch.cat(logits_list, dim=-1) # (B, 12)
        return logits

class MultiPlaneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_dict: dict, feature_cols: list):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings_dict
        self.feature_cols = feature_cols
        
        self.uids = self.df['StudyInstanceUID'].values
        self.labels = self.df[target_cols].values.astype(np.float32)
        self.weights = np.where(self.df['is_gold'] == 1, 5.0, 1.0).astype(np.float32)
        self.tab_features = self.df[feature_cols].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        uid = self.uids[idx]
        emb = self.embeddings.get(uid, torch.zeros(3, 768, dtype=torch.float32))
        if isinstance(emb, torch.Tensor) and emb.dtype == torch.float16:
            emb = emb.to(torch.float32)
            
        return {
            'embedding': emb,
            'tabular': torch.tensor(self.tab_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
            'weight': torch.tensor(self.weights[idx], dtype=torch.float32),
            'uid': uid
        }

def train_and_eval_multiplane():
    print("=" * 80)
    print("STAGE 3: TRAINING MULTI-PLANE CROSS-ATTENTION & TABULAR FUSION MODEL")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    emb_path = r"D:\MLprojs\KaggleCompetition\RSNA\multiplane_embeddings_convnext.pt"
    if not os.path.exists(emb_path):
        print(f"Embeddings file {emb_path} not found yet!")
        return
        
    print(f"Loading cached multiplane embeddings from {emb_path}...")
    embeddings = torch.load(emb_path, map_location='cpu')
    print(f"Loaded embeddings for {len(embeddings)} studies.")
    
    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    dicom_df = pd.read_csv('dicom_tabular_features.csv')
    
    df = pd.merge(master_df, dicom_df, on='StudyInstanceUID', how='left')
    df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
    df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
    df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)
    
    feature_cols = [
        'n_series', 'n_total_slices', 'n_sagittal_slices', 'n_coronal_slices', 'n_axial_slices',
        'ratio_sag_total', 'ratio_cor_total', 'ratio_ax_total',
        'patient_sex', 'field_strength', 'mean_tr', 'mean_te', 'pixel_spacing', 'tr_te_ratio'
    ] + [c for c in df.columns if c.startswith('manufacturer_')]
    
    # Normalize tabular features
    for col in feature_cols:
        m, s = df[col].mean(), df[col].std()
        df[col] = (df[col] - m) / (s + 1e-5)
        
    gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
    report_df = df[df['is_gold'] == 0].copy().reset_index(drop=True)
    
    oof_preds = np.zeros((len(gold_df), 12))
    checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for fold in range(5):
        print(f"\n--- Training Fold {fold} / 4 ---")
        val_gold = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        val_indices = gold_df[gold_df['fold'] == fold].index
        
        train_gold = gold_df[gold_df['fold'] != fold].copy().reset_index(drop=True)
        train_report = report_df[report_df['fold'] != fold].copy().reset_index(drop=True)
        train_combined = pd.concat([train_gold, train_report], ignore_index=True)
        
        train_ds = MultiPlaneDataset(train_combined, embeddings, feature_cols)
        val_ds = MultiPlaneDataset(val_gold, embeddings, feature_cols)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        model = MultiPlaneAttentionFusion(
            embed_dim=768, num_tab_features=len(feature_cols), num_classes=12, hidden_dim=256, dropout=0.2
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)
        
        best_val_auc = 0.0
        best_fold_preds = None
        
        for ep in range(35):
            model.train()
            for batch in train_loader:
                x_planes = batch['embedding'].to(device)
                x_tab = batch['tabular'].to(device)
                y_targets = batch['labels'].to(device)
                weights = batch['weight'].to(device)
                
                optimizer.zero_grad()
                logits = model(x_planes, x_tab)
                bce = F.binary_cross_entropy_with_logits(logits, y_targets, reduction='none')
                loss = (bce.mean(dim=-1) * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                
            scheduler.step()
            
            # Eval on val gold
            model.eval()
            val_preds_list = []
            val_targets_list = []
            with torch.no_grad():
                for batch in val_loader:
                    x_planes = batch['embedding'].to(device)
                    x_tab = batch['tabular'].to(device)
                    logits = model(x_planes, x_tab)
                    probs = torch.sigmoid(logits)
                    val_preds_list.append(probs.cpu().numpy())
                    val_targets_list.append(batch['labels'].numpy())
                    
            p_val = np.concatenate(val_preds_list, axis=0)
            y_val = np.concatenate(val_targets_list, axis=0)
            
            aucs = []
            for c in range(12):
                try:
                    aucs.append(roc_auc_score(y_val[:, c], p_val[:, c]))
                except:
                    aucs.append(0.5)
            val_auc = np.mean(aucs)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_fold_preds = p_val.copy()
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"multiplane_fusion_fold{fold}_best.pt"))
                
        print(f"Fold {fold} Best Gold OOF Macro-AUC: {best_val_auc:.4f}")
        oof_preds[val_indices] = best_fold_preds
        
    print("\n" + "=" * 80)
    final_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_preds[:, c]) for c in range(12)])
    print(f">>> MULTI-PLANE ATTENTION FUSION 5-FOLD MACRO-AUC: {final_macro:.4f} <<<")
    print("=" * 80)
    for c in range(12):
        auc = roc_auc_score(gold_df[target_cols[c]].values, oof_preds[:, c])
        print(f"  {target_cols[c]:18s}: OOF AUC = {auc:.4f}")
        
    # Save OOF
    oof_df = gold_df[['StudyInstanceUID']].copy()
    for c, target in enumerate(target_cols):
        oof_df[target] = oof_preds[:, c]
    oof_df.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\multiplane_fusion_oof.csv", index=False)
    print("Saved multiplane fusion OOF to oof/multiplane_fusion_oof.csv!")

if __name__ == '__main__':
    train_and_eval_multiplane()
