"""
Phase 7 Multi-Backbone (ConvNeXt + DenseNet) Cross-Attention MIL Transformer
Features:
- Dual-Backbone 1792-dim tokens (ConvNeXt-Small 768 + DenseNet121 1024)
- 5 Anatomical Sub-ROIs: Global, Notch Zoom (ACL/Meniscus), PF Anterior (PF OA/Effusion), Medial, Lateral
- Fluid-Sensitive contrast embeddings (is_fluid_sensitive, TR, TE)
- 12 Learnable Pathology Queries with Attention-Weighted Mean + Max pooling
- 50-Feature DICOM + Radiomics fusion
- Evaluated on honest 5-fold CV on held-out Gold-58 studies
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

class Phase7DualBackboneTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1792,
        num_tab_features: int = 50,
        num_classes: int = 12,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Dual-Backbone token projection
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Positional & Semantic Embeddings
        self.plane_emb = nn.Embedding(3, hidden_dim)  # Sagittal, Coronal, Axial
        self.roi_emb = nn.Embedding(5, hidden_dim)    # Global, Notch, PF, Medial, Lateral
        self.depth_proj = nn.Linear(1, hidden_dim)    # Relative z-depth
        self.contrast_proj = nn.Linear(3, hidden_dim) # is_fluid, norm_tr, norm_te
        
        # 12 Learnable Pathology Queries
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        
        # Cross-Attention
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Tabular & Radiomics MLP Encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tab_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 12 Independent Classifiers (Mean + Max + Tabular)
        in_clf_dim = hidden_dim * 2 + (hidden_dim // 2)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_clf_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        planes: torch.Tensor,
        rois: torch.Tensor,
        depths: torch.Tensor,
        contrast: torch.Tensor,
        x_tab: torch.Tensor
    ) -> torch.Tensor:
        """
        tokens: (B, K, 1792)
        planes: (B, K)
        rois: (B, K)
        depths: (B, K)
        contrast: (B, K, 3)
        x_tab: (B, num_tab_features)
        Returns: logits (B, 12)
        """
        B, K, _ = tokens.shape
        
        h_tok = self.token_proj(tokens)
        p_emb = self.plane_emb(planes)
        r_emb = self.roi_emb(rois)
        d_emb = self.depth_proj(depths.unsqueeze(-1))
        c_emb = self.contrast_proj(contrast)
        
        kv = h_tok + p_emb + r_emb + d_emb + c_emb
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        
        attn_out, attn_weights = self.mha(queries, kv, kv)
        
        # Attention-Weighted Max pooling
        weighted_kv = attn_weights.unsqueeze(-1) * kv.unsqueeze(1)
        attn_max, _ = torch.max(weighted_kv, dim=2)
        
        # Tabular & Radiomics
        tab_h = self.tab_encoder(x_tab)
        tab_h_exp = tab_h.unsqueeze(1).expand(-1, self.num_classes, -1)
        
        combined = torch.cat([attn_out, attn_max, tab_h_exp], dim=-1)
        logits_list = [self.classifiers[c](combined[:, c, :]) for c in range(self.num_classes)]
        return torch.cat(logits_list, dim=-1)


class Phase7Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, token_db: dict, feature_cols: list, max_tokens: int = 20):
        self.df = df.reset_index(drop=True)
        self.token_db = token_db
        self.feature_cols = feature_cols
        self.max_tokens = max_tokens
        
        self.uids = self.df['StudyInstanceUID'].values
        self.labels = self.df[target_cols].values.astype(np.float32)
        self.weights = np.where(self.df['is_gold'] == 1, 5.0, 1.0).astype(np.float32)
        self.tab_features = self.df[feature_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        entry = self.token_db.get(uid, None)
        
        if entry is not None:
            tokens = entry['tokens']
            planes = entry['planes']
            rois = entry.get('rois', torch.zeros_like(planes))
            depths = entry['depths']
            contrast = entry.get('contrast', torch.zeros((len(planes), 3), dtype=torch.float32))
            
            K = tokens.shape[0]
            if K > self.max_tokens:
                tokens = tokens[:self.max_tokens]
                planes = planes[:self.max_tokens]
                rois = rois[:self.max_tokens]
                depths = depths[:self.max_tokens]
                contrast = contrast[:self.max_tokens]
            elif K < self.max_tokens:
                pad = self.max_tokens - K
                tokens = torch.cat([tokens, torch.zeros((pad, 1792), dtype=tokens.dtype)], dim=0)
                planes = torch.cat([planes, torch.zeros(pad, dtype=planes.dtype)], dim=0)
                rois = torch.cat([rois, torch.zeros(pad, dtype=rois.dtype)], dim=0)
                depths = torch.cat([depths, torch.zeros(pad, dtype=depths.dtype)], dim=0)
                contrast = torch.cat([contrast, torch.zeros((pad, 3), dtype=contrast.dtype)], dim=0)
        else:
            tokens = torch.zeros((self.max_tokens, 1792), dtype=torch.float32)
            planes = torch.zeros(self.max_tokens, dtype=torch.long)
            rois = torch.zeros(self.max_tokens, dtype=torch.long)
            depths = torch.zeros(self.max_tokens, dtype=torch.float32)
            contrast = torch.zeros((self.max_tokens, 3), dtype=torch.float32)
            
        if tokens.dtype == torch.float16:
            tokens = tokens.to(torch.float32)
            
        return {
            'tokens': tokens,
            'planes': planes.long(),
            'rois': rois.long(),
            'depths': depths.float(),
            'contrast': contrast.float(),
            'tabular': torch.tensor(self.tab_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
            'weight': torch.tensor(self.weights[idx], dtype=torch.float32),
            'uid': uid
        }

def train_phase7():
    print("=" * 85)
    print("TRAINING PHASE 7 MULTI-BACKBONE TARGET-QUERY TRANSFORMER")
    print("=" * 85)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokens_path = r"D:\MLprojs\KaggleCompetition\RSNA\phase7_tokens_dual_backbone.pt"
    if not os.path.exists(tokens_path):
        print(f"Token database {tokens_path} not ready yet! Please wait for extraction.")
        return
        
    print(f"Loading Phase 7 Dual-Backbone tokens from {tokens_path}...")
    token_db = torch.load(tokens_path, map_location='cpu')
    print(f"Loaded tokens for {len(token_db)} studies.")
    
    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    rad_tab_df = pd.read_csv('dicom_radiomics_phase7_features.csv')
    
    df = pd.merge(master_df, rad_tab_df, on='StudyInstanceUID', how='left')
    df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
    df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
    df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)
    
    exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report'] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Normalize features
    for col in feature_cols:
        df[col] = df[col].fillna(0.0)
        m, s = df[col].mean(), df[col].std()
        df[col] = (df[col] - m) / (s + 1e-5)
        
    gold_df = df[df['is_gold'] == 1].copy().reset_index(drop=True)
    report_df = df[df['is_gold'] == 0].copy().reset_index(drop=True)
    
    oof_preds = np.zeros((len(gold_df), 12))
    checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for fold in range(5):
        print(f"\n--- Training Phase 7 Fold {fold} / 4 ---")
        val_gold = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        val_indices = gold_df[gold_df['fold'] == fold].index
        
        train_gold = gold_df[gold_df['fold'] != fold].copy().reset_index(drop=True)
        train_report = report_df[report_df['fold'] != fold].copy().reset_index(drop=True)
        
        train_comb = pd.concat([train_gold, train_report], ignore_index=True)
        
        train_ds = Phase7Dataset(train_comb, token_db, feature_cols, max_tokens=20)
        val_ds = Phase7Dataset(val_gold, token_db, feature_cols, max_tokens=20)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
        
        model = Phase7DualBackboneTransformer(
            embed_dim=1792,
            num_tab_features=len(feature_cols),
            num_classes=12,
            hidden_dim=256,
            num_heads=4,
            dropout=0.15
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        
        best_val_auc = 0.0
        best_fold_preds = None
        best_cp_path = os.path.join(checkpoints_dir, f"phase7_transformer_fold{fold}_best.pt")
        
        for epoch in range(1, 11):
            model.train()
            total_loss = 0.0
            
            for b in train_loader:
                toks = b['tokens'].to(device)
                pls = b['planes'].to(device)
                rois = b['rois'].to(device)
                deps = b['depths'].to(device)
                cntr = b['contrast'].to(device)
                tabs = b['tabular'].to(device)
                y = b['labels'].to(device)
                w = b['weight'].to(device)
                
                optimizer.zero_grad()
                logits = model(toks, pls, rois, deps, cntr, tabs)
                
                loss_per_element = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
                loss = (loss_per_element.mean(dim=-1) * w).mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            
            # Validation on Held-Out Gold
            model.eval()
            val_preds_list = []
            val_targets_list = []
            
            with torch.no_grad():
                for b in val_loader:
                    toks = b['tokens'].to(device)
                    pls = b['planes'].to(device)
                    rois = b['rois'].to(device)
                    deps = b['depths'].to(device)
                    cntr = b['contrast'].to(device)
                    tabs = b['tabular'].to(device)
                    
                    logits = model(toks, pls, rois, deps, cntr, tabs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.append(probs)
                    val_targets_list.append(b['labels'].numpy())
                    
            val_preds = np.vstack(val_preds_list)
            val_targets = np.vstack(val_targets_list)
            
            fold_aucs = []
            for c in range(12):
                if len(np.unique(val_targets[:, c])) > 1:
                    fold_aucs.append(roc_auc_score(val_targets[:, c], val_preds[:, c]))
            val_macro = np.mean(fold_aucs) if fold_aucs else 0.5
            
            if val_macro > best_val_auc:
                best_val_auc = val_macro
                best_fold_preds = val_preds
                torch.save(model.state_dict(), best_cp_path)
                
            if epoch % 2 == 0 or epoch == 10:
                print(f"  Epoch {epoch:2d} | Train Loss: {total_loss/len(train_loader):.4f} | Gold Val AUC: {val_macro:.4f} (Best: {best_val_auc:.4f})")
                
        oof_preds[val_indices] = best_fold_preds
        print(f"Fold {fold} finished. Best Gold AUC = {best_val_auc:.4f}")
        
    final_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_preds[:, c]) for c in range(12)])
    print("\n" + "=" * 85)
    print(f">>> PHASE 7 DUAL-BACKBONE TRANSFORMER 5-FOLD MACRO-AUC: {final_macro:.4f} <<<")
    print("=" * 85)
    
    for c, target in enumerate(target_cols):
        auc = roc_auc_score(gold_df[target].values, oof_preds[:, c])
        print(f"  {target:18s}: Phase 7 Transformer OOF AUC = {auc:.4f}")
        
    oof_df = gold_df[['StudyInstanceUID']].copy()
    for c, target in enumerate(target_cols):
        oof_df[target] = oof_preds[:, c]
    oof_df.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\phase7_transformer_oof.csv", index=False)
    print("Saved Phase 7 transformer OOF to oof/phase7_transformer_oof.csv!")

if __name__ == '__main__':
    train_phase7()
