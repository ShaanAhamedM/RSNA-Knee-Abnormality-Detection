"""
Upgrade 2: Multi-Slice 12-Query Cross-Attention MIL Transformer with Tabular & Radiomic Fusion.
Preserves individual slice tokens (12-18 per study), learned plane embeddings, and slice depth encodings.
Computes both Attention-Weighted Mean and Attention-Weighted Max per pathology query.
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

class MultiSliceTargetQueryTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_tab_features: int = 45,
        num_classes: int = 12,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Token projection from backbone dim (768) to hidden_dim (256)
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Plane embeddings (0=Sagittal, 1=Coronal, 2=Axial)
        self.plane_emb = nn.Embedding(3, hidden_dim)
        
        # Relative slice depth linear projection
        self.depth_proj = nn.Linear(1, hidden_dim)
        
        # 12 learnable pathology query vectors
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        
        # Cross-Attention: 12 queries attend to all slice tokens
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
        
        # 12 Independent Target Classifiers (Fuses Attended Mean + Attended Max + Radiomics)
        # Input dim: hidden_dim (mean) + hidden_dim (max) + hidden_dim // 2 (radiomics) = 2.5 * hidden_dim
        in_clf_dim = hidden_dim * 2 + (hidden_dim // 2)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_clf_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x_tokens: torch.Tensor, planes: torch.Tensor, depths: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: (B, K, 768)
        planes: (B, K)
        depths: (B, K)
        x_tab: (B, num_tab_features)
        Returns: logits (B, 12)
        """
        B, K, _ = x_tokens.shape
        
        # Token projection + plane embeddings + depth projection
        h_tokens = self.token_proj(x_tokens) # (B, K, hidden_dim)
        p_emb = self.plane_emb(planes) # (B, K, hidden_dim)
        d_emb = self.depth_proj(depths.unsqueeze(-1)) # (B, K, hidden_dim)
        
        # Positionalized tokens
        kv = h_tokens + p_emb + d_emb # (B, K, hidden_dim)
        
        # Target queries: (B, 12, hidden_dim)
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-Attention
        # attn_out: (B, 12, hidden_dim), attn_weights: (B, 12, K)
        attn_out, attn_weights = self.mha(queries, kv, kv)
        
        # Attention-Weighted Max pooling across slices
        # attn_weights: (B, 12, K, 1), kv: (B, 1, K, hidden_dim)
        weighted_kv = attn_weights.unsqueeze(-1) * kv.unsqueeze(1) # (B, 12, K, hidden_dim)
        attn_max, _ = torch.max(weighted_kv, dim=2) # (B, 12, hidden_dim)
        
        # Radiomics & Tabular representation
        tab_h = self.tab_encoder(x_tab) # (B, hidden_dim // 2)
        tab_h_expanded = tab_h.unsqueeze(1).expand(-1, self.num_classes, -1) # (B, 12, hidden_dim // 2)
        
        # Concatenate Attended Mean + Attended Max + Radiomics
        combined = torch.cat([attn_out, attn_max, tab_h_expanded], dim=-1) # (B, 12, in_clf_dim)
        
        logits_list = [self.classifiers[c](combined[:, c, :]) for c in range(self.num_classes)]
        logits = torch.cat(logits_list, dim=-1)
        return logits

class MultiSliceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, token_db: dict, feature_cols: list, max_slices: int = 16):
        self.df = df.reset_index(drop=True)
        self.token_db = token_db
        self.feature_cols = feature_cols
        self.max_slices = max_slices
        
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
            depths = entry['depths']
            
            K = tokens.shape[0]
            if K > self.max_slices:
                tokens = tokens[:self.max_slices]
                planes = planes[:self.max_slices]
                depths = depths[:self.max_slices]
            elif K < self.max_slices:
                pad_k = self.max_slices - K
                tokens = torch.cat([tokens, torch.zeros((pad_k, 768), dtype=tokens.dtype)], dim=0)
                planes = torch.cat([planes, torch.zeros(pad_k, dtype=planes.dtype)], dim=0)
                depths = torch.cat([depths, torch.zeros(pad_k, dtype=depths.dtype)], dim=0)
        else:
            tokens = torch.zeros((self.max_slices, 768), dtype=torch.float32)
            planes = torch.zeros(self.max_slices, dtype=torch.long)
            depths = torch.zeros(self.max_slices, dtype=torch.float32)

        if tokens.dtype == torch.float16:
            tokens = tokens.to(torch.float32)

        return {
            'tokens': tokens,
            'planes': planes.long(),
            'depths': depths.float(),
            'tabular': torch.tensor(self.tab_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
            'weight': torch.tensor(self.weights[idx], dtype=torch.float32),
            'uid': uid
        }

def train_and_eval_multislice():
    print("=" * 85)
    print("STAGE 3B: TRAINING MULTI-SLICE TARGET-QUERY TRANSFORMER")
    print("=" * 85)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    tokens_path = r"D:\MLprojs\KaggleCompetition\RSNA\multislice_tokens_convnext.pt"
    if not os.path.exists(tokens_path):
        print(f"Token database {tokens_path} not found yet!")
        return
        
    print(f"Loading cached slice tokens from {tokens_path}...")
    token_db = torch.load(tokens_path, map_location='cpu')
    print(f"Loaded tokens for {len(token_db)} studies.")
    
    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    rad_tab_df = pd.read_csv('dicom_and_radiomics_features.csv')
    
    df = pd.merge(master_df, rad_tab_df, on='StudyInstanceUID', how='left')
    df['ratio_sag_total'] = df['n_sagittal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_cor_total'] = df['n_coronal_slices'] / (df['n_total_slices'] + 1e-5)
    df['ratio_ax_total'] = df['n_axial_slices'] / (df['n_total_slices'] + 1e-5)
    df['tr_te_ratio'] = df['mean_tr'] / (df['mean_te'] + 1e-5)
    df = pd.get_dummies(df, columns=['manufacturer'], drop_first=True)
    
    exclude = ['StudyInstanceUID', 'fold', 'is_gold', 'Report', 'has_report'] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Fill NAs and normalize
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
        print(f"\n--- Training Fold {fold} / 4 ---")
        val_gold = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        val_indices = gold_df[gold_df['fold'] == fold].index
        
        train_gold = gold_df[gold_df['fold'] != fold].copy().reset_index(drop=True)
        train_report = report_df[report_df['fold'] != fold].copy().reset_index(drop=True)
        train_combined = pd.concat([train_gold, train_report], ignore_index=True)
        
        train_ds = MultiSliceDataset(train_combined, token_db, feature_cols, max_slices=16)
        val_ds = MultiSliceDataset(val_gold, token_db, feature_cols, max_slices=16)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        model = MultiSliceTargetQueryTransformer(
            embed_dim=768, num_tab_features=len(feature_cols), num_classes=12, hidden_dim=256, dropout=0.2
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
        
        best_val_auc = 0.0
        best_fold_preds = None
        
        for ep in range(40):
            model.train()
            for batch in train_loader:
                tokens = batch['tokens'].to(device)
                planes = batch['planes'].to(device)
                depths = batch['depths'].to(device)
                x_tab = batch['tabular'].to(device)
                y_targets = batch['labels'].to(device)
                weights = batch['weight'].to(device)
                
                optimizer.zero_grad()
                logits = model(tokens, planes, depths, x_tab)
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
                    tokens = batch['tokens'].to(device)
                    planes = batch['planes'].to(device)
                    depths = batch['depths'].to(device)
                    x_tab = batch['tabular'].to(device)
                    logits = model(tokens, planes, depths, x_tab)
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
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"multislice_transformer_fold{fold}_best.pt"))
                
        print(f"Fold {fold} Best Gold OOF Macro-AUC: {best_val_auc:.4f}")
        oof_preds[val_indices] = best_fold_preds
        
    print("\n" + "=" * 85)
    final_macro = np.mean([roc_auc_score(gold_df[target_cols[c]].values, oof_preds[:, c]) for c in range(12)])
    print(f">>> MULTI-SLICE TRANSFORMER 5-FOLD MACRO-AUC: {final_macro:.4f} <<<")
    print("=" * 85)
    for c in range(12):
        auc = roc_auc_score(gold_df[target_cols[c]].values, oof_preds[:, c])
        print(f"  {target_cols[c]:18s}: OOF AUC = {auc:.4f}")
        
    oof_df = gold_df[['StudyInstanceUID']].copy()
    for c, target in enumerate(target_cols):
        oof_df[target] = oof_preds[:, c]
    oof_df.to_csv(r"D:\MLprojs\KaggleCompetition\RSNA\oof\multislice_transformer_oof.csv", index=False)
    print("Saved multi-slice transformer OOF to oof/multislice_transformer_oof.csv!")

if __name__ == '__main__':
    train_and_eval_multislice()
