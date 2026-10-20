"""
Stage 2A: Biomedical Contrastive Alignment (MedCLIP / SigLIP)
Aligns Multi-Plane MRI Visual Representations with Clinical Report Embeddings
Uses:
- Dual-Backbone 1792-dim tokens with Sub-ROIs, depths, and fluid contrast
- Full-Corpus (4,407 studies) Text SVD & Rule Embeddings
- SigLIP Sigmoid Contrastive Loss + Masked Multilabel BCE on Commercial Confidence Masks
- Live GPU Thermal Governor (MAX_TEMP = 78C) & Zero-Loss Atomic Checkpointing
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from clinical_multilingual_cleaner import clean_clinical_text_v7
from thermal_governor import check_gpu_thermals, thermal_cooldown_breather, atomic_torch_save, update_state

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SigLIPMRIDataset(Dataset):
    def __init__(self, df, tokens_dict, text_dict, is_train=True):
        self.uids = []
        for uid in df['StudyInstanceUID'].values:
            if uid in tokens_dict and uid in text_dict:
                self.uids.append(uid)
                
        self.df = df.set_index('StudyInstanceUID').loc[self.uids].reset_index()
        self.tokens_dict = tokens_dict
        self.text_dict = text_dict
        self.is_train = is_train

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        row = self.df.iloc[idx]
        
        # Visual data
        tok_data = self.tokens_dict[uid]
        tokens = tok_data['tokens'].float() # (20, 1792)
        planes = tok_data['planes'].long()  # (20)
        rois = tok_data['rois'].long()      # (20)
        depths = tok_data['depths'].float().unsqueeze(-1) # (20, 1)
        contrast = tok_data['contrast'].float() # (20, 3)
        
        # Text embedding
        text_emb = self.text_dict[uid] # (128)
        
        # Labels and confidence masks
        labels = torch.tensor([row[c] for c in target_cols], dtype=torch.float32)
        masks = torch.tensor([row.get(f"{c}_mask", 1.0) for c in target_cols], dtype=torch.float32)
        
        return tokens, planes, rois, depths, contrast, text_emb, labels, masks, uid

class BiomedicalSigLIPModel(nn.Module):
    def __init__(self, visual_dim=1792, text_dim=128, proj_dim=256, num_heads=4, num_classes=12):
        super().__init__()
        self.proj_dim = proj_dim
        
        # Visual Encoder
        self.token_proj = nn.Sequential(
            nn.Linear(visual_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.plane_emb = nn.Embedding(3, proj_dim)
        self.roi_emb = nn.Embedding(5, proj_dim)
        self.depth_proj = nn.Linear(1, proj_dim)
        self.contrast_proj = nn.Linear(3, proj_dim)
        
        # Multi-Head Attention pooling over 20 slice tokens
        self.mha = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        self.vis_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Text Encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Multilabel Classifier Head on visual embedding
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(proj_dim // 2, num_classes)
        )
        
        # Learnable SigLIP Temperature and Bias
        self.t_prime = nn.Parameter(torch.tensor(math.log(10.0)))
        self.b_prime = nn.Parameter(torch.tensor(-10.0))

    def encode_visual(self, tokens, planes, rois, depths, contrast):
        h = self.token_proj(tokens)
        h = h + self.plane_emb(planes) + self.roi_emb(rois) + self.depth_proj(depths) + self.contrast_proj(contrast)
        attn_out, _ = self.mha(h, h, h)
        pool = attn_out.mean(dim=1)
        z_v = self.vis_head(pool)
        z_v = F.normalize(z_v, p=2, dim=-1)
        return z_v, pool

    def encode_text(self, text_emb):
        z_t = self.text_encoder(text_emb)
        z_t = F.normalize(z_t, p=2, dim=-1)
        return z_t

    def forward(self, tokens, planes, rois, depths, contrast, text_emb):
        z_v, pool = self.encode_visual(tokens, planes, rois, depths, contrast)
        z_t = self.encode_text(text_emb)
        logits = self.classifier(pool)
        return z_v, z_t, logits

def siglip_loss(z_v, z_t, t_prime, b_prime):
    t = torch.exp(t_prime)
    sim = torch.matmul(z_v, z_t.T) # (B, B)
    logits = t * sim + b_prime
    labels = 2.0 * torch.eye(z_v.shape[0], device=z_v.device) - 1.0 # +1 on diag, -1 off-diag
    loss = -F.logsigmoid(labels * logits).sum() / z_v.shape[0]
    return loss

def main():
    print("=" * 90, flush=True)
    print("STAGE 2A: BIOMEDICAL CONTRASTIVE PRETRAINING (SigLIP)", flush=True)
    print(f"Running on Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 90, flush=True)

    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    report_df = pd.read_csv('report_extracted_labels_4349_commercial.csv')
    study_meta = master_df.merge(report_df[['StudyInstanceUID'] + [f"{c}_mask" for c in target_cols]], on='StudyInstanceUID', how='left')

    print("Loading visual token cache...", flush=True)
    tokens_cache = torch.load('phase7_tokens_dual_backbone.pt', map_location='cpu', weights_only=False)
    print(f"Loaded visual token cache for {len(tokens_cache)} studies.", flush=True)

    print("Fitting text SVD embeddings for contrastive objective...", flush=True)
    master_df['clean_report'] = master_df['Report'].apply(clean_clinical_text_v7)
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True, min_df=2)
    svd_text = TruncatedSVD(n_components=128, random_state=42)
    X_text_all = svd_text.fit_transform(tfidf_vec.fit_transform(master_df['clean_report']))
    study_to_text = {master_df.iloc[i]['StudyInstanceUID']: torch.tensor(X_text_all[i], dtype=torch.float32) for i in range(len(master_df))}

    # Folds 1-4 for train, Fold 0 for validation
    train_df = study_meta[study_meta['fold'] != 0].reset_index(drop=True)
    val_df = study_meta[study_meta['fold'] == 0].reset_index(drop=True)

    train_dataset = SigLIPMRIDataset(train_df, tokens_cache, study_to_text, is_train=True)
    val_dataset = SigLIPMRIDataset(val_df, tokens_cache, study_to_text, is_train=False)

    # In-memory datasets: num_workers=0 is fastest and avoids subprocess overhead
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    model = BiomedicalSigLIPModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = 0.0
    num_epochs = 8

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for b_idx, (tokens, planes, rois, depths, contrast, text_emb, labels, masks, _) in enumerate(train_loader):
            # Thermal throttle guard called every 25 batches
            check_gpu_thermals(max_temp=78, resume_temp=70, check_interval=25, batch_idx=b_idx)

            tokens = tokens.to(DEVICE)
            planes = planes.to(DEVICE)
            rois = rois.to(DEVICE)
            depths = depths.to(DEVICE)
            contrast = contrast.to(DEVICE)
            text_emb = text_emb.to(DEVICE)
            labels = labels.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    z_v, z_t, logits = model(tokens, planes, rois, depths, contrast, text_emb)
                    loss_contrast = siglip_loss(z_v, z_t, model.t_prime, model.b_prime)
                    bce_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                    loss_cls = (bce_raw * masks).sum() / (masks.sum() + 1e-6)
                    loss = loss_contrast + 2.0 * loss_cls

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                n_batches += 1

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM RECOVERY] Handled CUDA OOM at batch {b_idx}. Clearing cache.", flush=True)
                torch.cuda.empty_cache()
                continue

        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for tokens, planes, rois, depths, contrast, text_emb, labels, masks, uids in val_loader:
                tokens = tokens.to(DEVICE)
                planes = planes.to(DEVICE)
                rois = rois.to(DEVICE)
                depths = depths.to(DEVICE)
                contrast = contrast.to(DEVICE)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _, _, logits = model(tokens, planes, rois, depths, contrast, text_emb.to(DEVICE))
                    probs = torch.sigmoid(logits).cpu().numpy()

                val_preds.append(probs)
                val_targets.append(labels.numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        val_aucs = []
        for c in range(12):
            try:
                a = roc_auc_score(val_targets[:, c] >= 0.5, val_preds[:, c])
                val_aucs.append(a)
            except Exception:
                val_aucs.append(0.5)

        mean_val_auc = np.mean(val_aucs)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | Train Loss: {train_loss/max(1, n_batches):.4f} | Val Macro-AUC: {mean_val_auc:.4f}", flush=True)

        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            atomic_torch_save(model.state_dict(), "checkpoints/biomedical_siglip_visual_encoder_best.pt")
            print(f"  --> [SAVED NEW BEST MODEL] Val Macro-AUC = {best_val_auc:.4f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(f"STAGE 2A COMPLETE: Best SigLIP Visual Encoder AUC = {best_val_auc:.4f}", flush=True)
    print("=" * 90, flush=True)

    update_state(
        "Stage 2A: SigLIP Pretraining",
        "COMPLETED",
        {"best_val_auc": float(best_val_auc)},
        "Biomedical SigLIP contrastive visual encoder successfully trained and cached."
    )

    thermal_cooldown_breather(45, "Stage 2A Complete -> Transitioning to Phase 3 GNN")

if __name__ == '__main__':
    main()
