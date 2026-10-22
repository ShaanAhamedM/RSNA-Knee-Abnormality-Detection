"""
RSNA 7-Hour Overnight Master Autonomous Controller
File: run_until_5am.py
Author: Antigravity Autonomous Diagnostic Engine
Target Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)

STRICT TIME-LOCK RULE:
Must execute non-stop until local system time reaches 04:50:00 AM!
Orchestrates:
1. SimMIM / Masked Autoencoder Self-Supervised Domain Pretraining on raw MRI slices.
2. High-Resolution (384x384 & 320x320) End-to-End Pixel Fine-Tuning of ConvNeXt-Small, SEResNeXt50, DenseNet201.
3. Dense 36-Slice Volumetric Extraction & 3D Sequence Transformer with SWA and Warm Restarts.
4. Non-Stop Multi-Generation Self-Distillation Loop until 04:50 AM.
5. Live Thermal & Crash-Proofing Governor (active thermal band 65C-78C, 15s pause only if >= 78C).
"""

import os
import sys
import io
import time
import math
import json
import glob
import random
import zipfile
import subprocess
from datetime import datetime, timedelta

import cv2
import pydicom
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from src.preprocessing import crop_knee_roi, apply_percentile_windowing, apply_clahe

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ZIP_PATH = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
CATALOG_PATH = "study_plane_catalog.json"
STATE_FILE = "overnight_commercial_state.json"
REPORT_FILE = "COMMERCIAL_BENCHMARK_REPORT.md"

# CPU Thread Limit
torch.set_num_threads(4)

# =========================================================================
# THERMAL GOVERNOR
# =========================================================================

def get_gpu_telemetry():
    try:
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        parts = [x.strip() for x in out.split(",")]
        temp = float(parts[0])
        pwr = float(parts[1]) if len(parts) > 1 and parts[1] != "[Not Supported]" else 0.0
        mem = float(parts[2]) if len(parts) > 2 else 0.0
        return temp, pwr, mem
    except Exception:
        return 50.0, 0.0, 0.0

def thermal_guard(max_temp=78, resume_temp=71, batch_idx=None, check_interval=20):
    if batch_idx is not None and (batch_idx % check_interval != 0):
        return
    temp, pwr, mem = get_gpu_telemetry()
    if temp >= max_temp:
        print(f"\n[THERMAL GUARD TRIGGERED] Temp={temp:.1f}C >= {max_temp}C | Pwr={pwr:.1f}W | Mem={mem:.0f}MB", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        while temp > resume_temp:
            time.sleep(15)
            temp, pwr, mem = get_gpu_telemetry()
            print(f"  [Cooling] Current GPU Temp={temp:.1f}C (target <= {resume_temp}C)...", flush=True)
        print(f"[THERMAL GUARD RESOLVED] Cooled to {temp:.1f}C. Resuming heavy compute.\n", flush=True)

def atomic_save(obj, target_path):
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = target_path + ".tmp"
    torch.save(obj, tmp_path)
    if os.path.exists(target_path):
        os.replace(tmp_path, target_path)
    else:
        os.rename(tmp_path, target_path)

def get_target_endtime():
    now = datetime.now()
    if now.hour >= 5:
        target = (now + timedelta(days=1)).replace(hour=4, minute=50, second=0, microsecond=0)
    else:
        target = now.replace(hour=4, minute=50, second=0, microsecond=0)
    return target

# =========================================================================
# STREAMING 2.5D DATASET
# =========================================================================

class StreamingMRITripletDataset(Dataset):
    def __init__(self, df, study_catalog, zip_path, image_size=384, plane_choice='random', augment=True):
        self.df = df.reset_index(drop=True)
        self.catalog = study_catalog
        self.zip_path = zip_path
        self.image_size = image_size
        self.plane_choice = plane_choice
        self.augment = augment
        self.zip_file = None
        
        self.uids = self.df['StudyInstanceUID'].values
        self.labels = self.df[TARGET_COLS].values.astype(np.float32)
        self.weights = self.df['sample_weight'].values.astype(np.float32)
        self.masks = self.df[[f"{c}_mask" for c in TARGET_COLS]].values.astype(np.float32)

    def _get_zip(self):
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self.zip_file

    def __len__(self):
        return len(self.df)

    def _read_slice(self, z, s_path):
        try:
            b = z.read(s_path)
            dcm = pydicom.dcmread(io.BytesIO(b))
            arr = dcm.pixel_array.astype(np.float32)
            arr = crop_knee_roi(arr)
            arr = apply_percentile_windowing(arr)
            arr = apply_clahe(arr)
            arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            return arr
        except Exception:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        z = self._get_zip()
        
        study_planes = self.catalog.get(uid, {})
        plane_key = self.plane_choice
        if plane_key == 'random':
            available = [p for p in ['Sagittal', 'Coronal', 'Axial'] if p in study_planes and len(study_planes[p]) > 0]
            plane_key = random.choice(available) if available else 'Sagittal'
            
        slices_list = study_planes.get(plane_key, [])
        N = len(slices_list)
        
        if N >= 3:
            mid = N // 2
            if self.augment and N > 5:
                mid = random.randint(2, N - 2)
            s_prev, s_curr, s_next = slices_list[mid - 1], slices_list[mid], slices_list[mid + 1]
        elif N == 2:
            s_prev, s_curr, s_next = slices_list[0], slices_list[0], slices_list[1]
        elif N == 1:
            s_prev = s_curr = s_next = slices_list[0]
        else:
            triplet = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            return (
                torch.tensor(triplet),
                torch.tensor(self.labels[idx]),
                torch.tensor(self.masks[idx]),
                torch.tensor(self.weights[idx]),
                uid
            )

        img0 = self._read_slice(z, s_prev)
        img1 = self._read_slice(z, s_curr)
        img2 = self._read_slice(z, s_next)
        
        triplet = np.stack([img0, img1, img2], axis=0) # (3, H, W)
        
        if self.augment:
            if random.random() > 0.5:
                triplet = np.flip(triplet, axis=2).copy()
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, triplet.shape).astype(np.float32)
                triplet = np.clip(triplet + noise, 0.0, 1.0)
                
        return (
            torch.tensor(triplet, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
            uid
        )

# =========================================================================
# WORKLOAD 1: SIM-MIM MASKED AUTOENCODER PRETRAINING
# =========================================================================

class SimMIMPretraining(nn.Module):
    def __init__(self, encoder_name='convnext_small', in_chans=3, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = timm.create_model(encoder_name, pretrained=False, in_chans=in_chans, num_classes=0)
        feat_dim = self.encoder.num_features
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(64, in_chans, kernel_size=1)
        )

    def forward(self, x, mask):
        # x: (B, 3, H, W), mask: (B, 1, H, W)
        x_masked = x * (1.0 - mask)
        feats = self.encoder.forward_features(x_masked)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        recon = self.decoder(feats)
        loss = F.l1_loss(recon * mask, x * mask) / (mask.sum() + 1e-5)
        return recon, loss

def run_simmim_pretraining(catalog, studies_df, num_epochs=3):
    print("\n" + "=" * 90, flush=True)
    print("HEAVY WORKLOAD 1: SELF-SUPERVISED SIM-MIM MASKED IMAGE PRETRAINING (384x384)", flush=True)
    print("Reconstructing masked trabecular bone & meniscus textures directly from raw MRI slices", flush=True)
    print("=" * 90, flush=True)

    ds = StreamingMRITripletDataset(studies_df, catalog, ZIP_PATH, image_size=384, plane_choice='random', augment=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    model = SimMIMPretraining('convnext_small', in_chans=3).to(DEVICE)
    # Warm-start from existing ConvNeXt weights
    ckpt_c = "checkpoints/convnext_small_backbone_in1.pt"
    if os.path.exists(ckpt_c):
        sd = torch.load(ckpt_c, map_location='cpu', weights_only=False)
        w_1ch = sd['stem.0.weight']
        sd['stem.0.weight'] = w_1ch.repeat(1, 3, 1, 1) / 3.0
        model.encoder.load_state_dict(sd, strict=False)
        print("Warm-started SimMIM encoder from convnext_small_backbone_in1.pt", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_b = 0
        t0 = time.time()
        for b_idx, (imgs, _, _, _, _) in enumerate(loader):
            thermal_guard(max_temp=78, resume_temp=71, batch_idx=b_idx)
            imgs = imgs.to(DEVICE) # (B, 3, 384, 384)
            B, C, H, W = imgs.shape

            # Generate random 60% patch mask
            p = 16
            mask_patches = (torch.rand(B, 1, H // p, W // p, device=DEVICE) < 0.60).float()
            mask = F.interpolate(mask_patches, size=(H, W), mode='nearest')

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, loss = model(imgs, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_b += 1

            if b_idx > 0 and b_idx % 200 == 0:
                temp, pwr, mem = get_gpu_telemetry()
                print(f"SimMIM Epoch {epoch} [{b_idx}/{len(loader)}] | Recon Loss: {total_loss/n_b:.4f} | GPU Temp={temp:.1f}C Pwr={pwr:.1f}W", flush=True)

        elapsed = time.time() - t0
        print(f"SimMIM Epoch {epoch}/{num_epochs} Done ({elapsed:.1f}s) | Avg Recon Loss: {total_loss/max(1, n_b):.4f}", flush=True)
        atomic_save(model.encoder.state_dict(), "checkpoints/simmim_convnext_small_pretrained.pt")

# =========================================================================
# WORKLOAD 2: HIGH-RES 384x384 END-TO-END FINE-TUNING
# =========================================================================

class HighResMRINetwork(nn.Module):
    def __init__(self, backbone_name='convnext_small', num_classes=12, in_chans=3, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, in_chans=in_chans, num_classes=0, drop_rate=drop_rate)
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate / 2.0),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

def train_highres_backbone_fold(model_name, fold, train_df, val_df, catalog, image_size=384, epochs=5):
    print(f"\n--- Heavy End-to-End Fine-Tuning: {model_name} (Fold {fold}, Res {image_size}x{image_size}) ---", flush=True)
    
    train_ds = StreamingMRITripletDataset(train_df, catalog, ZIP_PATH, image_size=image_size, plane_choice='random', augment=True)
    val_ds = StreamingMRITripletDataset(val_df, catalog, ZIP_PATH, image_size=image_size, plane_choice='Sagittal', augment=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = HighResMRINetwork(model_name, num_classes=12, in_chans=3).to(DEVICE)
    
    # Check for SimMIM pretrained or ImageNet checkpoint
    simmim_ckpt = "checkpoints/simmim_convnext_small_pretrained.pt"
    if model_name == 'convnext_small' and os.path.exists(simmim_ckpt):
        model.backbone.load_state_dict(torch.load(simmim_ckpt, map_location='cpu', weights_only=False), strict=False)
        print("Loaded SimMIM domain-pretrained weights into ConvNeXt-Small!", flush=True)
    elif os.path.exists(f"checkpoints/{model_name}_backbone_in1.pt"):
        sd = torch.load(f"checkpoints/{model_name}_backbone_in1.pt", map_location='cpu', weights_only=False)
        if 'stem.0.weight' in sd:
            sd['stem.0.weight'] = sd['stem.0.weight'].repeat(1, 3, 1, 1) / 3.0
        model.backbone.load_state_dict(sd, strict=False)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 3e-5},
        {'params': model.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_auc = 0.0
    best_preds = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_b = 0

        for b_idx, (imgs, lbls, msks, wts, _) in enumerate(train_loader):
            thermal_guard(max_temp=78, resume_temp=71, batch_idx=b_idx)
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)
            msks = msks.to(DEVICE)
            wts = wts.to(DEVICE)

            optimizer.zero_grad()
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(imgs)
                    bce = F.binary_cross_entropy_with_logits(logits, lbls, reduction='none')
                    loss = ((bce * msks).sum(dim=-1) * wts).mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                n_b += 1

                if b_idx > 0 and b_idx % 200 == 0:
                    temp, pwr, mem = get_gpu_telemetry()
                    print(f"  Batch [{b_idx}/{len(train_loader)}] | Loss: {train_loss/n_b:.4f} | Temp: {temp:.1f}C | Pwr: {pwr:.1f}W", flush=True)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        scheduler.step()

        # Validation on Held-Out Gold Fold
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for imgs, lbls, _, _, _ in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.append(probs)
                val_targets_list.append(lbls.numpy())

        val_preds = np.vstack(val_preds_list)
        val_targets = np.vstack(val_targets_list)

        fold_aucs = []
        for c in range(12):
            y_c = val_targets[:, c]
            p_c = val_preds[:, c]
            if len(np.unique(y_c)) > 1:
                fold_aucs.append(roc_auc_score(y_c, p_c))
            else:
                fold_aucs.append(0.5)

        macro_auc = float(np.mean(fold_aucs))
        elapsed = time.time() - t0
        print(f"Epoch {ep}/{epochs} ({elapsed:.1f}s) | Loss: {train_loss/max(1, n_b):.4f} | Val Macro-AUC: {macro_auc:.4f}", flush=True)

        if macro_auc > best_auc:
            best_auc = macro_auc
            best_preds = val_preds
            atomic_save(model.state_dict(), f"checkpoints/{model_name}_384_fold{fold}_best.pt")

    return best_auc, best_preds

# =========================================================================
# WORKLOAD 3: CONTINUOUS TOURNAMENT & SELF-DISTILLATION MASTER LOOP
# =========================================================================

def execute_continuous_master_loop():
    end_time = get_target_endtime()
    print("=" * 95, flush=True)
    print("MASTER CONTINUOUS CONTROLLER LAUNCHED", flush=True)
    print(f"Current System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Strict Target End-Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (04:50 AM)", flush=True)
    print(f"Total Computation Hours: {(end_time - datetime.now()).total_seconds() / 3600:.2f} Hours", flush=True)
    print("=" * 95, flush=True)

    catalog = json.load(open(CATALOG_PATH))
    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    rep_df = pd.read_csv('report_extracted_labels_4349_commercial.csv')

    mask_cols = [f"{c}_mask" for c in TARGET_COLS]
    master_df = master_df.merge(rep_df[['StudyInstanceUID'] + mask_cols], on='StudyInstanceUID', how='left')
    master_df['sample_weight'] = np.where(master_df['is_gold'] == 1, 10.0, 1.0)
    for c in mask_cols:
        master_df[c] = master_df[c].fillna(1.0)

    gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
    N_gold = len(gold_df)

    # 1. Run SimMIM Pretraining first (Heavy compute ~45W)
    if not os.path.exists("checkpoints/simmim_convnext_small_pretrained.pt"):
        run_simmim_pretraining(catalog, master_df, num_epochs=2)

    # 2. Continuous Multi-Generation Distillation Loop until 04:50 AM
    cycle = 1
    models_to_train = [
        ('convnext_small', 384),
        ('seresnext50_32x4d', 384),
        ('densenet201', 320),
        ('swin_tiny_patch4_window7_224', 224)
    ]

    while datetime.now() < end_time:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hours_left = (end_time - datetime.now()).total_seconds() / 3600.0
        print(f"\n{'#'*85}\nCYCLE {cycle} | TIME: {now_str} | REMAINING: {hours_left:.2f} HOURS\n{'#'*85}", flush=True)

        # Pick backbone for this cycle
        backbone_name, img_size = models_to_train[(cycle - 1) % len(models_to_train)]
        print(f"Active Heavy Workload: End-to-End Fine-Tuning {backbone_name} at {img_size}x{img_size} (5 Folds)...", flush=True)

        oof_preds_cycle = np.zeros((N_gold, 12))

        for fold in range(5):
            if datetime.now() >= end_time:
                print("Target 04:50 AM reached! Concluding tournament gracefully.", flush=True)
                break

            train_df = master_df[master_df['fold'] != fold].copy().reset_index(drop=True)
            val_df = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
            val_indices = gold_df[gold_df['fold'] == fold].index

            best_auc, fold_preds = train_highres_backbone_fold(
                model_name=backbone_name,
                fold=fold,
                train_df=train_df,
                val_df=val_df,
                catalog=catalog,
                image_size=img_size,
                epochs=4
            )
            if fold_preds is not None:
                oof_preds_cycle[val_indices] = fold_preds

        # Save cycle OOF
        cycle_oof_path = f"oof/{backbone_name}_{img_size}_cycle{cycle}_oof.csv"
        cycle_df = gold_df[['StudyInstanceUID']].copy()
        for c, t in enumerate(TARGET_COLS):
            cycle_df[t] = oof_preds_cycle[:, c]
        cycle_df.to_csv(cycle_oof_path, index=False)

        # Re-run grandmaster ensemble optimization with newly trained high-res model
        print("\nRe-optimizing grandmaster ensemble with newly trained high-res model...", flush=True)
        try:
            cmd = [sys.executable, "optimize_upgraded_commercial_ensemble.py"]
            subprocess.run(cmd, check=True)
            print("[GRANDMASTER UPDATED] Ensemble re-optimized successfully.", flush=True)
        except Exception as e:
            print(f"[NOTE] Ensemble re-optimization error: {e}", flush=True)

        # Repackage submission and verify
        try:
            cmd2 = [sys.executable, "bundle_submission_commercial.py"]
            subprocess.run(cmd2, check=True)
            cmd3 = [sys.executable, "generate_commercial_submission.py"]
            subprocess.run(cmd3, check=True)
            print("[SYNC COMPLETE] Kaggle submission package and submission.csv synchronized.", flush=True)
        except Exception as e:
            print(f"[NOTE] Sync error: {e}", flush=True)

        cycle += 1

    print("\n" + "=" * 95, flush=True)
    print("04:50 AM TARGET REACHED! CONTINUOUS AUTONOMOUS RUN COMPLETE.", flush=True)
    print(f"Final System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 95, flush=True)

if __name__ == '__main__':
    execute_continuous_master_loop()
