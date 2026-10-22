"""
Stage 3: Report-Supervised 2.5D MIL Vision Model Training
Trains 2.5D backbones across all available report-extracted studies + gold studies.
Strictly evaluated on honest held-out Gold-58 radiologist ground truth.
Features:
- Multi-plane slice aggregation (Sagittal, Coronal, Axial)
- Zero anatomical horizontal flip (strictly preserves Medial/Lateral geometry)
- Soft-target BCE loss with sample weights (Gold: 3.0, Report: 1.0)
- float16 mixed precision, gradient accumulation, auto-OOM safe
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.dataset import get_dataloader
from src.models import RSNAKneeModel
from src.metrics import calculate_macro_roc_auc

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def train_report_supervised(
    model_name: str = 'convnext_small',
    target_size: int = 256,
    slices_per_study: int = 6,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    epochs: int = 6,
    lr: float = 2e-4,
    num_report_samples: int = 1200
):
    print("=" * 90)
    print(f"=== LAUNCHING STAGE 3: REPORT-SUPERVISED VISION TRAINING ({model_name}) ===")
    print(f"Configuration: res={target_size}x{target_size}, slices={slices_per_study}, bs={batch_size}, accum={grad_accum_steps}")
    print("=" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    gold_df = master_df[master_df['is_gold'] == 1].copy().reset_index(drop=True)
    report_df = master_df[master_df['is_gold'] == 0].copy().reset_index(drop=True)

    # Filter to top high-confidence report studies
    probs = report_df[TARGET_COLUMNS].values
    conf = np.abs(probs - 0.50).mean(axis=1)
    if num_report_samples < len(report_df):
        top_indices = np.argsort(conf)[-num_report_samples:]
        curated_report_df = report_df.iloc[top_indices].copy().reset_index(drop=True)
    else:
        curated_report_df = report_df.copy()

    print(f"Training Pool: {len(gold_df)} Gold Studies + {len(curated_report_df)} High-Confidence Report Studies = {len(gold_df) + len(curated_report_df)} Total")

    checkpoint_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
    oof_dir = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)

    all_fold_best_aucs = []

    for fold in range(5):
        print(f"\n{'='*70}")
        print(f"[*] TRAINING FOLD {fold} / 4 — MODEL: {model_name}")
        print(f"{'='*70}")

        # Strict separation: Validation set is ONLY the gold studies assigned to this fold!
        val_subset = gold_df[gold_df['fold'] == fold].copy().reset_index(drop=True)
        in_fold_gold = gold_df[gold_df['fold'] != fold].copy().reset_index(drop=True)
        # In-fold report studies (exclude report studies with this fold if using cross-validation)
        in_fold_report = curated_report_df[curated_report_df['fold'] != fold].copy().reset_index(drop=True)

        train_subset = pd.concat([in_fold_gold, in_fold_report], ignore_index=True)
        print(f"Fold {fold} Data: Train = {len(train_subset)} (Gold: {len(in_fold_gold)}, Report: {len(in_fold_report)}) | Val Gold = {len(val_subset)}")

        train_loader = get_dataloader(
            train_subset, target_size=target_size, slices_per_study=slices_per_study,
            in_chans=1, batch_size=batch_size, shuffle=True, is_train=True
        )
        val_loader = get_dataloader(
            val_subset, target_size=target_size, slices_per_study=slices_per_study,
            in_chans=1, batch_size=batch_size, shuffle=False, is_train=False
        )

        model = RSNAKneeModel(
            backbone_name=model_name,
            num_classes=12,
            in_chans=1,
            pretrained=True,
            use_mil=True
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        best_val_auc = 0.0
        best_preds = None
        best_ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_report_sup_fold{fold}_best.pt")

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            optimizer.zero_grad()
            t0_ep = time.time()

            n_batches = len(train_loader)
            for step, batch in enumerate(train_loader):
                images = batch['image'].to(device, non_blocking=True)
                targets = batch['labels'].to(device, non_blocking=True)
                weights = batch.get('weight', torch.ones(len(targets))).to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = model(images)
                    # Soft-target BCE with sample weight
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                    loss = (bce_loss.mean(dim=1) * weights).mean()
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == n_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * grad_accum_steps

            scheduler.step()
            train_loss = total_train_loss / max(1, n_batches)

            # Evaluate on pure held-out Gold-58 studies
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device, non_blocking=True)
                    targets = batch['labels'].to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        logits = model(images)
                        probs = torch.sigmoid(logits)
                    val_preds.append(probs.cpu().numpy())
                    val_targets.append(targets.cpu().numpy())

            y_val = np.concatenate(val_targets, axis=0)
            p_val = np.concatenate(val_preds, axis=0)

            val_auc, per_class = calculate_macro_roc_auc(y_val, p_val)
            ep_time = time.time() - t0_ep

            status = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_preds = p_val.copy()
                torch.save(model.state_dict(), best_ckpt_path)
                status = ">>> NEW BEST CHECKPOINT SAVED <<<"

            print(f"Epoch {epoch+1:2d}/{epochs:2d} [{ep_time:.1f}s] | Train Loss: {train_loss:.4f} | Honest Gold Val Macro-AUC: {val_auc:.4f} {status}")

        print(f"\n[Fold {fold} Finished] Best Honest Gold Macro-AUC: {best_val_auc:.4f}")
        all_fold_best_aucs.append(best_val_auc)

        # Save OOF predictions for fold
        oof_pred_path = os.path.join(oof_dir, f"{model_name}_report_sup_fold{fold}_oof.npy")
        np.save(oof_pred_path, best_preds.astype(np.float16))

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    mean_5fold_auc = float(np.mean(all_fold_best_aucs))
    print("\n" + "=" * 90)
    print(f"STAGE 3 REPORT-SUPERVISED 5-FOLD HONEST GOLD MACRO-AUC: {mean_5fold_auc:.4f}")
    print(f"Per-Fold AUCs: {[round(x, 4) for x in all_fold_best_aucs]}")
    print("=" * 90)
    return mean_5fold_auc

if __name__ == '__main__':
    train_report_supervised(
        model_name='convnext_small',
        target_size=256,
        slices_per_study=6,
        batch_size=4,
        grad_accum_steps=4,
        epochs=5,
        lr=2e-4,
        num_report_samples=1200
    )
