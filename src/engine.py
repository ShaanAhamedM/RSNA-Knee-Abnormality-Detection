"""
Robust Deep Learning Training & Evaluation Engine
Features:
- Automatic Mixed Precision (torch.amp.autocast)
- Dynamic Gradient Accumulation
- Exponential Moving Average (EMA)
- Auto-OOM Recovery Wrapper
- Strict Disk Hygiene (single best checkpoint per fold)
- Exact Macro-ROC AUC Validation
"""

import os
import gc
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from tqdm import tqdm

from src.metrics import calculate_macro_roc_auc

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].copy_(self.decay * self.shadow[k] + (1.0 - self.decay) * v)

    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        model.load_state_dict(self.backup)

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 4,
    ema: EMA = None
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    n_batches = len(dataloader)
    for step, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['labels'].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)
            loss = loss / grad_accum_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if ema is not None:
                ema.update(model)
                
        total_loss += loss.item() * grad_accum_steps
        
    return total_loss / max(1, n_batches)

def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ema: EMA = None
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns (val_loss, macro_auc, y_true_all, y_pred_all)
    """
    if ema is not None:
        ema.apply_shadow(model)
        
    model.eval()
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['labels'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)
                probs = torch.sigmoid(logits)
                
            total_loss += loss.item()
            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(probs.cpu().numpy())
            
    if ema is not None:
        ema.restore(model)
        
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    val_loss = total_loss / max(1, len(dataloader))
    macro_auc, per_class = calculate_macro_roc_auc(y_true, y_pred)
    
    return val_loss, macro_auc, y_true, y_pred

def train_fold_with_auto_oom(
    model_name: str,
    fold: int,
    train_df,
    val_df,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Auto-OOM Recovery Wrapper: If CUDA OOM occurs, catches it, halves batch size,
    doubles gradient accumulation steps, and restarts fold automatically.
    """
    from src.dataset import get_dataloader
    from src.models import RSNAKneeModel
    from src.losses import Poly1Loss, AsymmetricFocalLoss, HomoscedasticTaskUncertaintyLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 4)
    grad_accum_steps = config.get('grad_accum_steps', 4)
    target_size = config.get('target_size', 256)
    slices_per_study = config.get('slices_per_study', 6)
    in_chans = config.get('in_chans', 1)
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 3e-4)
    loss_type = config.get('loss_type', 'poly')
    checkpoint_dir = config.get('checkpoint_dir', r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints")
    oof_dir = config.get('oof_dir', r"D:\MLprojs\KaggleCompetition\RSNA\oof")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_fold{fold}_best.pt")
    oof_pred_path = os.path.join(oof_dir, f"{model_name}_fold{fold}_oof.npy")
    oof_true_path = os.path.join(oof_dir, f"true_fold{fold}.npy")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n--- Starting {model_name} | Fold {fold} (Attempt {attempt+1}) ---")
            print(f"Batch Size: {batch_size} | Accumulation: {grad_accum_steps} | Res: {target_size}x{target_size} | Slices: {slices_per_study}")
            
            # DataLoaders
            train_loader = get_dataloader(
                train_df, target_size=target_size, slices_per_study=slices_per_study,
                in_chans=in_chans, batch_size=batch_size, shuffle=True, is_train=True
            )
            val_loader = get_dataloader(
                val_df, target_size=target_size, slices_per_study=slices_per_study,
                in_chans=in_chans, batch_size=batch_size, shuffle=False, is_train=False
            )
            
            # Model
            model = RSNAKneeModel(
                backbone_name=model_name,
                num_classes=12,
                in_chans=in_chans,
                pretrained=True,
                use_mil=True,
                grad_checkpointing=True
            ).to(device)
            
            # Loss formulation
            if loss_type == 'poly':
                criterion = Poly1Loss(epsilon=config.get('poly_eps', 1.0))
            elif loss_type == 'asl':
                criterion = AsymmetricFocalLoss(gamma_neg=config.get('gamma_neg', 3.0), gamma_pos=config.get('gamma_pos', 1.0))
            elif loss_type == 'uncertainty':
                criterion = HomoscedasticTaskUncertaintyLoss(num_tasks=12).to(device)
            else:
                criterion = nn.BCEWithLogitsLoss()
                
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
            ema = EMA(model, decay=0.99)
            
            best_auc = 0.0
            best_oof_preds = None
            best_y_true = None
            
            for epoch in range(1, epochs + 1):
                t_start = time.time()
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler, device,
                    grad_accum_steps=grad_accum_steps, ema=ema
                )
                val_loss, val_auc, y_true, y_pred = evaluate(model, val_loader, criterion, device, ema=ema)
                scheduler.step()
                
                t_epoch = time.time() - t_start
                print(f"Epoch {epoch:02d}/{epochs:02d} [{t_epoch:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-AUC: {val_auc:.4f}", flush=True)
                
                if val_auc > best_auc or epoch == 1:
                    best_auc = val_auc
                    best_oof_preds = y_pred
                    best_y_true = y_true
                    # Strict disk hygiene: overwrite single best checkpoint in float16
                    torch.save(model.state_dict(), best_checkpoint_path)
                    np.save(oof_pred_path, best_oof_preds)
                    np.save(oof_true_path, best_y_true)
                    print(f"  -> Checkpoint updated! New Best Macro-AUC: {best_auc:.4f}", flush=True)
                    
            return {
                'model_name': model_name,
                'fold': fold,
                'best_auc': best_auc,
                'checkpoint_path': best_checkpoint_path,
                'oof_pred_path': oof_pred_path,
                'oof_true_path': oof_true_path
            }
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                print(f"\n[AUTO-OOM DETECTED] GPU Out of Memory: {e}")
                print("Clearing VRAM cache, halving batch size, and auto-restarting fold...")
                gc.collect()
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                grad_accum_steps = grad_accum_steps * 2
                time.sleep(2)
            else:
                raise e
                
    raise RuntimeError(f"Training failed after {max_retries} attempts due to unrecoverable errors.")
