"""
Phase 7 Ultra-Fast Production Extractor:
1. Multi-Backbone: ConvNeXt-Small (768) + DenseNet121 (1024) -> 1792-dim slice tokens.
2. Targeted Anatomical Sub-ROIs (computed in 0.03ms via cv2.resize on normalized slice):
   - Global slice (256x256)
   - Intercondylar Notch / Joint-Line Zoom (center 45% x 45% for ACL & Meniscus)
   - Patellofemoral Anterior (anterior 50% for PF OA & Effusion)
   - Medial vs Lateral Compartment Split (for localized OA and Meniscus tears)
3. Fluid-Sensitive Sequence Indicators:
   - is_fluid_sensitive (T2/PD-FS vs T1)
   - normalized TR and TE
4. Enhanced Radiomics with Compartment & Fluid features.
5. Incremental checkpointing every 500 studies + flush=True logging.
"""

import io
import time
import json
import zipfile
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
import timm
from src.preprocessing import process_mri_slice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Phase 7 Extraction Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# Load both champion backbones offline
print("Loading ConvNeXt-Small and DenseNet121 backbones...", flush=True)
bb_c = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\convnext_small_backbone_in1.pt"
convnext = timm.create_model('convnext_small', pretrained=False, in_chans=1, num_classes=0)
convnext.load_state_dict(torch.load(bb_c, map_location=device))
convnext.to(device).eval()

bb_d = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\densenet121_backbone_in1.pt"
densenet = timm.create_model('densenet121.ra_in1k', pretrained=False, in_chans=1, num_classes=0)
densenet.load_state_dict(torch.load(bb_d, map_location=device))
densenet.to(device).eval()

with open('study_plane_catalog.json') as f:
    cat = json.load(f)

dicom_base_df = pd.read_csv('dicom_tabular_features.csv')
dicom_dict = dicom_base_df.set_index('StudyInstanceUID').to_dict('index')

study_uids = list(cat.keys())
print(f"Total studies to process: {len(study_uids)}", flush=True)
zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

def crop_sub_rois_fast(arr: np.ndarray, plane: str, laterality: str = 'L') -> list:
    """Ultra-fast sub-ROI cropping using pre-normalized 256x256 array (<0.1ms)."""
    H, W = arr.shape
    crops = [(arr, 0)] # (img, roi_type: 0=global)
    if plane in ['Sagittal', 'Coronal']:
        h0, h1 = int(0.275 * H), int(0.725 * H)
        w0, w1 = int(0.275 * W), int(0.725 * W)
        zoom = cv2.resize(arr[h0:h1, w0:w1], (256, 256), interpolation=cv2.INTER_LINEAR)
        crops.append((zoom, 1)) # 1=notch_zoom
    if plane in ['Axial', 'Sagittal']:
        pf = cv2.resize(arr[:int(0.55 * H), :], (256, 256), interpolation=cv2.INTER_LINEAR)
        crops.append((pf, 2)) # 2=pf_anterior
    if plane in ['Coronal', 'Axial']:
        left = cv2.resize(arr[:, :int(0.5 * W)], (256, 256), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(arr[:, int(0.5 * W):], (256, 256), interpolation=cv2.INTER_LINEAR)
        if laterality == 'L':
            crops.append((right, 3)) # 3=medial
            crops.append((left, 4))  # 4=lateral
        else:
            crops.append((left, 3))
            crops.append((right, 4))
    return crops

def extract_radiomics(img: np.ndarray) -> dict:
    p50, p75, p90, p95, p99 = np.percentile(img, [50, 75, 90, 95, 99])
    bright_fluid_ratio = float(np.mean(img > 0.85))
    mid_w = img.shape[1] // 2
    left_half = img[:, :mid_w]
    right_half = img[:, mid_w:]
    lr_mean_diff = float(np.mean(left_half) - np.mean(right_half))
    lr_p95_diff = float(np.percentile(left_half, 95) - np.percentile(right_half, 95))
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    grad_energy = float(np.mean(dx**2) + np.mean(dy**2))
    return {
        'p50': float(p50), 'p75': float(p75), 'p90': float(p90), 'p95': float(p95), 'p99': float(p99),
        'bright_fluid_ratio': bright_fluid_ratio,
        'lr_mean_diff': lr_mean_diff,
        'lr_p95_diff': lr_p95_diff,
        'grad_energy': grad_energy
    }

out_tokens_path = r"D:\MLprojs\KaggleCompetition\RSNA\phase7_tokens_dual_backbone.pt"
out_tab_path = r"D:\MLprojs\KaggleCompetition\RSNA\dicom_radiomics_phase7_features.csv"

token_db = {}
radiomics_rows = []
t_start = time.time()

with zipfile.ZipFile(zip_path, 'r') as z:
    for idx, uid in enumerate(study_uids):
        entry = cat[uid]
        d_meta = dicom_dict.get(uid, {})
        tr_val = float(d_meta.get('mean_tr', 2500.0) or 2500.0)
        te_val = float(d_meta.get('mean_te', 35.0) or 35.0)
        is_fluid = 1.0 if (tr_val > 1500 and te_val > 25) else 0.0
        norm_tr = tr_val / 3000.0
        norm_te = te_val / 50.0
        
        study_crops = []
        study_planes = []
        study_rois = []
        study_depths = []
        study_fluids = []
        study_trs = []
        study_tes = []
        
        rad_entry = {
            'StudyInstanceUID': uid,
            'is_fluid_sensitive': is_fluid,
            'tr_te_ratio': tr_val / (te_val + 1e-5)
        }
        
        for pl_idx, pl in enumerate(['Sagittal', 'Coronal', 'Axial']):
            slices = entry.get(pl, [])
            if not slices:
                for k in ['p50', 'p75', 'p90', 'p95', 'p99', 'bright_fluid_ratio', 'lr_mean_diff', 'lr_p95_diff', 'grad_energy']:
                    rad_entry[f"{pl}_{k}"] = 0.0
                continue
                
            n_sl = len(slices)
            indices = np.linspace(0, n_sl - 1, min(3, n_sl), dtype=int)
            
            # Radiomics on middle slice
            mid_sp = slices[n_sl // 2]
            try:
                raw_mid = z.read(mid_sp)
                dcm_mid = pydicom.dcmread(io.BytesIO(raw_mid))
                p_mid = process_mri_slice(dcm_mid.pixel_array, target_size=256)
                rad_vals = extract_radiomics(p_mid)
                for k, v in rad_vals.items():
                    rad_entry[f"{pl}_{k}"] = v
            except Exception:
                for k in ['p50', 'p75', 'p90', 'p95', 'p99', 'bright_fluid_ratio', 'lr_mean_diff', 'lr_p95_diff', 'grad_energy']:
                    rad_entry[f"{pl}_{k}"] = 0.0
                    
            for s_k in indices:
                sp = slices[s_k]
                try:
                    raw = z.read(sp)
                    dcm = pydicom.dcmread(io.BytesIO(raw))
                    p = process_mri_slice(dcm.pixel_array, target_size=256)
                    rois = crop_sub_rois_fast(p, pl, laterality='L')
                    rel_depth = s_k / max(1, n_sl - 1)
                    
                    for r_img, r_type in rois:
                        study_crops.append(r_img)
                        study_planes.append(pl_idx)
                        study_rois.append(r_type)
                        study_depths.append(rel_depth)
                        study_fluids.append(is_fluid)
                        study_trs.append(norm_tr)
                        study_tes.append(norm_te)
                except Exception:
                    pass
                    
        # Batch forward pass through both backbones
        if study_crops:
            if len(study_crops) > 20:
                study_crops = study_crops[:20]
                study_planes = study_planes[:20]
                study_rois = study_rois[:20]
                study_depths = study_depths[:20]
                study_fluids = study_fluids[:20]
                study_trs = study_trs[:20]
                study_tes = study_tes[:20]
                
            th_in = torch.tensor(np.array(study_crops), dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    c_feats = convnext(th_in) # (K, 768)
                    d_feats = densenet(th_in) # (K, 1024)
                    
            dual_tokens = torch.cat([c_feats, d_feats], dim=-1).cpu().to(torch.float16)
            planes_t = torch.tensor(study_planes, dtype=torch.long)
            rois_t = torch.tensor(study_rois, dtype=torch.long)
            depths_t = torch.tensor(study_depths, dtype=torch.float32)
            contrast_t = torch.tensor(list(zip(study_fluids, study_trs, study_tes)), dtype=torch.float32)
        else:
            dual_tokens = torch.zeros((1, 1792), dtype=torch.float16)
            planes_t = torch.zeros(1, dtype=torch.long)
            rois_t = torch.zeros(1, dtype=torch.long)
            depths_t = torch.zeros(1, dtype=torch.float32)
            contrast_t = torch.zeros((1, 3), dtype=torch.float32)
            
        token_db[uid] = {
            'tokens': dual_tokens,
            'planes': planes_t,
            'rois': rois_t,
            'depths': depths_t,
            'contrast': contrast_t
        }
        radiomics_rows.append(rad_entry)
        
        # Checkpoint every 500 studies
        if (idx + 1) % 500 == 0:
            torch.save(token_db, out_tokens_path + '.tmp')
            
        if (idx + 1) % 200 == 0 or (idx + 1) == len(study_uids):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(study_uids) - (idx + 1)) / max(rate, 1e-5)
            print(f"Extraction Progress: {idx + 1:4d} / {len(study_uids)} ({rate:.1f} studies/sec | Elapsed: {elapsed/60:.1f}m | Remaining: {rem/60:.1f}m)", flush=True)

# Save final token database
torch.save(token_db, out_tokens_path)
print(f"\n[SUCCESS] Saved Phase 7 Dual-Backbone Multi-ROI tokens for {len(token_db)} studies to {out_tokens_path}!", flush=True)

# Merge with DICOM metadata and save
rad_df = pd.DataFrame(radiomics_rows)
full_tab_df = pd.merge(dicom_base_df, rad_df, on='StudyInstanceUID', how='left')
full_tab_df.to_csv(out_tab_path, index=False)
print(f"[SUCCESS] Saved combined DICOM + Radiomics table ({full_tab_df.shape[1]} columns) to {out_tab_path}!", flush=True)
