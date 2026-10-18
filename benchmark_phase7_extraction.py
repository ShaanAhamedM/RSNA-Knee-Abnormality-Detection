"""
Benchmark Phase 7 Multi-ROI + Dual-Backbone Extraction Speed
Tests 30 full studies from zip archive to verify throughput (studies/sec) and VRAM usage.
"""

import io
import time
import json
import zipfile
import pydicom
import numpy as np
import torch
import timm
from src.preprocessing import process_mri_slice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Benchmark Device: {device}")

# Load backbones
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

study_uids = list(cat.keys())[:30]
zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

def crop_sub_rois(arr: np.ndarray, plane: str, laterality: str = 'L') -> list:
    H, W = arr.shape
    crops = [(arr, 0)] # (img, roi_type: 0=global)
    if plane in ['Sagittal', 'Coronal']:
        h0, h1 = int(0.275 * H), int(0.725 * H)
        w0, w1 = int(0.275 * W), int(0.725 * W)
        crops.append((process_mri_slice(arr[h0:h1, w0:w1], target_size=256), 1)) # 1=notch_zoom
    if plane in ['Axial', 'Sagittal']:
        crops.append((process_mri_slice(arr[:int(0.55 * H), :], target_size=256), 2)) # 2=pf_anterior
    if plane in ['Coronal', 'Axial']:
        left = process_mri_slice(arr[:, :int(0.5 * W)], target_size=256)
        right = process_mri_slice(arr[:, int(0.5 * W):], target_size=256)
        if laterality == 'L':
            crops.append((right, 3)) # 3=medial
            crops.append((left, 4))  # 4=lateral
        else:
            crops.append((left, 3))
            crops.append((right, 4))
    return crops

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for idx, uid in enumerate(study_uids):
        entry = cat[uid]
        study_crops = []
        for pl_idx, pl in enumerate(['Sagittal', 'Coronal', 'Axial']):
            slices = entry.get(pl, [])
            if not slices: continue
            n_sl = len(slices)
            indices = np.linspace(0, n_sl - 1, min(3, n_sl), dtype=int)
            for s_k in indices:
                try:
                    raw = z.read(slices[s_k])
                    dcm = pydicom.dcmread(io.BytesIO(raw))
                    p = process_mri_slice(dcm.pixel_array, target_size=256)
                    rois = crop_sub_rois(p, pl, laterality='L')
                    for r_img, r_type in rois:
                        study_crops.append(r_img)
                except Exception:
                    pass
        if study_crops:
            th_in = torch.tensor(np.array(study_crops), dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    c_f = convnext(th_in)
                    d_f = densenet(th_in)

t1 = time.time()
dt = t1 - t0
rate = len(study_uids) / dt
est_total = (4410 / rate) / 60
print(f"Processed {len(study_uids)} studies in {dt:.2f}s ({rate:.2f} studies/sec)")
print(f"Estimated full dataset (4,410 studies): {est_total:.1f} minutes")
print(f"Max VRAM allocated: {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB")
