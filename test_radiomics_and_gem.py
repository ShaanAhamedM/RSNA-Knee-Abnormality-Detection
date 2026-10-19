import io
import time
import json
import zipfile
import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.preprocessing import process_mri_slice

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

def extract_radiomics_from_slice(img: np.ndarray) -> dict:
    # img is (H, W) in [0, 1]
    p50, p75, p90, p95, p99 = np.percentile(img, [50, 75, 90, 95, 99])
    bright_fluid_ratio = float(np.mean(img > 0.85))
    
    # Left vs Right asymmetry (Coronal medial vs lateral)
    mid_w = img.shape[1] // 2
    left_half = img[:, :mid_w]
    right_half = img[:, mid_w:]
    lr_mean_diff = float(np.mean(left_half) - np.mean(right_half))
    lr_p95_diff = float(np.percentile(left_half, 95) - np.percentile(right_half, 95))
    
    # Gradient energy (Sobel/diff sharpness)
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

print("Testing GeM + Radiomics extraction on 10 studies...")
with open('study_plane_catalog.json') as f:
    cat = json.load(f)

sample_uids = list(cat.keys())[:10]
zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for s_idx, uid in enumerate(sample_uids):
        entry = cat[uid]
        study_rad = {}
        for pl in ['Sagittal', 'Coronal', 'Axial']:
            slices = entry.get(pl, [])
            if slices:
                raw = z.read(slices[len(slices)//2])
                dcm = pydicom.dcmread(io.BytesIO(raw))
                p = process_mri_slice(dcm.pixel_array, target_size=256, use_clahe=False, crop_roi=False)
                rad = extract_radiomics_from_slice(p)
                for k, v in rad.items():
                    study_rad[f"{pl}_{k}"] = v
            else:
                for k in ['p50', 'p75', 'p90', 'p95', 'p99', 'bright_fluid_ratio', 'lr_mean_diff', 'lr_p95_diff', 'grad_energy']:
                    study_rad[f"{pl}_{k}"] = 0.0
                    
        print(f"Study {s_idx} ({uid[:18]}...): Sag fluid={study_rad['Sagittal_bright_fluid_ratio']:.3f}, Cor asymmetry={study_rad['Coronal_lr_mean_diff']:.3f}, Ax p95={study_rad['Axial_p95']:.3f}")

el = time.time() - t0
print(f"10 studies processed in {el:.2f}s ({el/10:.3f}s per study)")
