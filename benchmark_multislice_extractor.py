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
print(f"Device: {device}")

backbone = timm.create_model('convnext_small', pretrained=True, in_chans=1, num_classes=0).to(device)
backbone.eval()

with open('study_plane_catalog.json') as f:
    cat = json.load(f)

sample_uids = list(cat.keys())[:20]
zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for uid in sample_uids:
        entry = cat[uid]
        study_slice_imgs = []
        study_planes = []
        study_depths = []
        
        for pl_idx, pl in enumerate(['Sagittal', 'Coronal', 'Axial']):
            slices = entry.get(pl, [])
            if not slices:
                continue
            n_sl = len(slices)
            # Pick 4 evenly spaced slices per plane
            indices = np.linspace(0, n_sl - 1, min(4, n_sl), dtype=int)
            for idx in indices:
                sp = slices[idx]
                try:
                    raw = z.read(sp)
                    dcm = pydicom.dcmread(io.BytesIO(raw))
                    p = process_mri_slice(dcm.pixel_array, target_size=256, use_clahe=False, crop_roi=False)
                    study_slice_imgs.append(p)
                    study_planes.append(pl_idx)
                    study_depths.append(idx / max(1, n_sl - 1))
                except Exception:
                    pass
                    
        if study_slice_imgs:
            tensor_batch = torch.tensor(np.array(study_slice_imgs), dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feats = backbone(tensor_batch) # (K, 768)

el = time.time() - t0
print(f"Processed 20 studies (12 slices per study) in {el:.2f}s ({el/20:.3f}s per study, {20/el:.1f} studies/sec)")
