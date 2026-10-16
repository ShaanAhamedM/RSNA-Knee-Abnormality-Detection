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

model = timm.create_model('convnext_small', pretrained=True, in_chans=1, num_classes=0).to(device)
model.eval()

with open('study_plane_catalog.json') as f:
    cat = json.load(f)

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
sample_uids = list(cat.keys())[:20]

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for s_idx, uid in enumerate(sample_uids):
        entry = cat[uid]
        plane_feats = []
        for pl in ['Sagittal', 'Coronal', 'Axial']:
            slices = entry.get(pl, [])
            if not slices:
                plane_feats.append(torch.zeros(768, device=device))
                continue
            # pick middle 2 slices
            sel = [slices[len(slices)//3], slices[2*len(slices)//3]] if len(slices) >= 2 else [slices[0]]
            imgs = []
            for sp in sel:
                raw_bytes = z.read(sp)
                dcm = pydicom.dcmread(io.BytesIO(raw_bytes))
                p = process_mri_slice(dcm.pixel_array, target_size=256, use_clahe=False, crop_roi=False)
                imgs.append(p)
            tensor = torch.tensor(np.array(imgs), dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feats = model(tensor) # (K, 768)
                    plane_feats.append(feats.mean(dim=0))
                    
        study_embed = torch.stack(plane_feats, dim=0) # (3, 768)

el = time.time() - t0
print(f"Extracted embeddings for 20 studies in {el:.2f}s ({el/20:.3f}s per study, {20/el:.1f} studies/sec)")
print(f"Embedding shape: {study_embed.shape}")
