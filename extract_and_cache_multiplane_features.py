"""
Multi-Plane Feature Pre-Extraction & Caching Engine
Extracts anatomical plane embeddings (Sagittal, Coronal, Axial) across all 4,410 studies
using pretrained ConvNeXt-Small (in_chans=1) with fp16 AMP.
Saves compact cached dictionary to D:\MLprojs\KaggleCompetition\RSNA\multiplane_embeddings_convnext.pt (~40 MB).
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
print(f"Extraction Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Pretrained ConvNeXt-Small backbone
print("Instantiating pretrained ConvNeXt-Small...")
backbone = timm.create_model('convnext_small', pretrained=True, in_chans=1, num_classes=0).to(device)
backbone.eval()

with open('study_plane_catalog.json') as f:
    cat = json.load(f)

print(f"Total studies to process: {len(cat)}")
zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

all_embeddings = {}
study_uids = list(cat.keys())

t_start = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for idx, uid in enumerate(study_uids):
        entry = cat[uid]
        plane_vectors = []
        
        for pl in ['Sagittal', 'Coronal', 'Axial']:
            slices = entry.get(pl, [])
            if not slices:
                plane_vectors.append(torch.zeros(768, dtype=torch.float16))
                continue
                
            # Sample central 2 slices
            if len(slices) >= 3:
                sel = [slices[len(slices)//3], slices[2*len(slices)//3]]
            else:
                sel = [slices[len(slices)//2]]
                
            slice_imgs = []
            for sp in sel:
                try:
                    raw = z.read(sp)
                    dcm = pydicom.dcmread(io.BytesIO(raw))
                    p = process_mri_slice(dcm.pixel_array, target_size=256, use_clahe=False, crop_roi=False)
                    slice_imgs.append(p)
                except Exception:
                    pass
                    
            if slice_imgs:
                t = torch.tensor(np.array(slice_imgs), dtype=torch.float32, device=device).unsqueeze(1) # (K, 1, 256, 256)
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        f = backbone(t) # (K, 768)
                        plane_vectors.append(f.mean(dim=0).cpu().to(torch.float16))
            else:
                plane_vectors.append(torch.zeros(768, dtype=torch.float16))
                
        # Stack 3 planes: shape (3, 768)
        all_embeddings[uid] = torch.stack(plane_vectors, dim=0)
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(study_uids):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            rem = (len(study_uids) - (idx + 1)) / max(rate, 1e-5)
            print(f"Progress: {idx + 1:4d} / {len(study_uids)} studies ({rate:.1f} studies/sec | Elapsed: {elapsed/60:.1f}m | Remaining: {rem/60:.1f}m)")

out_path = r"D:\MLprojs\KaggleCompetition\RSNA\multiplane_embeddings_convnext.pt"
torch.save(all_embeddings, out_path)
print(f"\n[SUCCESS] Saved multiplane embeddings for all {len(all_embeddings)} studies to {out_path}!")
print(f"Total size on disk: {len(all_embeddings) * 3 * 768 * 2 / (1024*1024):.1f} MB")
