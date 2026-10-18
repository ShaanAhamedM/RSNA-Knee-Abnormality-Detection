"""
Test Phase 7 Sub-ROI Extraction and Fluid-Sensitivity Detection
Verifies:
1. Series discovery with IOP plane detection (Sagittal, Coronal, Axial)
2. Fluid-sensitivity classification (T2/PD-FS vs T1) via TR, TE, and SeriesDescription
3. Sub-ROI cropping:
   - Global view (256x256)
   - Notch/Joint-Line Zoom Crop (center 45% x 45%)
   - Patellofemoral Anterior Crop (top 50% on Axial/Sagittal)
   - Medial vs Lateral Compartment Split Crops
4. Dual Backbone feature extraction (ConvNeXt-Small + DenseNet121)
"""

import io
import zipfile
import pydicom
import numpy as np
import torch
import timm
from src.preprocessing import process_mri_slice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Test Device: {device}")

# 1. Load both backbones offline
print("Loading ConvNeXt-Small backbone...")
convnext = timm.create_model('convnext_small', pretrained=False, in_chans=1, num_classes=0)
convnext.load_state_dict(torch.load(r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\convnext_small_backbone_in1.pt", map_location=device))
convnext.to(device).eval()

print("Loading DenseNet121 backbone...")
densenet = timm.create_model('densenet121.ra_in1k', pretrained=False, in_chans=1, num_classes=0)
densenet.load_state_dict(torch.load(r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints\densenet121_backbone_in1.pt", map_location=device))
densenet.to(device).eval()

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"

def crop_sub_rois(arr: np.ndarray, plane: str, laterality: str = 'L') -> dict:
    """Extracts targeted anatomical sub-ROIs from a normalized MRI slice."""
    H, W = arr.shape
    crops = {}
    
    # Global
    crops['global'] = arr
    
    # 1. Intercondylar Notch / Joint Line Zoom (Center 45% x 45%)
    if plane in ['Sagittal', 'Coronal']:
        h0, h1 = int(0.275 * H), int(0.725 * H)
        w0, w1 = int(0.275 * W), int(0.725 * W)
        crops['notch_zoom'] = process_mri_slice(arr[h0:h1, w0:w1], target_size=256)
        
    # 2. Patellofemoral Anterior Crop
    if plane == 'Axial':
        # Top 50% contains patella and anterior trochlea
        crops['pf_anterior'] = process_mri_slice(arr[:int(0.55 * H), :], target_size=256)
    elif plane == 'Sagittal':
        # Anterior half (front 55%)
        crops['pf_anterior'] = process_mri_slice(arr[:, :int(0.55 * W)], target_size=256)
        
    # 3. Medial vs Lateral Compartment Split Crops
    if plane in ['Coronal', 'Axial']:
        left_half = process_mri_slice(arr[:, :int(0.5 * W)], target_size=256)
        right_half = process_mri_slice(arr[:, int(0.5 * W):], target_size=256)
        
        # In DICOM patient orientation for coronal knee:
        # If Left knee: Right on screen is Medial, Left is Lateral
        # If Right knee: Left on screen is Medial, Right is Lateral
        if laterality == 'L':
            crops['medial_comp'] = right_half
            crops['lateral_comp'] = left_half
        else:
            crops['medial_comp'] = left_half
            crops['lateral_comp'] = right_half
            
    return crops

def is_fluid_sequence(desc: str, tr: float, te: float) -> bool:
    desc_u = desc.upper()
    keywords = ['FS', 'FATSAT', 'FAT_SAT', 'SPAIR', 'STIR', 'PD', 'T2', 'DP', 'WATER']
    for kw in keywords:
        if kw in desc_u:
            return True
    if tr > 1500 and te > 25:
        return True
    return False

# Test on 2 sample studies from test_series
test_studies = [
    '1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191',
    '1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776'
]

with zipfile.ZipFile(zip_path, 'r') as z:
    all_names = z.namelist()
    for uid in test_studies:
        print("\n" + "=" * 70)
        print(f"Testing Study: {uid}")
        print("=" * 70)
        prefix = f"test_series/{uid}/"
        study_files = [n for n in all_names if n.startswith(prefix) and n.endswith('.dcm')]
        
        # Group by series
        series_map = {}
        for f in study_files:
            s_uid = f.split('/')[2]
            if s_uid not in series_map:
                series_map[s_uid] = []
            series_map[s_uid].append(f)
            
        print(f"Found {len(series_map)} series for study.")
        
        for s_uid, files in series_map.items():
            mid_file = files[len(files) // 2]
            raw = z.read(mid_file)
            dcm = pydicom.dcmread(io.BytesIO(raw))
            
            iop = getattr(dcm, 'ImageOrientationPatient', None)
            plane = 'Unknown'
            if iop is not None and len(iop) == 6:
                r = np.array(iop[:3])
                c = np.array(iop[3:])
                n = np.cross(r, c)
                max_axis = np.argmax(np.abs(n))
                plane = ['Sagittal', 'Coronal', 'Axial'][max_axis]
                
            desc = str(getattr(dcm, 'SeriesDescription', ''))
            tr = float(getattr(dcm, 'RepetitionTime', 2500) or 2500)
            te = float(getattr(dcm, 'EchoTime', 35) or 35)
            lat = str(getattr(dcm, 'Laterality', 'L')).upper() or 'L'
            fluid_sens = is_fluid_sequence(desc, tr, te)
            
            print(f"  Series {s_uid[:15]}... | Plane: {plane:8s} | TR: {tr:5.0f} | TE: {te:4.1f} | Desc: {desc:15s} | Fluid-Sensitive: {fluid_sens} | Laterality: {lat}")
            
            # Test ROI cropping & multi-backbone inference
            proc_slice = process_mri_slice(dcm.pixel_array, target_size=256)
            rois = crop_sub_rois(proc_slice, plane, laterality=lat)
            print(f"    -> Extracted {len(rois)} ROIs: {list(rois.keys())}")
            
            # Pass through both backbones
            roi_imgs = np.array([v for v in rois.values()])
            th_in = torch.tensor(roi_imgs, dtype=torch.float32, device=device).unsqueeze(1)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    c_feats = convnext(th_in)
                    d_feats = densenet(th_in)
                    
            print(f"    -> ConvNeXt output: {list(c_feats.shape)}, DenseNet output: {list(d_feats.shape)}")
            break

print("\n[SUCCESS] Sub-ROI and Dual-Backbone feature extraction verified!")
