"""
Step 2 & 3: High-Speed DICOM Header & Plane Extraction Engine
Extracts:
1. Exact anatomical planes (Sagittal, Coronal, Axial) from ImageOrientationPatient normal vector.
2. Central slice paths for each plane (middle 35%-65% depth).
3. Rich study metadata (Sex, Field strength, Manufacturer, Series counts, Slice counts, TR/TE).
Saves:
- study_plane_catalog.json (Mapping each study to central slices for Sagittal, Coronal, Axial)
- dicom_tabular_features.csv (Tabular feature matrix for GBDT modeling)
"""

import io
import time
import json
import zipfile
import pydicom
import numpy as np
import pandas as pd
from collections import defaultdict

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
print("=" * 80)
print("EXTRACTING DICOM METADATA & PLANE CATALOG ACROSS ALL 4,410 STUDIES")
print("=" * 80)

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    all_files = [f for f in z.namelist() if f.endswith('.dcm')]
    print(f"Total DICOMs in zip: {len(all_files)} (Found in {time.time()-t0:.2f}s)")
    
    study_series_map = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        parts = f.split('/')
        if len(parts) >= 4:
            study_id = parts[1]
            series_id = parts[2]
            study_series_map[study_id][series_id].append(f)
            
    print(f"Total unique studies: {len(study_series_map)}")
    
    catalog = {}
    tabular_rows = []
    
    t_start = time.time()
    for s_idx, (study_id, series_dict) in enumerate(study_series_map.items()):
        study_entry = {
            'Sagittal': [],
            'Coronal': [],
            'Axial': [],
            'n_series': len(series_dict),
            'total_slices': sum(len(flist) for flist in series_dict.values())
        }
        
        sex_val = -1 # 0: F, 1: M, -1: unknown
        field_val = 1.5
        mfr_name = 'Other'
        tr_list = []
        te_list = []
        spacing_list = []
        
        plane_slice_map = {'Sagittal': [], 'Coronal': [], 'Axial': []}
        
        for series_id, files in series_dict.items():
            # sort files naturally
            sorted_files = sorted(files)
            mid_idx = len(sorted_files) // 2
            mid_file = sorted_files[mid_idx]
            
            try:
                raw_bytes = z.read(mid_file)
                dcm = pydicom.dcmread(io.BytesIO(raw_bytes), stop_before_pixels=True)
                iop = getattr(dcm, 'ImageOrientationPatient', None)
                plane = 'Unknown'
                if iop is not None and len(iop) == 6:
                    r = np.array(iop[:3])
                    c = np.array(iop[3:])
                    n = np.cross(r, c)
                    max_axis = np.argmax(np.abs(n))
                    plane = ['Sagittal', 'Coronal', 'Axial'][max_axis]
                
                if plane in plane_slice_map:
                    # select central 4 slices from this series
                    n_fl = len(sorted_files)
                    start_k = int(n_fl * 0.30)
                    end_k = max(start_k + 1, int(n_fl * 0.70))
                    step = max(1, (end_k - start_k) // 4)
                    selected_slices = sorted_files[start_k:end_k:step][:4]
                    plane_slice_map[plane].extend(selected_slices)
                    
                # metadata
                sex_str = str(getattr(dcm, 'PatientSex', '')).upper()
                if sex_str == 'M': sex_val = 1
                elif sex_str == 'F': sex_val = 0
                
                fld = getattr(dcm, 'MagneticFieldStrength', None)
                if fld is not None:
                    try: field_val = float(fld)
                    except: pass
                    
                mfr = str(getattr(dcm, 'Manufacturer', ''))
                if 'SIEMENS' in mfr.upper(): mfr_name = 'Siemens'
                elif 'GE' in mfr.upper() or 'GENERAL' in mfr.upper(): mfr_name = 'GE'
                elif 'PHILIPS' in mfr.upper(): mfr_name = 'Philips'
                elif 'TOSHIBA' in mfr.upper() or 'CANON' in mfr.upper(): mfr_name = 'Toshiba'
                
                tr = getattr(dcm, 'RepetitionTime', None)
                if tr is not None:
                    try: tr_list.append(float(tr))
                    except: pass
                    
                te = getattr(dcm, 'EchoTime', None)
                if te is not None:
                    try: te_list.append(float(te))
                    except: pass
                    
                spc = getattr(dcm, 'PixelSpacing', None)
                if spc is not None and len(spc) > 0:
                    try: spacing_list.append(float(spc[0]))
                    except: pass
            except Exception:
                pass
                
        # Curate top slices per plane
        for pl in ['Sagittal', 'Coronal', 'Axial']:
            slices = plane_slice_map[pl]
            if len(slices) > 6:
                # subsample evenly to 6
                step = len(slices) // 6
                study_entry[pl] = slices[::step][:6]
            else:
                study_entry[pl] = slices
                
        catalog[study_id] = study_entry
        
        tabular_rows.append({
            'StudyInstanceUID': study_id,
            'n_series': len(series_dict),
            'n_total_slices': sum(len(flist) for flist in series_dict.values()),
            'n_sagittal_slices': len(plane_slice_map['Sagittal']),
            'n_coronal_slices': len(plane_slice_map['Coronal']),
            'n_axial_slices': len(plane_slice_map['Axial']),
            'patient_sex': sex_val,
            'field_strength': field_val,
            'manufacturer': mfr_name,
            'mean_tr': np.mean(tr_list) if tr_list else 2500.0,
            'mean_te': np.mean(te_list) if te_list else 35.0,
            'pixel_spacing': np.mean(spacing_list) if spacing_list else 0.4
        })
        
        if (s_idx + 1) % 500 == 0 or (s_idx + 1) == len(study_series_map):
            elapsed = time.time() - t_start
            rate = (s_idx + 1) / elapsed
            print(f"Processed {s_idx + 1:4d} / {len(study_series_map)} studies ({rate:.1f} studies/sec, {elapsed:.1f}s elapsed)")

# Save plane catalog
out_catalog = r"D:\MLprojs\KaggleCompetition\RSNA\study_plane_catalog.json"
with open(out_catalog, 'w') as f:
    json.dump(catalog, f)
print(f"\n[SAVED] Plane catalog with {len(catalog)} studies saved to {out_catalog}!")

# Save tabular features
out_tab = r"D:\MLprojs\KaggleCompetition\RSNA\dicom_tabular_features.csv"
tab_df = pd.DataFrame(tabular_rows)
tab_df.to_csv(out_tab, index=False)
print(f"[SAVED] DICOM tabular features for {len(tab_df)} studies saved to {out_tab}!")
print(f"Sample features:\n{tab_df.head(3)}")
