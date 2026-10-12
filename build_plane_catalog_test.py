import io
import time
import json
import zipfile
import pydicom
import numpy as np
import pandas as pd
from collections import defaultdict

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
print("Scanning zip namelist...")
t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    all_files = [f for f in z.namelist() if f.endswith('.dcm')]
    print(f"Total DICOMs: {len(all_files)} in {time.time()-t0:.2f}s")
    
    # Group by study and series
    # Format: folder/study_id/series_id/instance.dcm
    study_series_map = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        parts = f.split('/')
        if len(parts) >= 4:
            study_id = parts[1]
            series_id = parts[2]
            study_series_map[study_id][series_id].append(f)
            
    print(f"Total unique studies discovered: {len(study_series_map)}")
    
    # Test plane parsing on first 20 studies
    sample_studies = list(study_series_map.keys())[:20]
    plane_counts = {'Sagittal': 0, 'Coronal': 0, 'Axial': 0, 'Unknown': 0}
    
    t_parse = time.time()
    for s_idx, study_id in enumerate(sample_studies):
        series_dict = study_series_map[study_id]
        study_planes = {}
        for series_id, files in series_dict.items():
            # read first file header
            sample_file = files[len(files)//2] # middle file
            raw_bytes = z.read(sample_file)
            dcm = pydicom.dcmread(io.BytesIO(raw_bytes), stop_before_pixels=True)
            iop = getattr(dcm, 'ImageOrientationPatient', None)
            plane = 'Unknown'
            if iop is not None and len(iop) == 6:
                r = np.array(iop[:3])
                c = np.array(iop[3:])
                n = np.cross(r, c)
                max_axis = np.argmax(np.abs(n))
                plane = ['Sagittal', 'Coronal', 'Axial'][max_axis]
            plane_counts[plane] += 1
            study_planes[series_id] = plane
            
        print(f"Study {s_idx:2d} ({study_id[:20]}...): {len(series_dict)} series -> {list(study_planes.values())}")
        
    print(f"\nPlane counts over 20 studies: {plane_counts}")
    print(f"Header parsing speed: {(time.time()-t_parse)/len(sample_studies):.3f}s per study")
