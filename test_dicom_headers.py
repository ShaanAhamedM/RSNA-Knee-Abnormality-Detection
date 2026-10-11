import io
import time
import zipfile
import pydicom
import numpy as np
import pandas as pd

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    all_names = [n for n in z.namelist() if n.endswith('.dcm')]
    print(f"Total DICOM files in zip: {len(all_names)} (Found in {time.time()-t0:.2f}s)")
    
    # Inspect first 5 files
    for n in all_names[:5]:
        b = z.read(n)
        dcm = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True)
        iop = getattr(dcm, 'ImageOrientationPatient', None)
        plane = "Unknown"
        if iop is not None and len(iop) == 6:
            r = np.array(iop[:3])
            c = np.array(iop[3:])
            normal = np.cross(r, c)
            max_axis = np.argmax(np.abs(normal))
            plane = ['Sagittal', 'Coronal', 'Axial'][max_axis]
            
        print(f"File: {n}")
        print(f"  Plane: {plane} | Orientation: {iop}")
        print(f"  SeriesDesc: {getattr(dcm, 'SeriesDescription', 'N/A')}")
        print(f"  Age: {getattr(dcm, 'PatientAge', 'N/A')} | Sex: {getattr(dcm, 'PatientSex', 'N/A')}")
        print(f"  Field: {getattr(dcm, 'MagneticFieldStrength', 'N/A')} | Manufacturer: {getattr(dcm, 'Manufacturer', 'N/A')}")
        print(f"  TR: {getattr(dcm, 'RepetitionTime', 'N/A')} | TE: {getattr(dcm, 'EchoTime', 'N/A')}")
        print(f"  PixelSpacing: {getattr(dcm, 'PixelSpacing', 'N/A')} | Rows/Cols: {getattr(dcm, 'Rows', 'N/A')}x{getattr(dcm, 'Columns', 'N/A')}")
