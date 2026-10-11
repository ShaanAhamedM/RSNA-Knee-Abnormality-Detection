"""
Test Streaming MRI Triplet Dataset
Validates in-memory streaming of 2.5D 3-slice triplets [z-1, z, z+1] at 384x384 directly from the ZIP.
"""

import os
import sys
import io
import json
import time
import zipfile
import cv2
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

from src.preprocessing import crop_knee_roi, apply_percentile_windowing, apply_clahe

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class StreamingMRITripletDataset(Dataset):
    def __init__(self, df, study_catalog, zip_path, image_size=384, plane_choice='Sagittal', augment=False):
        self.df = df.reset_index(drop=True)
        self.catalog = study_catalog
        self.zip_path = zip_path
        self.image_size = image_size
        self.plane_choice = plane_choice
        self.augment = augment
        self.zip_file = None  # Initialized per worker
        
        self.uids = self.df['StudyInstanceUID'].values
        self.labels = self.df[TARGET_COLS].values.astype(np.float32)
        self.weights = self.df['sample_weight'].values.astype(np.float32)
        self.masks = self.df[[f"{c}_mask" for c in TARGET_COLS]].values.astype(np.float32)

    def _get_zip(self):
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self.zip_file

    def __len__(self):
        return len(self.df)

    def _read_slice(self, z, s_path):
        try:
            b = z.read(s_path)
            dcm = pydicom.dcmread(io.BytesIO(b))
            arr = dcm.pixel_array.astype(np.float32)
            arr = crop_knee_roi(arr)
            arr = apply_percentile_windowing(arr)
            arr = apply_clahe(arr)
            arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            return arr
        except Exception:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        z = self._get_zip()
        
        study_planes = self.catalog.get(uid, {})
        # Pick plane
        plane_key = self.plane_choice
        if plane_key == 'random':
            available = [p for p in ['Sagittal', 'Coronal', 'Axial'] if p in study_planes and len(study_planes[p]) > 0]
            plane_key = np.random.choice(available) if available else 'Sagittal'
            
        slices_list = study_planes.get(plane_key, [])
        N = len(slices_list)
        
        if N >= 3:
            mid = N // 2
            if self.augment and N > 5:
                mid = np.random.randint(2, N - 2)
            s_prev = slices_list[mid - 1]
            s_curr = slices_list[mid]
            s_next = slices_list[mid + 1]
        elif N == 2:
            s_prev = slices_list[0]
            s_curr = slices_list[0]
            s_next = slices_list[1]
        elif N == 1:
            s_prev = s_curr = s_next = slices_list[0]
        else:
            triplet = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            return (
                torch.tensor(triplet),
                torch.tensor(self.labels[idx]),
                torch.tensor(self.masks[idx]),
                torch.tensor(self.weights[idx]),
                uid
            )

        img0 = self._read_slice(z, s_prev)
        img1 = self._read_slice(z, s_curr)
        img2 = self._read_slice(z, s_next)
        
        triplet = np.stack([img0, img1, img2], axis=0) # (3, H, W)
        
        # Augmentations
        if self.augment:
            if np.random.rand() > 0.5:
                triplet = np.flip(triplet, axis=2).copy() # Horizontal flip
            if np.random.rand() > 0.5:
                # Add subtle Rician MRI noise
                noise = np.random.normal(0, 0.02, triplet.shape).astype(np.float32)
                triplet = np.clip(triplet + noise, 0.0, 1.0)
                
        return (
            torch.tensor(triplet, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
            uid
        )

if __name__ == '__main__':
    print("Testing StreamingMRITripletDataset on 5 samples...")
    import pandas as pd
    master_df = pd.read_csv('full_dataset_4407_folds.csv')
    rep_df = pd.read_csv('report_extracted_labels_4349_commercial.csv')
    cat = json.load(open('study_plane_catalog.json'))
    zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
    
    mask_cols = [f"{c}_mask" for c in TARGET_COLS]
    master_df = master_df.merge(rep_df[['StudyInstanceUID'] + mask_cols], on='StudyInstanceUID', how='left')
    master_df['sample_weight'] = np.where(master_df['is_gold'] == 1, 10.0, 1.0)
    for c in mask_cols:
        master_df[c] = master_df[c].fillna(1.0)
        
    ds = StreamingMRITripletDataset(master_df.head(10), cat, zip_path, image_size=384, plane_choice='Sagittal', augment=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    
    t0 = time.time()
    for b_idx, (imgs, lbls, msks, wts, uids) in enumerate(loader):
        print(f"Batch {b_idx}: imgs shape={imgs.shape}, lbls shape={lbls.shape}, mean={imgs.mean():.4f}")
        if b_idx >= 2: break
    print(f"Elapsed: {time.time() - t0:.2f}s")
