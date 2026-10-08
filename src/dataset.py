"""
High-Performance Multiprocessing Zip-Safe Dataset & DataLoader Engine
Zero-Unzip In-Memory Streaming from rsna-knee-abnormality-detection.zip
"""

import os
import io
import json
import zipfile
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict

from src.preprocessing import process_mri_slice

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

_GLOBAL_SLICE_LOOKUP = None

def get_slice_lookup(lookup_json: str = r"D:\MLprojs\KaggleCompetition\RSNA\study_slice_lookup.json") -> Dict:
    global _GLOBAL_SLICE_LOOKUP
    if _GLOBAL_SLICE_LOOKUP is None:
        if os.path.exists(lookup_json):
            with open(lookup_json, 'r') as f:
                _GLOBAL_SLICE_LOOKUP = json.load(f)
        else:
            _GLOBAL_SLICE_LOOKUP = {}
    return _GLOBAL_SLICE_LOOKUP

class RSNAKneeZipDataset(Dataset):
    """
    Multiprocessing-safe, zero-unzip DICOM streaming dataset.
    Loads slices directly from D:\\RSNA\\rsna-knee-abnormality-detection.zip into RAM.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: str = r"D:\RSNA\rsna-knee-abnormality-detection.zip",
        target_size: int = 256,
        slices_per_study: int = 6,
        in_chans: int = 1,
        use_clahe: bool = True,
        crop_roi: bool = True,
        is_train: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.zip_path = zip_path
        self.target_size = target_size
        self.slices_per_study = slices_per_study
        self.in_chans = in_chans
        self.use_clahe = use_clahe
        self.crop_roi = crop_roi
        self.is_train = is_train
        
        # Fast cached lookup
        self.slice_lookup = get_slice_lookup()

        # Lazy per-worker zip handle placeholder (Windows multiprocessing safety)
        self.zip_handle = None

    def _get_zip(self):
        if self.zip_handle is None:
            self.zip_handle = zipfile.ZipFile(self.zip_path, 'r')
        return self.zip_handle

    def __len__(self):
        return len(self.df)

    def _load_dicom_slice(self, zip_obj, path: str) -> np.ndarray:
        global _SLICE_RAM_CACHE
        if '_SLICE_RAM_CACHE' not in globals():
            globals()['_SLICE_RAM_CACHE'] = {}
            
        cache_key = (path, self.target_size, self.use_clahe, self.crop_roi)
        if cache_key in globals()['_SLICE_RAM_CACHE']:
            return globals()['_SLICE_RAM_CACHE'][cache_key].copy()
            
        try:
            raw_bytes = zip_obj.read(path)
            dcm = pydicom.dcmread(io.BytesIO(raw_bytes))
            pixel_array = dcm.pixel_array
            processed = process_mri_slice(
                pixel_array,
                target_size=self.target_size,
                use_clahe=self.use_clahe,
                crop_roi=self.crop_roi
            )
            globals()['_SLICE_RAM_CACHE'][cache_key] = processed
            return processed.copy()
        except Exception:
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        study_id = row['StudyInstanceUID']
        z = self._get_zip()
        
        slice_paths = self.slice_lookup.get(study_id, [])
        if len(slice_paths) > self.slices_per_study:
            slice_paths = slice_paths[:self.slices_per_study]
        elif len(slice_paths) < self.slices_per_study and len(slice_paths) > 0:
            slice_paths = slice_paths + [slice_paths[-1]] * (self.slices_per_study - len(slice_paths))
            
        # Load and process slices
        slice_images = []
        if slice_paths:
            for sp in slice_paths:
                img_2d = self._load_dicom_slice(z, sp)
                # Training augmentations (photometric perturbation strictly preserving medial-lateral anatomical orientation)
                if self.is_train and np.random.rand() > 0.5:
                    alpha = float(np.random.uniform(0.90, 1.10))
                    beta = float(np.random.uniform(-0.05, 0.05))
                    img_2d = np.clip(alpha * img_2d + beta, 0.0, 1.0)
                slice_images.append(img_2d)
        else:
            # Fallback zero tensor
            slice_images = [np.zeros((self.target_size, self.target_size), dtype=np.float32)] * self.slices_per_study
            
        # Shape: (slices_per_study, in_chans, H, W)
        bag_tensor = torch.tensor(np.array(slice_images), dtype=torch.float32).unsqueeze(1) # (K, 1, H, W)
        
        if self.in_chans == 3:
            bag_tensor = bag_tensor.repeat(1, 3, 1, 1) # (K, 3, H, W)
            
        # Extract target labels
        labels = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
        for i, col in enumerate(TARGET_COLUMNS):
            if col in row and not pd.isna(row[col]):
                labels[i] = float(row[col])
                
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        weight = float(row['sample_weight']) if 'sample_weight' in row and not pd.isna(row['sample_weight']) else 1.0
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        
        return {
            'image': bag_tensor, # (Bag_size, C, H, W)
            'labels': labels_tensor, # (12,)
            'weight': weight_tensor,
            'study_id': study_id
        }

def get_dataloader(
    df: pd.DataFrame,
    target_size: int = 256,
    slices_per_study: int = 6,
    in_chans: int = 1,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    is_train: bool = True
) -> DataLoader:
    dataset = RSNAKneeZipDataset(
        df=df,
        target_size=target_size,
        slices_per_study=slices_per_study,
        in_chans=in_chans,
        is_train=is_train
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
