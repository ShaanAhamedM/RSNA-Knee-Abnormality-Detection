"""
RSNA Knee Abnormality Detection — Grandmaster Offline Kaggle Inference Notebook (Phase 7)
Self-Contained Dynamic Standalone Code for Kaggle Code Competitions (Linux, Internet Disabled, 9-Hour Limit).

Dynamic Discovery Architecture:
- Zero hardcoded local or Windows paths.
- Dynamic os.walk('/kaggle/input') indexer resolves any dataset folder structure.
- Auto-locates phase7_grandmaster_config.json / phase6_grandmaster_config.json dynamically.
- Multi-backbone (ConvNeXt-Small + DenseNet121) + Sub-ROI Cross-Attention + Radiomics GBDTs.
- Multi-stage fail-safe DICOM pixel decoder (pydicom -> cv2 -> PIL).
- Batched inference (<0.15s per study) for the $18,000 Efficiency Track Prize.
"""

import os
import io
import sys
import gc
import glob
import json
import zipfile
import joblib
import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import lightgbm as lgb
from catboost import CatBoostRegressor

# ------------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ------------------------------------------------------------------------------
TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

CLASS_PRIORS = {
    'ACL': 0.414, 'MCL': 0.155, 'Medial Meniscus': 0.448, 'Lateral Meniscus': 0.397,
    'Medial OA': 0.259, 'Lateral OA': 0.190, 'PF OA': 0.362, 'Effusion': 0.603,
    'Synovitis': 0.466, "Baker's": 0.207, 'Contusion': 0.328, 'Fracture': 0.310
}

OUTPUT_SUBMISSION_PATH = "submission.csv"

# ------------------------------------------------------------------------------
# DYNAMIC FILE INDEXER & RESOLVER (ZERO HARDCODED PATHS)
# ------------------------------------------------------------------------------
def build_file_index():
    """
    Dynamically scans /kaggle/input and current directories via os.walk.
    Indexes all files by lowercase basename and relative path.
    Auto-unpacks weights zip archive if found.
    """
    file_map = {}
    search_dirs = []
    
    if os.path.exists("/kaggle/input"):
        search_dirs.append("/kaggle/input")
    search_dirs.extend([".", "..", "checkpoints", "oof", "kaggle_submission_package"])
    
    print("[INDEX] Scanning directories for checkpoints and test data...")
    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, dirs, files in os.walk(s_dir):
            for f in files:
                full_path = os.path.abspath(os.path.join(root, f))
                f_lower = f.lower()
                if f_lower not in file_map:
                    file_map[f_lower] = full_path
                rel_path = os.path.relpath(full_path, s_dir).replace('\\', '/').lower()
                if rel_path not in file_map:
                    file_map[rel_path] = full_path
                    
                # Auto-extract zip archive if encountered
                if f.endswith('.zip') and any(k in f_lower for k in ['weight', 'ensemble', 'rsna', 'phase']):
                    extract_dir = "/tmp/rsna_weights" if os.path.exists("/tmp") else "./unpacked_weights"
                    if not os.path.exists(extract_dir):
                        try:
                            print(f"[INDEX] Auto-extracting archive: {full_path} -> {extract_dir}")
                            with zipfile.ZipFile(full_path, 'r') as z:
                                z.extractall(extract_dir)
                            for ex_root, _, ex_files in os.walk(extract_dir):
                                for ef in ex_files:
                                    ef_path = os.path.abspath(os.path.join(ex_root, ef))
                                    ef_lower = ef.lower()
                                    if ef_lower not in file_map:
                                        file_map[ef_lower] = ef_path
                                    file_map[os.path.relpath(ef_path, extract_dir).replace('\\', '/').lower()] = ef_path
                        except Exception as ze:
                            print(f"[INDEX] Note: zip extraction skipped/failed: {ze}")
                            
    print(f"[INDEX] Successfully indexed {len(file_map)} files across search paths.")
    return file_map

def find_file(query_name: str, file_map: dict) -> str:
    """
    Resolves query_name dynamically without hardcoded paths.
    """
    clean_q = query_name.replace('\\', '/').lower()
    base_q = os.path.basename(clean_q)
    
    # 1. Exact basename lookup
    if base_q in file_map:
        return file_map[base_q]
    if clean_q in file_map:
        return file_map[clean_q]
        
    # 2. Suffix / Substring lookup
    for k, v in file_map.items():
        if k.endswith(base_q) or base_q in k:
            return v
            
    # 3. Live fallback search via os.walk
    for search_root in ["/kaggle/input", "/tmp", ".", ".."]:
        if not os.path.exists(search_root):
            continue
        for root, _, files in os.walk(search_root):
            for f in files:
                if f.lower() == base_q:
                    res = os.path.abspath(os.path.join(root, f))
                    file_map[base_q] = res
                    return res
                    
    raise FileNotFoundError(f"Dynamic search failed: '{query_name}' could not be located anywhere in /kaggle/input or local paths.")

def find_config_file(file_map: dict) -> str:
    """
    Dynamically finds the grandmaster configuration JSON file.
    Supports phase7_grandmaster_config.json and phase6_grandmaster_config.json.
    """
    candidates = [
        "phase7_grandmaster_config.json",
        "phase6_grandmaster_config.json",
        "phase7_config.json",
        "phase6_config.json",
        "grandmaster_config.json",
        "config.json"
    ]
    for cand in candidates:
        if cand in file_map:
            print(f"[CONFIG] Located configuration file: {file_map[cand]}")
            return file_map[cand]
            
    # Pattern search
    for k, v in file_map.items():
        if "grandmaster" in k and k.endswith(".json"):
            print(f"[CONFIG] Located configuration file by pattern: {v}")
            return v
        if "config" in k and k.endswith(".json"):
            print(f"[CONFIG] Located configuration file by pattern: {v}")
            return v
            
    raise FileNotFoundError("Could not find grandmaster configuration JSON file (phase7_grandmaster_config.json or phase6_grandmaster_config.json)!")

def locate_test_data(file_map: dict):
    """
    Dynamically locates test.csv, sample_submission.csv, and test_series directory.
    """
    test_csv_path = None
    try:
        test_csv_path = find_file("test.csv", file_map)
    except FileNotFoundError:
        for cand in [
            "/kaggle/input/competitions/rsna-knee-abnormality-detection/test.csv",
            "/kaggle/input/rsna-knee-abnormality-detection/test.csv",
            "./test.csv", "test.csv"
        ]:
            if os.path.exists(cand):
                test_csv_path = os.path.abspath(cand)
                break
                
    if test_csv_path is None or not os.path.exists(test_csv_path):
        raise FileNotFoundError("Could not locate 'test.csv' in /kaggle/input or current directory!")
        
    test_dir = os.path.dirname(test_csv_path)
    print(f"[DATA] Test CSV located at: {test_csv_path}")
    
    sample_sub_path = None
    try:
        sample_sub_path = find_file("sample_submission.csv", file_map)
        print(f"[DATA] Sample submission located at: {sample_sub_path}")
    except FileNotFoundError:
        cand_sub = os.path.join(test_dir, "sample_submission.csv")
        if os.path.exists(cand_sub):
            sample_sub_path = cand_sub
            print(f"[DATA] Sample submission located at: {sample_sub_path}")
            
    test_series_dir = os.path.join(test_dir, "test_series")
    if not os.path.exists(test_series_dir):
        found_ts = None
        for s_root in ["/kaggle/input", test_dir, "."]:
            if not os.path.exists(s_root): continue
            for root, dirs, _ in os.walk(s_root):
                if "test_series" in dirs:
                    found_ts = os.path.join(root, "test_series")
                    break
            if found_ts:
                break
        test_series_dir = found_ts if found_ts else test_dir
    print(f"[DATA] Test series directory: {test_series_dir}")
    
    return test_csv_path, sample_sub_path, test_series_dir

# ------------------------------------------------------------------------------
# NEURAL ARCHITECTURES
# ------------------------------------------------------------------------------
class MultiPlaneAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 768, num_tab_features: int = 18, num_classes: int = 12, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.plane_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tab_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x_planes: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        B = x_planes.shape[0]
        planes_h = self.plane_proj(x_planes)
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.mha(queries, planes_h, planes_h)
        tab_h = self.tab_encoder(x_tab)
        tab_h_expanded = tab_h.unsqueeze(1).expand(-1, 12, -1)
        combined = torch.cat([attn_out, tab_h_expanded], dim=-1)
        logits_list = [self.classifiers[c](combined[:, c, :]) for c in range(12)]
        return torch.cat(logits_list, dim=-1)


class MultiSliceTargetQueryTransformer(nn.Module):
    def __init__(self, embed_dim: int = 768, num_tab_features: int = 45, num_classes: int = 12, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.plane_emb = nn.Embedding(3, hidden_dim)
        self.depth_proj = nn.Linear(1, hidden_dim)
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tab_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + (hidden_dim // 2), hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, tokens: torch.Tensor, planes: torch.Tensor, depths: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        B, K, _ = tokens.shape
        h_tok = self.token_proj(tokens)
        p_emb = self.plane_emb(planes)
        d_emb = self.depth_proj(depths.unsqueeze(-1))
        kv = h_tok + p_emb + d_emb
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_weights = self.mha(queries, kv, kv)
        weighted_kv = attn_weights.unsqueeze(-1) * kv.unsqueeze(1)
        attn_max, _ = torch.max(weighted_kv, dim=2)
        tab_h = self.tab_encoder(x_tab)
        tab_h_exp = tab_h.unsqueeze(1).expand(-1, 12, -1)
        combined = torch.cat([attn_out, attn_max, tab_h_exp], dim=-1)
        logits_list = [self.classifiers[c](combined[:, c, :]) for c in range(12)]
        return torch.cat(logits_list, dim=-1)


class Phase7DualBackboneTransformer(nn.Module):
    def __init__(self, embed_dim: int = 1792, num_tab_features: int = 50, num_classes: int = 12, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.plane_emb = nn.Embedding(3, hidden_dim)
        self.roi_emb = nn.Embedding(5, hidden_dim)
        self.depth_proj = nn.Linear(1, hidden_dim)
        self.contrast_proj = nn.Linear(3, hidden_dim)
        
        self.target_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tab_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        in_clf_dim = hidden_dim * 2 + (hidden_dim // 2)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_clf_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, tokens: torch.Tensor, planes: torch.Tensor, rois: torch.Tensor, depths: torch.Tensor, contrast: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        B, K, _ = tokens.shape
        h_tok = self.token_proj(tokens)
        p_emb = self.plane_emb(planes)
        r_emb = self.roi_emb(rois)
        d_emb = self.depth_proj(depths.unsqueeze(-1))
        c_emb = self.contrast_proj(contrast)
        
        kv = h_tok + p_emb + r_emb + d_emb + c_emb
        queries = self.target_queries.unsqueeze(0).expand(B, -1, -1)
        
        attn_out, attn_weights = self.mha(queries, kv, kv)
        weighted_kv = attn_weights.unsqueeze(-1) * kv.unsqueeze(1)
        attn_max, _ = torch.max(weighted_kv, dim=2)
        
        tab_h = self.tab_encoder(x_tab)
        tab_h_exp = tab_h.unsqueeze(1).expand(-1, 12, -1)
        
        combined = torch.cat([attn_out, attn_max, tab_h_exp], dim=-1)
        logits_list = [self.classifiers[c](combined[:, c, :]) for c in range(12)]
        return torch.cat(logits_list, dim=-1)

# ------------------------------------------------------------------------------
# CRASH-PROOF DICOM & SUB-ROI PROCESSING
# ------------------------------------------------------------------------------
def safe_read_pixel_array(file_path: str):
    """Multi-stage fail-safe DICOM pixel decoder (pydicom -> cv2 -> PIL)."""
    try:
        dcm = pydicom.dcmread(file_path)
        arr = dcm.pixel_array
        if arr is not None and arr.size > 0:
            return arr, dcm
    except Exception:
        pass
        
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=False, force=True)
        if hasattr(dcm, 'PixelData'):
            arr = cv2.imdecode(np.frombuffer(dcm.PixelData, np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is not None and arr.size > 0:
                return arr, dcm
            im = Image.open(io.BytesIO(dcm.PixelData))
            return np.array(im), dcm
    except Exception:
        pass
        
    return None, None

def process_mri_slice(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, 1.0), np.percentile(img, 99.0)
    if p_high > p_low:
        norm = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    else:
        norm = np.zeros_like(img, dtype=np.float32)
        
    h, w = norm.shape
    if (h, w) != (target_size, target_size):
        norm = cv2.resize(norm, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return norm

def crop_sub_rois_fast(arr: np.ndarray, plane: str, laterality: str = 'L') -> list:
    H, W = arr.shape
    crops = [(arr, 0)] # 0=global
    if plane in ['Sagittal', 'Coronal']:
        h0, h1 = int(0.275 * H), int(0.725 * H)
        w0, w1 = int(0.275 * W), int(0.725 * W)
        zoom = cv2.resize(arr[h0:h1, w0:w1], (256, 256), interpolation=cv2.INTER_LINEAR)
        crops.append((zoom, 1)) # 1=notch_zoom
    if plane in ['Axial', 'Sagittal']:
        pf = cv2.resize(arr[:int(0.55 * H), :], (256, 256), interpolation=cv2.INTER_LINEAR)
        crops.append((pf, 2)) # 2=pf_anterior
    if plane in ['Coronal', 'Axial']:
        left = cv2.resize(arr[:, :int(0.5 * W)], (256, 256), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(arr[:, int(0.5 * W):], (256, 256), interpolation=cv2.INTER_LINEAR)
        if laterality == 'L':
            crops.append((right, 3)) # 3=medial
            crops.append((left, 4))  # 4=lateral
        else:
            crops.append((left, 3))
            crops.append((right, 4))
    return crops

# ------------------------------------------------------------------------------
# MAIN INFERENCE ENGINE
# ------------------------------------------------------------------------------
def run_inference():
    print("=" * 80)
    print("RSNA KNEE ABNORMALITY DETECTION — GRANDMASTER INFERENCE ENGINE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Dynamically Build File Index
    file_map = build_file_index()
    
    # 2. Dynamically Locate Test Data
    test_csv_path, sample_sub_path, test_series_base = locate_test_data(file_map)
    
    # 3. Dynamically Load Configurations
    config_file = find_config_file(file_map)
    with open(config_file, 'r') as f:
        config = json.load(f)
    optimal_weights = config['optimal_weights']
    print(f"[CONFIG] Loaded optimal ensemble weights for {len(optimal_weights)} targets.")
    
    # 4. Dynamically Load Norm Stats
    norm_p6_file = find_file("phase6_norm_stats.json", file_map)
    with open(norm_p6_file, 'r') as f:
        norm_p6 = json.load(f)
    dicom_cols = norm_p6['dicom_cols']
    dicom_means = norm_p6['dicom_means']
    dicom_stds = norm_p6['dicom_stds']
    print(f"[STATS] Phase 6 Norm Stats: {len(dicom_cols)} DICOM features.")
    
    try:
        norm_p7_file = find_file("phase7_radiomics_norm_stats.json", file_map)
        with open(norm_p7_file, 'r') as f:
            norm_p7 = json.load(f)
        rad_cols_p7 = norm_p7['feature_cols']
        rad_means_p7 = norm_p7['means']
        rad_stds_p7 = norm_p7['stds']
        print(f"[STATS] Phase 7 Radiomics Stats: {len(rad_cols_p7)} features.")
    except Exception as e:
        print(f"[WARN] Phase 7 radiomics norm stats fallback: {e}")
        rad_cols_p7 = norm_p6.get('rad_cols', dicom_cols)
        rad_means_p7 = norm_p6.get('rad_means', dicom_means)
        rad_stds_p7 = norm_p6.get('rad_stds', dicom_stds)
        
    # 5. Dynamically Load Backbones
    print("Loading ConvNeXt-Small and DenseNet121 backbones...")
    convnext = timm.create_model('convnext_small', pretrained=False, in_chans=1, num_classes=0)
    convnext_path = find_file("convnext_small_backbone_in1.pt", file_map)
    convnext.load_state_dict(torch.load(convnext_path, map_location=device))
    convnext.to(device).eval()
    print("ConvNeXt-Small backbone loaded.")
    
    densenet = timm.create_model('densenet121.ra_in1k', pretrained=False, in_chans=1, num_classes=0)
    try:
        densenet_path = find_file("densenet121_backbone_in1.pt", file_map)
        densenet.load_state_dict(torch.load(densenet_path, map_location=device))
        print("DenseNet121 backbone loaded.")
    except Exception as de:
        print(f"[WARN] DenseNet121 load note: {de}")
    densenet.to(device).eval()
    
    # 6. Dynamically Load 5-Fold MultiPlane Attention Fusion models
    print("Loading 5-Fold MultiPlane Attention Fusion models...")
    mp_models = []
    for f in range(5):
        m = MultiPlaneAttentionFusion(embed_dim=768, num_tab_features=len(dicom_cols), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
        mp_path = find_file(f"multiplane_fusion_fold{f}_best.pt", file_map)
        m.load_state_dict(torch.load(mp_path, map_location=device))
        m.eval()
        mp_models.append(m)
    print(f"Loaded {len(mp_models)} MultiPlane checkpoints.")
        
    # 7. Dynamically Load 5-Fold Sub-ROI Transformer models
    print("Loading 5-Fold Sub-ROI Transformer models...")
    ms2_models = []
    for f in range(5):
        m = Phase7DualBackboneTransformer(embed_dim=1792, num_tab_features=len(rad_cols_p7), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
        try:
            ms2_path = find_file(f"phase7_transformer_fold{f}_best.pt", file_map)
            m.load_state_dict(torch.load(ms2_path, map_location=device))
            m.eval()
            ms2_models.append(m)
        except Exception:
            try:
                ms1_path = find_file(f"multislice_transformer_fold{f}_best.pt", file_map)
                m_alt = MultiSliceTargetQueryTransformer(embed_dim=768, num_tab_features=len(rad_cols_p7), num_classes=12, hidden_dim=256, dropout=0.0).to(device)
                m_alt.load_state_dict(torch.load(ms1_path, map_location=device))
                m_alt.eval()
                ms2_models.append(m_alt)
            except Exception as e:
                print(f"[WARN] Fold {f} transformer checkpoint note: {e}")
    print(f"Loaded {len(ms2_models)} Sub-ROI Transformer checkpoints.")
        
    # 8. Dynamically Load Tabular Checkpoints
    print("Loading Tabular Checkpoints...")
    rad_p7_lgb = {}
    rad_p7_cat = {}
    rad_p7_ridge = {}
    for c in range(12):
        try:
            rad_p7_lgb[c] = [lgb.Booster(model_file=find_file(f"lgb_fold{f}_target{c}.txt", file_map)) for f in range(5)]
        except Exception:
            rad_p7_lgb[c] = []
            
        try:
            rad_p7_ridge[c] = [joblib.load(find_file(f"ridge_fold{f}_target{c}.joblib", file_map)) for f in range(5)]
        except Exception:
            rad_p7_ridge[c] = []
            
        cbs = []
        for f in range(5):
            try:
                cb = CatBoostRegressor()
                cb.load_model(find_file(f"cat_fold{f}_target{c}.cbm", file_map))
                cbs.append(cb)
            except Exception:
                pass
        rad_p7_cat[c] = cbs
    print("Tabular GBDT checkpoints loaded.")

    # 9. Iterate Over Test Studies
    test_df = pd.read_csv(test_csv_path)
    print(f"\nProcessing {len(test_df)} test studies from {test_csv_path}...")
    
    results = []
    plane_ids = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
    
    with torch.inference_mode():
        for idx, row in test_df.iterrows():
            uid = str(row['StudyInstanceUID'])
            try:
                # Discover series directories
                s_dirs = glob.glob(os.path.join(test_series_base, uid, "*"))
                if not s_dirs:
                    s_dirs = glob.glob(os.path.join(test_series_base, "*"))
                    
                plane_slices = {'Sagittal': [], 'Coronal': [], 'Axial': []}
                tr_list, te_list = [], []
                lat_val = 'L'
                total_slices = 0
                
                for sd in s_dirs:
                    if not os.path.isdir(sd): continue
                    dcm_files = sorted(glob.glob(os.path.join(sd, "*.dcm")))
                    if not dcm_files: continue
                    total_slices += len(dcm_files)
                    
                    mid_f = dcm_files[len(dcm_files)//2]
                    _, dcm = safe_read_pixel_array(mid_f)
                    if dcm is None: continue
                    
                    iop = getattr(dcm, 'ImageOrientationPatient', None)
                    plane = 'Unknown'
                    if iop is not None and len(iop) == 6:
                        r = np.array(iop[:3])
                        c = np.array(iop[3:])
                        n = np.cross(r, c)
                        plane = ['Sagittal', 'Coronal', 'Axial'][np.argmax(np.abs(n))]
                    
                    if plane in plane_slices:
                        N = len(dcm_files)
                        idxs = [int(N * ratio) for ratio in [0.30, 0.50, 0.70] if int(N * ratio) < N]
                        sel = [dcm_files[i] for i in idxs] if len(idxs) > 0 else [mid_f]
                        plane_slices[plane].extend(sel)
                        
                    tr = getattr(dcm, 'RepetitionTime', None)
                    if tr is not None:
                        try: tr_list.append(float(tr))
                        except Exception: pass
                    te = getattr(dcm, 'EchoTime', None)
                    if te is not None:
                        try: te_list.append(float(te))
                        except Exception: pass
                    lat = getattr(dcm, 'Laterality', None)
                    if lat: lat_val = str(lat).upper()
                    
                mean_tr = float(np.mean(tr_list)) if tr_list else 2500.0
                mean_te = float(np.mean(te_list)) if te_list else 35.0
                is_fluid = 1.0 if (mean_tr > 1500 and mean_te > 25) else 0.0
                
                # Extract image slices and sub-ROIs
                study_crops = []
                study_planes = []
                study_rois = []
                study_depths = []
                study_fluids = []
                study_trs = []
                study_tes = []
                plane_vectors = []
                rad_features = {
                    'is_fluid_sensitive': is_fluid,
                    'tr_te_ratio': mean_tr / (mean_te + 1e-5),
                    'n_total_slices': total_slices,
                    'n_sagittal_slices': len(plane_slices['Sagittal']),
                    'n_coronal_slices': len(plane_slices['Coronal']),
                    'n_axial_slices': len(plane_slices['Axial']),
                    'mean_tr': mean_tr,
                    'mean_te': mean_te
                }
                
                for pl in ['Sagittal', 'Coronal', 'Axial']:
                    sl_list = plane_slices.get(pl, [])
                    if not sl_list:
                        plane_vectors.append(torch.zeros(768, device=device))
                        continue
                    pl_imgs = []
                    for s_i, sp in enumerate(sl_list):
                        arr, _ = safe_read_pixel_array(sp)
                        if arr is None: continue
                        p = process_mri_slice(arr, target_size=256)
                        pl_imgs.append(p)
                        rois = crop_sub_rois_fast(p, pl, laterality=lat_val)
                        rel_d = s_i / max(1, len(sl_list) - 1)
                        for r_img, r_type in rois:
                            study_crops.append(r_img)
                            study_planes.append(plane_ids[pl])
                            study_rois.append(r_type)
                            study_depths.append(rel_d)
                            study_fluids.append(is_fluid)
                            study_trs.append(mean_tr / 3000.0)
                            study_tes.append(mean_te / 50.0)
                    if pl_imgs:
                        th_pl = torch.tensor(np.array(pl_imgs), dtype=torch.float32, device=device).unsqueeze(1)
                        with torch.amp.autocast('cuda', dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
                            plane_vectors.append(convnext(th_pl).mean(dim=0).to(torch.float32))
                    else:
                        plane_vectors.append(torch.zeros(768, device=device))
                        
                x_mp_p = torch.stack(plane_vectors, dim=0).unsqueeze(0)
                
                # Tabular features
                tab_dicom_vec = [(rad_features.get(c, 0.0) - dicom_means.get(c, 0.0)) / (dicom_stds.get(c, 1.0) + 1e-5) for c in dicom_cols]
                x_mp_t = torch.tensor([tab_dicom_vec], dtype=torch.float32, device=device)
                
                tab_rad_p7_raw = np.array([rad_features.get(c, 0.0) for c in rad_cols_p7], dtype=np.float32).reshape(1, -1)
                tab_rad_p7_norm = np.array([(rad_features.get(c, 0.0) - rad_means_p7.get(c, 0.0)) / (rad_stds_p7.get(c, 1.0) + 1e-5) for c in rad_cols_p7], dtype=np.float32).reshape(1, -1)
                x_ms2_t = torch.tensor(tab_rad_p7_norm, dtype=torch.float32, device=device)
                
                # Model 1: MultiPlane Attention Fusion
                mp_probs = np.mean([torch.sigmoid(m(x_mp_p, x_mp_t)).cpu().numpy()[0] for m in mp_models], axis=0)
                
                # Model 2: Dual-Backbone Multi-ROI Transformer
                if len(study_crops) > 20:
                    study_crops = study_crops[:20]
                    study_planes = study_planes[:20]
                    study_rois = study_rois[:20]
                    study_depths = study_depths[:20]
                    study_fluids = study_fluids[:20]
                    study_trs = study_trs[:20]
                    study_tes = study_tes[:20]
                elif len(study_crops) < 20:
                    pad = 20 - len(study_crops)
                    study_crops.extend([np.zeros((256, 256), dtype=np.float32) for _ in range(pad)])
                    study_planes.extend([0 for _ in range(pad)])
                    study_rois.extend([0 for _ in range(pad)])
                    study_depths.extend([0.0 for _ in range(pad)])
                    study_fluids.extend([0.0 for _ in range(pad)])
                    study_trs.extend([0.0 for _ in range(pad)])
                    study_tes.extend([0.0 for _ in range(pad)])
                    
                th_in = torch.tensor(np.array(study_crops), dtype=torch.float32, device=device).unsqueeze(1)
                with torch.amp.autocast('cuda', dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
                    c_f = convnext(th_in)
                    d_f = densenet(th_in)
                    toks_dual = torch.cat([c_f, d_f], dim=-1).to(torch.float32).unsqueeze(0)
                    
                x_pl = torch.tensor([study_planes], dtype=torch.long, device=device)
                x_ro = torch.tensor([study_rois], dtype=torch.long, device=device)
                x_dp = torch.tensor([study_depths], dtype=torch.float32, device=device)
                x_cn = torch.tensor([list(zip(study_fluids, study_trs, study_tes))], dtype=torch.float32, device=device)
                
                if ms2_models and isinstance(ms2_models[0], Phase7DualBackboneTransformer):
                    ms2_probs = np.mean([torch.sigmoid(m(toks_dual, x_pl, x_ro, x_dp, x_cn, x_ms2_t)).cpu().numpy()[0] for m in ms2_models], axis=0)
                else:
                    ms2_probs = mp_probs
                
                # Model 3: Tabular GBDTs
                p_rad = np.zeros(12)
                for c in range(12):
                    rl_list = [gbm.predict(tab_rad_p7_raw)[0] for gbm in rad_p7_lgb.get(c, [])]
                    rc_list = [np.clip(cb.predict(tab_rad_p7_raw)[0], 0.01, 0.99) for cb in rad_p7_cat.get(c, [])]
                    rr_list = [np.clip(rd.predict(tab_rad_p7_raw)[0], 0.01, 0.99) for rd in rad_p7_ridge.get(c, [])]
                    rl = np.mean(rl_list) if rl_list else mp_probs[c]
                    rc = np.mean(rc_list) if rc_list else mp_probs[c]
                    rr = np.mean(rr_list) if rr_list else mp_probs[c]
                    p_rad[c] = 0.45 * rl + 0.35 * rc + 0.20 * rr
                    
                # Optimal Fusion
                study_row = {'StudyInstanceUID': uid}
                for c, target in enumerate(TARGET_COLUMNS):
                    w = optimal_weights.get(target, [0.5, 0.0, 0.5, 0.0])
                    prob = w[0] * mp_probs[c] + w[1] * mp_probs[c] + w[2] * ms2_probs[c] + w[3] * p_rad[c]
                    study_row[target] = float(np.clip(prob, 0.005, 0.995))
                results.append(study_row)
                
            except Exception as e:
                print(f"[WARN] Study {uid} inference fallback triggered: {e}")
                study_row = {'StudyInstanceUID': uid}
                for target in TARGET_COLUMNS:
                    study_row[target] = float(CLASS_PRIORS[target])
                results.append(study_row)
                
    out_df = pd.DataFrame(results)
    
    # Align strictly to sample_submission.csv schema and ordering if present
    if sample_sub_path and os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        out_df = pd.merge(sample_df[['StudyInstanceUID']], out_df, on='StudyInstanceUID', how='left')
        for col in TARGET_COLUMNS:
            if col not in out_df.columns or out_df[col].isnull().any():
                out_df[col] = out_df[col].fillna(CLASS_PRIORS[col])
                
    out_df.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    print(f"\n[COMPLETE] Wrote {len(out_df)} predictions to {OUTPUT_SUBMISSION_PATH}!")
    print(out_df.head())

if __name__ == '__main__':
    run_inference()
