"""
Radiometric Signal Processing & Knee Joint ROI Cropping Module
"""

import numpy as np
import cv2

def crop_knee_roi(image: np.ndarray, threshold: float = 0.05, margin: int = 4) -> np.ndarray:
    """
    Automatically detects and crops out empty black background borders
    around the knee joint to maximize effective spatial pixel resolution.
    """
    if image.max() <= 0:
        return image
    
    # Binary mask of non-background tissue
    mask = image > (image.max() * threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add margin with boundary checks
    h, w = image.shape[:2]
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(h, y_max + margin + 1)
    x_max = min(w, x_max + margin + 1)
    
    # Check if cropped area is reasonably sized (at least 20% of original)
    if (y_max - y_min) > 0.2 * h and (x_max - x_min) > 0.2 * w:
        return image[y_min:y_max, x_min:x_max]
    return image

def apply_percentile_windowing(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """
    Robust percentile intensity windowing (1% to 99%) eliminating outlier high-signal
    artifacts (e.g. skin fiducials) while stretching bone and soft tissue dynamic range.
    """
    p_low = np.percentile(image, low_pct)
    p_high = np.percentile(image, high_pct)
    if p_high > p_low:
        clipped = np.clip(image, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low)
        return norm.astype(np.float32)
    return np.zeros_like(image, dtype=np.float32)

def apply_clahe(image_norm: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) to sharpen
    subtle contrast in meniscus tears, cartilage defects, and trabecular contusions.
    Input must be in [0, 1] range. Output is in [0, 1] range.
    """
    img_uint8 = (np.clip(image_norm, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)
    return enhanced.astype(np.float32) / 255.0

def process_mri_slice(
    raw_pixel_array: np.ndarray,
    target_size: int = 256,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    use_clahe: bool = True,
    crop_roi: bool = True
) -> np.ndarray:
    """
    Full radiometric pipeline:
    1. Percentile windowing [0, 1]
    2. Optional CLAHE enhancement
    3. ROI border cropping
    4. Anti-aliased resizing to target_size x target_size
    """
    img = raw_pixel_array.astype(np.float32)
    norm = apply_percentile_windowing(img, low_pct=low_pct, high_pct=high_pct)
    
    if use_clahe:
        norm = apply_clahe(norm)
        
    if crop_roi:
        norm = crop_knee_roi(norm)
        
    # Resize to target dimension
    resized = cv2.resize(norm, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)
