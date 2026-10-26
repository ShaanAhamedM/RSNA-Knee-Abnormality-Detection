"""
Differential Volumetric Preprocessing (Δz and Δ²z Spatial Derivatives)
File: src/differential_preprocessing.py
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

def build_differential_slice_3ch(img_prev, img_curr, img_next):
    """
    Constructs a 3-channel mathematically orthogonal differential slice tensor:
    - Channel 0: Structural Intensity I_z (CLAHE / windowed normalized)
    - Channel 1: 1st-Order Inter-slice Velocity |I_{z+1} - I_{z-1}|
    - Channel 2: 2nd-Order Volumetric Curvature |2*I_z - I_{z-1} - I_{z+1}| + Sobel edge magnitude
    """
    # Ensure float32 in [0, 1]
    img_prev = np.clip(img_prev.astype(np.float32), 0.0, 1.0)
    img_curr = np.clip(img_curr.astype(np.float32), 0.0, 1.0)
    img_next = np.clip(img_next.astype(np.float32), 0.0, 1.0)
    
    # Channel 0: Structural Intensity
    ch0 = img_curr
    
    # Channel 1: 1st-Order Velocity (detects fiber tear clefts & termination)
    ch1 = np.abs(img_next - img_prev)
    
    # Channel 2: 2nd-Order Curvature + Sobel Edge Magnitude
    curvature = np.abs(2.0 * img_curr - img_prev - img_next)
    sobel_x = cv2.Sobel(img_curr, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_curr, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    max_mag = float(sobel_mag.max())
    if max_mag > 1e-6:
        sobel_mag = sobel_mag / max_mag
    ch2 = np.clip(0.5 * curvature + 0.5 * sobel_mag, 0.0, 1.0)
    
    # Stack into [3, H, W]
    diff_tensor = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)
    return diff_tensor

def build_differential_batch_gpu(slices_tensor):
    """
    GPU-accelerated differential volumetric tensor builder.
    Input: [B, K, 1, H, W] where K is sequence length.
    Output: [B, K, 3, H, W] differential 3-channel tensors.
    """
    B, K, C, H, W = slices_tensor.shape
    device = slices_tensor.device
    
    # Pad depth dimension for boundary slices
    padded = F.pad(slices_tensor, (0, 0, 0, 0, 0, 0, 1, 1), mode='replicate')  # [B, K+2, 1, H, W]
    
    prev_s = padded[:, :K, 0, :, :]
    curr_s = padded[:, 1:K+1, 0, :, :]
    next_s = padded[:, 2:K+2, 0, :, :]
    
    ch0 = curr_s
    ch1 = torch.abs(next_s - prev_s)
    ch2 = torch.abs(2.0 * curr_s - prev_s - next_s)
    
    # Stack channels along dim 2 -> [B, K, 3, H, W]
    out = torch.stack([ch0, ch1, ch2], dim=2)
    return out
