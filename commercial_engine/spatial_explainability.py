"""
Spatial Explainability: Layer-GradCAM, Eigen-CAM & Lesion Bounding Box Extractor
File: commercial_engine/spatial_explainability.py
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class SpatialAnomalyLocalizer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_spatial_cam(self, feature_map, target_weights=None):
        """
        Computes normalized 2D spatial activation heatmap using Eigen-CAM / Grad-CAM principles.
        Input: feature_map [1, C, H, W] (e.g., from last conv stage)
        Output: cam_norm [H_orig, W_orig] in range [0.0, 1.0]
        """
        if isinstance(feature_map, torch.Tensor):
            feat = feature_map.detach().float()
        else:
            feat = torch.from_numpy(feature_map).float()

        if feat.ndim == 3:
            feat = feat.unsqueeze(0)

        B, C, H, W = feat.shape
        if C == 0 or H == 0 or W == 0:
            return np.zeros((256, 256), dtype=np.float32)

        # Eigen-CAM: Principal projection along channel dimension
        # Reshape to [C, H*W]
        feat_reshaped = feat[0].reshape(C, -1)
        feat_centered = feat_reshaped - feat_reshaped.mean(dim=1, keepdim=True)
        
        # Singular Value Decomposition / 1st Principal Component
        try:
            U, S, V = torch.pca_lowrank(feat_centered.T, q=1, center=False)
            cam = V[:, 0].abs().reshape(H, W)
        except Exception:
            # Fallback to mean activation over channels
            cam = feat[0].abs().mean(dim=0)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-6:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    def extract_anomaly_bounding_box(self, cam_heatmap, target_shape=(256, 256), threshold_percentile=75):
        """
        Extracts peak lesion bounding box [xmin, ymin, xmax, ymax] normalized to [0, 1].
        """
        # Resize heatmap to target image resolution
        h_target, w_target = target_shape
        resized_cam = cv2.resize(cam_heatmap, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        
        # Threshold top activation region
        cutoff = float(np.percentile(resized_cam, threshold_percentile))
        binary_mask = (resized_cam >= cutoff).astype(np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Default center box if no contour found
            return [0.25, 0.25, 0.75, 0.75], resized_cam

        # Select largest contour by area
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)

        # Normalize to [0.0, 1.0]
        xmin = float(np.clip(x / w_target, 0.0, 1.0))
        ymin = float(np.clip(y / h_target, 0.0, 1.0))
        xmax = float(np.clip((x + w) / w_target, 0.0, 1.0))
        ymax = float(np.clip((y + h) / h_target, 0.0, 1.0))

        # Ensure minimal bounding box size for clinical visibility
        if xmax - xmin < 0.08:
            cx = (xmin + xmax) / 2.0
            xmin = max(0.0, cx - 0.05)
            xmax = min(1.0, cx + 0.05)
        if ymax - ymin < 0.08:
            cy = (ymin + ymax) / 2.0
            ymin = max(0.0, cy - 0.05)
            ymax = min(1.0, cy + 0.05)

        return [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)], resized_cam

    def render_overlay_color_map(self, original_slice_gray, cam_heatmap, alpha=0.45):
        """
        Creates an RGB heatmap overlay suitable for PACS viewer display.
        """
        h, w = original_slice_gray.shape[:2]
        cam_resized = cv2.resize(cam_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0.0, 1.0))
        
        # Apply Jet / Turbo colormap
        color_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Ensure base image is 3-channel BGR
        if original_slice_gray.ndim == 2:
            base_bgr = cv2.cvtColor(np.uint8(255 * np.clip(original_slice_gray, 0.0, 1.0)), cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = original_slice_gray.copy()

        overlay = cv2.addWeighted(color_cam, alpha, base_bgr, 1.0 - alpha, 0)
        return overlay
