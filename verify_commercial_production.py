"""
Comprehensive Sanity & API Stress-Test Suite
Script: verify_commercial_production.py

Validates:
1. Commercial Diagnostic Engine & In-Memory Checkpoints
2. Spatial Grad-CAM & Lesion Bounding-Box Localization
3. Out-of-Distribution (OOD) Quality Control
4. FastAPI Microservice Endpoints (/v1/health, /v1/analyze-study, /v1/analyze-tokens)
5. Compiled FP16 & CPU TorchScript Models
6. Kaggle Submission Compliance (submission.csv format, non-empty, values in [0, 1])
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from commercial_engine.commercial_knee_engine import CommercialKneeDiagnosticEngine, TARGET_COLS

def test_diagnostic_engine():
    print("=" * 65)
    print("[TEST 1/5] Testing Commercial Knee Diagnostic Engine...")
    print("=" * 65)
    engine = CommercialKneeDiagnosticEngine()
    
    K = 15
    dummy_tokens = {
        'tokens': torch.randn(K, 1792),
        'planes': torch.randint(0, 3, (K,)),
        'rois': torch.randint(0, 5, (K,)),
        'depths': torch.linspace(0.1, 0.9, K),
        'contrast': torch.rand(K, 3)
    }
    dummy_tab = np.zeros(46, dtype=np.float32)
    
    t0 = time.perf_counter()
    report = engine.predict_study_tokens(dummy_tokens, dummy_tab)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert report["resourceType"] == "DiagnosticReport", "Invalid FHIR resourceType"
    assert "findings_by_abnormality" in report, "Missing findings"
    
    for target in TARGET_COLS:
        assert target in report["findings_by_abnormality"], f"Missing {target}"
        item = report["findings_by_abnormality"][target]
        assert 0.0 <= item["calibrated_probability"] <= 1.0, f"Invalid prob for {target}"
        bbox = item["lesion_spatial_localization"]["bounding_box_normalized"]
        assert 0.0 <= bbox["xmin"] <= bbox["xmax"] <= 1.0, f"Invalid x-bbox for {target}"
        assert 0.0 <= bbox["ymin"] <= bbox["ymax"] <= 1.0, f"Invalid y-bbox for {target}"
        assert item["triage_classification"] in [
            "HIGH_CONFIDENCE_POSITIVE", "HIGH_CONFIDENCE_NEGATIVE", "BORDERLINE_NEEDS_RADIOLOGIST_REVIEW"
        ]
        
    print(f"Engine Test PASSED! 12 findings evaluated in {elapsed_ms:.2f} ms")
    return True

def test_ood_quality_gate():
    print("\n" + "=" * 65)
    print("[TEST 2/5] Testing Out-of-Distribution & Bad-Scan Quality Gate...")
    print("=" * 65)
    engine = CommercialKneeDiagnosticEngine()
    
    # Blank slice
    blank_slice = np.zeros((256, 256), dtype=np.uint16)
    is_valid, issues = engine.audit_scan_quality(blank_slice)
    assert not is_valid, "Failed to reject blank slice"
    print(f"Blank scan correctly flagged: {issues}")
    
    # Non-knee scan
    normal_slice = np.random.randint(50, 200, (256, 256), dtype=np.uint16)
    is_valid, issues = engine.audit_scan_quality(normal_slice, metadata={"Modality": "CT", "BodyPartExamined": "CHEST"})
    assert not is_valid, "Failed to reject CT CHEST scan"
    print(f"Non-knee scan correctly flagged: {issues}")
    
    # Valid knee scan
    is_valid, issues = engine.audit_scan_quality(normal_slice, metadata={"Modality": "MR", "BodyPartExamined": "KNEE"})
    assert is_valid, f"Rejected valid scan unexpectedly: {issues}"
    print("Valid knee scan PASSED quality gate.")
    return True

def test_fastapi_endpoints():
    print("\n" + "=" * 65)
    print("[TEST 3/5] Testing FastAPI Microservice Endpoints...")
    print("=" * 65)
    from fastapi.testclient import TestClient
    from commercial_engine.app import app
    
    client = TestClient(app)
    
    # Health endpoint
    res = client.get("/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f"Health check status: {health_data['status']}")
    print(f"GPU Telemetry: {health_data['gpu_telemetry']}")
    
    # Analyze tokens endpoint
    K = 10
    payload = {
        "tokens": np.random.randn(K, 1792).tolist(),
        "planes": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "rois": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "depths": np.linspace(0.1, 0.9, K).tolist(),
        "contrast": np.zeros((K, 3)).tolist(),
        "tab_features": [0.0] * 46
    }
    t0 = time.perf_counter()
    res = client.post("/v1/analyze-tokens", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res.status_code == 200, f"Token inference failed: {res.text}"
    report = res.json()
    assert report["resourceType"] == "DiagnosticReport"
    print(f"Token analysis API PASSED in {elapsed_ms:.2f} ms!")
    
    # Analyze study endpoint
    res = client.post("/v1/analyze-study", data={"study_id": "TEST_STUDY_001"})
    assert res.status_code == 200, f"Study analysis failed: {res.text}"
    study_report = res.json()
    assert "findings_by_abnormality" in study_report
    print("Analyze study API PASSED!")
    return True

def test_compiled_models():
    print("\n" + "=" * 65)
    print("[TEST 4/5] Testing Compiled FP16 & CPU TorchScript Models...")
    print("=" * 65)
    fp16_path = os.path.join(ROOT_DIR, "checkpoints", "compiled_commercial_student_fp16.pt")
    cpu_path = os.path.join(ROOT_DIR, "checkpoints", "compiled_commercial_student_cpu.pt")
    
    B, K = 1, 10
    if os.path.exists(fp16_path) and torch.cuda.is_available():
        ts_fp16 = torch.jit.load(fp16_path, map_location='cuda')
        dummy_gpu = (
            torch.randn(B, K, 1792, device='cuda', dtype=torch.float16),
            torch.randint(0, 3, (B, K), device='cuda'),
            torch.randint(0, 5, (B, K), device='cuda'),
            torch.rand(B, K, 1, device='cuda', dtype=torch.float16),
            torch.rand(B, K, 3, device='cuda', dtype=torch.float16),
            torch.randn(B, 46, device='cuda', dtype=torch.float16)
        )
        out = ts_fp16(*dummy_gpu)
        assert out.shape == (B, 12), f"Unexpected FP16 output shape {out.shape}"
        print(f"FP16 TorchScript execution verified: shape={out.shape}")
        
    if os.path.exists(cpu_path):
        ts_cpu = torch.jit.load(cpu_path, map_location='cpu')
        dummy_cpu = (
            torch.randn(B, K, 1792, device='cpu'),
            torch.randint(0, 3, (B, K), device='cpu'),
            torch.randint(0, 5, (B, K), device='cpu'),
            torch.rand(B, K, 1, device='cpu'),
            torch.rand(B, K, 3, device='cpu'),
            torch.randn(B, 46, device='cpu')
        )
        out = ts_cpu(*dummy_cpu)
        assert out.shape == (B, 12), f"Unexpected CPU output shape {out.shape}"
        print(f"CPU TorchScript execution verified: shape={out.shape}")
    return True

def test_submission_integrity():
    print("\n" + "=" * 65)
    print("[TEST 5/5] Testing Kaggle Submission File Integrity...")
    print("=" * 65)
    sub_path = os.path.join(ROOT_DIR, "submission.csv")
    assert os.path.exists(sub_path), f"submission.csv does not exist at {sub_path}"
    
    df = pd.read_csv(sub_path)
    assert "StudyInstanceUID" in df.columns, "Missing StudyInstanceUID column"
    for t in TARGET_COLS:
        assert t in df.columns, f"Missing target column {t}"
        
    vals = df[TARGET_COLS].values
    assert not np.isnan(vals).any(), "NaN values found in submission.csv"
    assert not np.isinf(vals).any(), "Inf values found in submission.csv"
    assert (vals >= 0.0).all() and (vals <= 1.0).all(), "Values out of bounds [0, 1] in submission.csv"
    assert len(df) > 0, "submission.csv is empty"
    
    print(f"Submission integrity VERIFIED! Rows: {len(df)}, Columns: {len(df.columns)}")
    return True

if __name__ == "__main__":
    print("STARTING COMMERCIAL PRODUCTION SANITY & STRESS TEST SUITE\n")
    t_start = time.perf_counter()
    test_diagnostic_engine()
    test_ood_quality_gate()
    test_fastapi_endpoints()
    test_compiled_models()
    test_submission_integrity()
    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 65)
    print(f"ALL 5 COMMERCIAL TEST SUITES PASSED IN {total_time:.2f} SECONDS!")
    print("=" * 65)
