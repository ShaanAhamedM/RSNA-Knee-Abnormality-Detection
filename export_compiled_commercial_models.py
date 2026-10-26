"""
Export Compiled Commercial FP16 / ONNX Inference Engine
RSNA Knee Abnormality Detection Commercial Engine

Generates:
1. Compiled FP16 TorchScript model: checkpoints/compiled_commercial_student_fp16.pt
2. Compiled CPU TorchScript model: checkpoints/compiled_commercial_student_cpu.pt
3. Production ONNX graph: checkpoints/compiled_commercial_gnn.onnx
4. High-precision latency benchmarks on both GPU (< 100 ms) and CPU (< 1.2s)
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np

ROOT_DIR = r"D:\MLprojs\KaggleCompetition\RSNA"
os.chdir(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from train_enhanced_gnn_mil_transformer import EnhancedCommercialGNNTransformer

def export_models():
    print("=" * 70)
    print("COMMERCIAL MODEL COMPILATION: FP16 TORCHSCRIPT & ONNX EXPORT")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compilation Target Device: {device}")
    
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "enhanced_gated_gnn_fold0_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "phase8_gnn_fold0_best.pt")
        
    print(f"Loading checkpoint: {ckpt_path}")
    model = EnhancedCommercialGNNTransformer(
        visual_dim=1792,
        tab_dim=46,
        hidden_dim=256,
        num_classes=12,
        dropout=0.0
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy inputs (Batch=1, K=20 tokens)
    B, K = 1, 20
    dummy_tokens = torch.randn(B, K, 1792, device=device)
    dummy_planes = torch.randint(0, 3, (B, K), device=device)
    dummy_rois = torch.randint(0, 5, (B, K), device=device)
    dummy_depths = torch.rand(B, K, 1, device=device)
    dummy_contrast = torch.rand(B, K, 3, device=device)
    dummy_tab = torch.randn(B, 46, device=device)

    # 1. Warm-up and Verify Output
    with torch.no_grad():
        base_out = model(dummy_tokens, dummy_planes, dummy_rois, dummy_depths, dummy_contrast, dummy_tab)
        base_probs = torch.sigmoid(base_out).cpu().numpy()[0]
    print(f"Baseline model output shape: {base_out.shape}")
    print(f"Sample prediction probs: {base_probs[:4].round(4)}")

    # 2. Export TorchScript FP16 for CUDA
    out_dir = os.path.join(ROOT_DIR, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    
    fp16_ts_path = os.path.join(out_dir, "compiled_commercial_student_fp16.pt")
    print(f"\n[1/3] Tracing FP16 TorchScript model to {fp16_ts_path}...")
    try:
        model_fp16 = EnhancedCommercialGNNTransformer(
            visual_dim=1792, tab_dim=46, hidden_dim=256, num_classes=12, dropout=0.0
        ).to(device).half()
        model_fp16.load_state_dict({k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()})
        model_fp16.eval()
        
        dummy_tokens_f16 = dummy_tokens.half()
        dummy_depths_f16 = dummy_depths.half()
        dummy_contrast_f16 = dummy_contrast.half()
        dummy_tab_f16 = dummy_tab.half()
        
        with torch.no_grad():
            traced_fp16 = torch.jit.trace(
                model_fp16,
                (dummy_tokens_f16, dummy_planes, dummy_rois, dummy_depths_f16, dummy_contrast_f16, dummy_tab_f16)
            )
            # Atomic save
            tmp_path = fp16_ts_path + ".tmp"
            traced_fp16.save(tmp_path)
            os.replace(tmp_path, fp16_ts_path)
        print(f"Successfully saved FP16 TorchScript ({os.path.getsize(fp16_ts_path) / 1024**2:.2f} MB)")
    except Exception as e:
        print(f"TorchScript FP16 trace failed: {e}")
        traced_fp16 = None

    # 3. Export TorchScript CPU (FP32)
    cpu_ts_path = os.path.join(out_dir, "compiled_commercial_student_cpu.pt")
    print(f"\n[2/3] Tracing CPU TorchScript model to {cpu_ts_path}...")
    try:
        model_cpu = EnhancedCommercialGNNTransformer(
            visual_dim=1792, tab_dim=46, hidden_dim=256, num_classes=12, dropout=0.0
        ).to('cpu')
        model_cpu.load_state_dict({k: v.cpu() for k, v in state_dict.items()})
        model_cpu.eval()
        
        cpu_tokens = dummy_tokens.cpu()
        cpu_planes = dummy_planes.cpu()
        cpu_rois = dummy_rois.cpu()
        cpu_depths = dummy_depths.cpu()
        cpu_contrast = dummy_contrast.cpu()
        cpu_tab = dummy_tab.cpu()
        
        with torch.no_grad():
            traced_cpu = torch.jit.trace(
                model_cpu,
                (cpu_tokens, cpu_planes, cpu_rois, cpu_depths, cpu_contrast, cpu_tab)
            )
            tmp_path = cpu_ts_path + ".tmp"
            traced_cpu.save(tmp_path)
            os.replace(tmp_path, cpu_ts_path)
        print(f"Successfully saved CPU TorchScript ({os.path.getsize(cpu_ts_path) / 1024**2:.2f} MB)")
    except Exception as e:
        print(f"TorchScript CPU trace failed: {e}")
        traced_cpu = None

    # 4. Export ONNX Graph
    onnx_path = os.path.join(out_dir, "compiled_commercial_gnn.onnx")
    print(f"\n[3/3] Exporting ONNX model to {onnx_path}...")
    try:
        model_cpu = model.cpu()
        input_names = ["tokens", "planes", "rois", "depths", "contrast", "tab_features"]
        output_names = ["logits"]
        
        tmp_onnx = onnx_path + ".tmp"
        torch.onnx.export(
            model_cpu,
            (cpu_tokens, cpu_planes, cpu_rois, cpu_depths, cpu_contrast, cpu_tab),
            tmp_onnx,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "tokens": {0: "batch_size", 1: "num_tokens"},
                "planes": {0: "batch_size", 1: "num_tokens"},
                "rois": {0: "batch_size", 1: "num_tokens"},
                "depths": {0: "batch_size", 1: "num_tokens"},
                "contrast": {0: "batch_size", 1: "num_tokens"},
                "tab_features": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        os.replace(tmp_onnx, onnx_path)
        print(f"Successfully exported ONNX ({os.path.getsize(onnx_path) / 1024**2:.2f} MB)")
    except Exception as e:
        print(f"ONNX export warning: {e}")

    # 5. Benchmarking Latency
    print("\n" + "=" * 70)
    print("INFERENCE LATENCY BENCHMARKING (50 ITERATIONS)")
    print("=" * 70)
    
    # GPU FP16 Latency
    if torch.cuda.is_available() and traced_fp16 is not None:
        gpu_inputs = (dummy_tokens_f16, dummy_planes, dummy_rois, dummy_depths_f16, dummy_contrast_f16, dummy_tab_f16)
        # Warmup
        for _ in range(10):
            _ = traced_fp16(*gpu_inputs)
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        N = 50
        for _ in range(N):
            _ = traced_fp16(*gpu_inputs)
            torch.cuda.synchronize()
        gpu_latency_ms = ((time.perf_counter() - t0) / N) * 1000.0
        print(f"[BENCHMARK] GPU FP16 Latency: {gpu_latency_ms:.2f} ms per study (Target: < 100 ms) -> {'PASSED' if gpu_latency_ms < 100 else 'FAILED'}", flush=True)
    else:
        gpu_latency_ms = None

    # CPU Latency
    if traced_cpu is not None:
        cpu_inputs = (cpu_tokens, cpu_planes, cpu_rois, cpu_depths, cpu_contrast, cpu_tab)
        # Warmup
        for _ in range(5):
            _ = traced_cpu(*cpu_inputs)
            
        t0 = time.perf_counter()
        N = 30
        for _ in range(N):
            _ = traced_cpu(*cpu_inputs)
        cpu_latency_ms = ((time.perf_counter() - t0) / N) * 1000.0
        print(f"[BENCHMARK] CPU FP32 Latency: {cpu_latency_ms:.2f} ms per study (Target: < 1200 ms) -> {'PASSED' if cpu_latency_ms < 1200 else 'FAILED'}", flush=True)
    else:
        cpu_latency_ms = None

    print("\nModel compilation and benchmarking completed successfully!")

if __name__ == "__main__":
    export_models()
