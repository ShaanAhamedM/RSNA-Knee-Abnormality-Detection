"""
Thermal & Hardware Governor for RSNA Commercial Engine (RTX 3050 6GB Laptop)
Strictly enforces:
1. Live nvidia-smi Thermal Throttle Guard (MAX_TEMP = 78C, resume <= 70C)
2. Cooldown breathers between stages and folds (45s)
3. Zero-loss atomic checkpointing via .tmp and os.replace
4. Persistent overnight run state tracking in overnight_commercial_state.json
5. CPU thread limitation (torch.set_num_threads(4)) to prevent shared heat-pipe CPU throttling
"""

import os
import sys
import time
import json
import subprocess
import torch

STATE_FILE = r"D:\MLprojs\KaggleCompetition\RSNA\overnight_commercial_state.json"

# Limit CPU threads to avoid CPU overheating on laptop shared heat-pipe
torch.set_num_threads(4)

def get_gpu_telemetry():
    """Returns (temperature_c, power_watts, memory_used_mb) via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        parts = [x.strip() for x in out.split(",")]
        temp = float(parts[0])
        pwr = float(parts[1]) if len(parts) > 1 and parts[1] != "[Not Supported]" else 0.0
        mem = float(parts[2]) if len(parts) > 2 else 0.0
        return temp, pwr, mem
    except Exception as e:
        return 50.0, 0.0, 0.0

def check_gpu_thermals(max_temp=78, resume_temp=70, check_interval=25, batch_idx=None, verbose=True):
    """
    Called every check_interval batches during training/inference.
    If GPU temperature >= max_temp, pauses until cooled down to resume_temp.
    """
    if batch_idx is not None and (batch_idx % check_interval != 0):
        return
        
    temp, pwr, mem = get_gpu_telemetry()
    if temp >= max_temp:
        print(f"\n[THERMAL GUARD TRIGGERED] GPU Temp={temp:.1f}C >= {max_temp}C | Power={pwr:.1f}W | Mem={mem:.0f}MB")
        print(f"Pausing execution and synchronizing CUDA until temperature drops <= {resume_temp}C...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
        pause_count = 0
        while temp > resume_temp:
            time.sleep(15)
            pause_count += 15
            temp, pwr, mem = get_gpu_telemetry()
            print(f"  [Cooling {pause_count}s] Current GPU Temp={temp:.1f}C (target <= {resume_temp}C)...")
            
        print(f"[THERMAL GUARD RESOLVED] GPU Cooled to {temp:.1f}C. Resuming execution safely.\n")

def thermal_cooldown_breather(seconds=45, reason="Stage/Fold Transition"):
    """
    Mandatory cooldown breather between folds, epochs, or pipeline stages
    to allow laptop heat pipes to dissipate accumulated heat.
    """
    print(f"\n{'='*70}\n[THERMAL BREATHER: {seconds}s] {reason}")
    temp, pwr, mem = get_gpu_telemetry()
    print(f"Starting Breather: Temp={temp:.1f}C | Power={pwr:.1f}W | Mem={mem:.0f}MB")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(seconds)
    temp, pwr, mem = get_gpu_telemetry()
    print(f"Ending Breather:   Temp={temp:.1f}C | Power={pwr:.1f}W | Mem={mem:.0f}MB\n{'='*70}\n")

def atomic_torch_save(obj, target_path):
    """
    Atomic zero-loss checkpointing.
    Writes to target_path + '.tmp', flushes, and atomically renames.
    """
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = target_path + ".tmp"
    torch.save(obj, tmp_path)
    # Flush and atomic replace
    if os.path.exists(target_path):
        os.replace(tmp_path, target_path)
    else:
        os.rename(tmp_path, target_path)

def load_or_create_state():
    """Loads persistent state file or creates a new one."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "mission_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_update_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phases_completed": {},
        "current_phase": "Phase 1: Clinical NLP Teacher",
        "highest_macro_auc": 0.7558,
        "history": []
    }

def update_state(phase_name, status, metrics=None, notes=None):
    """Atomically updates state file."""
    state = load_or_create_state()
    state["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["current_phase"] = phase_name
    
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase_name,
        "status": status,
        "metrics": metrics or {},
        "notes": notes or ""
    }
    state["history"].append(entry)
    if status == "COMPLETED":
        state["phases_completed"][phase_name] = {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics or {}
        }
        if metrics and "macro_auc" in metrics:
            if metrics["macro_auc"] > state.get("highest_macro_auc", 0.0):
                state["highest_macro_auc"] = metrics["macro_auc"]
                
    tmp_file = STATE_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_file, STATE_FILE)
    print(f"[STATE SAVED] Phase='{phase_name}' Status='{status}' Highest AUC={state['highest_macro_auc']:.4f}")

if __name__ == "__main__":
    t, p, m = get_gpu_telemetry()
    print(f"Telemetry test: Temp={t}C, Pwr={p}W, Mem={m}MB")
    check_gpu_thermals(batch_idx=0)
    update_state("Initialization", "COMPLETED", {"init_gpu_temp": t}, "Thermal governor verified")
