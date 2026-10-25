import os
import glob
import zipfile
import json
import torch

checkpoints_dir = r"D:\MLprojs\KaggleCompetition\RSNA\checkpoints"
oof_dir = r"D:\MLprojs\KaggleCompetition\RSNA\oof"
output_dir = r"D:\MLprojs\KaggleCompetition\RSNA\kaggle_submission_package"
zip_path = os.path.join(output_dir, "best_ensemble_weights_fp16.zip")

print("=" * 80)
print("PACKAGING FP16 OPTIMIZED KAGGLE ENSEMBLE WEIGHTS ARCHIVE")
print("=" * 80)

# Load grandmaster parameters
params_path = os.path.join(oof_dir, "grandmaster_params.json")
assert os.path.exists(params_path), f"Parameters file {params_path} not found!"

with open(params_path, 'r') as f:
    gm_params = json.load(f)

kept_models = gm_params['kept_models']
kept_folds = gm_params['kept_folds']

# Find all kept checkpoints + any report_sup checkpoints
checkpoints_to_pack = []
for m in kept_models:
    for fold in kept_folds[m]:
        cp = os.path.join(checkpoints_dir, f"{m}_fold{fold}_best.pt")
        if os.path.exists(cp):
            checkpoints_to_pack.append(cp)

# Also check for report-supervised checkpoints
report_ckpts = glob.glob(os.path.join(checkpoints_dir, "*_report_sup_*_best.pt"))
checkpoints_to_pack.extend(report_ckpts)

print(f"Total checkpoints to bundle: {len(checkpoints_to_pack)}")
for cp in checkpoints_to_pack:
    print(f"  - {os.path.basename(cp)} ({os.path.getsize(cp) / (1024*1024):.1f} MB)")

# Build zip archive with float16 weights
temp_dir = os.path.join(output_dir, "temp_fp16")
os.makedirs(temp_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    # Add grandmaster_params.json
    z.write(params_path, arcname="grandmaster_params.json")
    print("\nAdded grandmaster_params.json to archive.")
    
    # Process and add checkpoints
    for cp in checkpoints_to_pack:
        fname = os.path.basename(cp)
        print(f"Processing {fname} -> converting to FP16...")
        state_dict = torch.load(cp, map_location='cpu')
        
        # Convert float32 tensors to float16
        fp16_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                fp16_state_dict[k] = v.half()
            else:
                fp16_state_dict[k] = v
                
        temp_fp = os.path.join(temp_dir, fname)
        torch.save(fp16_state_dict, temp_fp)
        z.write(temp_fp, arcname=fname)
        
        orig_mb = os.path.getsize(cp) / (1024*1024)
        fp16_mb = os.path.getsize(temp_fp) / (1024*1024)
        print(f"  -> Compressed: {orig_mb:.1f} MB -> {fp16_mb:.1f} MB")
        os.remove(temp_fp)

try:
    os.rmdir(temp_dir)
except Exception:
    pass

final_zip_size = os.path.getsize(zip_path) / (1024*1024)
print("\n" + "=" * 80)
print(f"[SUCCESS] Kaggle Dataset Archive Created: {zip_path}")
print(f"Total Archive Size: {final_zip_size:.1f} MB")
print("=" * 80)
