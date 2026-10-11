import json
import time
import os
import numpy as np

input_json = r"D:\MLprojs\KaggleCompetition\RSNA\study_to_series_map.json"
output_lookup = r"D:\MLprojs\KaggleCompetition\RSNA\study_slice_lookup.json"

print("Pre-computing optimal multi-view slice paths for every study...")
t0 = time.time()

with open(input_json, 'r') as f:
    study_map = json.load(f)

print(f"Loaded {len(study_map)} studies from JSON in {time.time() - t0:.2f}s")

# We want 6 slices per study (distributed across Sagittal, Coronal, Axial series, middle 20-80% depth)
SLICES_PER_STUDY = 6
lookup = {}

for study_id, series_dict in study_map.items():
    all_series = list(series_dict.keys())
    if not all_series:
        lookup[study_id] = []
        continue
        
    selected_paths = []
    slices_per_series = max(1, SLICES_PER_STUDY // len(all_series))
    
    for sid in all_series:
        paths = sorted(series_dict[sid])
        n_paths = len(paths)
        if n_paths == 0:
            continue
            
        if n_paths <= slices_per_series:
            chosen = paths
        else:
            start_idx = int(0.20 * n_paths)
            end_idx = max(start_idx + 1, int(0.80 * n_paths))
            indices = np.linspace(start_idx, end_idx - 1, slices_per_series, dtype=int)
            chosen = [paths[i] for i in indices]
            
        selected_paths.extend(chosen)
        if len(selected_paths) >= SLICES_PER_STUDY:
            break
            
    if len(selected_paths) < SLICES_PER_STUDY:
        if selected_paths:
            selected_paths += [selected_paths[-1]] * (SLICES_PER_STUDY - len(selected_paths))
        else:
            selected_paths = []
            
    lookup[study_id] = selected_paths[:SLICES_PER_STUDY]

with open(output_lookup, 'w') as f:
    json.dump(lookup, f)

print(f"Compact lookup saved to {output_lookup} in {time.time() - t0:.2f}s!")
print(f"File size: {os.path.getsize(output_lookup) / (1024*1024):.2f} MB")
