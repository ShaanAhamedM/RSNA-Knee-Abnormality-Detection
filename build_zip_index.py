import zipfile
import time
import os
import pandas as pd
import json

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
output_parquet = r"D:\MLprojs\KaggleCompetition\RSNA\zip_index.parquet"
output_json = r"D:\MLprojs\KaggleCompetition\RSNA\study_to_series_map.json"

print("Starting Zip Indexing...")
t0 = time.time()

with zipfile.ZipFile(zip_path, 'r') as z:
    all_names = z.namelist()

print(f"Total names: {len(all_names)} read in {time.time() - t0:.2f}s")

records = []
for path in all_names:
    if not path.endswith('.dcm'):
        continue
    # Expected format: split_name/study_id/series_id/instance_id.dcm
    parts = path.split('/')
    if len(parts) >= 4:
        split_name = parts[0]
        study_id = parts[1]
        series_id = parts[2]
        instance_id = parts[3][:-4] # strip .dcm
        records.append({
            'split': split_name,
            'study_id': study_id,
            'series_id': series_id,
            'instance_id': instance_id,
            'zip_path': path
        })
    elif len(parts) == 3:
        # e.g., study_id/series_id/instance_id.dcm
        records.append({
            'split': 'unknown',
            'study_id': parts[0],
            'series_id': parts[1],
            'instance_id': parts[2][:-4],
            'zip_path': path
        })

print(f"Parsed {len(records)} DICOM slice records.")
df_index = pd.DataFrame(records)
print(f"Saving to {output_parquet}...")
df_index.to_parquet(output_parquet, index=False)
print(f"Parquet saved! Size: {os.path.getsize(output_parquet) / (1024*1024):.2f} MB")

# Also create a quick nested dictionary: study_id -> list of series_ids -> list of zip_paths
print("Building fast nested mapping study -> series -> slices...")
study_map = {}
for study_id, study_group in df_index.groupby('study_id'):
    study_map[study_id] = {}
    for series_id, series_group in study_group.groupby('series_id'):
        study_map[study_id][series_id] = series_group['zip_path'].tolist()

with open(output_json, 'w') as f:
    json.dump(study_map, f)

print(f"JSON map saved! Size: {os.path.getsize(output_json) / (1024*1024):.2f} MB")
print(f"Total index time: {time.time() - t0:.2f}s")
print(f"Unique studies indexed: {len(study_map)}")
