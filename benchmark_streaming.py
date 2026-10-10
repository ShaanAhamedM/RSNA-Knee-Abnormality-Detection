import zipfile
import io
import time
import pandas as pd
import pydicom
import numpy as np

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
index_df = pd.read_parquet(r"D:\MLprojs\KaggleCompetition\RSNA\zip_index.parquet")
series_df = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\train_series.csv")

print(f"Total slice index rows: {len(index_df)}")
print(f"Total series metadata rows: {len(series_df)}")

# Merge slice counts per series
series_counts = index_df.groupby('series_id').size().reset_index(name='slice_count')
series_summary = pd.merge(series_df, series_counts, left_on='SeriesInstanceUID', right_on='series_id', how='left')
print("\nSlice count statistics per series:")
print(series_summary['slice_count'].describe())

print("\nSeries count per study:")
studies_per_series = series_summary.groupby('StudyInstanceUID').size()
print(studies_per_series.describe())

# Benchmark direct in-memory streaming speed for 10 slices
print("\nBenchmarking in-memory streaming speed for 10 DICOM slices...")
sample_slices = index_df['zip_path'].sample(10, random_state=42).tolist()

t0 = time.time()
with zipfile.ZipFile(zip_path, 'r') as z:
    for s_path in sample_slices:
        b = z.read(s_path)
        dcm = pydicom.dcmread(io.BytesIO(b))
        arr = dcm.pixel_array
        # print first slice info
        if s_path == sample_slices[0]:
            print(f"Sample slice: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

t1 = time.time()
print(f"Loaded and decoded 10 DICOM slices in {t1 - t0:.3f}s ({((t1 - t0)/10)*1000:.1f} ms/slice)!")
