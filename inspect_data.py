import os
import pandas as pd
import numpy as np

print("--- Inspecting train.csv ---")
df = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\train.csv")
print(f"Shape: {df.shape}")
print(f"Unique StudyInstanceUID: {df['StudyInstanceUID'].nunique()}")

target_cols = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("\n--- Missing / Non-null Counts ---")
print(df[target_cols].notnull().sum())

print("\n--- Positive Label Sums (fillna(0)) ---")
df_filled = df[target_cols].fillna(0)
print(df_filled.sum())

print("\n--- Positive Rates (%) ---")
print((df_filled.mean() * 100).round(2))

print("\n--- Inspecting train_series.csv ---")
series_df = pd.read_csv(r"D:\MLprojs\KaggleCompetition\RSNA\train_series.csv")
print(f"Shape: {series_df.shape}")
print(f"Columns: {list(series_df.columns)}")
print(f"Unique Studies in series: {series_df['StudyInstanceUID'].nunique()}")
print(f"Unique Series: {series_df['SeriesInstanceUID'].nunique()}")
print("\nAnatomical_Plane value counts:")
print(series_df['Anatomical_Plane'].value_counts(dropna=False))
print("\nFluid_Sensitive value counts:")
print(series_df['Fluid_Sensitive'].value_counts(dropna=False))
print("\nFat_Suppression value counts:")
print(series_df['Fat_Suppression'].value_counts(dropna=False))
